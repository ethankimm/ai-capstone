#!/usr/bin/env python3
"""Laptop-side watcher for the pods running scripts/pod/codi_stab_arm.sh. Runs on its own (nohup), so it keeps working
when no Claude session is active. Every INTERVAL seconds, per live arm:
  * rsyncs checkpoints (with optimizer.pt, so a run can resume), logs and result records to DEST/<kind>/;
  * alerts on an unreachable pod, a crash, or the watchdog firing;
  * on /workspace/ARM_DONE or ARM_FAILED: final sync, verifies the sha256 of the newest or selected checkpoint against the
    pod, then terminates the pod so it stops billing. If verification fails the pod is kept and an alert raised.
Budget: projected need = sum over live arms of rate x (remaining steps x s/it + EVAL_H). If the balance is below
need + MARGIN it raises BUDGET_LOW (checkpoints are already synced at that point, so nothing beyond the current epoch
is at risk). Lines starting ALERT or ALL_DONE are what a waiting session looks for.

  nohup python3 scripts/pod/pair_watcher.py <dest> codi:<ip>:<port>:<pod_id>:<rate> codi0:<ip>:<port>:<pod_id>:<rate> >> <dest>/watcher.log 2>&1 &
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

INTERVAL, EVAL_H, MARGIN, TOTAL = 600, 0.45, 0.5, 17820
RUN = "/workspace/codi_runs/gpt2_l1to7w_stab_{k}/gpt2/ep_12/lr_0.002/seed_11"
LOG = "/workspace/logs/gpt2_l1to7w_stab_{k}.log"


def say(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def alert(msg: str) -> None:
    say(f"ALERT {msg}")
    subprocess.run(["osascript", "-e", f'display notification "{msg}" with title "RunPod watcher"'], capture_output=True)


def sh(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


class Arm:
    def __init__(self, spec: str, dest: Path):
        self.kind, self.host, self.port, self.pod, rate = spec.split(":")
        self.rate, self.dest, self.live, self.fails = float(rate), dest / self.kind, True, 0
        self.step, self.sit = 0, 2.2
        self.dest.mkdir(parents=True, exist_ok=True)
        self.e = f"ssh -p {self.port} -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"

    def ssh(self, cmd: str) -> tuple[int, str]:
        return sh(self.e.split() + [f"root@{self.host}", cmd], 120)

    def sync(self) -> bool:
        ok = True
        for src, sub in ((RUN.format(k=self.kind) + "/", "run/"), ("/workspace/logs/", "logs/"),
                         ("/workspace/ai-capstone/results/", "results/")):
            (self.dest / sub).mkdir(exist_ok=True)
            rc, out = sh(["rsync", "-a", "--partial", "-e", self.e, f"root@{self.host}:{src}", str(self.dest / sub)])
            if rc not in (0, 23, 24):  # 23: the run dir may not exist yet
                ok = False
                say(f"{self.kind} rsync {sub} rc={rc}: {out[-200:]}")
        return ok

    def poll(self) -> str:
        rc, out = self.ssh(f"tail -c 4000 {LOG.format(k=self.kind)} 2>/dev/null | tr '\\r' '\\n' | grep -E 'it\\]$' | tail -1;"
                           " echo; cat /workspace/ARM_DONE 2>/dev/null | sed 's/^/DONE /';"
                           " cat /workspace/ARM_FAILED 2>/dev/null | sed 's/^/FAILED /';"
                           f" cat {LOG.format(k=self.kind)}.watchdog 2>/dev/null | sed 's/^/WD /'; pgrep -f '[c]odi_stab_arm' >/dev/null && echo ARM_UP")
        if rc != 0:
            self.fails += 1
            if self.fails >= 3:
                alert(f"{self.kind} pod unreachable {self.fails} checks in a row")
            return ""
        self.fails = 0
        m = re.search(r"(\d+)/17820 \[.*?([0-9.]+)(s/it|it/s)\]", out)
        if m:
            self.step = int(m.group(1))
            self.sit = float(m.group(2)) if m.group(3) == "s/it" else 1 / float(m.group(2))
        return out

    def finish(self, out: str) -> None:
        good = self.sync()
        rc, sel = self.ssh("cat /workspace/SELECTED 2>/dev/null")
        rc2, rem = self.ssh(f"cd {RUN.format(k=self.kind)} && c=checkpoint-{sel.strip()}; [ -d $c ] || c=$(ls -d checkpoint-* | sort -V | tail -1);"
                            " echo $c; sha256sum $c/pytorch_model.bin | cut -d' ' -f1")
        parts = rem.split()
        same = False
        if rc2 == 0 and len(parts) == 2:
            _, loc = sh(["shasum", "-a", "256", str(self.dest / "run" / parts[0] / "pytorch_model.bin")], 300)
            same = loc.split()[:1] == [parts[1]]
        if not (good and same):
            alert(f"{self.kind} final sync/verify failed, pod {self.pod} kept billing")
            return
        rc, o = sh(["runpodctl", "pod", "delete", self.pod], 120)
        say(f"{self.kind} verified {parts[0]}, terminate rc={rc} {o[-120:].strip()}")
        if rc != 0:
            alert(f"{self.kind} terminate failed for {self.pod}")
            return
        self.live = False


def main() -> None:
    dest = Path(sys.argv[1]).expanduser()
    arms = [Arm(s, dest) for s in sys.argv[2:]]
    last_budget_alert = 0.0
    say(f"watcher up: {[a.kind for a in arms]}")
    while any(a.live for a in arms):
        for a in (x for x in arms if x.live):
            out = a.poll()
            if not out:
                continue
            if "\nFAILED " in "\n" + out or "\nWD " in "\n" + out:
                alert(f"{a.kind} failed: {[l for l in out.splitlines() if l.startswith(('FAILED', 'WD'))][:2]}")
                a.finish(out)
            elif "\nDONE " in "\n" + out:
                say(f"{a.kind} arm finished")
                a.finish(out)
            elif "ARM_UP" not in out:
                alert(f"{a.kind} arm script not running and no end marker")
                a.sync()
            else:
                a.sync()
        rc, u = sh(["runpodctl", "user"], 60)
        try:
            bal = float(json.loads(u)["clientBalance"])
        except (ValueError, KeyError):
            say(f"balance read failed: {u[:120]}")
            bal = None
        need = sum(a.rate * ((TOTAL - a.step) * a.sit / 3600 + EVAL_H) for a in arms if a.live)
        state = " ".join(f"{a.kind} {a.step}/{TOTAL} {a.sit:.2f}s/it{'' if a.live else ' (ended)'}" for a in arms)
        say(f"STATUS {state} | balance {bal} need {need:.2f}")
        if bal is not None and bal < need + MARGIN and time.time() - last_budget_alert > 3600:
            alert(f"BUDGET_LOW balance {bal:.2f} < projected need {need:.2f} + {MARGIN}: top up or stop an arm; checkpoints synced")
            last_budget_alert = time.time()
        if any(a.live for a in arms):
            time.sleep(INTERVAL)
    say("ALL_DONE")


if __name__ == "__main__":
    main()
