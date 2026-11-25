"""Score t58 pilot JSONs with the pre-registered gate: reach = interpolated k at which final-epoch test accuracy crosses the midpoint
between chance and accuracy at k = 1; PASS if acc(k=1) >= .90 and 1.5 <= reach <= 3.5.
  python scripts/necessity/third_task_score.py <json> [...]     (prints one line per file; exit 0 iff all files pass)"""
import json, sys


def score(path):
    d = json.load(open(path)); ep = d['by_epoch'][max(d['by_epoch'], key=int)]; t = {int(k): v for k, v in ep['test'].items()}
    a1, mid = t[1], (d['chance'] + t[1]) / 2; reach = None
    for k in range(2, 9):
        if t[k] < mid: reach = (k - 1) + (t[k - 1] - mid) / (t[k - 1] - t[k]); break
    passed = a1 >= .90 and reach is not None and 1.5 <= reach <= 3.5
    return dict(task=d['task'], acc1=a1, reach=round(reach, 2) if reach else '>8', passed=passed, test=t,
                train={int(k): v for k, v in ep['train'].items()}, path=ep.get('test_path'))


if __name__ == '__main__':
    ok = True
    for p in sys.argv[1:]:
        s = score(p); ok &= s['passed']; print(json.dumps(s))
    sys.exit(0 if ok else 1)
