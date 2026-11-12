"""t27a: more GSM8K twin pairs whose edit sits >= MIN_S steps before the answer (s = chain_len - step), for the hop count.

  uv run python scripts/necessity/gsm8k_long_twins.py <out json> [MIN_S=4]

Same generator as the t22 candidates (latentreasoning.data.minimal_pairs.generate_minimal_pair: perturb one question number that feeds
step `step`, re-execute the gold chain), but every delta in (1, -1, 2, -2, 3, -3) that gives a valid twin is kept (not only the first),
on GSM8K test only (the released models trained on GSM8K-Aug train). Twins whose answer equals the original's are dropped; twin questions
already in the t22 candidate file are dropped so the two sets do not overlap. Output rows have the t22 candidate format.
"""
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from latentreasoning.data.gsm8k_aug import load_gsm8k_aug  # noqa: E402
from latentreasoning.data.minimal_pairs import generate_minimal_pair  # noqa: E402
from latentreasoning.eval.counterfactual import parse_steps  # noqa: E402

out = sys.argv[1]; MIN_S = int(sys.argv[2]) if len(sys.argv) > 2 else 4
old = json.load(open(REPO.parent / 'codi_audit_artifacts/t22/gsm8k_twin_candidates.json')); seen = {c['twin_question'] for c in old}
ex = load_gsm8k_aug(split='test', n=None, seed=None); rng = random.Random(5980); rows = []
for e in ex:
    steps = parse_steps(e.rationale); n = len(steps)
    for step in range(n):
        if n - step < MIN_S: continue
        for delta in (1, -1, 2, -2, 3, -3):
            mp = generate_minimal_pair(e, step, delta, rng=rng)
            if mp is None or mp.twin.question in seen or str(mp.twin.answer) == str(e.answer): continue
            seen.add(mp.twin.question)
            rows.append(dict(recipient_idx=e.idx, step=mp.step, delta=mp.delta, perturbed_number=mp.perturbed_number, chain_len=n,
                             recipient_question=e.question, recipient_gold=str(e.answer), twin_question=mp.twin.question,
                             twin_answer=str(mp.twin.answer)))
json.dump(rows, open(out, 'w'))
from collections import Counter  # noqa: E402
print(len(rows), 'pairs; by s:', sorted(Counter(r['chain_len'] - r['step'] for r in rows).items()))
