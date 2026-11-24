"""t58 pilot: single-pass reach of pretrained GPT-2 fine-tuned WITHOUT CoT on two candidate third tasks.
Pre-registered in scripts/codi_null_audit/prereg/t_predictions.md (t58).

  ../codi/.venv/bin/python scripts/necessity/third_task_pilot.py <task sn|brew> <device> <out json> [n_per_k] [epochs]

sn    the S5 word problem over a fixed generating set (Liu et al. 2023; Merrill et al. 2024): k tokens, each one of 6 fixed
      generator permutations of 5 elements (single-token words, verified to generate all of S5); the answer is the single-token label
      of their product (120 labels, applied left to right). Meanings are fixed across items (720 transitions to learn).
brew  permutations given in context: 10 colours, 3 ingredients per item, each a random derangement of the colours printed as a rule
      table; a start colour and k stirs; the answer is the final colour. Our own generator, following the S_n word problem with
      in-context generators (all states on the path distinct, last-ingredient guess rejected, 8-reordering order filter).
k = 1..8, n_per_k training items per k pooled, 300 test items per k. Full fine-tune of HF gpt2, loss on the single answer token read
at the last prompt position; accuracy = argmax over the task's label set. Train accuracy per k is logged to detect underfitting.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

TASK, DEV, OUT = sys.argv[1:4]
NPK = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
KS = range(1, 9)
tok = GPT2TokenizerFast.from_pretrained('gpt2')
single = lambda w: len(tok.encode(' ' + w)) == 1

# ---------------------------------------------------------------- S5 word problem (fixed labels)
PERMS = list(itertools.permutations(range(5)))  # 120, lexicographic; PERMS[0] is the identity
def _labels():
    cand = sorted({tok.convert_tokens_to_string([t]).strip() for t in tok.get_vocab() if t.startswith('Ġ') and t[1:].isalpha()
                   and t[1:].islower() and 5 <= len(t) - 1 <= 7})
    cand = [w for w in cand if single(w)]
    return random.Random('t58-labels').sample(cand, 120)
SN_LAB = _labels() if TASK == 'sn' else None
compose = lambda p, q: tuple(q[p[i]] for i in range(5))  # apply p, then q


def _gens():
    r = random.Random('t58-gens')
    while True:
        g = [r.randrange(1, 120) for _ in range(6)]
        if len(set(g)) < 6: continue
        seen = {PERMS[0]}; frontier = [PERMS[0]]
        while frontier:
            nxt = []
            for p in frontier:
                for i in g:
                    q = compose(p, PERMS[i])
                    if q not in seen: seen.add(q); nxt.append(q)
            frontier = nxt
        if len(seen) == 120: return g
GENS = _gens() if TASK == 'sn' else None
GEN_LAB = SN_LAB[:6] if TASK == 'sn' else None   # generator tokens reuse the first six labels as words for the inputs


def sn_item(r, k):
    while True:
        idx = [GENS[r.randrange(6)] for _ in range(k)]
        prod = PERMS[0]
        for i in idx: prod = compose(prod, PERMS[i])
        if k >= 2:  # order filter: some reordering must change the product (8 tries)
            same = 0
            for _ in range(8):
                j = idx[:]; r.shuffle(j); q = PERMS[0]
                for i in j: q = compose(q, PERMS[i])
                same += q == prod
            if same == 8: continue
        a = PERMS.index(prod)
        return ' '.join(GEN_LAB[GENS.index(i)] for i in idx), SN_LAB[a]


def sn_prompt(seq): return f"Compose these permutations in order: {seq}. The product is:"

# ---------------------------------------------------------------- in-context permutations (brew surface, own generator)
COL = ('red', 'blue', 'green', 'gold', 'pink', 'gray', 'brown', 'black', 'white', 'purple')
ING = ('salt', 'ash', 'moss', 'clay', 'bark', 'sand', 'mint', 'chalk')


def derange(r, n):
    while True:
        p = list(range(n)); r.shuffle(p)
        if all(p[i] != i for i in range(n)): return p


def brew_item(r, k):
    while True:
        ings = r.sample(ING, 3); perm = {g: derange(r, 10) for g in ings}; s0 = r.randrange(10)
        seq = [r.choice(ings) for _ in range(k)]
        if k > 1 and len(set(seq)) < 2: continue
        st = [s0]
        for g in seq: st.append(perm[g][st[-1]])
        if len(set(st)) != k + 1: continue
        if k > 1 and perm[seq[-1]][s0] == st[-1]: continue
        bad = False
        for _ in range(8):
            q = seq[:]; r.shuffle(q)
            if q == seq: continue
            v = s0
            for g in q: v = perm[g][v]
            if v == st[-1]: bad = True; break
        if bad: continue
        order = list(range(10)); r.shuffle(order)
        rules = ' '.join(f"A {COL[c]} potion turns {COL[perm[ings[0]][c]]} with {ings[0]}, {COL[perm[ings[1]][c]]} with {ings[1]}, "
                         f"and {COL[perm[ings[2]][c]]} with {ings[2]}." for c in order)
        return (rules, ', then '.join(seq), COL[s0]), COL[st[-1]]


def brew_prompt(x):
    rules, seq, start = x
    return f"{rules} You stir in, one at a time: {seq}. The potion starts out {start}. What color is the potion at the end? Answer:"

# ---------------------------------------------------------------- data
item, prompt, labels = (sn_item, sn_prompt, SN_LAB) if TASK == 'sn' else (brew_item, brew_prompt, list(COL))
LID = torch.tensor([tok.encode(' ' + w)[0] for w in labels])
train, test = [], {}
for k in KS:
    r = random.Random(f't58-{TASK}-{k}')
    test[k] = [(k, *item(r, k)) for _ in range(300)]
    # keep test items out of training only where the item space is large (sn has 6^k sequences; brew is effectively unbounded)
    seen = {prompt(x) for _, x, _ in test[k]} if (TASK == 'brew' or 6 ** k >= 20 * NPK) else set()
    got = 0
    while got < NPK:
        x, a = item(r, k)
        if prompt(x) in seen: continue
        train.append((k, x, a)); got += 1
random.Random(f't58-{TASK}-shuffle').shuffle(train)
L = lambda rows: max(len(tok.encode(prompt(x))) for _, x, _ in rows)
print(TASK, 'train', len(train), 'max prompt tokens', L(train), flush=True)


def batch(rows):
    enc = [tok.encode(prompt(x)) for _, x, _ in rows]; n = max(map(len, enc))
    ids = torch.full((len(rows), n), tok.eos_token_id); m = torch.zeros((len(rows), n), dtype=torch.long)
    for i, e in enumerate(enc): ids[i, :len(e)] = torch.tensor(e); m[i, :len(e)] = 1
    last = torch.tensor([len(e) - 1 for e in enc]); y = torch.tensor([tok.encode(' ' + a)[0] for _, _, a in rows])
    return ids.to(DEV), m.to(DEV), last.to(DEV), y.to(DEV)


model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
BS = 16; steps = EPOCHS * (len(train) // BS)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * max(0.0, 1 - s / steps))


@torch.no_grad()
def acc(rows):
    model.eval(); out = {}
    for k in KS:
        rs = [x for x in rows if x[0] == k]; right = 0
        for i in range(0, len(rs), 32):
            ids, m, last, y = batch(rs[i:i + 32]); lg = model(input_ids=ids, attention_mask=m).logits
            lg = lg[torch.arange(len(last)), last][:, LID.to(DEV)]
            right += int((LID.to(DEV)[lg.argmax(-1)] == y).sum())
        out[k] = round(right / max(1, len(rs)), 3)
    model.train(); return out


res = {}; t0 = time.time(); trsub = [x for k in KS for x in [r for r in train if r[0] == k][:300]]
for ep in range(EPOCHS):
    model.train()
    for i in range(0, len(train) - BS + 1, BS):
        ids, m, last, y = batch(train[i:i + BS])
        lg = model(input_ids=ids, attention_mask=m).logits[torch.arange(len(last)), last]
        loss = torch.nn.functional.cross_entropy(lg, y); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if DEV == 'mps' and i % (BS * 200) == 0: torch.mps.empty_cache()
    res[ep] = dict(test=acc([x for k in KS for x in test[k]]), train=acc(trsub))
    print(TASK, 'ep', ep, f'{time.time() - t0:.0f}s', 'test', res[ep]['test'], 'train', res[ep]['train'], flush=True)
json.dump(dict(task=TASK, n_per_k=NPK, epochs=EPOCHS, chance=1 / len(labels), by_epoch=res), open(OUT, 'w'), indent=1)
print('DONE')
