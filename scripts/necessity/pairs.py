"""Minimal-pair generators for the route split. No model deps (pure Python; tokenizer passed in for the length check).

A pair = (recipient row, donor row, donor-implied answer). The donor differs from the recipient by one relation on the gold path
("hop k"), rewired to another concept named in the same question, so that the donor's answer differs from the recipient's.
Donor questions keep the recipient's token length (cache positions line up for the entries swap); edits that change it are skipped.

ProsQA ("Every X is a Y." / "Name is a Y." / "Is Name a A or B?"): answer = the option reachable from Name. Donor: edge on the path
  rewired so the other option is reachable and the gold one is not.
ProntoQA ("Every/Each X is a Y.", "Xes are Ys.", "... is [not] P.", "Name is a X.", "True or false: Name is [not] P."): answer by forward
  chaining from Name's concepts. Donor: path relation rewired to a concept whose chain gives the opposite polarity for P.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict, deque


def sentences(text):
    return [s for s in re.split(r'(?<=[.?:])\s+', text.strip()) if s]


# ---------------------------------------------------------------- ProsQA
PQ_EDGE = re.compile(r'^(Every (\w+)|([A-Z]\w*)) is an? (\w+)\.$')
PQ_Q = re.compile(r'^Is (\w+) an? (\w+) or (\w+)\?$')


def prosqa_parse(q):
    """-> (edges list of (src, dst, sentence index), name, opt1, opt2, sentence list)."""
    S = sentences(q); E = []
    for i, s in enumerate(S[:-1]):
        m = PQ_EDGE.match(s); assert m, s
        E.append(((m.group(2) or m.group(3)), m.group(4), i))
    m = PQ_Q.match(S[-1]); assert m, S[-1]
    return E, m.group(1), m.group(2), m.group(3), S


def reach(edges, src):
    adj = defaultdict(list)
    for a, b, _ in edges: adj[a].append(b)
    seen = {src}; dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in seen: seen.add(v); dq.append(v)
    return seen


def frontiers(edges, src, k_max=8):
    """Nodes at BFS distance exactly k from src, k = 0..k_max (Coconut's BFS claim: latent k ~ frontier k)."""
    adj = defaultdict(list)
    for a, b, _ in edges: adj[a].append(b)
    dist = {src: 0}; dq = deque([src])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in dist: dist[v] = dist[u] + 1; dq.append(v)
    return [sorted(n for n, d in dist.items() if d == k) for k in range(k_max + 1)]


def prosqa_answer(edges, name, o1, o2):
    R = reach(edges, name); a, b = o1 in R, o2 in R
    return f'{name} is a {o1 if a else o2}' if a != b else None


def prosqa_path(row):
    """Gold path as edge (src, dst) list from the steps ("Name is a X." then "Every X is a Y.")."""
    out = []
    for s in row['steps']:
        m = PQ_EDGE.match(s.strip()); out.append(((m.group(2) or m.group(3)), m.group(4)))
    return out


def norm(s):
    return (s or '').strip().rstrip('.').lower().replace(',', '')


def prosqa_pairs(rows, tok_len, seed=0, hops=None):
    """For each recipient and each hop k on its gold path: one donor (or None). Returns list of dicts
    {i, hop, donor (row dict), target (normalised answer), n_hops}."""
    rng = random.Random(seed); out = []
    for i, r in enumerate(rows):
        E, name, o1, o2, S = prosqa_parse(r['question']); path = prosqa_path(r)
        assert norm(prosqa_answer(E, name, o1, o2)) == norm(r['answer']), (i, r['answer'])
        L0 = tok_len(r['question']); concepts = sorted({x for a, b, _ in E for x in (a, b)} - {name})
        for k, (a, b) in enumerate(path):
            if hops is not None and k not in hops: continue
            idx = [e[2] for e in E if e[0] == a and e[1] == b]; assert idx, (i, a, b)
            si = idx[0]; cand = concepts[:]; rng.shuffle(cand); pick = None
            for w in cand:
                if w == b: continue
                E2 = [(x, (w if j == si else y), j) for x, y, j in E]
                ans = prosqa_answer(E2, name, o1, o2)
                if ans is None or norm(ans) == norm(r['answer']): continue
                S2 = S[:]; S2[si] = re.sub(rf'\b{b}\.$', f'{w}.', S[si])
                if S2[si][:-len(w) - 1].rstrip().endswith(' an'): continue  # keep "a"/"an" grammar simple
                q2 = ' '.join(S2)
                if tok_len(q2) != L0: continue
                pick = dict(i=i, hop=k, n_hops=len(path), donor=dict(r, question=q2, answer=ans + '.'), target=norm(ans), edit=(si, b, w)); break
            out.append(pick or dict(i=i, hop=k, n_hops=len(path), donor=None, target=None, edit=None))
    return out


# ---------------------------------------------------------------- ProntoQA
def _sing(w):
    return w[:-2] if w.endswith('puses') else w


PO_ISA = [re.compile(r'^(?:Every|Each) (\w+) is an? (\w+)\.$'), re.compile(r'^(\w+puses) are (\w+puses)\.$')]
PO_PROP = [re.compile(r'^(?:Every|Each) (\w+) is (not )?(\w+)\.$'), re.compile(r'^(\w+puses) are (not )?(\w+)\.$')]
PO_ENT = re.compile(r'^([A-Z][a-z]+) is an? (\w+)\.$')
PO_Q = re.compile(r'^([A-Z][a-z]+) is (not )?(\w+)\.$')


def prontoqa_parse(q):
    """-> (isa list of (src, dst, sent idx, form), props {concept: [(prop, positive, idx)]}, ent list of (name, concept, idx),
    (name, negated, prop), sentence list). form = 'sg' or 'pl' (object written singular / plural)."""
    body, query = q.split('True or false:')
    S = sentences(body); isa = []; props = defaultdict(list); ent = []
    for i, s in enumerate(S):
        m = PO_ENT.match(s)
        if m and not m.group(2).endswith('pus') and not m.group(2).endswith('puses'):
            m = None
        if m and not s.startswith(('Every', 'Each')):
            ent.append((m.group(1), m.group(2), i)); continue
        for f, P in zip(('sg', 'pl'), PO_ISA):
            m = P.match(s)
            if m and (m.group(2).endswith('pus') or m.group(2).endswith('puses')):
                isa.append((_sing(m.group(1).lower()), _sing(m.group(2)), i, f)); break
        else:
            for P in PO_PROP:
                m = P.match(s)
                if m: props[_sing(m.group(1).lower())].append((m.group(3), m.group(2) is None, i)); break
            else:
                raise AssertionError(s)
    m = PO_Q.match(query.strip()); assert m, query
    return isa, props, ent, (m.group(1), m.group(2) is not None, m.group(3)), S


def prontoqa_answer(isa, props, ent, query):
    name, neg, p = query
    E = [(a, b, i) for a, b, i, _ in isa]; vals = set()
    for n, c, _ in ent:
        if n != name: continue
        for x in reach(E, c):
            for pp, pos, _ in props.get(x, []):
                if pp == p: vals.add(pos)
    if len(vals) != 1: return None
    (pos,) = vals
    return 'True' if pos != neg else 'False'


def prontoqa_pairs(rows, tok_len, seed=0, hops=None):
    """Hop k = path relation k (0 = 'Name is a X.', 1.. = concept -> concept). Donor rewires its object to another concept of the same
    question so the answer flips. Returns list of dicts as prosqa_pairs."""
    rng = random.Random(seed); out = []
    for i, r in enumerate(rows):
        isa, props, ent, query, S = prontoqa_parse(r['question'])
        assert prontoqa_answer(isa, props, ent, query) == r['answer'], (i, r['answer'])
        name = query[0]
        chain = [s.split('. ')[0] for s in r['steps'][:-1]]  # "Stella is a zumpus" per hop
        cs = [re.match(r'^\w+ is an? (\w+)', c).group(1) for c in chain]  # path concepts c_0..c_{n-1}
        rel = [('ent', None, cs[0])] + [('isa', cs[k], cs[k + 1]) for k in range(len(cs) - 1)]
        concepts = sorted({x for a, b, _, _ in isa for x in (a, b)} | {c for _, c, _ in ent} | set(props))
        L0 = tok_len(r['question'])
        for k, (kind, a, b) in enumerate(rel):
            if hops is not None and k not in hops: continue
            if kind == 'ent':
                sel = [e for e in ent if e[0] == name and e[1] == b]; assert sel, (i, b); si = sel[0][2]; form = 'sg'
            else:
                sel = [e for e in isa if e[0] == a and e[1] == b]; assert sel, (i, a, b); si, form = sel[0][2], sel[0][3]
            cand = concepts[:]; rng.shuffle(cand); pick = None
            for w in cand:
                if w == b: continue
                if kind == 'ent':
                    ent2 = [(n, (w if j == si else c), j) for n, c, j in ent]; isa2 = isa
                else:
                    ent2 = ent; isa2 = [(x, (w if j == si else y), j, f) for x, y, j, f in isa]
                ans = prontoqa_answer(isa2, props, ent2, query)
                if ans is None or ans == r['answer']: continue
                old, new = (b + 'es', w + 'es') if form == 'pl' else (b, w)
                S2 = S[:]; S2[si] = re.sub(rf'\b{old}\.$', f'{new}.', S[si])
                if S2[si] == S[si] or re.search(r'\ban\b', S2[si]) and w[0] not in 'aeiou' or re.search(r'\ba\b', S2[si].split(' is ')[-1]) and w[0] in 'aeiou':
                    continue
                q2 = ' '.join(S2) + ' True or false:' + r['question'].split('True or false:')[1]
                if tok_len(q2) != L0: continue
                pick = dict(i=i, hop=k, n_hops=len(rel), donor=dict(r, question=q2, answer=ans), target=ans.lower(), edit=(si, b, w)); break
            out.append(pick or dict(i=i, hop=k, n_hops=len(rel), donor=None, target=None, edit=None))
    return out
