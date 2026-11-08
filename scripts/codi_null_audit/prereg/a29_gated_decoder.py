"""Tables for b27_gated_decoder.json. Usage: a29_gated_decoder.py <json>.
Exact read = decoded value equals the true step value. Headline cells fixed in b27's docstring before any read."""
import json, sys
import numpy as np
R = json.load(open(sys.argv[1])); SITES = ['t1'] + [f'h{s+1}' for s in range(6)] + ['colon']; LAY = [0, 2, 4, 8, 12]
def acc(pred, V, j): return float(np.mean(np.array(pred)[:, j] == V[:, j]))
def lens(o, si, j, l): return float(np.mean(np.array(o['lens'])[:, si, l] == np.array(o['V'])[:, j]))
models = [m for m in ('gated', 'ungated') if m in R]
print('## Answer accuracy (valid, this collector)')
for m in models: print(m, ' '.join(f"L{L} {np.mean(np.array(o['ans']) == np.array(o['V'])[:, -1]):.3f}" for L, o in R[m].items()))

print('\n## Headline: v_j from the latent z_j (h{j+1} layer 0) and from the pass that read rule j (h{j} layer 12); probe full / shuffled; logit lens')
for L in R[models[0]]:
    print(f'L{L}')
    for j in range(1, int(L)):
        row = []
        for m in models:
            o = R[m][L]; V = np.array(o['V']); c1 = o['cells'].get(f'h{j+1}|0'); c2 = o['cells'][f'h{j}|12']
            z = f"{acc(c1['full'], V, j):.2f}/{acc(c1['shuf'], V, j):.2f} lens {lens(o, SITES.index(f'h{j+1}'), j, 0):.2f}" if c1 else 'n/a'
            h = f"{acc(c2['full'], V, j):.2f}/{acc(c2['shuf'], V, j):.2f} lens {lens(o, SITES.index(f'h{j}'), j, 12):.2f}"
            row.append(f'{m}: z_{j} {z} | h{j}@12 {h}')
        mode = max(np.bincount(np.array(R[models[0]][L]['V'])[:, j]) / len(R[models[0]][L]['V']))
        print(f'  v_{j}  ' + '   ||   '.join(row) + f'   (mode share {mode:.2f})')

print('\n## Grid: v_j (rows) x site (cols), probe exact, best of 5 layers (best-of on scored items, so slightly high)')
for m in models:
    for L, o in R[m].items():
        V = np.array(o['V']); print(f'{m} L{L}   ' + ' '.join(f'{s:>6}' for s in SITES))
        for j in range(V.shape[1]):
            cells = [max(acc(o['cells'][f'{s}|{l}']['full'], V, j) for l in LAY) for s in SITES]
            print(f'  v_{j:<2}       ' + ' '.join(f'{c:6.2f}' for c in cells))

print('\n## Same grid, logit lens on number tokens, best of 13 layers')
for m in models:
    for L, o in R[m].items():
        V = np.array(o['V']); print(f'{m} L{L}   ' + ' '.join(f'{s:>6}' for s in SITES))
        for j in range(V.shape[1]):
            print(f'  v_{j:<2}       ' + ' '.join(f'{max(lens(o, si, j, l) for l in range(13)):6.2f}' for si in range(len(SITES))))

print('\n## Negative control (gated): v_j at sites before any pass has seen rule j (t1, h1..h{j-1}), all layers, probe exact; shuffled-label max anywhere')
for L, o in R.get('gated', {}).items():
    V = np.array(o['V']); bad = [acc(o['cells'][f'{s}|{l}']['full'], V, j) for j in range(2, V.shape[1] - 1) for s in ['t1'] + [f'h{k}' for k in range(1, j)] for l in LAY]
    sh = max(acc(c['shuf'], V, j) for c in o['cells'].values() for j in range(V.shape[1]))
    print(f'  L{L}: impossible cells max {max(bad):.2f} mean {np.mean(bad):.3f} (n={len(bad)}); shuffled max {sh:.2f}')

print('\n## Saturation: headline z_j cells, 4,000 vs 8,000 fit')
for m in models:
    for L, o in R[m].items():
        V = np.array(o['V']); print(f'  {m} L{L}: ' + ' '.join(f"v_{j} {acc(o['cells'][f'h{j+1}|0']['half'], V, j):.2f}->{acc(o['cells'][f'h{j+1}|0']['full'], V, j):.2f}" for j in range(1, min(int(L), 6))))

print('\n## Calibration against b23 (ungated, all items, value before the last rule at h4 layer 2; b23: L3 .49, L4 .21, L5 .12)')
for L, o in R.get('ungated', {}).items():
    V = np.array(o['V']); print(f"  L{L}: {acc(o['cells']['h4|2']['full'], V, V.shape[1] - 2):.2f}")
