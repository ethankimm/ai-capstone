"""Readable x used table (HANDOFF_2026-11-07 step 1). Usage: t1_table.py <audit dir>. Writes <audit dir>/t1_table.json, prints markdown.
Read side: b27_gated_decoder.json (gated, ungated = plain) and s2_sim_probe.json (sim), valid L3-L6, every item.
Use side: t1_{plain,gated,sim}.json (same items). Thresholds fixed before compiling (handoff):
  readable = probe exact >= .5 at site h{j+1} (best of layers 0/2/4/8/12, selected on the scored items, so slightly high) with the
             shuffled-label fit at that layer <= .06
  used     = single swap follows donor >= .5, or after same-level mean ablation <= .5 of the items answered right clean stay right.
             (Handoff wording was "mean ablation leaves accuracy <= .5"; changed before compiling the table because plain CODI's clean
             accuracy is itself below .5 at L4+, which would mark every plain latent used. Identical where clean is ~1, gated and SIM.)
Value read, kr: gated and sim min(j, L-1) (sim: z_j holds the operand v_j, z_{L-1}.. hold v_{L-1}); plain L-1.
For sim the value the swap moves is v_{min(j+1, L-1)}; its read is a second column, not used for the cell.
Levels relative to plain CODI-0's reach (d50 = 3, first level below .5): L3 at d50, L4-L6 past it. No below-reach level here."""
import json, sys
import numpy as np
W = sys.argv[1]; NL = 6; SITES = ['t1'] + [f'h{s+1}' for s in range(NL)] + ['colon']; LAY = [0, 2, 4, 8, 12]
RD = json.load(open(W + '/b27_gated_decoder.json')); RD = dict(plain=RD['ungated'], gated=RD['gated'], sim=json.load(open(W + '/s2_sim_probe.json'))['sim'])
US = {m: json.load(open(W + f'/t1_{m}.json')) for m in ('plain', 'gated', 'sim')}
B29 = json.load(open(W + '/b29_ungated_resample.json'))
READ_T, SHUF_T, USE_T = .5, .06, .5
def kr_of(m, j, L): return L - 1 if m == 'plain' else min(j, L - 1)
def ex(pred, V, k): return float(np.mean(np.array(pred)[:, k] == V[:, k]))

OUT = {}
for m in ('plain', 'gated', 'sim'):
    OUT[m] = {}
    for L in ('3', '4', '5', '6'):
        r = RD[m][L]; u = US[m][L]; V = np.array(r['V']); Lens = np.array(r['lens']); Li = int(L); cells = []
        assert len(V) == u['n'] and V[:, -1].tolist() == u['gold'], (m, L)
        for j in range(NL):
            kr = kr_of(m, j, Li); site = f'h{j+1}'; si = SITES.index(site)
            per = {l: ex(r['cells'][f'{site}|{l}']['full'], V, kr) for l in LAY}; bl = max(per, key=per.get)
            shuf = ex(r['cells'][f'{site}|{bl}']['shuf'], V, kr); shuf0 = ex(r['cells'][f'{site}|0']['shuf'], V, kr)
            lens = [float(np.mean(Lens[:, si, l] == V[:, kr])) for l in range(13)]
            sw = u['swap'][str(j)]; ma = u['mean_ablation'][str(j)]['acc']; rt = np.array(u['clean_ans']) == np.array(u['gold'])
            mk = float(np.mean(np.array(u['mean_ablation'][str(j)]['ans'])[rt] == np.array(u['gold'])[rt]))
            readable = per[bl] >= READ_T and shuf <= SHUF_T; readable0 = per[0] >= READ_T and shuf0 <= SHUF_T
            used = sw['follows_donor'] >= USE_T or mk <= USE_T
            c = dict(j=j, kr=kr, ks=sw['ks'], probe_best=round(per[bl], 3), best_layer=bl, shuf_best=round(shuf, 3), probe_l0=round(per[0], 3), shuf_l0=round(shuf0, 3),
                     lens_best=round(max(lens), 3), lens_layer=int(np.argmax(lens)), swap_n=sw['n'], follows=round(sw['follows_donor'], 3), keeps=round(sw['keeps_own'], 3),
                     mean_abl=round(ma, 3), mean_kept_right=round(mk, 3), readable=readable, readable_l0=readable0, lens_readable=max(lens) >= READ_T, used=used,
                     cell=('readable+used' if readable and used else 'readable, not used' if readable else 'used, not readable' if used else 'neither'))
            if m == 'sim':
                pks = {l: ex(r['cells'][f'{site}|{l}']['full'], V, sw['ks']) for l in LAY}; c['probe_ks_best'] = round(max(pks.values()), 3)
            cells.append(c)
        OUT[m][L] = dict(clean=round(u['clean'], 3), cells=cells, hide={c: round(h['acc'], 3) for c, h in u['hide'].items()})

for m, R in OUT.items():
    print(f'\n### {m}\n')
    print('| L | z_j | v read | probe best (layer) / shuf | probe L0 / shuf | lens best (layer) | swap n | follows / keeps | mean-abl acc (right kept) | cell |' + (' probe v_ks |' if m == 'sim' else ''))
    print('|---|---|---|---|---|---|---|---|---|---|' + ('---|' if m == 'sim' else ''))
    for L, o in R.items():
        for c in o['cells']:
            print(f"| {L} | z_{c['j']} | v_{c['kr']}" + (f" (swap v_{c['ks']})" if c['ks'] != c['kr'] else '') + f" | {c['probe_best']:.2f} ({c['best_layer']}) / {c['shuf_best']:.2f} | {c['probe_l0']:.2f} / {c['shuf_l0']:.2f} | "
                  f"{c['lens_best']:.2f} ({c['lens_layer']}) | {c['swap_n']} | {c['follows']:.2f} / {c['keeps']:.2f} | {c['mean_abl']:.2f} ({c['mean_kept_right']:.2f}) | {c['cell']} |" + (f" {c['probe_ks_best']:.2f} |" if m == 'sim' else ''))

print('\n### Four-cell counts (24 cells per model: L3-L6 x z_0..z_5). Probe best-of / probe layer 0 / logit lens as the decoder')
CATS = ['readable+used', 'readable, not used', 'used, not readable', 'neither']
def cat(c, key): return ('readable+used' if c[key] and c['used'] else 'readable, not used' if c[key] else 'used, not readable' if c['used'] else 'neither')
print('| model | levels | decoder | ' + ' | '.join(CATS) + ' |'); print('|---|---|---|' + '---|' * 4)
counts = {}
for m, R in OUT.items():
    for grp, Ls in (('L3 (at d50)', ['3']), ('L4-L6 (past)', ['4', '5', '6']), ('all', ['3', '4', '5', '6'])):
        for key, name in (('readable', 'probe best-of'), ('readable_l0', 'probe layer 0'), ('lens_readable', 'logit lens')):
            cs = [cat(c, key) for L in Ls for c in R[L]['cells']]; n = [cs.count(x) for x in CATS]; counts[f'{m}|{grp}|{name}'] = n
            print(f'| {m} | {grp} | {name} | ' + ' | '.join(map(str, n)) + ' |')
OUT['_counts'] = counts

print('\n### Hide (answer cannot see some latent passes; valid accuracy)')
print('| model | L | clean | block_used | block_upto (only idle left) | block_idle | block_all |'); print('|---|---|---|---|---|---|---|')
for m, R in OUT.items():
    if m.startswith('_'): continue
    for L, o in R.items():
        if o['hide']: print(f"| {m} | {L} | {o['clean']:.3f} | " + ' | '.join(f"{o['hide'][c]:.3f}" for c in ('block_used', 'block_upto', 'block_idle', 'block_all')) + ' |')

print('\n### Plain CODI, all six latents at once (b29, test split, for comparison with the near-copy single latents)')
print('| L | clean | diff donor n | follows / keeps | same-value donor keeps | same-level mean acc |'); print('|---|---|---|---|---|---|')
for L in ('3', '4', '5', '6'):
    o = B29[L]; print(f"| {L} | {o['clean']:.3f} | {o['diff']['n']} | {o['diff']['follows_donor']:.3f} / {o['diff']['keeps_own']:.3f} | {o['same']['keeps_own']:.3f} | {o['mean']['acc_all']:.3f} |")
json.dump(OUT, open(W + '/t1_table.json', 'w'), indent=1)
