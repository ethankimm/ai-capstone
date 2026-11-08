"""Tables for t7_partial_probe.json. Usage: t7_table.py <audit dir> [all|right]. Each cell: best over layers 0/1/2/3/4/8/12 (selected on
the scored items, slightly high), with the shuffled-label floor of that layer in brackets where shown. val = share within 2 of the scalar readout."""
import json, sys
import numpy as np
W = sys.argv[1]; KEY = sys.argv[2] if len(sys.argv) > 2 else 'all'; R = json.load(open(W + '/t7_partial_probe.json'))
LAY = [0, 1, 2, 3, 4, 8, 12]
def best(o, site, t):
    cs = [(o['cells'][f'{site}|{l}'][t][KEY], o['cells'][f'{site}|{l}'][t]['shuf'], l) for l in LAY]; return max(cs)
def sites(o): return sorted({k.split('|')[0] for k in o['cells']}, key=lambda s: (s[0] != 't', -int(s[1:]) if s[0] == 't' else (int(s[1:]) if s[0] == 'h' else 99)))
print(f'scored on: {KEY} items')
for L in ('3', '4', '5'):
    print(f'\n### L{L}: value before the last rule, v_{int(L) - 1}  (CODI right {R["codi"][L]["right"]}/{R["codi"][L]["n"]}, CODI-0 right {R["codi0"][L]["right"]}/{R["codi0"][L]["n"]})\n')
    j = int(L) - 1
    for mname in ('codi', 'codi0'):
        o = R[mname][L]; S = sites(o)
        print(f'{mname:6s} | target | ' + ' | '.join(S) + ' |'); print('|---|---|' + '---|' * len(S))
        for t in ('exact', 'tens', 'ones', 'par', 'val'):
            cells = [best(o, s, f'v{j}_{t}') for s in S]
            print(f'| {mname} | {t} | ' + ' | '.join(f'{a:.2f} ({sh:.2f})' for a, sh, l in cells) + ' |')
        print(f"majority: tens {o['majority'][f'v{j}_tens']:.2f} ones {o['majority'][f'v{j}_ones']:.2f} parity {o['majority'][f'v{j}_par']:.2f}\n")
    print(f'All steps, CODI, best site among latents h1..h6 vs best tail site (t10, t9, t1) vs CODI-0 best of its sites; tens / ones / parity / within 2\n')
    print('| v_j | CODI latents | CODI tail | CODI colon | CODI-0 tail | CODI-0 colon |'); print('|---|---|---|---|---|---|')
    for jj in range(int(L) + 1):
        def grp(m, S):
            return ' / '.join(f"{max(best(R[m][L], s, f'v{jj}_{t}')[0] for s in S):.2f}" for t in ('tens', 'ones', 'par', 'val'))
        print(f"| v{jj} | {grp('codi', [f'h{s}' for s in range(1, 7)])} | {grp('codi', ['t10', 't9', 't1'])} | {grp('codi', ['colon'])} | {grp('codi0', ['t10', 't9', 't1'])} | {grp('codi0', ['colon'])} |")
    print('\nBranch taken at rule j (majority in brackets), best site among latents / tail / colon\n')
    print('| rule | CODI latents | CODI tail | CODI colon | CODI-0 tail | CODI-0 colon | majority |'); print('|---|---|---|---|---|---|---|')
    for jj in range(1, int(L) + 1):
        t = f'b{jj}'; g = lambda m, S: f"{max(best(R[m][L], s, t)[0] for s in S):.2f}"
        print(f"| {jj} | {g('codi', [f'h{s}' for s in range(1, 7)])} | {g('codi', ['t10', 't9', 't1'])} | {g('codi', ['colon'])} | {g('codi0', ['t10', 't9', 't1'])} | {g('codi0', ['colon'])} | {R['codi'][L]['majority'][t]:.2f} |")
