"""Tables for t9_attn_flow.json. Usage: t9_table.py <audit dir>. Shares of each latent pass's attention (raw) and of attention output norm (norm),
mean over the 12 layers; sink removed and the rest renormalised for the 'non-sink' columns."""
import json, sys
import numpy as np
W = sys.argv[1]; R = json.load(open(W + '/t9_attn_flow.json'))
for key in ('norm', 'raw'):
    print(f'\n## {key}: share of each latent pass, mean over layers; question = rules + start + rest; non-sink renormalised\n')
    print('| model | L | pass | question | earlier latents | self | sink (raw total) |'); print('|---|---|---|---|---|---|---|')
    for m, RL in R.items():
        for L, o in RL.items():
            S = np.array(o[f'{key}_share']).mean(1)  # [pass, sets]
            for s in range(6):
                a = S[s]; ns = a[:5] / a[:5].sum(); q = ns[:3].sum()
                print(f'| {m} | {L} | {s} | {q:.2f} | {ns[3]:.2f} | {ns[4]:.2f} | {a[5]:.2f} |')
print('\n## norm share on earlier latents by distance (pass s reading pass s-d), L4, mean over layers, share of the earlier-latent total\n')
print('| model | pass | d=1 | d=2 | d=3 | d=4 | d=5 |'); print('|---|---|---|---|---|---|---|')
for m, RL in R.items():
    o = RL['4']; Dd = np.array(o['norm_by_distance']).mean(1)
    for s in range(1, 6):
        r = Dd[s, :s] / Dd[s, :s].sum(); print(f'| {m} | {s} | ' + ' | '.join(f'{x:.2f}' for x in r) + ' |' + ' - |' * (5 - s))
print('\n## norm share on earlier latents by layer, pass 4, L4\n')
for m, RL in R.items():
    S = np.array(RL['4']['norm_share'])[4]; ns = S[:, :5] / S[:, :5].sum(1, keepdims=True)
    print(f'{m:10s} ' + ' '.join(f'{x:.2f}' for x in ns[:, 3]))
