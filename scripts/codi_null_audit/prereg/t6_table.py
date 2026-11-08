"""Tables for t6_prefix_hide.json (start-last CODI). Usage: t6_table.py <audit dir>. Follows = answer equals the donor's (edited item's) gold;
keeps = recipient's gold; rows by rules left after the edited rule (0 = edit at the last rule). Hide rows: recipient accuracy (all are right clean)."""
import json, sys
import numpy as np
W = sys.argv[1]; D = json.load(open(W + '/t6_prefix_hide.json')); P = D['P']; st = D['store']['startlast CODI']; R = st['res']; nl = st['nl']
diff = [i for i in st['keep'] if P[i]['kind'] == 'diff']
assert all(R['self'][str(i)] == P[i]['gold'] for i in diff), 'own-latent control failed'
print(f'kept diff pairs {len(diff)}; own-latent control: every answer unchanged')
sw = ['all'] + [f'prefix_{s}' for s in range(1, nl)] + [f'suffix_{s}' for s in range(1, nl)]
print('\n## Swaps: follows donor / keeps own, by rules left after the edit\n')
print('| left | n | ' + ' | '.join(sw) + ' |'); print('|---|---|' + '---|' * len(sw))
for left in range(0, 6):
    ii = [i for i in diff if P[i]['left'] == left]
    if len(ii) < 15: continue
    cells = [f"{np.mean([R[c][str(i)] == P[i]['dgold'] for i in ii]):.2f} / {np.mean([R[c][str(i)] == P[i]['gold'] for i in ii]):.2f}" for c in sw]
    print(f'| {left} | {len(ii)} | ' + ' | '.join(cells) + ' |')
print('\n## Same, edits 1+ rules before the last, by level\n')
print('| L | n | ' + ' | '.join(sw) + ' |'); print('|---|---|' + '---|' * len(sw))
for L in range(2, 8):
    ii = [i for i in diff if P[i]['level'] == L and P[i]['left'] >= 1]
    if len(ii) < 15: continue
    print(f'| {L} | {len(ii)} | ' + ' | '.join(f"{np.mean([R[c][str(i)] == P[i]['dgold'] for i in ii]):.2f}" for c in sw) + ' |')
hd = [f'hide_ge_{s}' for s in range(1, nl)] + [f'only_{s}' for s in range(nl)] + ['hide_all']
print('\n## Which passes the answer reads: accuracy with the answer tokens cut off from some latent passes (clean latents)\n')
print('| L | n items | ' + ' | '.join(hd) + ' |'); print('|---|---|' + '---|' * len(hd))
for L in range(2, 8):
    seen = {};
    for i in diff:
        if P[i]['level'] == L and P[i]['rq'] not in seen: seen[P[i]['rq']] = i
    ii = list(seen.values())
    if len(ii) < 15: continue
    print(f'| {L} | {len(ii)} | ' + ' | '.join(f"{np.mean([R[c][str(i)] == P[i]['gold'] for i in ii]):.2f}" for c in hd) + ' |')
