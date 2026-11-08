"""Dynamics figure from t3_dynamics.json (HANDOFF_2026-11-07 step 3). Usage (repo env): uv run python t3_plot.py <audit dir> [z|h]
Small multiples: rows = metric, columns = level L3-L6; one line per model (median over items, interquartile band).
x = transition t (x_t -> x_{t+1}, the output of pass t); dotted line at t = L-1, the pass whose input z_{L-1} the answer uses
(gated, SIM). Placed after seeing the data: the pre-registered reference, the last rule-reading pass t = L-2, sits one pass early.
Writes t3_dynamics_<traj>.png and .pdf, and prints the medians as a table."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
W = sys.argv[1]; TR = sys.argv[2] if len(sys.argv) > 2 else 'z'
R = json.load(open(W + '/t3_dynamics.json'))
COL = dict(plain='#2a78d6', gated='#eb6834', sim='#1baf7a'); NAME = dict(plain='plain CODI', gated='gated CODI', sim='SIM-CODI')
INK, INK2, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#8a8984', '#fcfcfb', '#e6e5e0'
METRICS = [('rel_step', 'step size\n‖Δ_t‖ / ‖x_t‖', 0), ('dir_cos', 'turning\ncos(Δ_t, Δ_t−1)', 1), ('centred_cos', 'similarity\ncentred cos(x_t, x_t+1)', 0),
           ('log_ratio', 'sensitivity\nlog ‖Δ_t‖ / ‖Δ_t−1‖', 1)]
LV = ['3', '4', '5', '6']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})
fig, AX = plt.subplots(len(METRICS), len(LV), figsize=(7.2, 7.0), sharex=True, sharey='row', gridspec_kw=dict(hspace=.28, wspace=.08))
print(f'| traj | model | L | metric | ' + ' | '.join(f't{t}' for t in range(6)) + ' |')
for r, (key, label, off) in enumerate(METRICS):
    for c, L in enumerate(LV):
        ax = AX[r, c]; ax.grid(axis='y', color=GRID, lw=.6); ax.set_axisbelow(True)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        ax.axvline(int(L) - 1, color=MUTED, lw=.8, ls=':', zorder=1)
        for mdl in ('plain', 'gated', 'sim'):
            a = np.array(R[mdl][L][TR][key]); x = np.arange(a.shape[1]) + off
            med = np.median(a, 0); q1, q3 = np.percentile(a, [25, 75], 0)
            ax.fill_between(x, q1, q3, color=COL[mdl], alpha=.15, lw=0, zorder=2)
            ax.plot(x, med, color=COL[mdl], lw=2, marker='o', ms=3.5, mec=SURF, mew=1, zorder=3, solid_capstyle='round')
            print(f'| {TR} | {mdl} | {L} | {key} | ' + ' | '.join([''] * off + [f'{v:.3f}' for v in med]) + ' |')
        if r == 0: ax.set_title(f'L{L}', fontsize=8.5, fontweight='bold')
        if c == 0: ax.set_ylabel(label, fontsize=7.5)
        if r == len(METRICS) - 1: ax.set_xticks(range(6))
# direct labels at the right end of the first row's last panel, plus a legend (identity never by colour alone)
ax = AX[0, -1]
for mdl in ('plain', 'gated', 'sim'):
    y = np.median(np.array(R[mdl]['6'][TR]['rel_step']), 0)[-1]
    ax.annotate(NAME[mdl], (5, y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=7, color=INK2, annotation_clip=False)
handles = [plt.Line2D([], [], color=COL[m], lw=2, marker='o', ms=3.5, label=NAME[m]) for m in ('plain', 'gated', 'sim')]
handles.append(plt.Line2D([], [], color=MUTED, lw=.8, ls=':', label='pass the answer reads (t = L−1)'))
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, .995), labelcolor=INK2)
fig.supxlabel('transition t  (x_t → x_t+1, the output of latent pass t)', fontsize=7.5, color=INK2, y=.045)
accs = '; '.join(f"{NAME[m]} " + ' / '.join(f"{R[m][L]['acc']:.2f}" for L in LV) for m in ('plain', 'gated', 'sim'))
fig.text(.5, .01, f'valid accuracy L3 / L4 / L5 / L6: {accs}', ha='center', fontsize=7, color=INK2)
traj = 'latent inputs z_t = prj(h_t)' if TR == 'z' else 'last-layer states h_t (before prj)'
fig.suptitle(f'Latent dynamics, start-last models, valid L3–L6: {traj}. Median over items, band = IQR', y=.955, fontsize=8.5, color=INK)
for ext in ('png', 'pdf'): fig.savefig(W + f'/t3_dynamics_{TR}.{ext}', dpi=200, bbox_inches='tight')
print('saved', W + f'/t3_dynamics_{TR}.png')
