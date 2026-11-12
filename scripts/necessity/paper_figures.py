"""Draft figures for the hop-count paper (Nov 16 draft). Reads saved outputs only; runs no model.

  python3 scripts/necessity/paper_figures.py [out dir]      (default: ~/Documents/Penn/CIS5980/paper-draft/figures)

Inputs (codi_audit_artifacts):
  t26/{gated,sim,simsf,startlast,relay,stab}.json      hop counts + clean accuracy (t26)
  t26/ref_{startlast0,stab0}.json                      matched CODI-0 accuracy (t27c)
  t26/gsm8k_depth_hops.json                            GSM8K pairs with dependency depth (export of t26 + t27a pairs)
  t26/rcread_*.json, lens_gsm8k_codi.json, sim_parallel_b.json   readability by refresh (t29, t30, t28b)
  t15_*.json                                           RuleChain route split (record 20261108-073431)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ART = Path.home() / 'Documents/Penn/CIS5980/codi_audit_artifacts'; T26 = ART / 't26'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'Documents/Penn/CIS5980/paper-draft/figures'
OUT.mkdir(parents=True, exist_ok=True)
K = 6; RNG = np.random.default_rng(0)

# ---------------------------------------------------------------- style (validated categorical slots, light mode, print)
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
C = dict(gated='#2a78d6', sim='#eb6834', simsf='#1baf7a', startlast='#eda100', relay='#e87ba4', stab='#008300', codi0='#8a8984',
         gsm8k_codi='#6250d6', gsm8k_sim='#eb6834', gsm8k_coconut='#1baf7a')
NAME = dict(gated='Rule-gated CODI', sim='SIM-CODI', simsf='SIM-CODI (start first)', startlast='CODI', relay='CODI (relay)',
            stab='CODI (stabilised)', codi0='CODI-0 (no latents)', gsm8k_codi='CODI', gsm8k_sim='SIM-CODI', gsm8k_coconut='Coconut')
MK = dict(gated='o', sim='s', simsf='D', startlast='^', relay='v', stab='P', codi0='x', gsm8k_codi='o', gsm8k_sim='s', gsm8k_coconut='^')
RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']  # sequential blue, light -> dark
ORD = [None, '#86b6ef', '#5598e7', '#2a78d6', '#104281']  # ordinal steps (250..650), lightest clears 2:1 on white
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .5,
    'axes.axisbelow': True, 'lines.linewidth': 1.6, 'lines.markersize': 4, 'legend.frameon': False, 'pdf.fonttype': 42,
    'axes.titleweight': 'normal', 'axes.titlelocation': 'left'})
load = lambda p: json.load(open(p))
W = 6.5  # text width (in), 1in margins on letter


def boot_ci(v, n=2000):
    v = np.asarray(v, float)
    if len(v) < 2: return v.mean(), v.mean()
    m = RNG.choice(v, (n, len(v))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def title(ax, letter, text, **kw):
    ax.set_title(r'$\bf{' + letter + '}$   ' + text, **kw)


def end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords='offset points', fontsize=6.8, color=INK2, va='center', ha='left')
    ax.plot([], [])


# ---------------------------------------------------------------- Fig 1: the refresh procedure (schematic)
def fig_schematic():
    fig, ax = plt.subplots(figsize=(W, 2.35)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.grid(False)
    colA, colB, colMix = '#d6d5d0', C['gated'], '#b7d3f6'
    rows = [(0, 'M$_0$', 0, 'A'), (1, 'M$_1$', 1, 'A'), (2, 'M$_2$', 2, 'B'), (3, 'M$_3$', 3, 'B'), (6, 'M$_6$', 6, 'B')]
    x0, w, gap = 22, 6.2, 1.2; ys = [33, 26, 19, 12, 3]
    ax.text(3, 38.5, 'run', fontsize=7, color=INK2); ax.text(9.5, 38.5, 'question', fontsize=7, color=INK2)
    ax.text(x0, 38.5, 'latent inputs fed to passes 1-6', fontsize=7, color=INK2); ax.text(70.5, 38.5, 'answer', fontsize=7, color=INK2)
    for (n, lab, nB, ans), y in zip(rows, ys):
        ax.text(3, y + 1.6, lab, fontsize=8.5, va='center')
        ax.add_patch(FancyBboxPatch((9.5, y), 9, 3.2, boxstyle='round,pad=0,rounding_size=0.6', fc='white', ec=colB, lw=1))
        ax.text(14, y + 1.6, "B's Q", fontsize=7, ha='center', va='center', color=INK)
        for j in range(K):
            if n == 0: fc, t = colA, 'A'
            elif j < nB: fc, t = colB, 'B'
            else: fc, t = colMix, 'mix'
            ax.add_patch(FancyBboxPatch((x0 + j * (w + gap), y), w, 3.2, boxstyle='round,pad=0,rounding_size=0.6', fc=fc, ec='white', lw=1.2))
            ax.text(x0 + j * (w + gap) + w / 2, y + 1.6, f'z$_{j + 1}$' if t != 'mix' else 'mix', fontsize=6.8, ha='center', va='center',
                    color='white' if t == 'B' else INK)
        ax.add_patch(FancyBboxPatch((70.5, y), 8, 3.2, boxstyle='round,pad=0,rounding_size=0.6',
                                    fc=colB if ans == 'B' else colA, ec='white', lw=1.2))
        ax.text(74.5, y + 1.6, f"{ans}'s", fontsize=7, ha='center', va='center', color='white' if ans == 'B' else INK)
        if n in (1, 2, 3):
            ax.annotate('', xy=(x0 - 1.5, y + 3.0), xytext=(x0 - 1.5, y + 6.6),
                        arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=.8, mutation_scale=7))
    ax.text(15, 9.0, '...', fontsize=10, ha='center', color=MUTED); ax.text(47, 9.0, '...', fontsize=10, ha='center', color=MUTED)
    ax.text(x0, 30.95, 'each run is fed the latents the previous run produced (arrows)', fontsize=6.3, color=INK2, va='center')
    ax.annotate('hops = 2:\nfirst run whose\nanswer is B\'s', xy=(78.8, ys[2] + 1.6), xytext=(84, ys[2] + 1.6), fontsize=7, va='center',
                arrowprops=dict(arrowstyle='-', color=INK2, lw=.6))
    ax.text(84, ys[4] + 1.6, "= B's own run\n(checked on every pair)", fontsize=6.5, va='center', color=INK2)
    for i, (fc, t) in enumerate([(colA, "A's latents"), (colB, "exactly B's"), (colMix, "B's question, stale inputs")]):
        ax.add_patch(FancyBboxPatch((22 + i * 17, -0.6), 2.4, 1.8, boxstyle='round,pad=0,rounding_size=0.4', fc=fc, ec='none'))
        ax.text(25.2 + i * 17, 0.3, t, fontsize=6.5, va='center', color=INK2)
    fig.savefig(OUT / 'fig1_refresh_schematic.pdf', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- Fig 2: accuracy by chain length vs the matched CODI-0
def fig_accuracy():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.2), sharey=True)
    groups = [('startlast0', ('gated', 'sim', 'startlast'), 'Start value last'), ('stab0', ('simsf', 'relay', 'stab'), 'Start value first')]
    for ax, (ref, ms, ttl), pl in zip(axs, groups, 'ab'):
        Ls = [2, 3, 4, 5, 6]
        acc0 = load(T26 / f'ref_{ref}.json')['valid_acc']
        ax.plot(Ls, [acc0[str(L)] for L in Ls], color=C['codi0'], marker=MK['codi0'], ls='--', lw=1.2, label=NAME['codi0'])
        for m in ms:
            a = load(T26 / f'{m}.json')['clean_acc']
            ax.plot(Ls, [a[str(L)] for L in Ls], color=C[m], marker=MK[m], label=NAME[m])
        ax.set_xticks(Ls); ax.set_xlabel('rules in the chain (L)'); ax.set_ylim(0, 1.04); title(ax, pl, ttl)
        ax.legend(loc='center right' if pl == 'a' else 'lower left', handlelength=2.2)
        if pl == 'a': ax.text(6, .955, 'rule-gated and SIM-CODI overlap (≥ .996)', fontsize=6.3, color=INK2, ha='right', va='top')
    axs[0].set_ylabel('validation accuracy')
    fig.tight_layout(w_pad=1.5); fig.savefig(OUT / 'fig2_accuracy.pdf', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- Fig 3: hop count by steps from edit to answer
def fig_hops():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.6), sharey=True)
    ax = axs[0]; xs = np.arange(1, 7)
    ax.fill_between(xs, xs - 1, xs, color='#eeedea', lw=0, zorder=0)
    ax.text(5.75, 3.45, 'one crossing\nper step', fontsize=6.3, color=INK2, ha='center', va='center')
    for m in ('gated', 'sim', 'simsf', 'startlast', 'relay', 'stab'):
        by = defaultdict(list)
        for p in load(T26 / f'{m}.json')['pairs']:
            if p['qualified']: by[p['s']].append(p['hops'])
        ss = [s for s in sorted(by) if len(by[s]) >= 15]; mu = [np.mean(by[s]) for s in ss]; ci = np.array([boot_ci(by[s]) for s in ss])
        ax.errorbar(ss, mu, yerr=[np.array(mu) - ci[:, 0], ci[:, 1] - np.array(mu)], color=C[m], marker=MK[m], capsize=0, elinewidth=.8,
                    label=NAME[m])
    ax.set_xlabel('steps from the edited rule to the answer (s)'); ax.set_ylabel('loop crossings used (mean hops)')
    ax.set_xticks(xs); ax.set_ylim(-0.2, 6.3); title(ax, 'a', 'RuleChain (trained here)')
    ax.legend(loc='upper left', handlelength=2.2, ncol=1)
    ax = axs[1]; D = load(T26 / 'gsm8k_depth_hops.json')
    ax.fill_between(xs, xs - 1, xs, color='#eeedea', lw=0, zorder=0)
    for m in ('gsm8k_codi', 'gsm8k_sim', 'gsm8k_coconut'):
        by = defaultdict(list)
        for p in D[m]:
            if p['depth'] is not None: by[p['depth']].append(p['hops'])
        ss = [s for s in sorted(by) if len(by[s]) >= 15]; mu = [np.mean(by[s]) for s in ss]; ci = np.array([boot_ci(by[s]) for s in ss])
        ax.errorbar(ss, mu, yerr=[np.array(mu) - ci[:, 0], ci[:, 1] - np.array(mu)], color=C[m], marker=MK[m], capsize=0, elinewidth=.8,
                    label=f'{NAME[m]} (n={sum(len(by[s]) for s in ss)})')
    ax.set_xlabel('dependency depth of the edit (steps)'); ax.set_xticks(range(2, 6)); ax.set_xlim(1.5, 5.5)
    title(ax, 'b', 'GSM8K-Aug (released checkpoints)'); ax.legend(loc='upper left', handlelength=2.2)
    ax.text(4.75, 4.3, 'one crossing\nper step', fontsize=6.3, color=INK2, ha='center', va='center')
    fig.tight_layout(w_pad=1.5); fig.savefig(OUT / 'fig3_hops.pdf', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- Fig 4: GSM8K hop distribution by dependency depth
def fig_gsm8k_dist():
    D = load(T26 / 'gsm8k_depth_hops.json'); fig, axs = plt.subplots(1, 3, figsize=(W, 1.9), sharey=True)
    cols = [RAMP[0], RAMP[2], RAMP[4], RAMP[6]]; labs = ['0', '1', '2', '3+']
    for ax, m, pl in zip(axs, ('gsm8k_codi', 'gsm8k_sim', 'gsm8k_coconut'), 'abc'):
        by = defaultdict(list)
        for p in D[m]:
            if p['depth'] is not None: by[p['depth']].append(min(p['hops'], 3))
        ds = [d for d in sorted(by) if len(by[d]) >= 15]; bottom = np.zeros(len(ds))
        for h in range(4):
            v = np.array([np.mean(np.array(by[d]) == h) for d in ds])
            ax.bar(ds, v, bottom=bottom, color=cols[h], width=.72, edgecolor='white', linewidth=1, label=f"{labs[h]} hop{'' if labs[h] == '1' else 's'}")
            bottom += v
        for d in ds: ax.text(d, 1.02, f'n={len(by[d])}', ha='center', fontsize=6.2, color=INK2)
        ax.set_xticks(ds); ax.set_xlabel('dependency depth'); title(ax, pl, NAME[m], pad=11); ax.set_ylim(0, 1.0); ax.grid(axis='x', visible=False)
    axs[0].set_ylabel('share of pairs')
    axs[2].legend(loc='center left', bbox_to_anchor=(1.0, .5), handlelength=1.2)
    fig.tight_layout(w_pad=1); fig.savefig(OUT / 'fig4_gsm8k_hop_distribution.pdf', bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- Fig 5: readability by refresh, aligned at the crossing
NUMRE = re.compile(r'<<([^=<>]*)=([^<>]*)>>')


def _num_eq(a, b):
    try: return abs(float(str(a).replace(',', '')) - float(str(b).replace(',', ''))) < 1e-6
    except ValueError: return False


def sim_decoder_summary():
    """t28b items rescored: B read, stale A read, chance (next item's B at the same depth), per refresh 0..6, given own-readable."""
    d = load(T26 / 'sim_parallel_b.json'); items = []
    for p in d['pairs']:
        R = {c: [m.group(2).strip() if (m := NUMRE.search(s or '')) else None for s in p['decoded'][c]] for c in p['decoded']}
        for t in p['steps']:
            rd = lambda v: [any(r is not None and _num_eq(r, v) for r in R[f'refresh{n}']) for n in range(K + 1)]
            items.append(dict(depth=t['depth'], b=t['b'], B=rd(t['b']), A=rd(t['a']), own=t['read_own'], rd=rd))
    grp = defaultdict(list)
    for it in items: grp[it['depth']].append(it)
    out = {}
    for dd, g in grp.items():
        for x, y in zip(g, g[1:] + g[:1]): x['C'] = x['rd'](y['b']) if not _num_eq(y['b'], x['b']) else [False] * (K + 1)
        own = [it for it in g if it['own']]
        out[dd] = dict(n_own=len(own), B=[np.mean([it['B'][n] for it in own]) for n in range(K + 1)],
                       A=[np.mean([it['A'][n] for it in own]) for n in range(K + 1)], chance=[np.mean([it['C'][n] for it in own]) for n in range(K + 1)])
    return out


def fig_readability():
    cols = [('gated', 'RuleChain\nrule-gated CODI', 'probe'), ('sim', 'RuleChain\nSIM-CODI', 'probe'),
            ('relay', 'RuleChain\nCODI (relay)', 'probe'), ('gsm8k_sim', 'GSM8K\nSIM-CODI', 'decoder'), ('gsm8k_codi', 'GSM8K\nCODI', 'lens')]
    fig, axs = plt.subplots(2, 5, figsize=(W, 3.7), sharex=True, sharey=True); NS = {}
    for c, (m, ttl, how) in enumerate(cols):
        if how == 'probe': S = {int(k): v for k, v in load(T26 / f'rcread_{m}.json')['summary'].items()}
        elif how == 'lens': S = {int(k): v for k, v in load(T26 / 'lens_gsm8k_codi.json')['summary'].items()}
        else: S = sim_decoder_summary()
        ds = [d for d in sorted(S) if S[d]['n_own'] >= 25][:4]; NS[m] = {d: S[d]['n_own'] for d in ds}
        for d in ds:
            x = np.arange(K + 1) - d; keep = (x >= -3) & (x <= 2)
            for r, key in ((0, 'B'), (1, 'A')):
                axs[r, c].plot(x[keep], np.array(S[d][key])[keep], color=ORD[d], marker='o', ms=3, lw=1.3, label=f'd = {d}')
        w = np.array([S[d]['n_own'] for d in ds], float)
        ch = max(float(np.dot(w, [S[d]['chance'][n] for d in ds]) / w.sum()) for n in range(K + 1))
        axs[0, c].axhline(ch, color=MUTED, ls=':', lw=1); axs[0, c].text(2.1, ch + .035, f'chance {ch:.2f}', fontsize=5.8, color=INK2, ha='right')
        axs[0, c].set_title(ttl, fontsize=7.2, loc='center'); axs[1, c].set_xlabel('refresh − d')
        axs[1, c].text(2.1, .93, how, fontsize=6.2, color=INK2, ha='right')
        for r in (0, 1): axs[r, c].axvline(0, color=INK2, lw=.6, alpha=.5)
    axs[0, 0].set_ylabel("B's new value\nreadable"); axs[1, 0].set_ylabel("A's old value\nreadable")
    for ax in axs.flat: ax.set_xticks([-3, -2, -1, 0, 1, 2]); ax.set_ylim(-0.03, 1.05)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, [x + (' (crossings needed)' if i == 0 else '') for i, x in enumerate(l)], loc='lower center', ncol=4,
               bbox_to_anchor=(.5, -0.03), handlelength=1.8)
    fig.tight_layout(h_pad=.5, w_pad=.4, rect=(0, .04, 1, 1)); fig.savefig(OUT / 'fig5_readability.pdf', bbox_inches='tight'); plt.close(fig)
    print('fig5 n own-readable per depth:', NS)


# ---------------------------------------------------------------- Fig 6: RuleChain route split (recurrent input vs attention entries)
def fig_route_split(L='4'):
    ms = ('gated', 'sim', 'simsf', 'startlast', 'relay', 'stab')
    conds = [('both', 'full swap of z$_j$'), ('input', 'recurrent input only'), ('kv', 'attention entries only')]
    cmap = LinearSegmentedColormap.from_list('ramp', ['#fcfcfb'] + RAMP)
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.3), sharey=True)
    for ax, (cnd, ttl), pl in zip(axs, conds, 'abc'):
        M = np.full((len(ms), K + 1), np.nan)
        for i, m in enumerate(ms):
            r = load(ART / f't15_{m}.json')[L]['res']
            for j in range(K): M[i, j] = r[f'{cnd}_{j}']['follows']
            if f'{cnd}_all' in r: M[i, K] = r[f'{cnd}_all']['follows']
        ax.imshow(np.nan_to_num(M, nan=0), cmap=cmap, vmin=0, vmax=1, aspect='auto')
        for i in range(len(ms)):
            for j in range(K + 1):
                v = M[i, j]
                ax.text(j, i, '–' if np.isnan(v) else (f'{v:.2f}'.lstrip('0') if v < .995 else '1'), ha='center', va='center',
                        fontsize=6, color='white' if (not np.isnan(v) and v > .55) else INK2)
        ax.set_xticks(range(K + 1)); ax.set_xticklabels([f'z$_{j + 1}$' for j in range(K)] + ['all'])
        ax.set_yticks(range(len(ms))); ax.set_yticklabels([NAME[m] for m in ms]); ax.grid(False)
        ax.axvline(K - .5, color='white', lw=2); title(ax, pl, ttl)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout(w_pad=.6); fig.savefig(OUT / 'fig6_route_split.pdf', bbox_inches='tight'); plt.close(fig)


for f in (fig_schematic, fig_accuracy, fig_hops, fig_gsm8k_dist, fig_readability, fig_route_split):
    f(); print('wrote', f.__name__)
