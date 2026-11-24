"""Draft figures for the hop-count paper (Nov 16 draft). Reads saved outputs only; runs no model.

  python3 scripts/necessity/paper_figures.py [out dir]      (default: ~/Documents/Penn/CIS5980/paper-draft/figures)

Inputs (codi_audit_artifacts):
  t26/{gated,sim,simsf,startlast,relay,stab}.json      hop counts + clean accuracy (t26)
  t26/ref_{startlast0,stab0}.json                      matched CODI-0 accuracy (t27c)
  t26/gsm8k_depth_hops.json                            GSM8K pairs with dependency depth (export of t26 + t27a pairs)
  t26/rcread_*.json, lens_gsm8k_codi.json, sim_parallel_b.json   readability by refresh (t29, t30, t28b)
  t15_*.json                                           RuleChain route split (record 20261108-073431)
  t37/hops.json, t41/*.json                            RuleChain SIM-Coconut hops (t37); L7 hops + accuracy (t41)
  ai-capstone/results/<record>/manifest.json           RuleChain Coconut hops (t33), GSM8K hops by serial distance (t42),
                                                       hidden-path monitor cells (t34, t37)

Main-paper figures: schematic, hops, depth_failure, readability, monitor. depth_failure shows the matched start-last
models plus the relay control (and, for accuracy, the no-loop references: CODI-0 and the pause-token model); hops shows the
released GSM8K checkpoints. depth_failure_all (every model) and hops_by_s (RuleChain by steps from the edit) go in the appendix. The GSM8K hop distribution and the route
split are still written (fig4_*, fig6_*) but the paper no longer includes them.
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
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ART = Path.home() / 'Documents/Penn/CIS5980/codi_audit_artifacts'; T26 = ART / 't26'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'Documents/Penn/CIS5980/paper-draft/figures'
OUT.mkdir(parents=True, exist_ok=True)
K = 6; RNG = np.random.default_rng(0)

# ---------------------------------------------------------------- style (validated categorical slots, light mode, print)
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
# Colour follows the method across panels: CODI yellow, SIM-CODI orange, Coconut red, SIM-Coconut violet.
C = dict(gated='#2a78d6', sim='#eb6834', simsf='#1baf7a', startlast='#eda100', relay='#e87ba4', stab='#008300', codi0='#8a8984',
         simcoconut='#6250d6', coconut='#e34948',
         gsm8k_codi='#eda100', gsm8k_sim='#eb6834', gsm8k_coconut='#e34948', gsm8k_simcoconut='#6250d6',
         gated_rule='#2a78d6', gated_start='#2a78d6', codi0_sl='#6f6e69', pause='#6f6e69', codi0_sf='#b3b2ad')
NAME = dict(gated='Rule-gated CODI', sim='SIM-CODI', simsf='SIM-CODI (start first)', startlast='CODI (start last)', relay='CODI (relay)',
            stab='CODI (stabilised)', codi0='CODI-0 (no latents)', simcoconut='SIM-Coconut', coconut='Coconut',
            gsm8k_codi='CODI', gsm8k_sim='SIM-CODI', gsm8k_coconut='Coconut', gsm8k_simcoconut='SIM-Coconut',
            gated_rule='Rule-gated CODI, rule edits', gated_start='Rule-gated CODI, start edits',
            codi0_sl='CODI-0 (start last)', pause='Pause-token CODI (start last)', codi0_sf='CODI-0 (start first)')
MK = dict(gated='o', sim='s', simsf='D', startlast='^', relay='v', stab='P', codi0='x', simcoconut='X', coconut='h',
          gsm8k_codi='^', gsm8k_sim='s', gsm8k_coconut='h', gsm8k_simcoconut='X',
          gated_rule='o', gated_start='o', codi0_sl='x', pause='+', codi0_sf='x')
# Recipe group is the comparison the paper makes: per-step structure is drawn solid, standard recipes dashed.
PERSTEP = ('gated', 'gated_rule', 'gated_start', 'sim', 'simsf', 'simcoconut', 'gsm8k_sim', 'gsm8k_simcoconut')
REFS = ('codi0_sl', 'pause', 'codi0_sf')   # no loop: zero latent passes, or six positions with constant inputs
OPEN = ('gated_start',)                    # drawn with open markers
# The main figures carry the matched start-last models plus the relay control; the appendix versions show every model.
MAIN = ('gated', 'sim', 'startlast', 'relay')
ALL = ('gated', 'sim', 'simsf', 'simcoconut', 'startlast', 'relay', 'stab', 'coconut')
GROUP = lambda m: 'per-step' if m in PERSTEP else 'ref' if m in REFS else 'standard'
GTITLE = {'per-step': 'Per-step training (solid)', 'standard': 'Standard recipe (dashed)', 'ref': 'No loop (dotted)'}
LS = lambda m: '-' if m in PERSTEP else (0, (1, 1.6)) if m in REFS else (0, (4, 2))
RES = Path(__file__).resolve().parents[2] / 'results'
manifest = lambda rec: json.load(open(RES / rec / 'manifest.json'))['metrics']['extra']
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
    fig, ax = plt.subplots(figsize=(W, 2.65)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 45); ax.grid(False)
    colA, colB, colMix = '#d6d5d0', C['gated'], '#b7d3f6'
    rows = [(0, 'M$_0$', 0, 'A'), (1, 'M$_1$', 1, 'A'), (2, 'M$_2$', 2, 'B'), (3, 'M$_3$', 3, 'B'), (6, 'M$_6$', 6, 'B')]
    x0, w, gap = 22, 6.2, 1.2; ys = [37, 29, 21, 13, 3]
    ax.text(3, 43, 'run', fontsize=7, color=INK2); ax.text(9.5, 43, 'question', fontsize=7, color=INK2)
    ax.text(x0, 43, 'latent inputs fed to passes 1-6', fontsize=7, color=INK2); ax.text(70.5, 43, 'answer', fontsize=7, color=INK2)
    # Arrows: within a run every input is fixed. The question pass's output and pass j's output (input z_j) become
    # z_1 and z_(j+1) of the NEXT run, so each arrow goes one row down and one slot right. z_6's output is not fed on.
    arr = dict(arrowstyle='-|>', color=MUTED, lw=.7, mutation_scale=6, shrinkA=0, shrinkB=0)
    for r in range(3):
        y_lo, y_hi = ys[r], ys[r + 1] + 3.2
        ax.annotate('', xy=(x0 + .3 * w, y_hi + .2), xytext=(17.0, y_lo - .2), arrowprops=arr)
        for j in range(K - 1):
            xa = x0 + j * (w + gap) + .7 * w; xb = x0 + (j + 1) * (w + gap) + .3 * w
            ax.annotate('', xy=(xb, y_hi + .2), xytext=(xa, y_lo - .2), arrowprops=arr)
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
    ax.text(15, 9.6, '...', fontsize=10, ha='center', color=MUTED); ax.text(47, 9.6, '...', fontsize=10, ha='center', color=MUTED)
    ax.text(84, ys[0] - 2.4, "arrows: pass j's output\nbecomes input j+1\nof the next run", fontsize=6.3, va='center', color=INK2)
    ax.annotate('hops = 2:\nfirst run whose\nanswer is B\'s', xy=(78.8, ys[2] + 1.6), xytext=(84, ys[2] + 1.6), fontsize=7, va='center',
                arrowprops=dict(arrowstyle='-', color=INK2, lw=.6))
    ax.text(84, ys[4] + 1.6, "= B's own run\n(checked on every pair)", fontsize=6.5, va='center', color=INK2)
    for i, (fc, t) in enumerate([(colA, "A's latents"), (colB, "exactly B's"), (colMix, "made last run from a stale input")]):
        ax.add_patch(FancyBboxPatch((22 + i * 17, -0.6), 2.4, 1.8, boxstyle='round,pad=0,rounding_size=0.4', fc=fc, ec='none'))
        ax.text(25.2 + i * 17, 0.3, t, fontsize=6.5, va='center', color=INK2)
    fig.savefig(OUT / 'fig1_refresh_schematic.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'fig1_refresh_schematic.png', bbox_inches='tight', dpi=220); plt.close(fig)


# ---------------------------------------------------------------- hops: the headline figure (all models, both tasks)
def _line(ax, m, xs, mu, ci=None):
    kw = dict(color=C[m], marker=MK[m], ls=LS(m))
    if m in OPEN: kw.update(mfc='white', lw=1.1)
    if m in REFS: kw.update(lw=1.2)
    if ci is not None:
        ci = np.array(ci); h = ax.errorbar(xs, mu, yerr=[np.array(mu) - ci[:, 0], ci[:, 1] - np.array(mu)], capsize=0, elinewidth=.8, **kw)
    else: h, = ax.plot(xs, mu, **kw)
    ax._series = getattr(ax, '_series', []) + [(m, h)]


def _group_legend(ax, series, group, x, y=-.27, ncol=1):
    """One legend per recipe group, below the axis, so the solid / dashed split is named and no data is covered."""
    hs = [(h, NAME[m]) for m, h in series if GROUP(m) == group]
    lg = Legend(ax, *zip(*hs), loc='upper left', bbox_to_anchor=(x, y), ncol=ncol, handlelength=2.8, fontsize=6.8, title_fontsize=7,
                   alignment='left', labelspacing=.3, columnspacing=1.2, borderaxespad=0,
                   title=GTITLE[group])
    lg.set_in_layout(False); ax.add_artist(lg); ax.figure._legs = getattr(ax.figure, '_legs', []) + [lg]


def fig_hops():
    """Main figure: released GSM8K checkpoints, mean hops by serial distance (t42). The RuleChain counts by chain length
    are panel b of depth_failure; the RuleChain counts by steps from the edit are fig_hops_by_s (appendix)."""
    fig, ax = plt.subplots(figsize=(W * .68, 2.5)); fig._legs = []; xs = np.arange(2, 6)
    ax.fill_between(xs, xs - 1, xs, color='#eeedea', lw=0, zorder=0)
    ax.text(4.55, 4.75, 'one crossing\nper step', fontsize=6.3, color=INK2, ha='center', va='center')
    G = manifest('20261114-040437_codi_gsm8k-serial-depth-reanalysis')['strict']['models']           # t42
    for m, k in (('gsm8k_sim', 'sim'), ('gsm8k_simcoconut', 'simcoconut'), ('gsm8k_codi', 'codi'), ('gsm8k_coconut', 'coconut')):
        h = G[k]['hops_by_serial_s']; ss = [int(s) for s in sorted(h) if h[s]['n'] >= 15]
        _line(ax, m, ss, [h[str(s)]['mean'] for s in ss], [h[str(s)]['ci'] for s in ss])
    ax.set_xlabel('serial distance from the edit to the answer'); ax.set_ylabel('loop crossings used (mean hops)')
    ax.set_xticks(xs); ax.set_xlim(1.7, 5.3); ax.set_ylim(-0.25, 6.3)
    _group_legend(ax, ax._series, 'per-step', 1.04, y=1.0); _group_legend(ax, ax._series, 'standard', 1.04, y=.6)
    fig.tight_layout(); fig.savefig(OUT / 'hops.pdf', bbox_inches='tight', bbox_extra_artists=fig._legs); plt.close(fig)


def fig_hops_by_s():
    """Appendix: RuleChain, every model, mean hops by the number of steps s from the edit to the answer, for edits at every
    position, L = 2-6. Own-answer pairs, as in every hop count of the paper: a pair counts if the model's own answers to A and
    B differ, right or wrong (t31 for the CODI models and Coconut, t37 for SIM-Coconut). Rule-gated CODI is split by edit
    type, because its attention mask fixes the count for rule edits at s - 1."""
    fig, ax = plt.subplots(figsize=(W * .68, 2.7)); fig._legs = []; xs = np.arange(1, 7)
    ax.fill_between(xs, xs - 1, xs, color='#eeedea', lw=0, zorder=0)
    ax.text(5.2, 3.0, 'one crossing\nper step', fontsize=6.3, color=INK2, ha='center', va='center')
    src = {m: ART / f't31/{m}.json' for m in ('gated', 'sim', 'simsf', 'startlast', 'relay', 'stab')}
    src.update(simcoconut=ART / 't37/hops.json', coconut=ART / 't31/coconut_rc.json')
    for m in ALL:
        by = defaultdict(lambda: defaultdict(list))
        for p in load(src[m])['pairs']:
            if p['qualified']: by[m if m != 'gated' else 'gated_start' if p['edit'] == 'start' else 'gated_rule'][p['s']].append(p['hops'])
        for key in sorted(by):
            ss = [s for s in sorted(by[key]) if len(by[key][s]) >= 15]
            _line(ax, key, ss, [np.mean(by[key][s]) for s in ss], [boot_ci(by[key][s]) for s in ss])
            print(f'  hops by s {key}:', {s: (len(by[key][s]), round(float(np.mean(by[key][s])), 3)) for s in ss})
    ax.set_xlabel('steps from the edit to the answer (s)'); ax.set_ylabel('loop crossings used (mean hops)')
    ax.set_xticks(xs); ax.set_ylim(-0.25, 6.3)
    _group_legend(ax, ax._series, 'per-step', 1.04, y=1.0); _group_legend(ax, ax._series, 'standard', 1.04, y=.42)
    fig.tight_layout(); fig.savefig(OUT / 'hops_by_s.pdf', bbox_inches='tight', bbox_extra_artists=fig._legs); plt.close(fig)


# ---------------------------------------------------------------- depth_failure: accuracy falls, the count does not rise
# Test record of each model's selected checkpoint (500 items per level; TRAINING_PROTOCOL.md: test touched once per model).
TEST_RECORD = dict(gated='20261107-204814_codi_v2-l1to7w-startlast-gated-depth2-codi',
                   sim='20261107-225603_codi_v2-l1to7w-startlast-sim-depth2-codi',
                   simsf='20261108-130905_codi_v2-l1to7w-sim-depth2-codi',
                   simcoconut='20261104-144235_coconut_v2-depth2-pilot_k8_simcot_c1m6',
                   startlast='20261106-083843_codi_v2-l1to7w-startlast-depth2-codi',
                   relay='20261103-193425_codi_v2-l1to7w-depth2-codi',
                   stab='20261105-044444_codi_v2-l1to7w-stab-depth2-codi',
                   coconut='20261102-022150_coconut_v2-depth2-gpt2_depth2_coconut_c1m6',
                   codi0_sl='20261106-052644_codi_v2-l1to7w-startlast-depth2-codi0',
                   codi0_sf='20261105-024111_codi_v2-l1to7w-stab-depth2-codi0',
                   pause='20261108-182654_codi_v2-l1to7w-startlast-pause-depth2-codi')
test_acc = lambda m: manifest(TEST_RECORD[m])['per_level_accuracy']


def fig_depth_failure(models=MAIN, out='depth_failure.pdf'):
    """By chain length L = 2..7: test accuracy (a), from each model's test record, and mean hops when the start value is
    edited (b), so that the whole chain lies between the edit and the answer. Hops use own-answer pairs throughout: CODI
    models t31 (L2-6) + t41 (L7); Coconut t31/coconut_rc (t33), SIM-Coconut t37, both L2-6 only (no L7 audit)."""
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.4)); Ls = [2, 3, 4, 5, 6, 7]; fig._legs = []
    for m in REFS:   # no-loop references, accuracy only (they have no crossings to count)
        acc = test_acc(m); _line(axs[0], m, Ls, [acc[str(L)] for L in Ls])
        print(f'  depth_failure {m}: test acc', {L: round(acc[str(L)], 3) for L in Ls})
    axs[1].fill_between(Ls, [L - 1 for L in Ls], Ls, color='#eeedea', lw=0, zorder=0)
    axs[1].text(6.1, 3.7, 'one crossing\nper step', fontsize=6.3, color=INK2, ha='center', va='center')
    axs[1].axhline(K, color=MUTED, ls=':', lw=.8); axs[1].text(2.05, K + .12, 'six latents', fontsize=6.3, color=INK2, va='bottom')
    hop_src = {m: [ART / f't31/{m}.json', ART / f't41/{m}.json'] for m in ('gated', 'sim', 'simsf', 'startlast', 'relay', 'stab')}
    hop_src.update(simcoconut=[ART / 't37/hops.json'], coconut=[ART / 't31/coconut_rc.json'])
    for m in models:
        acc = test_acc(m)
        by = defaultdict(list)
        for f in hop_src[m]:
            for p in load(f)['pairs']:
                if p['qualified'] and p['edit'] == 'start': by[p['level']].append(p['hops'])
        hL = [L for L in Ls if len(by[L]) >= 15]
        print(f'  depth_failure {m}: test acc', {L: round(acc[str(L)], 3) for L in Ls}, '| n start-edit pairs', {L: len(by[L]) for L in hL})
        _line(axs[0], m, Ls, [acc[str(L)] for L in Ls])
        _line(axs[1], m, hL, [np.mean(by[L]) for L in hL], [boot_ci(by[L]) for L in hL])
    for ax in axs: ax.set_xticks(Ls); ax.set_xlabel('rules in the chain (L)')
    axs[0].set_ylim(0, 1.04); axs[0].set_ylabel('test accuracy'); title(axs[0], 'a', 'Accuracy')
    axs[1].set_ylim(-0.2, 7.2); axs[1].set_ylabel('mean hops, start value edited'); title(axs[1], 'b', 'Crossings used for the whole chain')
    _group_legend(axs[0], axs[0]._series, 'per-step', 0, y=-.3); _group_legend(axs[0], axs[0]._series, 'ref', .52, y=-.3)
    _group_legend(axs[1], axs[1]._series, 'standard', 0, y=-.3)
    fig.tight_layout(w_pad=1.8); fig.savefig(OUT / out, bbox_inches='tight', bbox_extra_artists=fig._legs); plt.close(fig)


def fig_hops_vs_accuracy():
    """Appendix: the two panels of depth_failure against each other. One point per model and chain length: test accuracy
    against mean hops when the start value is edited (same sources as depth_failure). The ends of each line carry L."""
    fig, ax = plt.subplots(figsize=(W * .68, 2.9)); fig._legs = []
    hop_src = {m: [ART / f't31/{m}.json', ART / f't41/{m}.json'] for m in ('gated', 'sim', 'simsf', 'startlast', 'relay', 'stab')}
    hop_src.update(simcoconut=[ART / 't37/hops.json'], coconut=[ART / 't31/coconut_rc.json'])
    for m in ALL:
        acc = test_acc(m); by = defaultdict(list)
        for f in hop_src[m]:
            for p in load(f)['pairs']:
                if p['qualified'] and p['edit'] == 'start': by[p['level']].append(p['hops'])
        Ls = [L for L in range(2, 8) if len(by[L]) >= 15]; h = [float(np.mean(by[L])) for L in Ls]; a = [acc[str(L)] for L in Ls]
        _line(ax, m, h, a)
        for i in (0, -1):   # the three per-step CODI lines overlap at accuracy 1, so they share one label below
            if m not in ('gated', 'sim', 'simsf'):
                ax.annotate(f'L={Ls[i]}', (h[i], a[i]), xytext=(4, 5 if i == 0 else -8), textcoords='offset points', fontsize=5.6, color=C[m])
    ax.annotate('L = 2', (1.85, 1.0), xytext=(0, 6), textcoords='offset points', fontsize=5.6, color=INK2, ha='center')
    ax.annotate('L = 7', (5.95, 1.0), xytext=(0, 6), textcoords='offset points', fontsize=5.6, color=INK2, ha='center')
    ax.set_xlabel('mean hops, start value edited'); ax.set_ylabel('test accuracy'); ax.set_ylim(-0.03, 1.1); ax.set_xlim(-0.25, 6.5)
    _group_legend(ax, ax._series, 'per-step', 1.04, y=1.0); _group_legend(ax, ax._series, 'standard', 1.04, y=.5)
    fig.tight_layout(); fig.savefig(OUT / 'hops_vs_accuracy.pdf', bbox_inches='tight', bbox_extra_artists=fig._legs); plt.close(fig)


# ---------------------------------------------------------------- monitor: what the latents add over question + answer
def fig_monitor():
    """Branch taken at a middle rule, balanced accuracy. Baseline = question pass + own answer; with latents = best of +Z / +H.
    Dot = mean over (level, rule) cells, thin line = range over cells."""
    t34 = manifest('20261112-130851_codi_v2-l1to7w-hidden-path-detection')['by_model']
    t37 = manifest('20261112-211705_coconut_v2-rulechain-simcot-coconut-hops-hidden-path')['hidden_path']
    rows = [('gated', t34['gated']), ('sim', t34['sim']), ('simsf', t34['simsf']), ('simcoconut', t37),
            ('startlast', t34['startlast']), ('relay', t34['relay']), ('stab', t34['stab'])]
    fig, ax = plt.subplots(figsize=(W, 2.5)); ys = [7.3, 6.3, 5.3, 4.3, 2.4, 1.4, 0.4]; cb, cl = '#8a8984', C['gated']
    for (m, lv), y in zip(rows, ys):
        base, lat = [], []
        for x in lv.values():
            c = x['cells']
            for r in sorted({k.split('|')[1] for k in c}):
                base.append(c[f'QA|{r}']['all']); lat.append(max(c[f'{f}|{r}']['all'] for f in ('QAZ', 'QAH')))
        b, l = np.mean(base), np.mean(lat)
        ax.plot([b, l], [y, y], color='#c9c8c3', lw=2.2, zorder=1, solid_capstyle='round')
        ax.plot([min(base), max(base)], [y - .17, y - .17], color=cb, lw=.8); ax.plot([min(lat), max(lat)], [y + .17, y + .17], color=cl, lw=.8)
        ax.plot(b, y, 'o', ms=6.5, color=cb, mec='white', mew=.8, zorder=3, label='question + answer' if m == 'gated' else None)
        ax.plot(l, y, 'o', ms=6.5, color=cl, mec='white', mew=.8, zorder=3, label='with latents' if m == 'gated' else None)
        ax.text(1.035, y, f'+{l - b:.2f}'.replace('+0.', '+.'), fontsize=7.5, va='center', ha='left', color=INK)
        print(f'  monitor {m}: base {b:.3f} [{min(base):.2f}, {max(base):.2f}]  latents {l:.3f} [{min(lat):.2f}, {max(lat):.2f}]  gain {l - b:+.3f}')
    ax.set_yticks(ys); ax.set_yticklabels([NAME[m] for m, _ in rows]); ax.set_ylim(-.3, 8.6); ax.set_xlim(.4, 1.02)
    ax.axvline(.5, color=MUTED, ls=':', lw=1); ax.text(.505, 3.35, 'chance', fontsize=6.5, color=INK2, va='center')
    for yy, lab in ((8.25, 'Per-step training'), (3.35, 'Standard recipe')):
        ax.text(-.012, yy, lab, transform=ax.get_yaxis_transform(), fontsize=7.5, fontweight='bold', color=INK, ha='right', va='center')
    ax.text(1.035, 8.25, 'gain', fontsize=7, color=INK2, va='center', ha='left')
    ax.set_xlabel('balanced accuracy: which branch did a middle rule take?'); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.0), ncol=2, handletextpad=.3, columnspacing=1.6)
    fig.tight_layout(); fig.savefig(OUT / 'monitor.pdf', bbox_inches='tight'); plt.close(fig)


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


def rc_read_summary(m, edit=None):
    """t29 items of a RuleChain model, rescored as in rc_readability.py, optionally for one edit type ('start' / 'rule').
    The saved items carry no edit type, so it is recovered by replaying the script's pair selection; every item is checked
    against the replay (level, step, d and both values)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codi_null_audit'))
    import rc as RC
    vals = lambda q: RC.walk(*RC.parse(q))[0]
    items = load(T26 / f'rcread_{m}.json')['items']; cnt = defaultdict(int); n = 0
    for p in load(T26 / f'{m}.json')['pairs']:
        if not (p['qualified'] and p['level'] in (2, 3, 4, 5, 6)) or cnt[(p['level'], p['edit'])] >= 60: continue
        cnt[(p['level'], p['edit'])] += 1
        L = p['level']; va, vb = vals(p['R']['question']), vals(p['D']['question']); i = 0 if p['edit'] == 'start' else int(p['edit'][4:])
        for k in range(1, L):
            if vb[k] == va[k]: continue
            it = items[n]; n += 1
            assert (it['L'], it['k'], it['d'], it['a'], it['b']) == (L, k, k - max(i, 1) + 1, int(va[k]), int(vb[k])), (m, n)
            it['edit'] = 'start' if i == 0 else 'rule'
    assert n == len(items), (m, n, len(items))
    read = lambda it, r, v: it['pred'] is not None and v is not None and v in it['pred'][r]
    S = {}
    for d in sorted({it['d'] for it in items}):
        xs = [it for it in items if it['d'] == d and edit in (None, it['edit'])]; own = [it for it in xs if read(it, K, it['b'])]
        if len(xs) < 5: continue
        S[d] = dict(n_own=len(own), B=[sum(read(it, r, it['b']) for it in own) / max(len(own), 1) for r in range(K + 1)],
                    A=[sum(read(it, r, it['a']) for it in own) / max(len(own), 1) for r in range(K + 1)],
                    chance=[sum(read(it, r, it['chance']) for it in xs) / len(xs) for r in range(K + 1)])
    return S


def fig_readability():
    # Rule-gated CODI is shown for rule edits only: for start edits d is one too small on most pairs (the start value usually
    # takes one crossing to enter the first latent), which the text states in one sentence.
    cols = [('gated', 'RuleChain\nrule-gated CODI\nrule edits', 'probe', 'rule'), ('sim', 'RuleChain\nSIM-CODI\nall edits', 'probe', None),
            ('gsm8k_sim', 'GSM8K\nSIM-CODI\n(released)', 'decoder', None), ('gsm8k_codi', 'GSM8K\nCODI\n(released)', 'lens', None)]
    fig, axs = plt.subplots(2, 4, figsize=(W, 3.5), sharex=True, sharey=True); NS = {}
    for e in ('rule', 'start'):   # the numbers quoted in the text: share readable at refresh d - 1, d, d + 1
        for m in ('gated', 'sim', 'simsf'):
            S = rc_read_summary(m, e)
            print(f'  readability {m} {e} edits, B at d-1 / d / d+1:', {d: [round(S[d]['B'][d + o], 2) for o in (-1, 0, 1)] for d in S})
    for c, (m, ttl, how, edit) in enumerate(cols):
        if how == 'probe':
            S = rc_read_summary(m, edit); pooled = {int(k): v for k, v in load(T26 / f'rcread_{m}.json')['summary'].items()}
            if edit is None: assert all(np.allclose(S[d]['B'], pooled[d]['B']) and np.allclose(S[d]['A'], pooled[d]['A']) for d in pooled), m
        elif how == 'lens': S = {int(k): v for k, v in load(T26 / 'lens_gsm8k_codi.json')['summary'].items()}
        else: S = sim_decoder_summary()
        ds = [d for d in sorted(S) if S[d]['n_own'] >= 25][:4]; NS[(m, edit)] = {d: S[d]['n_own'] for d in ds}
        for d in ds:
            x = np.arange(K + 1) - d; keep = (x >= -3) & (x <= 2)
            for r, key in ((0, 'B'), (1, 'A')):
                axs[r, c].plot(x[keep], np.array(S[d][key])[keep], color=ORD[d], marker='o', ms=3, lw=1.3, label=f'd = {d}')
        w = np.array([S[d]['n_own'] for d in ds], float)
        ch = max(float(np.dot(w, [S[d]['chance'][n] for d in ds]) / w.sum()) for n in range(K + 1))
        axs[0, c].axhline(ch, color=MUTED, ls=':', lw=1); axs[0, c].text(2.1, ch + .035, f'chance {ch:.2f}', fontsize=5.6, color=INK2, ha='right')
        axs[0, c].set_title(ttl, fontsize=6.6, loc='center'); axs[1, c].set_xlabel('refresh − d')
        axs[1, c].text(2.1, .93, how, fontsize=6.2, color=INK2, ha='right')
        for r in (0, 1): axs[r, c].axvline(0, color=INK2, lw=.6, alpha=.5)
    axs[0, 0].set_ylabel("B's new value\nreadable"); axs[1, 0].set_ylabel("A's old value\nreadable")
    for ax in axs.flat: ax.set_xticks([-3, -2, -1, 0, 1, 2]); ax.set_ylim(-0.03, 1.05); ax.tick_params(labelsize=6.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, [x + (' (crossings needed)' if i == 0 else '') for i, x in enumerate(l)], loc='lower center', ncol=4,
               bbox_to_anchor=(.5, -0.03), handlelength=1.8)
    fig.tight_layout(h_pad=.5, w_pad=.4, rect=(0, .04, 1, 1)); fig.savefig(OUT / 'readability.pdf', bbox_inches='tight'); plt.close(fig)
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


def fig_gsm8k_vs_rulechain():
    """Main figure (Nov 24, GSM8K subsection): mean hops by steps from the edit to the answer, pairs answered right on both
    sides. Released GSM8K CODI: edit at the start of the longest dependency path, so the distance is the item's serial depth
    (steps_per_pass/gsm8k_steps_per_pass.json, start_of_path; unaffected by the twin-label artefact). RuleChain start-last
    CODI and SIM-CODI: t31 pairs by s. Cells with fewer than 15 pairs are not drawn."""
    fig, ax = plt.subplots(figsize=(W * .62, 2.5)); xs = np.arange(1, 7); std = (0, (4, 2))
    ax.fill_between(xs, xs - 1, xs, color='#eeedea', lw=0, zorder=0)
    ax.text(5.35, 3.3, 'one crossing\nper step', fontsize=6.3, color=INK2, ha='center', va='center')
    for m, lab, mk, ls in (('sim', 'RuleChain SIM-CODI (per-step)', 's', '-'), ('startlast', 'RuleChain CODI, start last', '^', std)):
        by = defaultdict(list)
        for p in load(ART / f't31/{m}.json')['pairs']:
            if p.get('qualified') and p.get('hops') is not None and p.get('R_right') and p.get('D_right'): by[p['s']].append(p['hops'])
        ss = [s for s in sorted(by) if len(by[s]) >= 15]; mu = np.array([np.mean(by[s]) for s in ss]); ci = np.array([boot_ci(by[s]) for s in ss])
        ax.errorbar(ss, mu, yerr=[mu - ci[:, 0], ci[:, 1] - mu], color=C[m], marker=mk, ls=ls, capsize=0, elinewidth=.8, label=lab)
        print('  gsm8k_vs_rulechain', m, {s: (len(by[s]), round(float(np.mean(by[s])), 2)) for s in ss})
    G = load(ART / 'steps_per_pass/gsm8k_steps_per_pass.json')['codi']['start_of_path']
    ds = sorted(int(d) for d in G if G[d]['n'] >= 15); mu = np.array([G[str(d)]['hops'] for d in ds]); ci = np.array([G[str(d)]['hops_ci'] for d in ds])
    ax.errorbar(ds, mu, yerr=[mu - ci[:, 0], ci[:, 1] - mu], color=INK, marker='o', mfc='white', ls=std, capsize=0, elinewidth=.8,
                label='GSM8K CODI (released)')
    print('  gsm8k_vs_rulechain gsm8k_codi', {d: (G[str(d)]['n'], round(G[str(d)]['hops'], 2)) for d in ds})
    ax.set_xlabel('steps from the edit to the answer'); ax.set_ylabel('loop crossings used (mean hops)')
    ax.set_xticks(xs); ax.set_xlim(.7, 6.3); ax.set_ylim(-0.25, 6.3); ax.legend(loc='upper left', fontsize=6.8, handlelength=2.8)
    fig.tight_layout(); fig.savefig(OUT / 'gsm8k_vs_rulechain.pdf', bbox_inches='tight'); plt.close(fig)


import os  # noqa: E402
if os.environ.get('ONLY'):  # ONLY=<function name>: write one figure, skip the rest
    globals()[os.environ['ONLY']](); print('wrote', os.environ['ONLY']); sys.exit()
for f in (fig_schematic, fig_hops, fig_depth_failure, fig_monitor, fig_readability, fig_gsm8k_dist, fig_route_split):
    f(); print('wrote', f.__name__)
fig_hops_by_s(); fig_hops_vs_accuracy(); fig_depth_failure(ALL, 'depth_failure_all.pdf'); print('wrote appendix versions (all models)')
