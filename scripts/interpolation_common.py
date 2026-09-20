"""Shared math/plotting for "smooth path interpolation patching" (Task 3): pure
functions with no mechanism-specific model code, used by both
`interpolation_patch_codi.py` and `interpolation_patch_coconut.py` so the alpha-sweep
classification and the trajectory-plot format are identical across mechanisms and
therefore comparable. Everything mechanism-specific (how to run a forward pass, how to
override a latent, what "the answer logits" means for that architecture) stays in each
script -- this module only knows about the resulting (alpha, P(yA), P(yB), entropy)
arrays.

For a (Recipient A, Donor B) pair with different gold answers, at a chosen latent slot
z_i, interpolate z_i(alpha) = (1-alpha) z_i^A + alpha z_i^B for alpha in
[0.0, 0.1, ..., 1.0], patch it in, and read the vocabulary distribution over the
"answer token" position immediately after the latent computation finishes (each
script's own single logits readout -- CODI's post-loop pre-eot-generation logits,
Coconut's first post-latent-sequence logits). P(yA)/P(yB) = softmax probability mass
on the first token of each example's tokenized gold answer at that single readout;
H = Shannon entropy (nats) of that same distribution.
"""
from __future__ import annotations

import math

ALPHAS: tuple[float, ...] = tuple(round(i / 10, 1) for i in range(11))


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def entropy_from_probs(probs) -> float:
    """`probs` is any iterable of non-negative floats summing to ~1 (a torch tensor
    works fine as an iterable of 0-d floats via .tolist())."""
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def classify_trajectory(
    alphas: list[float], pa: list[float], pb: list[float], h: list[float],
    collapse_prob: float = 0.01, collapse_entropy_jump: float = 1.5, mono_thresh: float = 0.5,
    step_dominance_thresh: float = 0.5,
) -> str:
    """One of "off_manifold_collapse", "step_function_invariance", "smooth_transition",
    "ambiguous". Checked in that priority order.

    - collapse: BOTH P(yA) and P(yB) drop under `collapse_prob` at some interior alpha
      (excluding the endpoints, where one of them is expected to be small if the model
      is confident) while entropy jumps by more than `collapse_entropy_jump` nats above
      the boundary entropies -- "the distribution goes somewhere neither answer, and it
      goes uniform/garbage while doing it", not just "the model changed its mind".
    - step_function_invariance: sign(P(yA)-P(yB)) flips exactly once across the sweep
      AND that flip is concentrated in a single alpha-step -- the biggest one-step
      change in (P(yA)-P(yB)) accounts for at least `step_dominance_thresh` of the total
      variation (sum of |consecutive changes|). Checked BEFORE the smooth-transition
      trend test: a genuine step function still has strongly-correlated endpoints
      (P(yA) high at alpha=0, low at alpha=1) and would otherwise pass that test too --
      concentration, not just net direction, is what actually distinguishes the two.
    - smooth_transition: not a concentrated single-step flip, but P(yA) trends down and
      P(yB) trends up across the sweep (Pearson correlation with alpha beyond
      +/-`mono_thresh` in the expected direction) -- change spread across the path.
    - ambiguous: none of the above (multiple flips, no trend, no collapse -- a noisy or
      genuinely mixed trajectory).
    """
    baseline_h = max(h[0], h[-1])
    interior = range(1, len(alphas) - 1)
    collapse = any(
        pa[i] < collapse_prob and pb[i] < collapse_prob and h[i] > baseline_h + collapse_entropy_jump
        for i in interior
    )
    if collapse:
        return "off_manifold_collapse"

    diff = [pa[i] - pb[i] for i in range(len(alphas))]
    signs = [1 if d > 0 else (-1 if d < 0 else 0) for d in diff]
    nonzero = [s for s in signs if s != 0]
    flips = sum(1 for i in range(1, len(nonzero)) if nonzero[i] != nonzero[i - 1])
    steps = [abs(diff[i + 1] - diff[i]) for i in range(len(diff) - 1)]
    total_variation = sum(steps)
    step_fraction = (max(steps) / total_variation) if total_variation > 0 else 0.0
    if flips == 1 and step_fraction >= step_dominance_thresh:
        return "step_function_invariance"

    corr_a = pearson_corr(alphas, pa)
    corr_b = pearson_corr(alphas, pb)
    if corr_a <= -mono_thresh and corr_b >= mono_thresh:
        return "smooth_transition"

    return "ambiguous"


def classification_summary(labels: list[str]) -> dict:
    n = len(labels)
    keys = ("smooth_transition", "off_manifold_collapse", "step_function_invariance", "ambiguous")
    counts = {k: labels.count(k) for k in keys}
    return {
        "n": n,
        "counts": counts,
        "pct": {k: (counts[k] / n if n else 0.0) for k in keys},
    }


def plot_interpolation_curves(
    curves: dict, alphas: tuple[float, ...], out_path: str,
    decodable_positions: tuple, nondecodable_positions: tuple, position_label,
    title: str = "Smooth path interpolation patching",
) -> None:
    """`curves[pos]` = {"mean_pa": [...], "mean_pb": [...], "mean_h": [...]} (one entry
    per swept position). Plots mean P(yA), P(yB), entropy vs alpha, one line per
    position, decodable positions solid and non-decodable dashed, so the two groups the
    task description asks for are visually separable in a single figure per metric.
    Requires matplotlib (`pip install matplotlib` in whichever venv runs the script --
    not a base project dependency, see pyproject.toml)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [("mean_pa", "P(y_A)  (recipient's own answer)"), ("mean_pb", "P(y_B)  (donor's answer)"),
               ("mean_h", "Entropy H(P(y))  [nats]")]
    positions = sorted(curves.keys())
    cmap = plt.get_cmap("tab10")
    for ax, (key, ylabel) in zip(axes, metrics):
        for i, pos in enumerate(positions):
            style = "--" if pos in nondecodable_positions else "-"
            marker = "o" if pos in decodable_positions else "s" if pos in nondecodable_positions else "."
            ax.plot(alphas, curves[pos][key], style, marker=marker, color=cmap(i % 10),
                     label=position_label(pos), linewidth=1.8, markersize=4)
        ax.set_xlabel("alpha (0=recipient A, 1=donor B)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
