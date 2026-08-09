"""Generate the CXT-Fish manuscript figure system.

The figures are rebuilt from the frozen CSV artifacts and vector primitives.
No scientific image is generated or redistributed.  The visual system is
deliberately restrained: compact panels, direct labels, white background,
colour-blind-safe accents, and editable PDF/SVG text.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Ellipse

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "reports" / "figures" / "manuscript_submission_v2"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1d2a35"
MUTED = "#687582"
GRID = "#dce3e8"
NAVY = "#315f8a"
TEAL = "#168b83"
AMBER = "#c88935"
BURGUNDY = "#ae5a5d"
PURPLE = "#756fa3"
PALE_BLUE = "#eaf1f6"
PALE_TEAL = "#e7f3f1"
PALE_AMBER = "#fbf3e6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#82909b",
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.55,
    "grid.alpha": 0.8,
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save(fig: plt.Figure, name: str) -> None:
    stem = Path(name).stem
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    plt.close(fig)


def panel(ax, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=10, weight="bold", color=INK)


def clean_axis(ax, grid=True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#82909b")
    ax.spines["bottom"].set_color("#82909b")
    ax.grid(grid, axis="y" if grid else "both")
    ax.tick_params(length=3, width=0.7, labelsize=8)


def box(ax, xy, width, height, text, edge=NAVY, fill="white", fs=8.0,
        lw=1.0, text_color=INK):
    x, y = xy
    p = Rectangle((x, y), width, height, linewidth=lw, edgecolor=edge,
                  facecolor=fill, joinstyle="round")
    ax.add_patch(p)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fs, color=text_color, linespacing=1.2)
    return p


def arrow(ax, start, end, color=MUTED, lw=1.0, style="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, linestyle=style, color=color,
                                 shrinkA=2, shrinkB=2))


def fig1_protocol() -> None:
    fig = plt.figure(figsize=(10.5, 3.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 0.72], hspace=0.23)
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel(ax, "a")
    ax.text(0, 1.0, "Evidence chain", transform=ax.transAxes, fontsize=10,
            weight="bold", color=INK, va="bottom")
    labels = ["F4K-16T\nconstruction", "Group-aware\nsplit", "Context\ninterventions", "Foreground\nsufficiency", "Frozen\nre-evaluation"]
    fills = [PALE_BLUE, PALE_BLUE, PALE_TEAL, PALE_TEAL, PALE_BLUE]
    edges = [NAVY, NAVY, TEAL, TEAL, NAVY]
    xs = np.linspace(0.02, 0.80, 5)
    for i, (x, lab) in enumerate(zip(xs, labels), 1):
        ax.add_patch(Rectangle((x, 0.30), 0.145, 0.36, facecolor=fills[i-1],
                               edgecolor=edges[i-1], linewidth=1.1))
        ax.add_patch(Ellipse((x + 0.018, 0.70), 0.034, 0.034,
                             facecolor=edges[i-1], edgecolor="white", linewidth=0.6))
        ax.text(x + 0.018, 0.70, str(i), ha="center", va="center", color="white",
                fontsize=7, weight="bold")
        ax.text(x + 0.0725, 0.48, lab, ha="center", va="center", fontsize=8.0,
                color=INK, linespacing=1.2)
        if i < 5:
            arrow(ax, (x + 0.145, 0.48), (xs[i] - 0.012, 0.48), color="#7c8993")
    ax.text(0.98, 0.10, "same corpus", ha="right", va="center", fontsize=8,
            color=MUTED)
    ax.plot([0.02, 0.98], [0.10, 0.10], color=GRID, linewidth=0.8)

    ax2 = fig.add_subplot(gs[1]); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    panel(ax2, "b")
    ax2.text(0, 1.0, "Bounded post-hoc controls", transform=ax2.transAxes,
             fontsize=10, weight="bold", color=INK, va="bottom")
    controls = [("donor realization", "5 deterministic seeds", AMBER),
                ("two-RGB control", "duplicated supervised views", NAVY),
                ("construct validity", "subject-suppressed sensitivity", TEAL)]
    for i, (head, sub, col) in enumerate(controls):
        x = 0.02 + i * 0.325
        ax2.add_patch(Rectangle((x, 0.19), 0.27, 0.45, facecolor="white",
                                edgecolor=col, linewidth=1.0))
        ax2.add_patch(Rectangle((x, 0.19), 0.012, 0.45, facecolor=col,
                                edgecolor=col, linewidth=0))
        ax2.text(x + 0.03, 0.49, head, ha="left", va="center", fontsize=8.5,
                 weight="bold", color=INK)
        ax2.text(x + 0.03, 0.31, sub, ha="left", va="center", fontsize=7.6,
                 color=MUTED)
    ax2.text(0.98, 0.05, "post-hoc; not independent validation", ha="right",
             va="center", fontsize=7.6, color=MUTED, style="italic")
    save(fig, "fig1_protocol.png")


def fig2_gate0() -> None:
    d = pd.read_csv(EXP / "ctp_fish_gate0_summary.csv")
    means = d.groupby("split").mean(numeric_only=True)
    pc = pd.read_csv(EXP / "ctp_fish_gate0_per_class_summary.csv")
    tier_means = pc.groupby(["split", "tier"])["f1"].mean().unstack("split")
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1), gridspec_kw={"wspace": 0.36})
    # a: paired protocol estimate
    ax = axs[0]; panel(ax, "a")
    vals = {"macro-F1": [means.loc["image", "macro_f1"], means.loc["track", "macro_f1"]],
            "group-balanced accuracy": [means.loc["image", "track_balanced_accuracy"], means.loc["track", "track_balanced_accuracy"]]}
    for name, ys, col in [("macro-F1", vals["macro-F1"], NAVY),
                          ("group-balanced accuracy", vals["group-balanced accuracy"], AMBER)]:
        ax.plot([0, 1], ys, color=col, linewidth=2.0, marker="o", markersize=5,
                label=name)
    ax.set_xticks([0, 1], ["image-level", "group-disjoint"])
    ax.set_ylim(0.90, 1.005); ax.set_ylabel("score"); clean_axis(ax)
    ax.legend(frameon=False, fontsize=7.3, loc="lower left")
    # b: class strata
    ax = axs[1]; panel(ax, "b")
    for tier, col in [("head", NAVY), ("mid", TEAL), ("tail", AMBER)]:
        y = [tier_means.loc[tier, "image"], tier_means.loc[tier, "track"]]
        ax.plot([0, 1], y, marker="o", linewidth=1.8, markersize=5, label=tier, color=col)
    ax.set_xticks([0, 1], ["image-level", "group-disjoint"])
    ax.set_ylim(0.88, 1.005); ax.set_ylabel("per-class F1"); clean_axis(ax)
    ax.legend(frameon=False, fontsize=7.3, loc="lower left")
    # c: nearest-neighbour audit
    ax = axs[2]; panel(ax, "c")
    ax.barh([1, 0], [0.4821, 0.0], color=[NAVY, AMBER], height=0.42)
    ax.set_yticks([1, 0], ["image-level", "group-disjoint"])
    ax.set_xlim(0, 0.55); ax.set_xlabel("same-group nearest-neighbour fraction")
    clean_axis(ax, grid=False)
    for y, v in [(1, 0.4821), (0, 0.0)]:
        ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=8, color=INK)
    save(fig, "fig2_gate0.png")


def fig3_context() -> None:
    labels = ["Original RGB", "Foreground-only", "Constant-fill", "Shuffled-mask",
              "Inpainted", "Background-only", "Mask-only", "Geometry-only"]
    vals = [0.937, 0.938, 0.895, 0.897, 0.805, 0.799, 0.713, 0.077]
    colors = [NAVY, TEAL, "#93a95d", "#93a95d", AMBER, AMBER, PURPLE, MUTED]
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    panel(ax, "a")
    y = np.arange(len(labels))
    for yi, idx in enumerate(order):
        ax.hlines(yi, 0, vals[idx], color=GRID, linewidth=3.5, zorder=1)
        ax.plot(vals[idx], yi, "o", color=colors[idx], markersize=7, zorder=3)
        ax.text(vals[idx] + 0.015, yi, f"{vals[idx]:.3f}", va="center", fontsize=8,
                color=INK)
    ax.set_yticks(y, [labels[i] for i in order])
    ax.set_xlim(0, 1.05); ax.set_xlabel("historical development macro-F1")
    ax.set_title("Context-diagnostic views", loc="left", fontsize=10, weight="bold", color=INK)
    ax.axvline(0.06, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.text(0.06, 1.01, "chance reference", transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=7.4, color=MUTED)
    clean_axis(ax)
    save(fig, "fig3_context.png")


def _fish(ax, cx=0.50, cy=0.50, scale=1.0, fill="#7896a6", edge="#415563"):
    ax.add_patch(Ellipse((cx, cy), 0.46 * scale, 0.16 * scale, angle=-8,
                         facecolor=fill, edgecolor=edge, linewidth=0.9))
    ax.add_patch(Polygon([[cx - 0.21 * scale, cy + 0.01 * scale],
                          [cx - 0.31 * scale, cy - 0.07 * scale],
                          [cx - 0.27 * scale, cy + 0.10 * scale]],
                         facecolor=fill, edgecolor=edge, linewidth=0.8))
    ax.plot(cx + 0.18 * scale, cy + 0.02 * scale, "o", color=INK, ms=2.0)


def fig4_donor() -> None:
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.8), gridspec_kw={"wspace": 0.20})
    titles = ["recipient RGB", "recipient mask", "donor context", "composite"]
    for ax, title in zip(axs, titles):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5, weight="bold", color=INK)
        ax.add_patch(Rectangle((0.08, 0.19), 0.84, 0.60, facecolor="white",
                               edgecolor="#9aa6b2", linewidth=0.8))
    for y, c in [(0.31, "#d7e6eb"), (0.45, "#b9d5dc"), (0.60, "#d7e6eb")]:
        axs[0].plot([0.10, 0.90], [y, y], color=c, linewidth=5, solid_capstyle="butt")
    _fish(axs[0])
    _fish(axs[1], fill=TEAL, edge="#206b60")
    axs[1].text(0.50, 0.08, r"$m_r$", ha="center", fontsize=9, color="#206b60")
    for y, c in [(0.31, "#e6d9c9"), (0.45, "#d8c7af"), (0.60, "#eadfd1")]:
        axs[2].plot([0.10, 0.90], [y, y], color=c, linewidth=5, solid_capstyle="butt")
    axs[2].text(0.50, 0.08, "different species / group", ha="center", fontsize=6.8,
                color=MUTED)
    for y, c in [(0.31, "#d8c4af"), (0.45, "#c9b08f"), (0.60, "#e1d1bd")]:
        axs[3].plot([0.10, 0.90], [y, y], color=c, linewidth=5, solid_capstyle="butt")
    _fish(axs[3])
    for ax in axs[:-1]: arrow(ax, (0.93, 0.49), (1.04, 0.49), color="#7c8993")
    fig.text(0.50, 0.035, r"$x_{r\leftarrow d}=m_r\odot x_r+(1-m_r)\odot R(x_d)$",
             ha="center", fontsize=8.5, color=INK)
    fig.text(0.50, 0.005, "vector schematic; context intervention is synthetic and mask-defined",
             ha="center", fontsize=7.2, color=MUTED)
    save(fig, "fig4_donor_intervention.png")


def fig5_method() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.1), gridspec_kw={"wspace": 0.28})
    # training panel
    ax = axs[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); panel(ax, "a")
    ax.text(0, 1.0, "training-time objective", transform=ax.transAxes,
            fontsize=10, weight="bold", color=INK, va="bottom")
    box(ax, (0.02, 0.60), 0.20, 0.16, "ordinary RGB\n$x$", NAVY, PALE_BLUE)
    box(ax, (0.02, 0.28), 0.20, 0.16, "foreground-sufficient\n$x_{fg}$", TEAL, PALE_TEAL)
    box(ax, (0.36, 0.42), 0.25, 0.28, "shared\nResNet18", NAVY, "white", fs=9)
    box(ax, (0.76, 0.60), 0.20, 0.16, r"$CE(f(x),y)$", NAVY, PALE_BLUE, fs=9)
    box(ax, (0.76, 0.28), 0.20, 0.16, r"$\lambda CE(f(x_{fg}),y)$", TEAL, PALE_TEAL, fs=8.1)
    arrow(ax, (0.22, 0.68), (0.36, 0.60), NAVY); arrow(ax, (0.22, 0.36), (0.36, 0.52), TEAL)
    arrow(ax, (0.61, 0.60), (0.76, 0.68), NAVY); arrow(ax, (0.61, 0.52), (0.76, 0.36), TEAL)
    ax.text(0.50, 0.12, r"$L=CE(f(x),y)+\lambda CE(f(x_{fg}),y)$;  $\lambda=1$",
            ha="center", fontsize=8.3, color=INK)
    ax.text(0.50, 0.02, "two supervised views, one concatenated forward", ha="center",
            fontsize=7.6, color=MUTED)
    # inference panel
    ax = axs[1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); panel(ax, "b")
    ax.text(0, 1.0, "inference-time graph", transform=ax.transAxes,
            fontsize=10, weight="bold", color=INK, va="bottom")
    box(ax, (0.08, 0.48), 0.22, 0.18, "ordinary RGB\n$x$", NAVY, PALE_BLUE, fs=9)
    box(ax, (0.40, 0.48), 0.28, 0.18, "ResNet18\nclassifier", NAVY, "white", fs=9)
    box(ax, (0.80, 0.48), 0.14, 0.18, r"$\hat y$", TEAL, PALE_TEAL, fs=10)
    arrow(ax, (0.30, 0.57), (0.40, 0.57), NAVY); arrow(ax, (0.68, 0.57), (0.80, 0.57), TEAL)
    ax.text(0.51, 0.28, "one image  ·  one forward pass", ha="center", fontsize=8.5, color=INK)
    ax.text(0.51, 0.17, "no mask  ·  no donor selection  ·  no auxiliary branch", ha="center",
            fontsize=7.5, color=MUTED)
    save(fig, "fig5_method.png")


def fig6_main() -> None:
    labels = ["clean", "foreground", "same-class\ncomposite", "cross-class\ncomposite"]
    f0 = np.array([0.9577, 0.8331, 0.9229, 0.5189])
    f1 = np.array([0.9556, 0.9550, 0.9291, 0.5913])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.45), gridspec_kw={"wspace": 0.32})
    ax = axs[0]; panel(ax, "a")
    for i in range(4):
        ax.plot([0, 1], [f0[i], f1[i]], color=GRID, linewidth=2.6, zorder=1)
        ax.scatter([0, 1], [f0[i], f1[i]], s=34, color=[NAVY, TEAL], zorder=2)
        ax.text(-0.06, f0[i], f0[i], ha="right", va="center", fontsize=7.2, color=NAVY)
        ax.text(1.06, f1[i], f1[i], ha="left", va="center", fontsize=7.2, color=TEAL)
    ax.set_xlim(-0.27, 1.28); ax.set_ylim(0.46, 1.01)
    ax.set_xticks([0, 1], ["F0", "CXT-Fish"]); ax.set_ylabel("macro-F1")
    ax.set_yticks([0.5, 0.7, 0.9, 1.0]); clean_axis(ax)
    label_positions = [(0.53, 0.986), (0.50, 0.845), (0.50, 0.944), (0.50, 0.555)]
    for (xlab, ylab), lab in zip(label_positions, labels):
        ax.text(xlab, ylab, lab, ha="center", va="center", fontsize=7.3,
                color=INK, linespacing=1.05,
                bbox={"boxstyle": "round,pad=0.10", "fc": "white", "ec": "none", "alpha": 0.82})
    # b: focal effect + guardrails
    ax = axs[1]; panel(ax, "b")
    names = ["cross-class\ncomposite", "group-balanced\naccuracy", "tail F1", "clean", "DAR-flip"]
    delta = np.array([7.24, 0.62, 0.26, -0.21, -3.54])
    cols = [TEAL, TEAL, TEAL, BURGUNDY, TEAL]
    y = np.arange(len(names))[::-1]
    ax.axvline(0, color="#8a969f", linewidth=0.8)
    for yi, v, c, lab in zip(y, delta, cols, names):
        ax.plot([0, v], [yi, yi], color=c, linewidth=2.5)
        ax.scatter(v, yi, s=38, color=c, zorder=3)
        ax.text(v + (0.20 if v >= 0 else -0.20), yi, f"{v:+.2f} pp", ha="left" if v >= 0 else "right",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(y, names); ax.set_xlabel("CXT-Fish − F0 (percentage points)")
    ax.set_xlim(-4.7, 8.8); clean_axis(ax, grid=False)
    ax.text(0.02, 1.03, "equal-weight fold-seed effects", transform=ax.transAxes,
            fontsize=7.5, color=MUTED)
    save(fig, "fig6_main_results.png")


def fig7_controls() -> None:
    donor = pd.read_csv(EXP / "cxt_fish_donor_sensitivity_summary.csv")
    rgb2 = pd.read_csv(EXP / "cxt_fish_rgb2_control_summary.csv")
    fish = pd.read_csv(EXP / "cxt_fish_fish_suppressed_donor_summary.csv")
    fig, axs = plt.subplots(1, 3, figsize=(10.8, 3.25), gridspec_kw={"wspace": 0.38})
    # donor forest
    ax = axs[0]; panel(ax, "a")
    y = np.arange(len(donor))[::-1]
    eff = donor["delta_f1_f0_equal_weight_cell_mean"].to_numpy() * 100
    spread = donor["delta_sd_across_9_cells"].to_numpy() * 100
    ax.axvline(0, color="#8a969f", linewidth=0.8)
    ax.errorbar(eff, y, xerr=spread, fmt="o", color=TEAL, ecolor="#82b9b3",
                capsize=2.5, markersize=4, linewidth=1.2)
    ax.set_yticks(y, donor["donor_seed"].astype(str).tolist()[::-1])
    ax.set_xlabel("CXT-Fish − F0 (pp)"); ax.set_title("donor realization", loc="left", fontsize=9.5, weight="bold")
    ax.set_xlim(0, 10); clean_axis(ax, grid=False)
    # two-RGB control
    ax = axs[1]; panel(ax, "b")
    rows = {r.method: r for r in rgb2.itertuples()}
    names = ["F0", "F0-2RGB", "CXT-Fish"]
    clean = [0.9577, float(rows["F0_2RGB"].clean_macro_f1), float(rows["CXT-Fish"].clean_macro_f1)]
    cross = [0.5189, float(rows["F0_2RGB"].cross_composite_macro_f1), float(rows["CXT-Fish"].cross_composite_macro_f1)]
    cols = [NAVY, AMBER, TEAL]
    ax.plot(clean, cross, color=GRID, linewidth=1.2, zorder=1)
    for x, yv, n, c in zip(clean, cross, names, cols):
        ax.scatter(x, yv, s=48, color=c, zorder=3)
        ax.text(x + 0.0008, yv + 0.008, n, fontsize=7.6, color=INK)
    ax.set_xlabel("clean macro-F1"); ax.set_ylabel("cross-composite macro-F1")
    ax.set_xlim(0.95, 0.96); ax.set_ylim(0.50, 0.61); clean_axis(ax)
    ax.set_title("two-RGB control", loc="left", fontsize=9.5, weight="bold")
    # suppression paired points
    ax = axs[2]; panel(ax, "c")
    means = fish.groupby("method")["cross_suppressed_macro_f1"].mean()
    sd = fish.groupby("method")["cross_suppressed_macro_f1"].std()
    ax.errorbar([0, 1], [means["F0"], means["F1"]], yerr=[sd["F0"], sd["F1"]], fmt="o",
                color=TEAL, ecolor="#82b9b3", capsize=3, markersize=5, linewidth=1.2)
    ax.set_xticks([0, 1], ["F0", "CXT-Fish"]); ax.set_ylabel("subject-suppressed\ncross macro-F1")
    ax.set_ylim(0.62, 0.84); clean_axis(ax)
    ax.set_title("construct-validity sensitivity", loc="left", fontsize=9.5, weight="bold")
    save(fig, "fig7_controls.png")


def fig8_architecture() -> None:
    d = pd.read_csv(EXP / "cxt_fish_mobilenet_per_fold_results.csv")
    cell = d[d["row_type"].astype(str).str.lower() == "cell"].copy()
    p = cell.pivot_table(index="fold", columns="method", values=["original_macro_f1", "foreground_macro_f1", "cross_swap_macro_f1"])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.35), gridspec_kw={"wspace": 0.34})
    ax = axs[0]; panel(ax, "a")
    for metric, label, color in [("original_macro_f1", "clean", BURGUNDY),
                                 ("foreground_macro_f1", "foreground", TEAL),
                                 ("cross_swap_macro_f1", "cross-class composite", AMBER)]:
        vals = (p[(metric, "MV1")] - p[(metric, "MV0")]) * 100
        ax.plot(vals.index, vals.values, marker="o", linewidth=1.8, markersize=5,
                label=label, color=color)
    ax.axhline(0, color="#8a969f", linewidth=0.8)
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("outer fold"); ax.set_ylabel("MV1 − MV0 (pp)")
    ax.legend(frameon=False, fontsize=7.2, loc="lower left"); clean_axis(ax)
    ax.set_title("fold-level effects", loc="left", fontsize=9.5, weight="bold")
    # clean-robustness plane with paired arrows
    ax = axs[1]; panel(ax, "b")
    for fold in [1, 2, 3]:
        a = cell[(cell.fold == fold) & (cell.method == "MV0")].iloc[0]
        b = cell[(cell.fold == fold) & (cell.method == "MV1")].iloc[0]
        ax.annotate("", xy=(b.original_macro_f1, b.cross_swap_macro_f1),
                    xytext=(a.original_macro_f1, a.cross_swap_macro_f1),
                    arrowprops={"arrowstyle": "->", "color": "#9aa6af", "lw": 1.0})
        ax.text(a.original_macro_f1 - 0.0007, a.cross_swap_macro_f1 - 0.006, f"{fold}", color=NAVY, fontsize=7)
        ax.text(b.original_macro_f1 + 0.0007, b.cross_swap_macro_f1 + 0.004, f"{fold}", color=TEAL, fontsize=7)
    ax.scatter(cell[cell.method == "MV0"].original_macro_f1, cell[cell.method == "MV0"].cross_swap_macro_f1,
               s=38, color=NAVY, label="MV0")
    ax.scatter(cell[cell.method == "MV1"].original_macro_f1, cell[cell.method == "MV1"].cross_swap_macro_f1,
               s=38, color=TEAL, label="MV1")
    ax.set_xlabel("clean macro-F1"); ax.set_ylabel("cross-class composite macro-F1")
    ax.legend(frameon=False, fontsize=7.2, loc="lower left"); clean_axis(ax)
    ax.set_title("architecture sensitivity", loc="left", fontsize=9.5, weight="bold")
    save(fig, "fig8_architecture_sensitivity.png")


if __name__ == "__main__":
    fig1_protocol(); fig2_gate0(); fig3_context(); fig4_donor()
    fig5_method(); fig6_main(); fig7_controls(); fig8_architecture()
