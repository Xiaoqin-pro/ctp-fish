"""Reproducible, non-generative figures for the CXT-Fish manuscript.

All quantitative panels are read from frozen CSV/JSON artifacts. Schematic
panels use vector primitives only and contain no generated scientific imagery.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Polygon

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "reports" / "figures" / "manuscript_final"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2b5c7d"
TEAL = "#3a9988"
ORANGE = "#e1842f"
RED = "#b85b59"
GREY = "#6f7b8a"
PALETTE = [BLUE, TEAL, ORANGE, RED]

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d8dde3",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.65,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    stem = Path(name).stem
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, edge=BLUE, fill="white", fs=8.5, lw=1.1, radius=0.008):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={radius}",
                       linewidth=lw, edgecolor=edge, facecolor=fill)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#203244",
            fontsize=fs, weight="normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GREY, style="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, linestyle=style, color=color))


def fig1_protocol():
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.94, "Study design and evidence hierarchy", fontsize=11, weight="bold", color="#203244")
    labels = ["Correlation\naudit", "F4K–16T\nrecorded-group\nprotocol", "Context diagnostics\nand donor stress test",
              "Foreground-\nsufficiency\ntraining", "Same-corpus frozen\ngroup-disjoint\nre-evaluation"]
    xs = [0.02, 0.215, 0.41, 0.605, 0.80]
    for i, (x, lab) in enumerate(zip(xs, labels), 1):
        box(ax, x, 0.53, 0.16, 0.20, lab, BLUE if i in (1, 5) else TEAL, fill="white", fs=8.1)
        ax.add_patch(Ellipse((x + 0.018, 0.76), 0.032, 0.032, facecolor=BLUE if i in (1,5) else TEAL, edgecolor="white", linewidth=0.7))
        ax.text(x + 0.018, 0.76, str(i), ha="center", va="center", color="white", fontsize=7.5, weight="bold")
    for x in xs[:-1]: arrow(ax, x + 0.16, 0.63, x + 0.195, 0.63, color="#52606d")
    # bounded controls: compact dashed container, no slogan text
    ax.add_patch(Rectangle((0.16, 0.14), 0.68, 0.22, fill=False, edgecolor=ORANGE, linewidth=1.0, linestyle=(0, (3, 2))))
    ax.text(0.18, 0.32, "Bounded post-hoc controls", fontsize=8.5, color=ORANGE, weight="bold")
    controls = ["5 donor realizations", "F0–2RGB control", "donor-subject-suppressed sensitivity"]
    for x, lab in zip([0.20, 0.43, 0.66], controls):
        box(ax, x, 0.19, 0.18, 0.08, lab, edge=ORANGE, fill="#fffaf2", fs=7.2, lw=0.8, radius=0.004)
    arrow(ax, 0.88, 0.53, 0.83, 0.36, color=ORANGE, style="--")
    ax.text(0.02, 0.05, "solid arrows: main evidence chain   |   dashed arrows: bounded post-hoc checks   |   same corpus throughout",
            color=GREY, fontsize=7.5)
    save(fig, "fig1_protocol.png")


def fig2_gate0():
    d = pd.read_csv(EXP / "ctp_fish_gate0_summary.csv")
    means = d.groupby("split").mean(numeric_only=True)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"wspace": 0.30})
    x = np.arange(2); width = 0.36
    axs[0].bar(x - width / 2, means.loc[["image", "track"], "macro_f1"], width, label="Macro-F1", color=BLUE)
    axs[0].bar(x + width / 2, means.loc[["image", "track"], "track_balanced_accuracy"], width, label="Group-balanced accuracy", color=ORANGE)
    axs[0].set_xticks(x, ["Image-level", "Group-disjoint"]); axs[0].set_ylim(0.9, 1.01); axs[0].set_ylabel("Score"); axs[0].legend(frameon=False, fontsize=8)
    pc = pd.read_csv(EXP / "ctp_fish_gate0_per_class_summary.csv")
    tier_means = pc.groupby(["split", "tier"])["f1"].mean().unstack("split")
    head = [float(tier_means.loc["head", "image"]), float(tier_means.loc["head", "track"])]
    mid = [float(tier_means.loc["mid", "image"]), float(tier_means.loc["mid", "track"])]
    tail = [float(tier_means.loc["tail", "image"]), float(tier_means.loc["tail", "track"])]
    for vals, label, c in [(head, "Head", BLUE), (mid, "Mid", TEAL), (tail, "Tail", ORANGE)]:
        axs[1].plot([0, 1], vals, marker="o", linewidth=2, label=label, color=c)
    axs[1].set_xticks([0, 1], ["Image-level", "Group-disjoint"]); axs[1].set_ylim(0.88, 1.01); axs[1].set_ylabel("Mean per-class F1"); axs[1].legend(frameon=False, fontsize=8)
    axs[2].bar([0, 1], [0.4821, 0.0], color=[BLUE, ORANGE]); axs[2].set_xticks([0, 1], ["Image-level\nnearest group", "Group-disjoint\nnearest group"]); axs[2].set_ylim(0, 0.55); axs[2].set_ylabel("Fraction")
    axs[2].text(0, 0.49, "48.21%", ha="center", weight="bold")
    axs[2].text(1, 0.02, "0%", ha="center", weight="bold")
    for i, title in enumerate(["Performance estimate", "Class strata", "Same-group NN audit"]): axs[i].set_title(title, loc="left", weight="bold")
    save(fig, "fig2_gate0.png")


def fig3_context():
    d = pd.read_csv(EXP / "cxt_fish_context_sanity_summary.csv")
    labels = ["Original RGB", "Foreground-only", "Background-only", "Mask-only", "Geometry-only", "Constant-fill", "Inpainted", "Shuffled-mask"]
    vals = [0.937, 0.938, 0.799, 0.713, 0.077, 0.895, 0.805, 0.897]
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    colors = [BLUE, TEAL, ORANGE, "#7568a5", "#8a9098", "#d4a51d", "#5694a5", RED]
    bars = ax.bar(np.arange(len(labels)), vals, color=colors, edgecolor="white", linewidth=0.7)
    ax.axhline(0.06, color=GREY, linestyle="--", linewidth=1, label="chance reference")
    ax.set_xticks(np.arange(len(labels)), [s.replace(" ", "\n") for s in labels]); ax.set_ylim(0, 1.05); ax.set_ylabel("Validation macro-F1"); ax.set_title("Context-diagnostic views on the historical development validation set", loc="left", weight="bold")
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    save(fig, "fig3_context.png")


def fig4_donor():
    fig, axs = plt.subplots(1, 4, figsize=(10.5, 2.8), gridspec_kw={"wspace": 0.18})
    titles = ["(a) Recipient RGB", "(b) Foreground mask", "(c) Donor context source", "(d) Composite stress test"]
    # schematic-only panels: no synthetic fish imagery or redistributed raw data
    for ax, title in zip(axs, titles):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top", fontsize=8.2, weight="bold", color="#203244")
        ax.add_patch(Rectangle((0.08, 0.20), 0.84, 0.58, facecolor="white", edgecolor="#9aa6b2", linewidth=0.9))
    # explicit vector schematic textures, not AI-generated imagery
    for y, c in [(0.31, "#d7e6eb"), (0.45, "#b9d5dc"), (0.60, "#d7e6eb")]: axs[0].plot([0.11, 0.89], [y, y], color=c, linewidth=5, solid_capstyle="butt")
    axs[0].add_patch(Ellipse((0.50, 0.49), 0.46, 0.16, angle=-8, facecolor="#7896a6", edgecolor="#415563", linewidth=0.9)); axs[0].add_patch(Polygon([[0.29,0.50],[0.20,0.43],[0.23,0.56]], facecolor="#7896a6", edgecolor="#415563", linewidth=0.8)); axs[0].plot(0.68,0.51,"o",color="#203244",ms=2)
    axs[1].add_patch(Ellipse((0.50, 0.49), 0.46, 0.16, angle=-8, facecolor="#3a9988", edgecolor="#206b60", linewidth=0.9)); axs[1].add_patch(Polygon([[0.29,0.50],[0.20,0.43],[0.23,0.56]], facecolor="#3a9988", edgecolor="#206b60", linewidth=0.8)); axs[1].text(0.50, 0.09, "mᵣ", ha="center", fontsize=9, color="#206b60")
    for y, c in [(0.30, "#e6d9c9"), (0.43, "#d8c7af"), (0.58, "#eadfd1")]: axs[2].plot([0.11, 0.89], [y, y], color=c, linewidth=5, solid_capstyle="butt")
    axs[2].text(0.50, 0.10, "species ≠ recipient; group ≠ recipient", ha="center", fontsize=6.5, color=GREY)
    axs[3].imshow(np.zeros((1, 1, 3)), extent=(0.08, 0.92, 0.20, 0.78), visible=False)
    for y, c in [(0.31, "#d8c4af"), (0.45, "#c9b08f"), (0.60, "#e1d1bd")]: axs[3].plot([0.11, 0.89], [y, y], color=c, linewidth=5, solid_capstyle="butt")
    axs[3].add_patch(Ellipse((0.50, 0.49), 0.46, 0.16, angle=-8, facecolor="#7896a6", edgecolor="#415563", linewidth=0.9)); axs[3].add_patch(Polygon([[0.29,0.50],[0.20,0.43],[0.23,0.56]], facecolor="#7896a6", edgecolor="#415563", linewidth=0.8))
    for ax in axs[:-1]: arrow(ax, 0.93, 0.49, 1.04, 0.49, color="#52606d")
    fig.text(0.5, 0.04, r"$x_{dc}=m_r\odot x_r+(1-m_r)\odot R(x_d)$    |    schematic construction; synthetic diagnostic, not photorealistic reconstruction",
             ha="center", fontsize=7.5, color=GREY)
    save(fig, "fig4_donor_intervention.png")


def fig5_method():
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"wspace": 0.28})
    for ax in axs: ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    axs[0].text(0.02, 0.96, "(a) Training-time pipeline", fontsize=9.5, weight="bold", color="#203244")
    box(axs[0], 0.03, 0.57, 0.22, 0.16, "ordinary RGB\n$x$", edge=BLUE, fs=8.8)
    box(axs[0], 0.03, 0.27, 0.22, 0.16, "foreground-sufficient\n$x_{fg}$", edge=TEAL, fs=8.5)
    box(axs[0], 0.36, 0.39, 0.27, 0.26, "shared ResNet18\nencoder + classifier\nconcatenated 2B forward", edge=BLUE, fs=8.5)
    box(axs[0], 0.74, 0.57, 0.22, 0.16, r"$CE(f(x),y)$", edge=BLUE, fs=9)
    box(axs[0], 0.74, 0.27, 0.22, 0.16, r"$\lambda CE(f(x_{fg}),y)$", edge=TEAL, fs=8.2)
    arrow(axs[0], 0.25, 0.65, 0.36, 0.56, BLUE); arrow(axs[0], 0.25, 0.35, 0.36, 0.47, TEAL)
    arrow(axs[0], 0.63, 0.56, 0.74, 0.65, BLUE); arrow(axs[0], 0.63, 0.47, 0.74, 0.35, TEAL)
    axs[0].text(0.50, 0.08, r"$L=CE(f(x),y)+\lambda CE(f(x_{fg}),y)$,  $\lambda=1.0$", ha="center", fontsize=8.5, color="#203244")
    axs[1].text(0.02, 0.96, "(b) Inference-time pipeline", fontsize=9.5, weight="bold", color="#203244")
    box(axs[1], 0.08, 0.48, 0.24, 0.18, "ordinary RGB\n$x$", edge=BLUE, fs=9)
    box(axs[1], 0.43, 0.48, 0.25, 0.18, "shared ResNet18\nclassifier", edge=BLUE, fs=9)
    box(axs[1], 0.80, 0.48, 0.16, 0.18, r"$\hat y$", edge=TEAL, fs=10)
    arrow(axs[1], 0.32, 0.57, 0.43, 0.57, BLUE); arrow(axs[1], 0.68, 0.57, 0.80, 0.57, TEAL)
    axs[1].text(0.52, 0.25, "one ordinary RGB image\nno mask · no auxiliary branch\none forward pass", ha="center", va="center", fontsize=8.5, color=GREY)
    save(fig, "fig5_method.png")


def fig6_main():
    labels = ["Original", "Foreground", "Same-class\ncomposite", "Cross-class\ncomposite"]
    f0 = [0.9577, 0.8331, 0.9229, 0.5189]
    f1 = [0.9556, 0.9550, 0.9291, 0.5913]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(4); w = 0.35
    axs[0].bar(x - w/2, f0, w, label="F0", color=BLUE)
    axs[0].bar(x + w/2, f1, w, label="CXT-Fish", color=ORANGE)
    axs[0].set_xticks(x, labels); axs[0].set_ylim(0.45, 1.02); axs[0].set_ylabel("Macro-F1"); axs[0].legend(frameon=False)
    delta = [-0.21, 0.26, 0.62, 7.24, -3.54]
    names = ["Clean", "Tail", "Group-bal.\naccuracy", "Cross-class\ncomposite", "DAR-flip"]
    c = [RED if v < 0 else TEAL for v in delta]
    axs[1].axvline(0, color=GREY, linewidth=1); axs[1].scatter(delta, np.arange(5), s=65, c=c, zorder=3)
    for y, v in enumerate(delta): axs[1].text(v + (0.18 if v >= 0 else -0.18), y, f"{v:+.2f} pp", va="center", ha="left" if v >= 0 else "right", color=c[y])
    axs[1].set_yticks(np.arange(5), names); axs[1].set_xlabel("CXT-Fish − F0 (percentage points)"); axs[1].set_xlim(-5, 8.5); axs[1].set_title("Equal-weight cell means", loc="left", weight="bold")
    save(fig, "fig6_main_results.png")


def fig7_controls():
    donor = pd.read_csv(EXP / "cxt_fish_donor_sensitivity_summary.csv")
    rgb2 = pd.read_csv(EXP / "cxt_fish_rgb2_control_summary.csv")
    fish = pd.read_csv(EXP / "cxt_fish_fish_suppressed_donor_summary.csv")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"wspace": 0.32})
    axs[0].plot(donor["donor_seed"].astype(str), donor["delta_f1_f0_equal_weight_cell_mean"] * 100, marker="o", color=TEAL, linewidth=2)
    axs[0].axhline(0, color=GREY, linewidth=1); axs[0].set_ylabel("CXT-Fish − F0 (pp)"); axs[0].set_xlabel("Deterministic donor seed"); axs[0].set_title("Donor realization", loc="left", weight="bold")
    for x, y in enumerate(donor["delta_f1_f0_equal_weight_cell_mean"] * 100): axs[0].text(x, y + .15, f"{y:.2f}", ha="center", fontsize=8)
    names = ["F0", "F0-2RGB", "CXT-Fish"]; clean = [0.9577, float(rgb2.loc[rgb2.method == "F0_2RGB", "clean_macro_f1"].iloc[0]), float(rgb2.loc[rgb2.method == "CXT-Fish", "clean_macro_f1"].iloc[0])]; cross = [0.5189, float(rgb2.loc[rgb2.method == "F0_2RGB", "cross_composite_macro_f1"].iloc[0]), float(rgb2.loc[rgb2.method == "CXT-Fish", "cross_composite_macro_f1"].iloc[0])]
    axs[1].scatter(clean, cross, s=80, c=[BLUE, ORANGE, TEAL])
    for x, y, n in zip(clean, cross, names): axs[1].text(x + .0001, y + .001, n, fontsize=9)
    axs[1].set_xlabel("Original-view macro-F1"); axs[1].set_ylabel("Cross-class composite macro-F1"); axs[1].set_title("Two-view control", loc="left", weight="bold")
    f0s = fish[fish.method == "F0"]["cross_suppressed_macro_f1"].mean(); f1s = fish[fish.method == "F1"]["cross_suppressed_macro_f1"].mean(); axs[2].bar([0,1], [f0s,f1s], color=[BLUE,TEAL]); axs[2].set_xticks([0,1], ["F0", "CXT-Fish"]); axs[2].set_ylim(0.6, 0.82); axs[2].set_ylabel("Fish-suppressed cross macro-F1"); axs[2].set_title("Donor-subject suppression", loc="left", weight="bold")
    for x, y in enumerate([f0s, f1s]): axs[2].text(x, y + .005, f"{y:.3f}", ha="center")
    save(fig, "fig7_controls.png")


def fig8_architecture():
    d = pd.read_csv(EXP / "cxt_fish_mobilenet_per_fold_results.csv")
    # The file contains one row per method and fold; compute fold deltas.
    p = d.pivot_table(index="fold", columns="method", values=["original_macro_f1", "foreground_macro_f1", "cross_swap_macro_f1", "dar_flip"])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw={"width_ratios": [1.15, 1]})
    for metric, label, color in [("original_macro_f1", "Clean", RED), ("foreground_macro_f1", "Foreground", TEAL), ("cross_swap_macro_f1", "Cross composite", ORANGE)]:
        vals = (p[(metric, "MV1")] - p[(metric, "MV0")]) * 100
        axs[0].plot(vals.index, vals.values, marker="o", linewidth=2, label=label, color=color)
    axs[0].axhline(0, color=GREY, linewidth=1); axs[0].set_xticks([1,2,3]); axs[0].set_xlabel("Outer fold"); axs[0].set_ylabel("MV1 − MV0 (pp)"); axs[0].legend(frameon=False, fontsize=8); axs[0].set_title("MobileNetV3-Large fold effects", loc="left", weight="bold")
    for method, color in [("MV0", BLUE), ("MV1", TEAL)]:
        axs[1].scatter(d.loc[d.method == method, "original_macro_f1"], d.loc[d.method == method, "cross_swap_macro_f1"], s=65, label=method, color=color)
    for fold in [1,2,3]:
        a = d[(d.fold == fold) & (d.method == "MV0")].iloc[0]; b = d[(d.fold == fold) & (d.method == "MV1")].iloc[0]
        axs[1].annotate("", xy=(b.original_macro_f1, b.cross_swap_macro_f1), xytext=(a.original_macro_f1, a.cross_swap_macro_f1), arrowprops={"arrowstyle":"->", "color":GREY})
    axs[1].set_xlabel("Original-view macro-F1"); axs[1].set_ylabel("Cross-class composite macro-F1"); axs[1].set_title("Architecture sensitivity", loc="left", weight="bold")
    save(fig, "fig8_architecture_sensitivity.png")


if __name__ == "__main__":
    fig1_protocol(); fig2_gate0(); fig3_context(); fig4_donor(); fig5_method(); fig6_main(); fig7_controls(); fig8_architecture()
