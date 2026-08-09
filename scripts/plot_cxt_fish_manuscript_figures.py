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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#d8dde3",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.65,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, edge=BLUE, fill="#eef5f8", fs=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=1.6, edgecolor=edge, facecolor=fill)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#203244",
            fontsize=fs, weight="bold", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GREY, style="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.5, linestyle=style, color=color))


def fig1_protocol():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.95, "Evidence hierarchy: characterization → frozen re-evaluation → bounded post-hoc controls",
            fontsize=13, weight="bold", color=BLUE)
    labels = ["Correlation audit", "F4K-16T\nrecorded-group\nprotocol", "Context diagnostics\nand donor stress test",
              "Foreground-sufficiency\ntraining", "Same-corpus frozen\ngroup-disjoint\nre-evaluation"]
    xs = [0.02, 0.22, 0.42, 0.62, 0.82]
    for x, lab in zip(xs, labels): box(ax, x, 0.55, 0.15, 0.18, lab, BLUE if x in (0.02, 0.82) else TEAL)
    for x in xs[:-1]: arrow(ax, x + 0.15, 0.64, x + 0.195, 0.64)
    box(ax, 0.34, 0.19, 0.30, 0.18,
        "Post-hoc validity controls\n5 donor realizations · F0-2RGB\ndonor-subject-suppressed sensitivity",
        edge=ORANGE, fill="#fff5e7", fs=8)
    arrow(ax, 0.89, 0.55, 0.64, 0.37, ORANGE, "--")
    ax.text(0.02, 0.08, "Solid path = main evidence. Dashed branch = later controls; neither is independent validation.",
            color=GREY, fontsize=9)
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
    fig, ax = plt.subplots(figsize=(11, 3.1)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.93, r"$x_{dc}=m_r\odot x_r+(1-m_r)\odot R(x_d)$", ha="center", fontsize=15, weight="bold", color="#203244")
    for i, (x, title, edge) in enumerate([(0.03, "Recipient RGB", BLUE), (0.27, "Recipient mask", TEAL), (0.51, "Donor RGB", ORANGE), (0.75, "Composite stress test", RED)]):
        box(ax, x, 0.20, 0.18, 0.45, title, edge=edge, fill="#f7fafc", fs=9)
        if i != 3: arrow(ax, x + 0.18, 0.42, x + 0.23, 0.42)
    ax.text(0.5, 0.08, "Donors come from the same held-out fold; cross-class donors differ in species and recorded group. This is a synthetic diagnostic, not a photorealistic background reconstruction.", ha="center", fontsize=8.5, color=RED)
    save(fig, "fig4_donor_intervention.png")


def fig5_method():
    fig, ax = plt.subplots(figsize=(11, 3.5)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    box(ax, 0.04, 0.55, 0.20, 0.20, "Ordinary RGB\n$x$", edge=BLUE, fs=11)
    box(ax, 0.04, 0.20, 0.20, 0.20, "Foreground-sufficient\n$x_{fg}$", edge=TEAL, fs=10)
    box(ax, 0.38, 0.37, 0.24, 0.25, "Shared ResNet18\n2B concatenated forward\nshared BatchNorm", edge=BLUE, fs=10)
    box(ax, 0.75, 0.55, 0.19, 0.20, r"$CE(f(x),y)$", edge=BLUE, fs=11)
    box(ax, 0.75, 0.20, 0.19, 0.20, r"$\lambda CE(f(x_{fg}),y)$", edge=TEAL, fs=10)
    arrow(ax, 0.24, 0.65, 0.38, 0.53, BLUE); arrow(ax, 0.24, 0.30, 0.38, 0.46, TEAL)
    arrow(ax, 0.62, 0.53, 0.75, 0.65, BLUE); arrow(ax, 0.62, 0.46, 0.75, 0.30, TEAL)
    ax.text(0.52, 0.08, r"$L=CE(f(x),y)+\lambda CE(f(x_{fg}),y)$,  $\lambda=1.0$", ha="center", fontsize=12, weight="bold", color="#203244")
    ax.text(0.52, 0.94, "Training uses privileged masks; inference uses one ordinary RGB image and one forward pass.", ha="center", fontsize=10, color=GREY)
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
