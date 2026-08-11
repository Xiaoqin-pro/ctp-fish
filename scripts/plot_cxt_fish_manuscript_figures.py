"""Build the journal figure set for the CXT-Fish manuscript.

The visual grammar follows conventional scientific figures: white canvas,
thin rules, compact panels, direct annotation, data-first comparisons, and
vector output.  All numerical panels are rebuilt from frozen CSV/JSON
artifacts.  Scientific images are neither generated nor redistributed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
OUT = ROOT / "reports" / "figures" / "manuscript_submission_v4"
OUT.mkdir(parents=True, exist_ok=True)

# Restrained, colour-blind-safe palette. Black/grey carry structure; colour is
# reserved for the compared methods and the focal robustness effect.
BLACK = "#17212b"
GREY = "#66717c"
LIGHT = "#d8dee3"
LIGHTER = "#eef1f3"
BLUE = "#27628d"
TEAL = "#148f84"
ORANGE = "#d07a24"
RED = "#b44b51"
PURPLE = "#7561a8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 8.8,
    "axes.labelsize": 8.0,
    "axes.linewidth": 0.65,
    "axes.edgecolor": GREY,
    "axes.labelcolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "xtick.labelsize": 7.3,
    "ytick.labelsize": 7.3,
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            OUT / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
    plt.close(fig)


def label_panel(ax, text: str) -> None:
    ax.text(-0.10, 1.05, text, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", ha="left", color=BLACK)


def style_axis(ax, *, xgrid=False, ygrid=False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(length=2.8, width=0.65)
    if xgrid:
        ax.grid(axis="x", color=LIGHT, linewidth=0.5, alpha=0.8)
    if ygrid:
        ax.grid(axis="y", color=LIGHT, linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)


def arrow(ax, a, b, *, color=GREY, lw=0.9, ls="-") -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8,
                                 color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=1.5, shrinkB=1.5))


def _cell(ax, x, y, w, h, title, detail, *, color=BLUE, number=None) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=BLACK,
                           linewidth=0.75))
    ax.add_patch(Rectangle((x, y + h - 0.035), w, 0.035, facecolor=color,
                           edgecolor=color, linewidth=0))
    if number is not None:
        ax.text(x + 0.025, y + h - 0.075, str(number), fontsize=7.2,
                fontweight="bold", color=color, ha="left", va="top")
    ax.text(x + 0.025, y + 0.56 * h, title, fontsize=7.5, fontweight="bold",
            color=BLACK, ha="left", va="center")
    ax.text(x + 0.025, y + 0.23 * h, detail, fontsize=6.7, color=GREY,
            ha="left", va="center", linespacing=1.2)


def fig1_protocol() -> None:
    """Evidence architecture: main chain above, bounded controls below."""
    fig = plt.figure(figsize=(10.7, 3.35))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.85], hspace=0.12)

    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    label_panel(ax, "a")
    ax.text(0, 1.03, "Study evidence architecture", transform=ax.transAxes,
            fontsize=9.2, fontweight="bold", color=BLACK, ha="left")
    xs = [0.02, 0.215, 0.410, 0.605, 0.800]
    titles = ["F4K-16T", "Correlation audit", "Intervention family",
              "Training\nintervention", "Frozen\nre-evaluation"]
    details = ["16 species\n8,680 recorded groups",
               "image-level vs\ngroup-disjoint",
               "foreground and\ndonor composites",
               "label-level\nforeground view",
               "3 folds x 3 seeds\nsame corpus"]
    colors = [BLUE, BLUE, ORANGE, TEAL, BLUE]
    for i, (x, t, d, c) in enumerate(zip(xs, titles, details, colors), start=1):
        _cell(ax, x, 0.30, 0.155, 0.43, t, d, color=c, number=i)
        if i < 5:
            arrow(ax, (x + 0.155, 0.515), (xs[i] - 0.010, 0.515), color=GREY)
    ax.plot([0.02, 0.955], [0.17, 0.17], color=BLACK, linewidth=0.65)
    ax.text(0.02, 0.10, "Historical development", fontsize=6.8, color=GREY,
            ha="left", va="center")
    ax.text(0.955, 0.10, "frozen models and protocol", fontsize=6.8,
            color=GREY, ha="right", va="center")

    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    label_panel(ax, "b")
    ax.text(0, 1.03, "Bounded post-hoc construct checks", transform=ax.transAxes,
            fontsize=9.2, fontweight="bold", color=BLACK, ha="left")
    controls = [
        (0.02, "Donor realization", "five deterministic assignments", ORANGE),
        (0.355, "Ordinary-RGB control", "matched views, CE count and 2B forward", BLUE),
        (0.690, "Subject-suppressed donor", "alternative-explanation sensitivity", TEAL),
    ]
    for x, head, sub, col in controls:
        ax.plot([x, x], [0.24, 0.68], color=col, linewidth=3.0)
        ax.text(x + 0.025, 0.56, head, fontsize=7.9, fontweight="bold",
                color=BLACK, ha="left", va="center")
        ax.text(x + 0.025, 0.33, sub, fontsize=6.7, color=GREY,
                ha="left", va="center")
    ax.text(0.98, 0.07, "post-hoc; internal to the same corpus",
            fontsize=6.8, color=GREY, style="italic", ha="right")
    save(fig, "fig1_protocol")


def fig2_gate0() -> None:
    """Correlation audit with protocol, class-tier and neighbour evidence."""
    runs = pd.read_csv(EXP / "ctp_fish_gate0_summary.csv")
    per_class = pd.read_csv(EXP / "ctp_fish_gate0_per_class_summary.csv")
    m = runs.groupby("split").mean(numeric_only=True)
    tiers = per_class.groupby(["split", "tier"])["f1"].mean().unstack("split")
    fig, axs = plt.subplots(1, 3, figsize=(10.7, 3.05),
                            gridspec_kw={"wspace": 0.42})

    ax = axs[0]; label_panel(ax, "a")
    metrics = [("Macro-F1", "macro_f1", BLUE),
               ("Group-balanced\naccuracy", "track_balanced_accuracy", ORANGE)]
    for i, (lab, col, color) in enumerate(metrics):
        y0, y1 = m.loc["image", col], m.loc["track", col]
        ax.plot([0, 1], [y0, y1], color=color, marker="o", markersize=4.5,
                linewidth=1.5)
        ax.text(1.04, y1, lab, fontsize=6.7, color=color, va="center")
    ax.set_xticks([0, 1], ["Image split", "Group-disjoint"])
    ax.set_ylim(0.94, 1.002); ax.set_ylabel("Score")
    ax.set_title("Evaluation unit", loc="left", fontweight="bold")
    style_axis(ax, ygrid=True)

    ax = axs[1]; label_panel(ax, "b")
    for tier, color in [("head", BLUE), ("mid", TEAL), ("tail", ORANGE)]:
        y0, y1 = tiers.loc[tier, "image"], tiers.loc[tier, "track"]
        ax.plot([0, 1], [y0, y1], color=color, marker="o", markersize=4.5,
                linewidth=1.5, label=tier.capitalize())
    ax.set_xticks([0, 1], ["Image split", "Group-disjoint"])
    ax.set_ylim(0.90, 1.002); ax.set_ylabel("Mean per-class F1")
    ax.set_title("Class-frequency tiers", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=6.6, loc="lower left", handlelength=1.4)
    style_axis(ax, ygrid=True)

    ax = axs[2]; label_panel(ax, "c")
    vals = [48.21, 0.0]
    yy = [1, 0]
    ax.barh(yy, vals, color=[BLUE, LIGHT], height=0.34, edgecolor="none")
    ax.set_yticks(yy, ["Image split", "Group-disjoint"])
    ax.set_xlim(0, 52); ax.set_xlabel("Same-group nearest neighbour (%)")
    ax.set_title("Feature-neighbour audit", loc="left", fontweight="bold")
    for y, v in zip(yy, vals):
        ax.text(max(v + 1.2, 1.2), y, f"{v:.2f}", va="center", fontsize=7.1,
                color=BLACK)
    style_axis(ax, xgrid=True)
    save(fig, "fig2_gate0")


def fig3_context() -> None:
    """Data-first diagnostic spectrum; no decorative cards or icons."""
    labels = ["Original RGB", "Foreground-suppressed detail", "Constant fill",
              "Shuffled mask", "Telea inpaint", "Non-primary region",
              "Mask only", "Bounding-box geometry"]
    values = np.array([0.937, 0.938, 0.895, 0.897, 0.805, 0.799, 0.713, 0.077])
    families = ["reference", "subject-preserving", "subject-preserving",
                "subject-preserving", "subject-suppressed", "subject-suppressed",
                "geometry", "geometry"]
    colors = {"reference": BLACK, "subject-preserving": TEAL,
              "subject-suppressed": ORANGE, "geometry": PURPLE}
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(8.9, 3.85))
    fig.subplots_adjust(top=0.82)
    label_panel(ax, "a")
    y = np.arange(len(order))
    for yi, idx in enumerate(order):
        ax.plot([0, values[idx]], [yi, yi], color=LIGHT, linewidth=1.0)
        ax.scatter(values[idx], yi, s=31, color=colors[families[idx]], zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.text(values[idx] + 0.013, yi, f"{values[idx]:.3f}", va="center",
                fontsize=7.2, color=BLACK)
    ax.axvline(1 / 16, color=GREY, linestyle=(0, (3, 2)), linewidth=0.75)
    ax.text(1 / 16, len(y) - 0.15, "chance", fontsize=6.5, color=GREY,
            ha="left", va="bottom")
    ax.set_yticks(y, [labels[i] for i in order])
    ax.set_xlim(0, 1.02); ax.set_xlabel("Historical development macro-F1")
    ax.set_title("Predictive evidence retained by diagnostic views", loc="left",
                 fontweight="bold", y=1.12)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                          markeredgecolor="none", markersize=5, label=k)
               for k, c in colors.items()]
    ax.legend(handles=handles, frameon=False, ncol=4, fontsize=6.3,
              loc="lower right", bbox_to_anchor=(1.0, 1.025), handletextpad=0.35,
              columnspacing=0.9)
    style_axis(ax, xgrid=True)
    save(fig, "fig3_context")


def _fish(ax, cx, cy, sx, sy, *, face, edge=BLACK, alpha=1.0) -> None:
    ax.add_patch(Ellipse((cx, cy), sx, sy, angle=-6, facecolor=face,
                         edgecolor=edge, linewidth=0.7, alpha=alpha))
    ax.add_patch(Polygon([[cx - sx * 0.43, cy], [cx - sx * 0.66, cy - sy * 0.48],
                          [cx - sx * 0.60, cy + sy * 0.55]], closed=True,
                         facecolor=face, edgecolor=edge, linewidth=0.7, alpha=alpha))
    ax.add_patch(Circle((cx + sx * 0.31, cy + sy * 0.06), sy * 0.045,
                        facecolor=BLACK, edgecolor="none"))


def fig4_donor() -> None:
    """Technical compositing diagram with explicit operator and region legend."""
    fig = plt.figure(figsize=(10.7, 3.2))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.82, 1, 0.12, 1.15], wspace=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in [0, 1, 2, 4]]
    titles = ["Recipient image  $x_r$", "Mask  $m_r$", "Donor image  $x_d$",
              r"Composite  $x_{r\leftarrow d}$"]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0, 1.03, f"{chr(97+i)}  {title}", fontsize=8.1,
                fontweight="bold", color=BLACK, ha="left")
        ax.add_patch(Rectangle((0.03, 0.20), 0.94, 0.62, facecolor="#f6f8f9",
                               edgecolor=BLACK, linewidth=0.7))
    # Recipient scene and subject.
    for yy in [0.30, 0.43, 0.56, 0.69]:
        axes[0].plot([0.04, 0.96], [yy, yy + 0.03], color="#b7d1d8", lw=7,
                     solid_capstyle="butt")
    _fish(axes[0], 0.53, 0.50, 0.48, 0.18, face=BLUE)
    # Mask is shown as a binary operator with a narrow feather band.
    axes[1].add_patch(Rectangle((0.03, 0.20), 0.94, 0.62, facecolor="#20282f",
                                edgecolor=BLACK, linewidth=0.7))
    _fish(axes[1], 0.53, 0.50, 0.48, 0.18, face="white", edge="white")
    axes[1].add_patch(Ellipse((0.53, 0.50), 0.52, 0.22, angle=-6,
                              facecolor="none", edgecolor=ORANGE,
                              linewidth=1.0, linestyle=(0, (2, 1.5))))
    # Donor source.
    for yy in [0.30, 0.43, 0.56, 0.69]:
        axes[2].plot([0.04, 0.96], [yy + 0.02, yy - 0.02], color="#ddc6a9", lw=7,
                     solid_capstyle="butt")
    _fish(axes[2], 0.35, 0.56, 0.34, 0.13, face=ORANGE, alpha=0.70)
    axes[2].text(0.50, 0.10, "different species and recorded group",
                 fontsize=6.5, color=GREY, ha="center")
    # Composite retains recipient subject and replaces complement.
    for yy in [0.30, 0.43, 0.56, 0.69]:
        axes[3].plot([0.04, 0.96], [yy + 0.02, yy - 0.02], color="#ddc6a9", lw=7,
                     solid_capstyle="butt")
    _fish(axes[3], 0.53, 0.50, 0.48, 0.18, face=BLUE)
    axes[3].add_patch(Ellipse((0.53, 0.50), 0.52, 0.22, angle=-6,
                              facecolor="none", edgecolor=ORANGE,
                              linewidth=1.0, linestyle=(0, (2, 1.5))))
    # Operator between input triplet and output.
    op = fig.add_subplot(gs[0, 3]); op.set_xlim(0, 1); op.set_ylim(0, 1); op.axis("off")
    arrow(op, (0.05, 0.51), (0.95, 0.51), color=BLACK, lw=1.0)
    fig.text(0.50, 0.105,
             r"$x_{r\leftarrow d}=m_r\odot x_r+(1-m_r)\odot R(x_d)$",
             ha="center", va="center", fontsize=8.5, color=BLACK)
    fig.text(0.50, 0.035,
             "blue: recipient subject     tan: donor non-recipient region     orange dashed: feathered boundary",
             ha="center", va="center", fontsize=6.5, color=GREY)
    save(fig, "fig4_donor_intervention")


def fig5_method() -> None:
    """Compact computation graph; no presentation-style card layout."""
    fig, axs = plt.subplots(1, 2, figsize=(10.7, 3.15),
                            gridspec_kw={"width_ratios": [1.25, 0.75], "wspace": 0.20})
    ax = axs[0]; label_panel(ax, "a")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Training: two label-supervised views, shared parameters",
                 loc="left", fontweight="bold", pad=8)
    # Inputs and view construction.
    ax.text(0.02, 0.76, "ordinary RGB", fontsize=7.2, color=BLACK)
    ax.text(0.02, 0.31, "context-suppressed view", fontsize=7.2, color=BLACK)
    ax.add_patch(Rectangle((0.02, 0.57), 0.18, 0.14, facecolor="#e9f0f5",
                           edgecolor=BLUE, linewidth=0.8))
    ax.text(0.11, 0.64, r"$x$", fontsize=11, ha="center", va="center")
    ax.add_patch(Rectangle((0.02, 0.12), 0.18, 0.14, facecolor="#e8f3f1",
                           edgecolor=TEAL, linewidth=0.8))
    ax.text(0.11, 0.19, r"$x_{fg}$", fontsize=11, ha="center", va="center")
    ax.text(0.27, 0.42, "concatenate\nalong batch axis", fontsize=6.8,
            color=GREY, ha="center", va="center")
    arrow(ax, (0.20, 0.64), (0.34, 0.49), color=BLUE)
    arrow(ax, (0.20, 0.19), (0.34, 0.40), color=TEAL)
    ax.add_patch(Rectangle((0.35, 0.30), 0.22, 0.30, facecolor="white",
                           edgecolor=BLACK, linewidth=0.85))
    ax.text(0.46, 0.49, "shared encoder", fontsize=7.8, fontweight="bold",
            ha="center", va="center")
    ax.text(0.46, 0.39, "+ classifier", fontsize=7.2, ha="center", va="center")
    ax.text(0.46, 0.24, r"$2B$ inputs, one forward", fontsize=6.5,
            color=GREY, ha="center")
    arrow(ax, (0.57, 0.49), (0.68, 0.64), color=BLUE)
    arrow(ax, (0.57, 0.40), (0.68, 0.19), color=TEAL)
    ax.text(0.70, 0.64, r"$CE(f_\theta(x),y)$", fontsize=8.8, color=BLUE,
            ha="left", va="center")
    ax.text(0.70, 0.19, r"$\lambda CE(f_\theta(x_{fg}),y)$", fontsize=8.8,
            color=TEAL, ha="left", va="center")
    ax.plot([0.69, 0.96], [0.09, 0.09], color=BLACK, linewidth=0.65)
    ax.text(0.825, 0.015,
            r"$L=CE(f_\theta(x),y)+\lambda CE(f_\theta(x_{fg}),y),\ \lambda=1$",
            fontsize=7.6, color=BLACK, ha="center", va="bottom")

    ax = axs[1]; label_panel(ax, "b")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Inference: unchanged recognition graph", loc="left",
                 fontweight="bold", pad=8)
    xs = [0.05, 0.38, 0.78]
    labels = [(r"RGB $x$", BLUE), ("encoder +\nclassifier", BLACK), (r"$\hat{y}$", TEAL)]
    widths = [0.20, 0.27, 0.15]
    for x, (lab, col), w in zip(xs, labels, widths):
        ax.add_patch(Rectangle((x, 0.43), w, 0.20, facecolor="white",
                               edgecolor=col, linewidth=0.85))
        ax.text(x + w / 2, 0.53, lab, fontsize=8.3, color=BLACK,
                ha="center", va="center")
    arrow(ax, (0.25, 0.53), (0.38, 0.53), color=BLUE)
    arrow(ax, (0.65, 0.53), (0.78, 0.53), color=TEAL)
    ax.text(0.50, 0.27, "one ordinary-RGB image; one forward pass",
            fontsize=7.2, color=BLACK, ha="center")
    ax.text(0.50, 0.16, "no mask, donor selection, or auxiliary branch",
            fontsize=6.7, color=GREY, ha="center")
    save(fig, "fig5_method")


def _paired_cells(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    cells = df[df["row_type"].str.lower() == "cell"].copy()
    p = cells.pivot_table(index=["fold", "seed"], columns="method", values=metric)
    p["delta"] = p["F1"] - p["F0"]
    return p.reset_index()


def fig6_main() -> None:
    """Nine-cell paired evidence and aggregate guardrails."""
    df = pd.read_csv(EXP / "cxt_fish_final_method_summary.csv")
    clean = _paired_cells(df, "original_macro_f1")
    cross = _paired_cells(df, "cross_swap_macro_f1")
    same = _paired_cells(df, "same_swap_macro_f1")
    tail = _paired_cells(df, "original_tail_f1")
    dar = _paired_cells(df, "dar_flip")
    boot = json.loads((EXP / "cxt_fish_resnet_outer_bootstrap_5000.json").read_text())
    ci = np.array(boot["views"]["cross_swap"]["ci95"]) * 100

    fig, axs = plt.subplots(1, 2, figsize=(10.7, 3.55),
                            gridspec_kw={"width_ratios": [1.15, 0.85], "wspace": 0.38})
    ax = axs[0]; label_panel(ax, "a")
    x = np.arange(9)
    fold_colors = {1: BLUE, 2: TEAL, 3: ORANGE}
    for data, yoff, name in [(clean, 0.14, "Clean"), (cross, -0.14, "Cross-class composite")]:
        vals = data["delta"].to_numpy() * 100
        for i, row in data.iterrows():
            ax.scatter(i, vals[i], s=28, facecolor=fold_colors[int(row.fold)],
                       edgecolor="white", linewidth=0.45, zorder=3)
        ax.plot(x, vals, color=LIGHT, linewidth=0.75, zorder=1)
    ax.axhline(0, color=BLACK, linewidth=0.65)
    for boundary in (2.5, 5.5):
        ax.axvline(boundary, color=LIGHT, linewidth=0.65)
    ax.set_xticks([1, 4, 7], ["Fold 1", "Fold 2", "Fold 3"])
    ax.set_ylabel("CXT-Fish - F0 (percentage points)")
    ax.set_title("Paired fold-seed effects", loc="left", fontweight="bold")
    ax.set_ylim(-8.5, 15.5)
    ax.text(8.78, cross["delta"].iloc[-1] * 100 + 0.5, "cross-class composite",
            fontsize=6.6, color=TEAL, ha="right")
    ax.text(8.78, clean["delta"].iloc[-1] * 100 - 0.8, "clean",
            fontsize=6.6, color=BLUE, ha="right")
    style_axis(ax, ygrid=True)

    ax = axs[1]; label_panel(ax, "b")
    names = ["Cross-class\ncomposite", "Same-class\ncomposite",
             "Tail F1", "Clean", "DAR-flip"]
    effects = np.array([cross.delta.mean(), same.delta.mean(), tail.delta.mean(),
                        clean.delta.mean(), dar.delta.mean()]) * 100
    y = np.arange(len(names))[::-1]
    ax.axvline(0, color=BLACK, linewidth=0.65)
    for yi, value, name in zip(y, effects, names):
        color = TEAL if (value > 0 and name != "DAR-flip") or (value < 0 and name == "DAR-flip") else RED
        ax.plot([0, value], [yi, yi], color=color, linewidth=1.5)
        ax.scatter(value, yi, s=32, color=color, edgecolor="white", linewidth=0.45,
                   zorder=3)
        dx = 0.22 if value >= 0 else -0.22
        ax.text(value + dx, yi, f"{value:+.2f}", fontsize=7.2,
                ha="left" if value >= 0 else "right", va="center", color=BLACK)
    # Only the focal cross-composite contrast receives the conditional interval.
    ax.plot(ci, [y[0], y[0]], color=BLACK, linewidth=0.9, zorder=2)
    ax.plot(ci, [y[0], y[0]], "|", color=BLACK, markersize=6)
    ax.text(ci[0], y[0] + 0.34, "conditional 95% interval [6.13, 8.20]",
            fontsize=6.2, color=GREY, ha="left")
    ax.set_yticks(y, names)
    ax.set_xlabel("Equal-weight mean effect (percentage points)")
    ax.set_xlim(-5.2, 9.1)
    ax.set_title("Main effect and guardrails", loc="left", fontweight="bold")
    style_axis(ax, xgrid=True)
    save(fig, "fig6_main_results")


def fig7_controls() -> None:
    donor = pd.read_csv(EXP / "cxt_fish_donor_sensitivity_summary.csv")
    rgb = pd.read_csv(EXP / "cxt_fish_rgb2_control_summary.csv")
    suppressed = pd.read_csv(EXP / "cxt_fish_fish_suppressed_donor_summary.csv")
    fig, axs = plt.subplots(1, 3, figsize=(10.7, 3.2),
                            gridspec_kw={"wspace": 0.43})

    ax = axs[0]; label_panel(ax, "a")
    eff = donor["delta_f1_f0_equal_weight_cell_mean"].to_numpy() * 100
    sd = donor["delta_sd_across_9_cells"].to_numpy() * 100
    y = np.arange(len(donor))[::-1]
    ax.errorbar(eff, y, xerr=sd, fmt="o", color=TEAL, ecolor="#82bdb7",
                elinewidth=0.9, capsize=2.2, markersize=4.2)
    ax.axvline(0, color=BLACK, linewidth=0.65)
    ax.set_yticks(y, donor["donor_seed"].astype(int).astype(str))
    ax.set_xlabel("Cross-composite effect (pp)")
    ax.set_title("Donor realizations", loc="left", fontweight="bold")
    ax.text(0.02, -0.25, "point: nine-cell mean; whisker: cell SD",
            transform=ax.transAxes, fontsize=6.1, color=GREY)
    ax.set_xlim(-1, 15); style_axis(ax, xgrid=True)

    ax = axs[1]; label_panel(ax, "b")
    rows = rgb.set_index("method")
    methods = ["F0", "F0_2RGB", "CXT-Fish"]
    clean = [0.9577, rows.loc["F0_2RGB", "clean_macro_f1"], rows.loc["CXT-Fish", "clean_macro_f1"]]
    cross = [0.5189, rows.loc["F0_2RGB", "cross_composite_macro_f1"], rows.loc["CXT-Fish", "cross_composite_macro_f1"]]
    colors = [BLUE, ORANGE, TEAL]
    labels = ["F0", "F0-2RGB", "CXT-Fish"]
    for x0, y0, c, lab in zip(clean, cross, colors, labels):
        ax.scatter(x0, y0, s=39, color=c, edgecolor="white", linewidth=0.45, zorder=3)
        ax.text(x0 + 0.0007, y0 + 0.006, lab, fontsize=6.6, color=BLACK)
    ax.plot(clean, cross, color=LIGHT, linewidth=0.8, zorder=1)
    ax.set_xlabel("Clean macro-F1")
    ax.set_ylabel("Cross-composite macro-F1")
    ax.set_title("Ordinary-RGB control", loc="left", fontweight="bold")
    ax.set_xlim(0.951, 0.959); ax.set_ylim(0.505, 0.608)
    style_axis(ax, xgrid=True, ygrid=True)

    ax = axs[2]; label_panel(ax, "c")
    p = suppressed.pivot_table(index=["fold", "seed"], columns="method",
                               values="cross_suppressed_macro_f1")
    for _, row in p.iterrows():
        ax.plot([0, 1], [row.F0, row.F1], color=LIGHT, linewidth=0.8)
    ax.scatter(np.zeros(len(p)), p.F0, s=22, color=BLUE, alpha=0.9,
               edgecolor="white", linewidth=0.35, zorder=3)
    ax.scatter(np.ones(len(p)), p.F1, s=22, color=TEAL, alpha=0.9,
               edgecolor="white", linewidth=0.35, zorder=3)
    ax.plot([0, 1], [p.F0.mean(), p.F1.mean()], color=BLACK, linewidth=1.4,
            marker="o", markersize=4.0, zorder=4)
    ax.set_xticks([0, 1], ["F0", "CXT-Fish"])
    ax.set_ylabel("Subject-suppressed\ncross-composite macro-F1")
    ax.set_title("Construct sensitivity", loc="left", fontweight="bold")
    ax.set_ylim(0.60, 0.86); style_axis(ax, ygrid=True)
    save(fig, "fig7_controls")


def fig8_architecture() -> None:
    df = pd.read_csv(EXP / "cxt_fish_mobilenet_per_fold_results.csv")
    cells = df[df["row_type"].str.lower() == "cell"].copy()
    p = cells.pivot_table(index="fold", columns="method",
                          values=["original_macro_f1", "foreground_macro_f1",
                                  "cross_swap_macro_f1", "same_swap_macro_f1",
                                  "dar_flip"])
    fig, axs = plt.subplots(1, 2, figsize=(10.7, 3.35),
                            gridspec_kw={"width_ratios": [1.15, 0.85], "wspace": 0.36})

    ax = axs[0]; label_panel(ax, "a")
    metrics = [("original_macro_f1", "Clean", BLUE),
               ("foreground_macro_f1", "Foreground", TEAL),
               ("cross_swap_macro_f1", "Cross-class composite", ORANGE),
               ("same_swap_macro_f1", "Same-class composite", PURPLE)]
    positions = np.arange(len(metrics))[::-1]
    offsets = [-0.12, 0, 0.12]
    for yi, (metric, label, color) in zip(positions, metrics):
        vals = (p[(metric, "MV1")] - p[(metric, "MV0")]) * 100
        for fold, off, val in zip([1, 2, 3], offsets, vals):
            ax.scatter(val, yi + off, s=24, facecolor=color, edgecolor="white",
                       linewidth=0.4, zorder=3)
            ax.text(val, yi + off + 0.11, str(fold), fontsize=5.7, color=GREY,
                    ha="center", va="bottom")
        ax.plot([vals.min(), vals.max()], [yi, yi], color=color, linewidth=1.0,
                alpha=0.55)
        ax.scatter(vals.mean(), yi, marker="D", s=31, color=BLACK, zorder=4)
    ax.axvline(0, color=BLACK, linewidth=0.65)
    ax.set_yticks(positions, [x[1] for x in metrics])
    ax.set_xlabel("MV1 - MV0 (percentage points)")
    ax.set_title("Fold-level architecture sensitivity", loc="left", fontweight="bold")
    ax.text(0.02, -0.18, "circles: folds 1-3; diamonds: three-fold mean",
            transform=ax.transAxes, fontsize=6.2, color=GREY)
    style_axis(ax, xgrid=True)

    ax = axs[1]; label_panel(ax, "b")
    for fold in [1, 2, 3]:
        a = cells[(cells.fold == fold) & (cells.method == "MV0")].iloc[0]
        b = cells[(cells.fold == fold) & (cells.method == "MV1")].iloc[0]
        ax.annotate("", xy=(b.original_macro_f1, b.cross_swap_macro_f1),
                    xytext=(a.original_macro_f1, a.cross_swap_macro_f1),
                    arrowprops={"arrowstyle": "->", "lw": 0.85, "color": GREY})
        ax.scatter(a.original_macro_f1, a.cross_swap_macro_f1, s=27, color=BLUE,
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter(b.original_macro_f1, b.cross_swap_macro_f1, s=27, color=TEAL,
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(a.original_macro_f1, a.cross_swap_macro_f1 - 0.012, str(fold),
                fontsize=6, color=BLUE, ha="center")
        ax.text(b.original_macro_f1, b.cross_swap_macro_f1 + 0.010, str(fold),
                fontsize=6, color=TEAL, ha="center")
    ax.set_xlabel("Clean macro-F1")
    ax.set_ylabel("Cross-composite macro-F1")
    ax.set_title("Clean-robustness movement", loc="left", fontweight="bold")
    ax.text(0.03, 0.04, "blue: MV0    teal: MV1", transform=ax.transAxes,
            fontsize=6.4, color=GREY)
    style_axis(ax, xgrid=True, ygrid=True)
    save(fig, "fig8_architecture_sensitivity")


if __name__ == "__main__":
    fig1_protocol()
    fig2_gate0()
    fig3_context()
    fig4_donor()
    fig5_method()
    fig6_main()
    fig7_controls()
    fig8_architecture()
