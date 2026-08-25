#!/usr/bin/env python3
"""
Generate paper figures v4 — fixes all label/legend overlap issues.

Fixes:
  fault_family_detection.pdf  : CI bars no longer obscure labels; n= on separate line
  accidental_corrects.pdf     : legend moved outside/below plot area
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
BLUE   = "#2563EB"   # MORPH-DA
ORANGE = "#D97706"   # Universal baseline
GREY   = "#6B7280"   # Neutral / truly-correct
GREEN  = "#059669"   # Truly-correct / caught
RED    = "#DC2626"   # Accidental / uncaught
LABEL  = "#111827"   # Near-black for all text labels

FIGDIR = Path("paper/submission_v07/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 : Fault-family detection  (fault_family_detection.pdf)
# ─────────────────────────────────────────────────────────────────────────────
def fig_fault_family():
    # Data from fig4_kill_matrix.csv
    families = ["Aggregation", "Filter", "Grouping", "Hardcoding", "Ranking"]
    n_vals   = [131, 173, 21, 129, 109]          # totals
    kill     = [0.2824, 0.6763, 0.8095, 0.8527, 0.7615]
    ci_lo    = [0.172,  0.571,  0.545,  0.776,  0.658]
    ci_hi    = [0.390,  0.775,  1.000,  0.920,  0.856]
    univ     = [0.0610, 0.0058, 0.0000, 0.0000, 0.0000]

    x = np.arange(len(families))
    BAR_W = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.subplots_adjust(bottom=0.22, top=0.93, left=0.10, right=0.97)

    # ── bars ──────────────────────────────────────────────────────────────────
    bars_morph = ax.bar(x - BAR_W/2, kill, BAR_W,
                        color=BLUE, label="MORPH-DA", zorder=3)
    bars_univ  = ax.bar(x + BAR_W/2, univ, BAR_W,
                        color=ORANGE, label="Universal only", zorder=3)

    # ── CI error bars for MORPH-DA (plotted separately so they sit on top) ────
    yerr_lo = [k - l for k, l in zip(kill, ci_lo)]
    yerr_hi = [h - k for k, h in zip(kill, ci_hi)]
    ax.errorbar(x - BAR_W/2, kill,
                yerr=[yerr_lo, yerr_hi],
                fmt="none", color="#1E40AF", capsize=4, capthick=1.5,
                linewidth=1.5, zorder=5)

    # ── value labels ABOVE each bar (including CI cap) ───────────────────────
    for i, (k, hi) in enumerate(zip(kill, ci_hi)):
        label_y = hi + 0.03          # always above the top CI cap
        ax.text(x[i] - BAR_W/2, label_y,
                f"{k:.0%}", ha="center", va="bottom",
                fontsize=9, color=LABEL, fontweight="bold", zorder=6)

    for i, u in enumerate(univ):
        if u > 0:
            ax.text(x[i] + BAR_W/2, u + 0.025,
                    f"{u:.1%}", ha="center", va="bottom",
                    fontsize=8, color=LABEL, zorder=6)
        else:
            ax.text(x[i] + BAR_W/2, 0.025,
                    "0%", ha="center", va="bottom",
                    fontsize=8, color=LABEL, zorder=6)

    # ── x-axis labels: family name on line 1, n= on line 2 ───────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{fam}\n(n={n})" for fam, n in zip(families, n_vals)],
        fontsize=9, color=LABEL
    )

    # ── axes & grid ───────────────────────────────────────────────────────────
    ax.set_ylim(0, 1.18)       # extra headroom so 100% label isn't clipped
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=9)
    ax.set_ylabel("Mutation Detection Rate", fontsize=10)
    ax.set_title("Detection Rate by Fault Family", fontsize=11, pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # ── legend inside plot, upper-left corner (well away from bars) ───────────
    ax.legend(fontsize=9, loc="upper left",
              framealpha=0.9, edgecolor="#D1D5DB")

    out = FIGDIR / "fault_family_detection.pdf"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 : Accidental corrects  (accidental_corrects.pdf)
# ─────────────────────────────────────────────────────────────────────────────
def fig_accidental_corrects():
    models        = ["Haiku 4.5", "Sonnet 4.6", "Opus 4.5"]
    seed42        = [61, 65, 62]
    truly_correct = [45, 52, 49]
    lucky         = [16, 13, 13]        # accidental corrects = seed42 - truly_correct
    morph_catches = [13,  7, 10]
    catch_rate    = [0.8125, 0.5385, 0.7692]
    n_tasks       = 101

    x = np.arange(len(models))
    BAR_W = 0.55

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(bottom=0.26, top=0.88, left=0.12, right=0.97)

    # ── stacked bars: truly correct (green) + accidental (red) ───────────────
    bars_true = ax.bar(x, truly_correct, BAR_W, color=GREEN,
                       label="Truly correct", zorder=3)
    bars_acc  = ax.bar(x, lucky, BAR_W, bottom=truly_correct, color=RED,
                       label="Accidental correct (wrong on held-out seeds)", zorder=3)

    # ── labels on truly-correct segment (centred inside if tall enough) ───────
    for i, (tc, s42) in enumerate(zip(truly_correct, seed42)):
        mid = tc / 2
        ax.text(x[i], mid, str(tc),
                ha="center", va="center",
                fontsize=10, color="white", fontweight="bold", zorder=5)

    # ── labels on accidental segment (centred inside) ─────────────────────────
    for i, (tc, lk) in enumerate(zip(truly_correct, lucky)):
        mid = tc + lk / 2
        ax.text(x[i], mid, str(lk),
                ha="center", va="center",
                fontsize=10, color="white", fontweight="bold", zorder=5)

    # ── total seed-42 count above each bar ────────────────────────────────────
    for i, s42 in enumerate(seed42):
        ax.text(x[i], s42 + 1.5, f"{s42}/{n_tasks}",
                ha="center", va="bottom",
                fontsize=9, color=LABEL, fontweight="bold", zorder=6)

    # ── MORPH-DA catch annotation below x-axis (no arrow, no overlap) ─────────
    catch_lines = []
    for i, (catches, rate) in enumerate(zip(morph_catches, catch_rate)):
        catch_lines.append(
            f"MORPH-DA catches {catches}/{lucky[i]} ({rate:.0%})"
        )

    # Three separate text annotations below the bars
    for i, line in enumerate(catch_lines):
        ax.text(x[i], -8, line,
                ha="center", va="top",
                fontsize=8, color="#1E40AF", style="italic",
                transform=ax.get_xaxis_transform() if False else ax.transData)

    # ── x-axis ────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_xlim(-0.55, len(models) - 0.45)

    # ── y-axis ────────────────────────────────────────────────────────────────
    ax.set_ylim(-14, 80)       # negative room for catch text
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels(["0", "20", "40", "60", "80"], fontsize=9)
    ax.set_ylabel("Programs (out of 101)", fontsize=10)
    ax.set_title("Accidental Corrects by Model\n(seed-42 program re-executed on seeds 7 & 123)",
                 fontsize=10, pad=6)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # Hide negative y-tick labels (they're in the annotation space)
    ax.axhline(0, color="#9CA3AF", linewidth=0.8, zorder=1)

    # ── legend BELOW the plot (outside axes, no overlap) ─────────────────────
    ax.legend(loc="upper center",
              bbox_to_anchor=(0.5, -0.20),   # below axes
              ncol=1, fontsize=9,
              framealpha=0.9, edgecolor="#D1D5DB")

    out = FIGDIR / "accidental_corrects.pdf"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures v4...")
    fig_fault_family()
    fig_accidental_corrects()
    print("Done.")
