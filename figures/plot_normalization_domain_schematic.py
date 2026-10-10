#!/usr/bin/env python3
"""CNN schematic comparing LN, GN, PLN, and mixed-axis grouping."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


OUTPUT_PDF = Path(__file__).resolve().parent / "generated" / "fig_normalization_domains.pdf"
OUTPUT_PNG = OUTPUT_PDF.with_suffix(".png")


COLORS = ["#DCEAF7", "#FDE4D2", "#DDF1E5", "#E9E1F4", "#F6EDC8", "#DDECEF"]
GRID = "#A9ADB3"
BOUNDARY = "#222222"
THEORY = "#007A5E"
EMPIRICAL = "#B45F06"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def group_edges(length: int, groups: int) -> list[int]:
    """Balanced integer group boundaries, including 0 and length."""
    return [round(i * length / groups) for i in range(groups + 1)]


def draw_partition(
    ax: plt.Axes,
    *,
    rows: int,
    cols: int,
    row_groups: int,
    col_groups: int,
    title: str,
    notation: str,
    y_label: str | None = None,
    badge: str | None = None,
    badge_color: str = THEORY,
) -> None:
    """Draw a matrix whose thick boxes are independent normalization domains."""
    r_edges = group_edges(rows, row_groups)
    c_edges = group_edges(cols, col_groups)

    block_index = 0
    for ri in range(row_groups):
        for ci in range(col_groups):
            r0, r1 = r_edges[ri], r_edges[ri + 1]
            c0, c1 = c_edges[ci], c_edges[ci + 1]
            ax.add_patch(
                Rectangle(
                    (c0, rows - r1),
                    c1 - c0,
                    r1 - r0,
                    facecolor=COLORS[block_index % len(COLORS)],
                    edgecolor="none",
                )
            )
            ax.add_patch(
                Rectangle(
                    (c0, rows - r1),
                    c1 - c0,
                    r1 - r0,
                    facecolor="none",
                    edgecolor=BOUNDARY,
                    linewidth=1.35,
                    zorder=4,
                )
            )
            block_index += 1

    for c in range(1, cols):
        ax.plot([c, c], [0, rows], color=GRID, linewidth=0.35, zorder=3)
    for r in range(1, rows):
        ax.plot([0, cols], [r, r], color=GRID, linewidth=0.35, zorder=3)
    ax.add_patch(
        Rectangle((0, 0), cols, rows, facecolor="none", edgecolor=BOUNDARY, linewidth=1.35, zorder=5)
    )

    ax.set_xlim(-0.08, cols + 0.08)
    ax.set_ylim(-0.08, rows + 0.08)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, pad=5, fontweight="bold")
    ax.text(0.5, -0.13, notation, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    ax.text(0.5, -0.27, "channels / features", transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
    if y_label:
        ax.text(-0.18, 0.5, y_label, transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=7.5)
    if badge:
        ax.text(
            0.5,
            1.22,
            badge,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color=badge_color,
            fontsize=6.8,
            fontweight="bold",
        )


def main() -> None:
    configure()
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.9))
    plt.subplots_adjust(left=0.06, right=0.995, top=0.76, bottom=0.15, wspace=0.34)

    rows, cols = 6, 8

    # Image tensors are flattened only for visualization: P = H W.
    draw_partition(
        axes[0], rows=rows, cols=cols, row_groups=1, col_groups=1,
        title="LayerNorm (LN)", notation=r"$G_P=1,\ G_C=1$", y_label=r"spatial positions $P=HW$",
    )
    draw_partition(
        axes[1], rows=rows, cols=cols, row_groups=1, col_groups=2,
        title="GroupNorm (GN)", notation=r"$G_P=1,\ G_C>1$",
    )
    draw_partition(
        axes[2], rows=rows, cols=cols, row_groups=rows, col_groups=2,
        title="PLN (CPLN)", notation=r"$G_P=P,\ G_C>1$",
        badge="EXACT REPRESENTATION", badge_color=THEORY,
    )
    draw_partition(
        axes[3], rows=rows, cols=cols, row_groups=2, col_groups=cols,
        title="Mixed grouping", notation=r"$G_P=2,\ G_C=C=8$",
        badge="EMPIRICAL DOMAIN", badge_color=EMPIRICAL,
    )
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, facecolor="white")
    fig.savefig(OUTPUT_PNG, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
