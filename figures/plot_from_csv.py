"""Regenerate the CNN/ViT appendix figures from bundled per-seed CSVs.

Requires matplotlib and numpy. Outputs go to figures/generated/ so the
committed paper figures in pic/ are not overwritten during verification.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "generated"
GC = [1, 2, 4, 8, 16, 32, 64, 128]
K = [1, 2, 4, 8, 16, 32]
GT = [197, 128, 64, 32, 16, 8, 4, 2, 1]
VIT_GC = {
    "allnorm": [1, 2, 4, 8, 12, 24, 48, 192],
    "activation": [1, 2, 4, 8, 12, 24, 48, 96, 192],
}
LN_COLOR = "#FF2D95"
GN_COLOR = "#00E5FF"


def records(filename: str) -> list[dict[str, str]]:
    with (HERE / "data" / filename).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def values_by_cell(rows: list[dict[str, str]], keys: tuple[str, ...], metric: str):
    grouped = defaultdict(list)
    seeds = defaultdict(set)
    for row in rows:
        if row[metric] != "":
            key = tuple(row[key] for key in keys)
            grouped[key].append(float(row[metric]))
            seeds[key].add(row["seed"])
    for key, values in grouped.items():
        if len(values) != 3 or len(seeds[key]) != 3:
            raise ValueError(f"Expected three distinct seeds for {key}")
    return grouped


def grid(rows, keys, metric, ys, xs):
    grouped = values_by_cell(rows, keys, metric)
    mean = np.empty((len(ys), len(xs)))
    sd = np.empty_like(mean)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            key = (str(y), str(x))
            if key not in grouped:
                raise ValueError(f"Missing cell {key} for metric {metric}")
            values = grouped[key]
            mean[i, j] = np.mean(values)
            sd[i, j] = np.std(values, ddof=1)
    return mean, sd


def annotate(ax, matrix, image, size=18):
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            rgba = image.cmap(image.norm(value))
            light = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            label = f"{value:.0f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=size, color="black" if light > 0.58 else "white")


def outline(ax, row, col, color, width=2.7):
    ax.add_patch(Rectangle((col - 0.43, row - 0.43), 0.86, 0.86,
                           fill=False, edgecolor=color, linewidth=width))


def heatmap(ax, matrix, ys, xs, title, *, vmin, vmax, cmap, position_axis,
            show_values=True, aspect="equal", annotation_size=18):
    image = ax.imshow(matrix, cmap=cmap, norm=Normalize(vmin, vmax), aspect=aspect)
    ax.set_xticks(range(len(xs)), xs)
    ax.set_yticks(range(len(ys)), ys)
    ax.tick_params(axis="both", labelsize=17)
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("Channel groups $G_C$" if position_axis == "token" else
                  r"Spatial splits per axis $K$ ($K\times K$ blocks)", fontsize=17)
    ax.set_ylabel("Token groups $G_T$" if position_axis == "token" else
                  "Channel groups $G_C$", fontsize=17)
    if show_values:
        annotate(ax, matrix, image, size=annotation_size)
    if position_axis == "token":
        ax.add_patch(Rectangle((-0.5, -0.5), len(xs), 1, fill=False,
                               edgecolor="white", linewidth=2, linestyle="--"))
    else:
        ax.add_patch(Rectangle((len(xs) - 1.5, -0.5), 1, len(ys), fill=False,
                               edgecolor="white", linewidth=2, linestyle="--"))
        ax.add_patch(Rectangle((-0.43, -0.43), 0.86, len(ys) - 0.14,
                               fill=False, edgecolor=GN_COLOR, linewidth=3))
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    outline(ax, i, j, "white")
    return image


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cnn_heatmap(cnn):
    panels = (
        ("random50k", "final_train_pct", "Random-label fit", 10, 100, "viridis"),
        ("clean50k", "final_train_pct", "Clean-label train", 75, 100, "viridis"),
        ("clean50k", "final_test_pct", "Clean-label test", 40, 80, "magma"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 10.2))
    for ax, (dataset, metric, title, vmin, vmax, cmap) in zip(axes, panels):
        rows = [r for r in cnn if r["dataset"] == dataset]
        matrix, _ = grid(rows, ("channel_groups", "spatial_splits"), metric, GC, K)
        image = heatmap(ax, matrix, GC, K, title, vmin=vmin, vmax=vmax,
                        cmap=cmap, position_axis="spatial", aspect="auto",
                        annotation_size=17)
        if ax is not axes[-1]:
            ax.set_xlabel("")
        outline(ax, 0, 0, LN_COLOR)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
        colorbar.set_label("Accuracy (%)", fontsize=16)
        colorbar.ax.tick_params(labelsize=15)
    fig.suptitle("CNN normalization domains on CIFAR-10", y=0.99,
                 fontsize=17)
    fig.legend(handles=[Patch(facecolor="none", edgecolor=GN_COLOR, linewidth=3,
                              label="GN family ($K=1$)"),
                        Patch(facecolor="none", edgecolor=LN_COLOR, linewidth=3,
                              label="LN-equivalent endpoint")],
               loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=2, frameon=False, fontsize=16)
    fig.subplots_adjust(left=0.13, right=0.92, bottom=0.06, top=0.86,
                        hspace=0.38)
    save(fig, "fig_cnn_d4_grouping_3seed.pdf")


def plot_vit_heatmaps(vit, dataset):
    rows = [r for r in vit if r["dataset"] == dataset]
    mode_labels = {"allnorm": "Normalization replacement",
                   "activation": "Activation replacement"}
    if dataset == "clean50k":
        fig, axes = plt.subplots(2, 2, figsize=(15.4, 12.0), constrained_layout=True)
        items = [("final_train_pct", "Train", 75, 100, "viridis"),
                 ("final_test_pct", "Test", 60, 75, "magma")]
        for i, mode in enumerate(("allnorm", "activation")):
            for j, (metric, label, vmin, vmax, cmap) in enumerate(items):
                subset = [r for r in rows if r["mode"] == mode]
                matrix, _ = grid(subset, ("token_groups", "channel_groups"),
                                 metric, GT, VIT_GC[mode])
                image = heatmap(axes[i, j], matrix, GT, VIT_GC[mode],
                                f"{'Normalization' if mode == 'allnorm' else 'Activation'}: {label.lower()}",
                                vmin=vmin, vmax=vmax, cmap=cmap, position_axis="token")
                if mode == "allnorm":
                    outline(axes[i, j], 0, 0, LN_COLOR)
                colorbar = fig.colorbar(image, ax=axes[i, j], fraction=0.046,
                                        pad=0.035)
                colorbar.set_label("Accuracy (%)", fontsize=16)
                colorbar.ax.tick_params(labelsize=15)
        name = "fig_vit_clean50k_two_modes_3seed.pdf"
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.8), constrained_layout=True)
        for ax, mode in zip(axes, ("allnorm", "activation")):
            subset = [r for r in rows if r["mode"] == mode]
            matrix, _ = grid(subset, ("token_groups", "channel_groups"),
                             "best_fit_pct", GT, VIT_GC[mode])
            image = heatmap(ax, matrix, GT, VIT_GC[mode],
                            mode_labels[mode], vmin=10, vmax=100,
                            cmap="viridis", position_axis="token")
            if mode == "allnorm":
                outline(ax, 0, 0, LN_COLOR)
        colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
        colorbar.set_label("Best fit (%)", fontsize=16)
        colorbar.ax.tick_params(labelsize=15)
        name = f"fig_vit_{dataset}_two_modes_3seed.pdf"
    if dataset == "clean50k":
        fig.suptitle("ViT clean-label accuracy on CIFAR-10", fontsize=17)
    else:
        sample_count = {"random6k": "6,000", "random20k": "20,000"}[dataset]
        fig.suptitle(f"ViT random-label fit on CIFAR-10 ({sample_count} examples)",
                     fontsize=17)
    save(fig, name)


def plot_cnn_direction(cnn):
    panels = (("random50k", "final_train_pct", "Random-label fit", (7, 104)),
              ("clean50k", "final_train_pct", "Clean-label train", (7, 104)),
              ("clean50k", "final_test_pct", "Clean-label test", (7, 78)))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(K)))
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.85))
    for ax, (dataset, metric, title, ylim) in zip(axes, panels):
        rows = [r for r in cnn if r["dataset"] == dataset]
        mean, sd = grid(rows, ("channel_groups", "spatial_splits"), metric, GC, K)
        for j, (k, color) in enumerate(zip(K, colors)):
            xs = np.array([gc * k * k for gc in GC])
            label = f"$K={k}$" + (" (GN)" if k == 1 else
                                    " (PLN)" if k == 32 else "")
            ax.errorbar(xs, mean[:, j], yerr=sd[:, j], color=color,
                        marker=("o", "s", "D", "^", "v", ">")[j],
                        markersize=3, linewidth=1.25 if k in (1, 32) else 0.72,
                        elinewidth=0.42, capsize=0.9, label=label)
        ax.scatter(1, mean[0, 0], marker="*", color="#d62728", s=54, label="LN")
        ax.set(xscale="log", ylim=ylim, title=title, xlabel=r"Total groups $G_CK^2$")
        ax.tick_params(axis="both", labelsize=9.5)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0.005, 0.06, 0.995, 0.82))
    save(fig, "fig_cnn_grouping_direction.pdf")


def plot_vit_direction(vit):
    random = [r for r in vit if r["dataset"] == "random20k"]
    clean = [r for r in vit if r["dataset"] == "clean50k"]
    specs = ((random, "best_fit_pct", "Random-20k fit", (7, 98)),
             (clean, "final_train_pct", "Clean-label train", (7, 104)),
             (clean, "final_test_pct", "Clean-label test", (7, 77)))
    colors = plt.get_cmap("viridis")(np.linspace(0.07, 0.94, len(GT)))
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.7))
    baseline = []
    for source, metric, _, _ in specs:
        rows = [r for r in source if r["mode"] == "allnorm"]
        mean, _ = grid(rows, ("token_groups", "channel_groups"),
                       metric, GT, VIT_GC["allnorm"])
        baseline.append(mean[0, 0])
    for i, mode in enumerate(("allnorm", "activation")):
        for j, (source, metric, title, ylim) in enumerate(specs):
            rows = [r for r in source if r["mode"] == mode]
            mean, sd = grid(rows, ("token_groups", "channel_groups"),
                            metric, GT, VIT_GC[mode])
            ax = axes[i, j]
            for r, (gt, color) in enumerate(zip(GT, colors)):
                xs = np.array([gt * gc for gc in VIT_GC[mode]])
                label = f"$G_T={gt}$" + (" (PLN)" if gt == 197 else "")
                ax.errorbar(xs, mean[r], yerr=sd[r], color=color,
                            marker=("o", "s", "D", "^", "v", "P", "X", "<", ">")[r],
                            markersize=3.1 if gt == 197 else 2.5,
                            linewidth=1.25 if gt == 197 else 0.62,
                            elinewidth=0.36, capsize=0.8, label=label)
            ax.axhline(baseline[j], color="#d62728", linestyle="--", linewidth=0.82)
            ax.set(xscale="log", ylim=ylim, title=title,
                   xlabel=r"Total groups $G_TG_C$")
            ax.tick_params(axis="both", labelsize=9.5)
            ax.grid(alpha=0.18)
    axes[0, 0].set_ylabel("All norms replaced\nAccuracy (%)")
    axes[1, 0].set_ylabel("FFN activations replaced\nAccuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#d62728", linestyle="--", label="LN-GELU"))
    labels.append("LN-GELU")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.005, 0.04, 0.995, 0.86))
    save(fig, "fig_vit_grouping_direction_20k.pdf")


def main():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    cnn = records("cnn_runs.csv")
    vit = records("vit_runs.csv")
    plot_cnn_heatmap(cnn)
    plot_cnn_direction(cnn)
    for dataset in ("random6k", "random20k", "clean50k"):
        plot_vit_heatmaps(vit, dataset)
    plot_vit_direction(vit)
    print(f"Wrote six figures to {OUTPUT}")


if __name__ == "__main__":
    main()
