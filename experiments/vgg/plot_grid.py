"""Plot a completed *new* VGG paper grid from summarize_grid.py output."""

import argparse
import csv
import math
from pathlib import Path

from experiments.vgg.make_grid_plan import ACTIVATIONS, INIT_SCALES, LEARNING_RATES


def load_summary(path):
    expected = {(activation, lr, scale) for activation in ACTIVATIONS
                for lr in LEARNING_RATES for scale in INIT_SCALES}
    cells = {}
    metrics = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["activation"], row["learning_rate"],
                   row["initialization_multiplier"])
            if key not in expected or key in cells:
                raise ValueError(f"unexpected or duplicate VGG cell: {key}")
            value = float(row["mean_test_accuracy"])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"invalid test accuracy for {key}: {value}")
            cells[key] = value
            metrics.add(row["metric"])
    if set(cells) != expected:
        raise ValueError(f"incomplete VGG grid: {len(cells)}/{len(expected)} cells")
    if len(metrics) != 1 or next(iter(metrics)) not in ("final", "best"):
        raise ValueError(f"invalid or mixed epoch metric: {metrics}")
    return cells, metrics.pop()


def plot_summary(cells, metric, output):
    import matplotlib.pyplot as plt

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    scales_top_down = tuple(reversed(INIT_SCALES))
    fig, axes = plt.subplots(1, len(ACTIVATIONS), figsize=(16, 4.8),
                             sharex=True, sharey=True, constrained_layout=True)
    for axis, activation in zip(axes, ACTIVATIONS):
        matrix = [[cells[(activation, lr, scale)] for lr in LEARNING_RATES]
                  for scale in scales_top_down]
        image = axis.imshow(matrix, cmap="Greens", vmin=0, vmax=100,
                            aspect="auto", interpolation="nearest")
        axis.set_title({"pln": "PLN[8]", "relu": "ReLU", "silu": "SiLU",
                        "tanh": "Tanh", "sat": "SAT"}[activation])
        axis.set_xticks(range(len(LEARNING_RATES)), LEARNING_RATES,
                        rotation=45, ha="right")
        axis.set_yticks(range(len(scales_top_down)), scales_top_down)
        axis.set_xlabel("Initial learning rate")
    axes[0].set_ylabel("He initialization multiplier")
    fig.colorbar(image, ax=axes, label=f"Mean CIFAR-10 test accuracy (%) — {metric} epoch",
                 shrink=0.75)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new PDF or PNG file; existing files are not overwritten")
    args = parser.parse_args(argv)
    cells, metric = load_summary(args.summary)
    plot_summary(cells, metric, args.output)
    print(f"Plotted {len(cells)} {metric}-epoch cells to {args.output}")


if __name__ == "__main__":
    main()
