"""Audit and aggregate a newly completed paper-protocol VGG grid.

This does not recover the original manuscript figure's missing run records.
Choose ``final`` or ``best`` explicitly: the historical figure's epoch
selection rule has not been verified from source data.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from experiments.vgg.make_grid_plan import (
    ACTIVATIONS, INIT_SCALES, LEARNING_RATES,
)


def run_id(activation, lr, init_scale, seed):
    return f"{activation}_lr{lr}_init{init_scale}_seed{seed}"


def read_accuracy(directory, activation, lr, init_scale, seed, metric):
    """Read one complete run and reject settings that are not paper-comparable."""
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    expected = {
        "activation": activation, "lr": float(lr),
        "init_scale": float(init_scale), "seed": seed,
        "epochs": 200, "warmup_epochs": 10, "batch_size": 256,
        "weight_decay": 1e-4, "max_train_batches": None,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"{directory}: {name}={config.get(name)!r}; expected {value!r}")
    records = [json.loads(line) for line in
               (directory / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != 200 or any(row.get("epoch") != i
                                  for i, row in enumerate(records)):
        raise ValueError(f"{directory}: expected 200 consecutive epoch records")
    field = "test_accuracy" if metric == "final" else "best_test_accuracy"
    value = records[-1].get(field)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(f"{directory}: invalid {field}={value!r}")
    return float(value)


def summarize(run_root, seeds, metric):
    if metric not in ("final", "best"):
        raise ValueError("metric must be 'final' or 'best'")
    seeds = tuple(seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("paper-protocol summaries require three distinct seeds")
    rows = []
    for activation in ACTIVATIONS:
        for init_scale in INIT_SCALES:
            for lr in LEARNING_RATES:
                values = [read_accuracy(
                    run_root / run_id(activation, lr, init_scale, seed),
                    activation, lr, init_scale, seed, metric,
                ) for seed in seeds]
                rows.append({
                    "activation": activation,
                    "learning_rate": lr,
                    "initialization_multiplier": init_scale,
                    "mean_test_accuracy": sum(values) / len(values),
                    "metric": metric,
                    "seed_accuracies_json": json.dumps(dict(zip(seeds, values))),
                })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs=3, required=True)
    parser.add_argument("--metric", choices=("final", "best"), required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new summary CSV; an existing file is not overwritten")
    args = parser.parse_args(argv)
    rows = summarize(args.run_root, args.seeds, args.metric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} verified three-seed grid cells to {args.output}")


if __name__ == "__main__":
    main()
