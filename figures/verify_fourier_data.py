"""Recompute the manuscript's Dense Fourier table from sanitized run records."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from itertools import product
from pathlib import Path


DATA = Path(__file__).resolve().parent / "data"
ACTIVATIONS = ("identity", "ln", "pln4", "pln8", "relu", "silu", "tanh", "sat")
TARGETS = (20260903, 20260913, 20260923, 20260933, 20260943)
LEARNING_RATES = (0.0001, 0.0003, 0.001)

# Mean and sample SD printed in the arXiv appendix, independent of the two
# summary CSVs. Keep the displayed decimal precision, including trailing zeros.
PAPER_FOURIER = {
    4: {
        ("identity", 1): ("0.9982", "0.0071"),
        ("identity", 4): ("0.9979", "0.0072"),
        ("ln", 1): ("0.9979", "0.0071"),
        ("ln", 4): ("0.8793", "0.0885"),
        ("pln4", 1): ("0.0048", "0.0013"),
        ("pln4", 4): ("0.0018", "0.0006"),
        ("pln8", 1): ("0.0079", "0.0030"),
        ("pln8", 4): ("0.00075", "0.00027"),
        ("relu", 1): ("0.0118", "0.0043"),
        ("relu", 4): ("0.00081", "0.00025"),
        ("silu", 1): ("0.9985", "0.0072"),
        ("silu", 4): ("0.00037", "0.00026"),
        ("tanh", 1): ("0.3574", "0.0425"),
        ("tanh", 4): ("0.0014", "0.0006"),
        ("sat", 1): ("0.2302", "0.0127"),
        ("sat", 4): ("0.0011", "0.0003"),
    },
    8: {
        ("identity", 1): ("0.9989", "0.0159"),
        ("identity", 4): ("0.9972", "0.0144"),
        ("ln", 1): ("0.9990", "0.0158"),
        ("ln", 4): ("0.9808", "0.0099"),
        ("pln4", 1): ("0.1171", "0.0197"),
        ("pln4", 4): ("0.1034", "0.0244"),
        ("pln8", 1): ("0.0803", "0.0068"),
        ("pln8", 4): ("0.0660", "0.0215"),
        ("relu", 1): ("0.1196", "0.0082"),
        ("relu", 4): ("0.0282", "0.0036"),
        ("silu", 1): ("0.9986", "0.0156"),
        ("silu", 4): ("0.0269", "0.0143"),
        ("tanh", 1): ("0.9384", "0.0242"),
        ("tanh", 4): ("0.0345", "0.0075"),
        ("sat", 1): ("0.7131", "0.0359"),
        ("sat", 4): ("0.0335", "0.0079"),
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def check_printed(value: float, printed: str, label: str) -> None:
    decimals = len(printed.partition(".")[2])
    observed = f"{value:.{decimals}f}"
    require(observed == printed,
            f"{label}: manuscript prints {printed}, reconstructed {observed}")


def verify_dimension(dimension: int) -> None:
    runs = read_csv(f"dense_fourier_d{dimension}_runs.csv")
    table = read_csv(f"dense_fourier_d{dimension}_table.csv")
    require(len(runs) == 720,
            f"d={dimension}: expected 720 runs, got {len(runs)}")
    require(len(table) == 16, f"d={dimension}: expected 16 table rows")

    by_config: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    observed: set[tuple[str, int, float, int, int]] = set()
    for row in runs:
        require(int(row["input_dim"]) == dimension, f"d={dimension}: wrong input dimension")
        require(int(row["width"]) == 1024, f"d={dimension}: wrong width")
        require(row["lr_schedule"] == "warmup-cosine", f"d={dimension}: wrong LR schedule")
        require(int(row["warmup_steps"]) == 500, f"d={dimension}: wrong warmup steps")
        require(int(row["steps"]) == 10000, f"d={dimension}: wrong training steps")
        require(int(row["data_seed"]) == 20260904, f"d={dimension}: wrong data seed")
        activation = row["activation"]
        depth = int(row["depth"])
        lr = float(row["lr"])
        target = int(row["target_seed"])
        seed = int(row["model_seed"])
        require(activation in ACTIVATIONS, f"d={dimension}: unknown activation {activation}")
        require(depth in (1, 4), f"d={dimension}: invalid depth {depth}")
        require(lr in LEARNING_RATES, f"d={dimension}: invalid learning rate {lr}")
        require(target in TARGETS, f"d={dimension}: invalid target seed {target}")
        require(seed in (0, 1, 2), f"d={dimension}: invalid model seed {seed}")
        identity = (activation, depth, lr, target, seed)
        require(identity not in observed, f"d={dimension}: duplicate {identity}")
        observed.add(identity)
        by_config[(activation, depth, lr)].append(row)

    expected_runs = set(product(ACTIVATIONS, (1, 4), LEARNING_RATES,
                                TARGETS, (0, 1, 2)))
    require(observed == expected_runs, f"d={dimension}: incomplete run grid")
    expected_table = set(product(ACTIVATIONS, (1, 4)))
    actual_table = [(row["activation"], int(row["depth"])) for row in table]
    require(len(set(actual_table)) == len(actual_table) and
            set(actual_table) == expected_table,
            f"d={dimension}: incomplete or duplicate summary rows")

    for paper_row in table:
        activation = paper_row["activation"]
        depth = int(paper_row["depth"])
        validation_means = {
            lr: statistics.mean(
                float(row["validation_mse"])
                for row in by_config[(activation, depth, lr)]
            )
            for lr in LEARNING_RATES
        }
        selected_lr = min(validation_means, key=validation_means.get)
        require(math.isclose(selected_lr, float(paper_row["selected_lr"])),
                f"d={dimension}, {activation}, depth={depth}: selected LR differs: {selected_lr}")
        selected_runs = by_config[(activation, depth, selected_lr)]
        require(len(selected_runs) == 15,
                f"d={dimension}, {activation}, depth={depth}: expected 15 selected runs")
        per_target = [
            statistics.mean(
                float(row["test_mse"])
                for row in selected_runs
                if int(row["target_seed"]) == target
            )
            for target in TARGETS
        ]
        require(int(paper_row["n_targets"]) == len(per_target),
                f"d={dimension}, {activation}, depth={depth}: wrong target count")
        reconstructed = (
            ("test_mse_mean", statistics.mean(per_target)),
            ("test_mse_sd_across_targets", statistics.stdev(per_target)),
        )
        for (field, actual), printed in zip(
            reconstructed, PAPER_FOURIER[dimension][(activation, depth)]
        ):
            expected = float(paper_row[field])
            require(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=5e-7),
                    f"d={dimension}, {activation}, depth={depth}, {field}: "
                    f"observed {actual}, expected {expected}")
            check_printed(actual, printed,
                          f"d={dimension}, {activation}, depth={depth}, {field}")


def main() -> None:
    for dimension in (4, 8):
        verify_dimension(dimension)
    print("Verified both 720-run Dense Fourier grids and paper summaries")


if __name__ == "__main__":
    main()
