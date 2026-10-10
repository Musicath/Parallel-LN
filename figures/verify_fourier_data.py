"""Recompute the manuscript's Dense Fourier table from sanitized run records."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


DATA = Path(__file__).resolve().parent / "data"
ACTIVATIONS = ("identity", "ln", "pln4", "pln8", "relu", "silu", "tanh", "sat")
TARGETS = (20260903, 20260913, 20260923, 20260933, 20260943)
LEARNING_RATES = (0.0001, 0.0003, 0.001)


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def verify_dimension(dimension: int) -> None:
    runs = read_csv(f"dense_fourier_d{dimension}_runs.csv")
    table = read_csv(f"dense_fourier_d{dimension}_table.csv")
    assert len(runs) == 720, f"d={dimension}: expected 720 runs, got {len(runs)}"
    assert len(table) == 16, f"d={dimension}: expected 16 table rows"

    by_config: dict[tuple[str, int, float], list[dict[str, str]]] = defaultdict(list)
    observed: set[tuple[str, int, float, int, int]] = set()
    for row in runs:
        assert int(row["input_dim"]) == dimension
        assert int(row["width"]) == 1024
        assert row["lr_schedule"] == "warmup-cosine"
        assert int(row["warmup_steps"]) == 500
        assert int(row["steps"]) == 10000
        assert int(row["data_seed"]) == 20260904
        activation = row["activation"]
        depth = int(row["depth"])
        lr = float(row["lr"])
        target = int(row["target_seed"])
        seed = int(row["model_seed"])
        assert activation in ACTIVATIONS
        assert depth in (1, 4)
        assert lr in LEARNING_RATES
        assert target in TARGETS
        assert seed in (0, 1, 2)
        identity = (activation, depth, lr, target, seed)
        assert identity not in observed, f"d={dimension}: duplicate {identity}"
        observed.add(identity)
        by_config[(activation, depth, lr)].append(row)

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
        assert math.isclose(selected_lr, float(paper_row["selected_lr"])), (
            dimension, activation, depth, selected_lr
        )
        selected_runs = by_config[(activation, depth, selected_lr)]
        assert len(selected_runs) == 15
        per_target = [
            statistics.mean(
                float(row["test_mse"])
                for row in selected_runs
                if int(row["target_seed"]) == target
            )
            for target in TARGETS
        ]
        assert int(paper_row["n_targets"]) == len(per_target)
        for field, actual in (
            ("test_mse_mean", statistics.mean(per_target)),
            ("test_mse_sd_across_targets", statistics.stdev(per_target)),
        ):
            expected = float(paper_row[field])
            assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=5e-7), (
                dimension, activation, depth, field, actual, expected
            )


def main() -> None:
    for dimension in (4, 8):
        verify_dimension(dimension)
    print("Verified both 720-run Dense Fourier grids and paper summaries")


if __name__ == "__main__":
    main()
