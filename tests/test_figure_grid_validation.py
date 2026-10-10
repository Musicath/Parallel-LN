"""Regression checks for the figure-data integrity gate."""

import csv
import importlib
from itertools import product
from pathlib import Path
import sys
import unittest
from unittest import mock

from figures.validate_grid import (
    require_exact_keys,
    require_percentages,
    require_unique_sources,
)


ROOT = Path(__file__).resolve().parents[1]


def load_rows(filename):
    with (ROOT / "figures" / "data" / filename).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class FigureGridValidationTests(unittest.TestCase):
    def test_cnn_exact_grid_and_duplicate_replacing_missing_cell(self):
        rows = load_rows("cnn_runs.csv")
        fields = ("dataset", "seed", "channel_groups", "spatial_splits")
        expected = set(product(("clean50k", "random50k"), ("0", "1", "2"),
                               map(str, (1, 2, 4, 8, 16, 32, 64, 128)),
                               map(str, (1, 2, 4, 8, 16, 32))))
        require_exact_keys(rows, fields, expected, "CNN")
        corrupted = rows.copy()
        corrupted[1] = corrupted[0]
        with self.assertRaisesRegex(ValueError, "missing=1.*duplicate=1"):
            require_exact_keys(corrupted, fields, expected, "CNN")

    def test_vit_exact_grid_and_unknown_cell(self):
        rows = load_rows("vit_runs.csv")
        fields = ("dataset", "mode", "seed", "token_groups", "channel_groups")
        groups = {"allnorm": (1, 2, 4, 8, 12, 24, 48, 192),
                  "activation": (1, 2, 4, 8, 12, 24, 48, 96, 192)}
        expected = (key for dataset in ("clean50k", "random6k", "random20k")
                    for mode in groups
                    for key in product((dataset,), (mode,), ("40", "41", "42"),
                                       map(str, (197, 128, 64, 32, 16, 8, 4, 2, 1)),
                                       map(str, groups[mode])))
        require_exact_keys(rows, fields, expected, "ViT")
        corrupted = rows.copy()
        corrupted[0] = dict(corrupted[0], token_groups="999")
        expected = (key for dataset in ("clean50k", "random6k", "random20k")
                    for mode in groups
                    for key in product((dataset,), (mode,), ("40", "41", "42"),
                                       map(str, (197, 128, 64, 32, 16, 8, 4, 2, 1)),
                                       map(str, groups[mode])))
        with self.assertRaisesRegex(ValueError, "missing=1.*unexpected=1"):
            require_exact_keys(corrupted, fields, expected, "ViT")

    def test_rejects_bad_accuracy_and_reused_source(self):
        rows = load_rows("cnn_runs.csv")
        require_percentages(rows, ("final_train_pct",), "CNN")
        require_unique_sources(rows, "CNN")
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            require_percentages([dict(rows[0], final_train_pct="nan")],
                                ("final_train_pct",), "CNN")
        with self.assertRaisesRegex(ValueError, "unique"):
            require_unique_sources([rows[0], rows[0]], "CNN")

    def test_plot_entrypoint_rejects_corruption_before_drawing(self):
        figures_dir = str(ROOT / "figures")
        sys.path.insert(0, figures_dir)
        try:
            plotter = importlib.import_module("plot_from_csv")
        finally:
            sys.path.remove(figures_dir)
        cnn = load_rows("cnn_runs.csv")
        cnn[1] = cnn[0]
        vit = load_rows("vit_runs.csv")
        with mock.patch.object(plotter, "records", side_effect=(cnn, vit)), \
             mock.patch.object(plotter, "plot_cnn_heatmap") as draw:
            with self.assertRaisesRegex(ValueError, "incomplete or repeated"):
                plotter.main()
            draw.assert_not_called()

    def test_vit_manuscript_table_checks_both_replacement_modes(self):
        figures_dir = str(ROOT / "figures")
        sys.path.insert(0, figures_dir)
        try:
            verifier = importlib.import_module("verify_data")
        finally:
            sys.path.remove(figures_dir)
        rows = load_rows("vit_runs.csv")
        verifier.verify_vit_summary(rows)
        for mode, token_groups, channel_groups, metric in (
            ("allnorm", "197", "1", "final_test_pct"),
            ("activation", "64", "192", "best_fit_pct"),
        ):
            corrupted = [dict(row) for row in rows]
            target = next(row for row in corrupted
                          if row["dataset"] == ("clean50k" if mode == "allnorm" else "random20k")
                          and row["mode"] == mode and row["seed"] == "40"
                          and row["token_groups"] == token_groups
                          and row["channel_groups"] == channel_groups)
            target[metric] = "0"
            with self.assertRaisesRegex(AssertionError, "ViT table"):
                verifier.verify_vit_summary(corrupted)

    def test_cnn_manuscript_table_checks_pln4_endpoint(self):
        figures_dir = str(ROOT / "figures")
        sys.path.insert(0, figures_dir)
        try:
            verifier = importlib.import_module("verify_data")
        finally:
            sys.path.remove(figures_dir)
        rows = load_rows("cnn_runs.csv")
        verifier.verify_cnn_summary(rows)
        corrupted = [dict(row) for row in rows]
        target = next(row for row in corrupted
                      if row["dataset"] == "random50k" and row["seed"] == "0"
                      and row["channel_groups"] == "32"
                      and row["spatial_splits"] == "32")
        target["final_train_pct"] = "0"
        with self.assertRaisesRegex(AssertionError, "PLN\\[4\\]"):
            verifier.verify_cnn_summary(corrupted)


if __name__ == "__main__":
    unittest.main()
