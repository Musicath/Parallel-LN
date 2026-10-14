"""CPU checks for the framework-independent VGG-16 extraction."""

import unittest
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.vgg.model import ACTIVATIONS, PaperVGG16, SaturatingActivation
from experiments.vgg.make_grid_plan import (
    INIT_SCALES, LEARNING_RATES, build_plan, main as write_grid_plan,
)
from experiments.vgg.train_cifar10 import learning_rate_for_epoch, parse_args, run
from experiments.vgg.summarize_grid import summarize
from experiments.vgg.plot_grid import load_summary, plot_summary
from pln import ParallelLN


class VGGExtractionTests(unittest.TestCase):
    def test_grid_plan_enumerates_paper_factors_without_training(self):
        self.assertEqual(LEARNING_RATES,
                         ("0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1"))
        self.assertEqual(INIT_SCALES,
                         ("0.25", "0.35", "0.5", "0.7", "1.0", "1.4", "2.0", "2.8", "4.0"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seeds, expected in (((20, 30), 630), ((20, 30, 40), 945)):
                plan = list(build_plan(seeds, root / "data", root / "runs"))
                self.assertEqual(len(plan), expected)
                self.assertEqual(len({row["run_id"] for row in plan}), expected)
                self.assertTrue(all(row["argv"][1:3] == ["-m", "experiments.vgg.train_cifar10"]
                                    for row in plan))
                self.assertTrue(all(row["argv"][-2:] == ["--device", "cuda:0"]
                                    for row in plan))
            parsed = parse_args(plan[0]["argv"][3:])
            self.assertEqual(parsed.activation, "pln")
            self.assertEqual(parsed.epochs, 200)
            self.assertEqual(parsed.warmup_epochs, 10)
            self.assertEqual(parsed.batch_size, 256)
            self.assertEqual(parsed.weight_decay, 1e-4)
            with self.assertRaises(ValueError):
                list(build_plan((20, 20), root / "data", root / "runs"))

            plan_path = root / "plan.jsonl"
            write_grid_plan(["--seeds", "20", "30", "40", "--plan", str(plan_path)])
            self.assertEqual(len(plan_path.read_text(encoding="utf-8").splitlines()), 945)
            with self.assertRaises(FileExistsError):
                write_grid_plan(["--seeds", "20", "30", "40", "--plan", str(plan_path)])

    def test_historical_feature_layout_and_five_activations(self):
        self.assertEqual(ACTIVATIONS, ("pln", "relu", "silu", "tanh", "sat"))
        for name in ACTIVATIONS:
            with self.subTest(activation=name):
                model = PaperVGG16(activation=name, init_scale=0.7)
                self.assertEqual(
                    sum(isinstance(m, nn.Conv2d) for m in model.modules()), 13
                )
                self.assertEqual(
                    sum(isinstance(m, nn.MaxPool2d) for m in model.modules()), 4
                )
                self.assertEqual(model.classifier[0].in_features, 512)
                if name == "pln":
                    groups = [m for m in model.modules() if isinstance(m, ParallelLN)]
                    self.assertEqual(len(groups), 13)
                    self.assertTrue(all(m.num_per_group == 8 for m in groups))

    def test_forward_backward_and_saturating_formula(self):
        torch.manual_seed(20)
        model = PaperVGG16(activation="pln")
        output = model(torch.randn(1, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (1, 10))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.features[0].weight.grad).all())
        x = torch.tensor([-2.0, 0.0, 2.0])
        torch.testing.assert_close(
            SaturatingActivation()(x), x / torch.sqrt(x.square() + 1)
        )

    def test_paper_learning_rate_epoch_boundaries(self):
        expected = {
            1: 0.001, 9: 0.009, 10: 0.01, 11: 0.01,
            49: 0.01, 50: 0.004, 89: 0.004,
            90: 0.0016, 130: 0.00064, 170: 0.000256,
        }
        for epoch_number, rate in expected.items():
            with self.subTest(epoch=epoch_number):
                self.assertAlmostEqual(learning_rate_for_epoch(0.01, epoch_number), rate)

    def test_summary_requires_three_complete_paper_runs_and_explicit_metric(self):
        with tempfile.TemporaryDirectory() as temporary, patch(
            "experiments.vgg.summarize_grid.ACTIVATIONS", ("pln",)
        ), patch("experiments.vgg.summarize_grid.INIT_SCALES", ("1.0",)), patch(
            "experiments.vgg.summarize_grid.LEARNING_RATES", ("0.01",)
        ):
            root = Path(temporary)
            for seed, accuracy in ((20, 60.0), (30, 70.0), (40, 80.0)):
                directory = root / f"pln_lr0.01_init1.0_seed{seed}"
                directory.mkdir()
                config = dict(activation="pln", lr=0.01, init_scale=1.0,
                              seed=seed, epochs=200, warmup_epochs=10,
                              batch_size=256, weight_decay=1e-4,
                              max_train_batches=None)
                (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
                records = [dict(epoch=i, test_accuracy=accuracy,
                                best_test_accuracy=accuracy + 1)
                           for i in range(200)]
                (directory / "epochs.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in records), encoding="utf-8"
                )
            rows = summarize(root, (20, 30, 40), "final")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["mean_test_accuracy"], 70.0)
            self.assertEqual(summarize(root, (20, 30, 40), "best")[0][
                "mean_test_accuracy"], 71.0)
            with self.assertRaises(ValueError):
                summarize(root, (20, 30), "final")
            with self.assertRaises(ValueError):
                summarize(root, (20, 30, 40), "unspecified")
            (root / "pln_lr0.01_init1.0_seed40" / "epochs.jsonl").unlink()
            with self.assertRaises(FileNotFoundError):
                summarize(root, (20, 30, 40), "final")

    def test_plotter_rejects_duplicate_grid_cell_and_plots_complete_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.csv"
            with summary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    "activation", "learning_rate", "initialization_multiplier",
                    "mean_test_accuracy", "metric",
                ))
                writer.writeheader()
                for activation in ACTIVATIONS:
                    for scale in INIT_SCALES:
                        for lr in LEARNING_RATES:
                            writer.writerow(dict(activation=activation,
                                                 learning_rate=lr,
                                                 initialization_multiplier=scale,
                                                 mean_test_accuracy=65.0,
                                                 metric="final"))
            cells, metric = load_summary(summary)
            self.assertEqual(len(cells), 315)
            figure = root / "figure.pdf"
            plot_summary(cells, metric, figure)
            self.assertGreater(figure.stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                plot_summary(cells, metric, figure)
            with summary.open("a", encoding="utf-8") as stream:
                stream.write("pln,0.01,1.0,65,final\n")
            with self.assertRaises(ValueError):
                load_summary(summary)

    def test_one_epoch_output_contract_with_synthetic_data(self):
        # Exercises file creation and the loop, not CIFAR-10 accuracy.
        samples = TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([1, 2]))
        loader = DataLoader(samples, batch_size=1)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cell"
            args = parse_args([
                "--data-root", temporary, "--output", str(output),
                "--device", "cpu", "--activation", "sat", "--epochs", "1",
                "--warmup-epochs", "1", "--batch-size", "1", "--workers", "0",
                "--max-train-batches", "1",
            ])
            with patch("experiments.vgg.train_cifar10.make_loaders",
                       return_value=(loader, loader)):
                run(args)
            record = json.loads((output / "epochs.jsonl").read_text().strip())
            self.assertEqual(record["epoch"], 0)
            self.assertEqual(record["epoch_number"], 1)
            self.assertTrue(0 <= record["test_accuracy"] <= 100)
            self.assertEqual(json.loads((output / "config.json").read_text())["activation"], "sat")
            with self.assertRaises(FileExistsError), patch(
                "experiments.vgg.train_cifar10.make_loaders",
                return_value=(loader, loader),
            ):
                run(args)


if __name__ == "__main__":
    unittest.main()
