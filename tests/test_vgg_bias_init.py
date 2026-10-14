"""CPU checks for the separate historical 108-cell VGG-Bias extraction."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.vgg_bias_init.make_plan import build_plan, main as write_plan
from experiments.vgg_bias_init.model import VGGBiasInit, apply_parametric_init
from experiments.vgg_bias_init.train import (
    learning_rate_for_epoch, parse_args, run,
)


class VGGBiasInitTests(unittest.TestCase):
    def test_grid_is_108_cells_and_never_trains(self):
        plan = list(build_plan("data/cifar10", "outputs/vgg_bias_init"))
        self.assertEqual(len(plan), 108)
        self.assertEqual(len({item["run_id"] for item in plan}), 108)
        self.assertEqual({item["seed"] for item in plan}, {42})
        self.assertEqual({item["activation"] for item in plan}, {"relu", "tanh"})
        self.assertEqual({item["norm"] for item in plan}, {"No", "LN", "BN"})
        self.assertTrue(all(item["argv"][1:3] == ["-m", "experiments.vgg_bias_init.train"]
                            for item in plan))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "plan.jsonl"
            write_plan(["--data-root", "data/cifar10", "--plan", str(target)])
            self.assertEqual(len(target.read_text(encoding="utf-8").splitlines()), 108)
            with self.assertRaises(FileExistsError):
                write_plan(["--data-root", "data/cifar10", "--plan", str(target)])

    def test_architecture_and_norms(self):
        for norm in ("No", "LN", "BN"):
            model = VGGBiasInit(norm=norm, activation="tanh")
            with self.subTest(norm=norm):
                self.assertEqual(sum(isinstance(x, nn.Conv2d) for x in model.modules()), 13)
                self.assertEqual(sum(isinstance(x, nn.MaxPool2d) for x in model.modules()), 4)
                self.assertEqual(sum(isinstance(x, nn.Linear) for x in model.modules()), 2)
                self.assertEqual(sum(p.numel() for p in model.parameters()),
                                 14982474 + (9472 if norm != "No" else 0))
                self.assertTrue(all(x.bias is not None for x in model.modules()
                                    if isinstance(x, (nn.Conv2d, nn.Linear))))
                if norm == "LN":
                    self.assertEqual(sum(isinstance(x, nn.GroupNorm) for x in model.modules()), 14)
                if norm == "BN":
                    self.assertEqual(sum(isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d))
                                         for x in model.modules()), 14)
        with torch.no_grad():
            self.assertEqual(tuple(VGGBiasInit()(torch.zeros(1, 3, 32, 32)).shape), (1, 10))

    def test_initializer_endpoints_and_schedule(self):
        small = nn.Sequential(nn.Conv2d(3, 2, 3), nn.Linear(2, 1))
        apply_parametric_init(small, scalar=2.0, beta=1.0)
        self.assertEqual(torch.count_nonzero(small[0].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(small[1].weight).item(), 0)
        apply_parametric_init(small, scalar=2.0, beta=0.0)
        self.assertEqual(torch.count_nonzero(small[0].bias).item(), 0)
        self.assertEqual(torch.count_nonzero(small[1].bias).item(), 0)
        self.assertAlmostEqual(learning_rate_for_epoch(0), 0.03)
        self.assertGreater(learning_rate_for_epoch(1), 0.029)
        self.assertAlmostEqual(learning_rate_for_epoch(100), 0.0)
        self.assertAlmostEqual(learning_rate_for_epoch(1, profile="yaml-intended"), 0.0057)
        self.assertAlmostEqual(learning_rate_for_epoch(10, profile="yaml-intended"), 0.03)
        self.assertAlmostEqual(learning_rate_for_epoch(30, profile="yaml-intended"), 0.03)
        self.assertAlmostEqual(learning_rate_for_epoch(31, profile="yaml-intended"), 0.003)
        self.assertAlmostEqual(learning_rate_for_epoch(51, profile="yaml-intended"), 0.0003)
        intended = parse_args(["--data-root", "data", "--output", "out",
                               "--profile", "yaml-intended"])
        self.assertEqual((intended.batch_size, intended.workers), (512, 4))

    def test_one_synthetic_epoch_output_contract(self):
        class TinyNet(nn.Module):
            def __init__(self, norm, activation):
                super().__init__()
                self.classifier = nn.Linear(3 * 4 * 4, 10)

            def forward(self, x):
                return self.classifier(x.flatten(1))

        loader = DataLoader(TensorDataset(torch.randn(4, 3, 4, 4),
                                          torch.tensor([0, 1, 2, 3])), batch_size=2)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cell"
            args = parse_args(["--data-root", temporary, "--output", str(output),
                               "--device", "cpu", "--workers", "0", "--epochs", "1",
                               "--max-train-batches", "1"])
            self.assertEqual(args.batch_size, 64)
            with patch("experiments.vgg_bias_init.train.make_loaders", return_value=(loader, loader)), \
                    patch("experiments.vgg_bias_init.train.VGGBiasInit", TinyNet):
                run(args)
            record = json.loads((output / "epochs.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["epoch"], 0)
            self.assertEqual(record["lr"], 0.03)
            self.assertTrue(0 <= record["test_accuracy"] <= 100)
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["source_commit"], "6647574352bbbb3e9b76449097647c0d21b91fe0")
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
