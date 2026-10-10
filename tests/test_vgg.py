"""CPU checks for the framework-independent VGG-16 extraction."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.vgg.model import ACTIVATIONS, PaperVGG16, SaturatingActivation
from experiments.vgg.train_cifar10 import parse_args, run
from pln import ParallelLN


class VGGExtractionTests(unittest.TestCase):
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
            self.assertTrue(0 <= record["test_accuracy"] <= 100)
            self.assertEqual(json.loads((output / "config.json").read_text())["activation"], "sat")
            with self.assertRaises(FileExistsError), patch(
                "experiments.vgg.train_cifar10.make_loaders",
                return_value=(loader, loader),
            ):
                run(args)


if __name__ == "__main__":
    unittest.main()
