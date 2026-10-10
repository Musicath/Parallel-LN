"""CNN release entry points must protect records and avoid surprise downloads."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from experiments.cnn import run_cifar10_conv_gn_clean_labels as clean
from experiments.cnn import run_cifar10_conv_gn_random_labels as random_labels
from experiments.cnn.run_output import check_existing, metadata, save_exclusive


class CNNRunnerOutputTests(unittest.TestCase):
    def test_existing_result_requires_exact_parameters(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            expected = {"seed": 1, "batch_size": 128}
            save_exclusive(path, expected)
            with self.assertRaises(FileExistsError):
                check_existing(path, expected, skip_existing=False)
            self.assertTrue(check_existing(path, expected, skip_existing=True))
            with self.assertRaisesRegex(ValueError, "does not match"):
                check_existing(path, {"seed": 1, "batch_size": 256},
                               skip_existing=True)
            with self.assertRaises(FileExistsError):
                save_exclusive(path, expected)

    def test_skip_matching_run_does_not_require_gpu_or_dataset(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "dataset"
            cases = (
                (random_labels, random_labels.Config(
                    n_train=6000, channels=128, operator="pln4", lr=0.001,
                    model_seed=0, data_seed=20260902, label_seed=20260903,
                    epochs=200, batch_size=128, pool_size=4, depth=1,
                    hidden_operator_count=1),
                 ["--operators", "pln4", "--lrs", "0.001", "--depths", "1"],
                 "make_random_label_data"),
                (clean, clean.Config(
                    n_train=50000, channels=128, operator="bgn8k4", lr=0.0003,
                    model_seed=0, data_seed=20260902, epochs=200,
                    batch_size=128, pool_size=4, depth=2),
                 ["--operators", "bgn8k4"], "make_clean_label_data"),
            )
            for index, (module, cfg, options, loader_name) in enumerate(cases):
                with self.subTest(module=module.__name__):
                    output = root / f"out{index}"
                    output.mkdir()
                    saved = metadata(cfg, eval_every=10, device="cuda:0",
                                     data_root=data_root)
                    save_exclusive(output / f"{cfg.key}.json", saved)
                    argv = ["runner", "--data-root", str(data_root),
                            "--output", str(output), "--skip-existing", *options]
                    with mock.patch.object(sys, "argv", argv), \
                         mock.patch.object(torch.cuda, "is_available", return_value=False), \
                         mock.patch.object(module, loader_name) as load:
                        module.main()
                        load.assert_not_called()

    def test_dataset_download_is_opt_in(self):
        marker = RuntimeError("dataset constructor reached")
        with mock.patch.object(random_labels, "CIFAR10", side_effect=marker) as dataset:
            with self.assertRaisesRegex(RuntimeError, "constructor reached"):
                random_labels.make_random_label_data(Path("data"), 1, 1, 2,
                                                     torch.device("cpu"))
            self.assertFalse(dataset.call_args.kwargs["download"])
        with mock.patch.object(clean, "CIFAR10", side_effect=marker) as dataset:
            with self.assertRaisesRegex(RuntimeError, "constructor reached"):
                clean.make_clean_label_data(Path("data"), 1, 1,
                                            torch.device("cpu"))
            self.assertFalse(dataset.call_args.kwargs["download"])

    def test_duplicate_requested_cells_fail_before_dataset_access(self):
        with TemporaryDirectory() as tmp:
            cases = (
                (random_labels, ["--operators", "pln4,pln4"],
                 "make_random_label_data"),
                (clean, ["--operators", "bgn8k4,bgn8k4"],
                 "make_clean_label_data"),
            )
            for index, (module, options, loader_name) in enumerate(cases):
                with self.subTest(module=module.__name__):
                    argv = ["runner", "--data-root", str(Path(tmp) / "data"),
                            "--output", str(Path(tmp) / f"out{index}"), *options]
                    with mock.patch.object(sys, "argv", argv), \
                         mock.patch.object(module, loader_name) as load:
                        with self.assertRaisesRegex(ValueError, "duplicate requested run key"):
                            module.main()
                        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
