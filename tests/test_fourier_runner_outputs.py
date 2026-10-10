"""A repeated Dense Fourier CLI call must not overwrite or mislabel a run."""

from dataclasses import asdict
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from experiments.dense_fourier import run_dense_fourier_mlp as runner


class FourierRunnerOutputTests(unittest.TestCase):
    def test_existing_result_is_never_overwritten_or_mis_skipped(self):
        def fake_train(cfg, _device, _eval_every):
            return {**asdict(cfg), "key": cfg.key, "test_relative_l2": 0.5,
                    "train_mse": 0.25, "test_mse": 0.25,
                    "best_step": 1, "elapsed_sec": 0.01}

        with TemporaryDirectory() as tmp, mock.patch.object(runner, "train", side_effect=fake_train) as train:
            argv = ["fourier", "--output", str(Path(tmp) / "results"),
                    "--support-size", "2", "--activation", "relu",
                    "--lr", "0.001", "--device", "cpu", "--steps", "1"]

            def invoke(*extra):
                with mock.patch.object(sys, "argv", argv + list(extra)):
                    runner.main()

            invoke()
            result_path = next((Path(tmp) / "results").glob("*.json"))
            original = result_path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                invoke()
            invoke("--skip-existing")
            with self.assertRaisesRegex(ValueError, "does not match every requested parameter"):
                invoke("--skip-existing", "--batch-size", "32")
            with self.assertRaisesRegex(ValueError, "does not match every requested parameter"):
                invoke("--skip-existing", "--eval-every", "1")
            self.assertEqual(result_path.read_bytes(), original)
            self.assertEqual(train.call_count, 1)


if __name__ == "__main__":
    unittest.main()
