"""Existing fixed-label artifacts must include consistent metadata as well as tensors."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import torch

from experiments.vit import prepare_fixed_labels


class FixedLabelIntegrityTests(unittest.TestCase):
    def test_existing_directory_checks_summary_and_hash_metadata(self):
        count = 6000
        clean = torch.arange(count, dtype=torch.long) % 10
        noisy = (clean + 1) % 10
        tensors = {
            "subset_indices.pt": torch.arange(count, dtype=torch.long),
            "clean_labels.pt": clean,
            "noise1/noisy_labels.pt": noisy,
            "noise1/noise_mask.pt": torch.ones(count, dtype=torch.bool),
        }
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "fixed6k"
            argv = ["prepare_fixed_labels", "--size", "6000",
                    "--output", str(output)]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(prepare_fixed_labels, "build_artifacts",
                                   return_value=tensors):
                prepare_fixed_labels.main()
                prepare_fixed_labels.main()
                metadata_path = output / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["sha256"]["clean_labels.pt"] = "incorrect"
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "existing artifact differs: .*metadata.json"):
                    prepare_fixed_labels.main()
                summary_path = output / "noise1" / "summary.json"
                summary_path.unlink()
                with self.assertRaisesRegex(ValueError, "existing artifact differs: .*summary.json"):
                    prepare_fixed_labels.main()


if __name__ == "__main__":
    unittest.main()
