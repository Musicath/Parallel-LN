"""Rebuild the historical fixed 6k/20k CIFAR-10 random-label artifacts.

The tracked CSV preserves the exact first 6,000 train indices and random
labels. Clean labels come from the user's own CIFAR-10 copy and are checked
against the archived digest; they are not redistributed in the CSV.
For 20k, the original preparation protocol retains the random labels and draws
14,000 additional labels with a separate CPU generator seeded 20260629.
Torchvision obtains CIFAR-10 itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10


MANIFEST = Path(__file__).resolve().parent / "data" / "fixed_6k_labels.csv"
ARCHIVED_CLEAN6_DIGEST = "95699b3bbd09ec4fc1ee037df638e54ccf3f36ebbbf9398ae143d2fa670cbd32"


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def read_manifest() -> tuple[torch.Tensor, torch.Tensor]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["cifar10_train_index", "random_label"]:
            raise ValueError("fixed-label manifest must contain only index and random label")
        rows = list(reader)
    if len(rows) != 6000:
        raise ValueError(f"expected 6000 manifest rows, got {len(rows)}")
    indices = torch.tensor(
        [int(row["cifar10_train_index"]) for row in rows], dtype=torch.long
    )
    random = torch.tensor([int(row["random_label"]) for row in rows], dtype=torch.long)
    if not torch.equal(indices, torch.arange(6000)):
        raise ValueError("historical subset must be the first 6000 CIFAR-10 images")
    if not torch.all((0 <= random) & (random < 10)):
        raise ValueError("random labels must be CIFAR-10 class indices")
    return indices, random


def build_artifacts(size: int, data_root: Path, download: bool) -> dict[str, torch.Tensor]:
    indices6, random6 = read_manifest()
    dataset = CIFAR10(str(data_root), train=True, download=download)
    clean = torch.tensor(dataset.targets[:size], dtype=torch.long)
    if digest(clean[:6000]) != ARCHIVED_CLEAN6_DIGEST:
        raise ValueError("CIFAR-10 training labels do not match the archived 6k digest")
    if size == 6000:
        indices = indices6
        labels = random6
    else:
        indices = torch.arange(20000)
        added = torch.randint(
            0, 10, (14000,), generator=torch.Generator().manual_seed(20260629)
        )
        labels = torch.cat((random6, added))
    return {
        "subset_indices.pt": indices,
        "clean_labels.pt": clean,
        "noise1/noisy_labels.pt": labels,
        "noise1/noise_mask.pt": torch.ones(size, dtype=torch.bool),
    }


def describe_artifacts(size: int, tensors: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    clean = tensors["clean_labels.pt"]
    random = tensors["noise1/noisy_labels.pt"]
    summary = {
        "num_samples": size,
        "noise_rate": 1.0,
        "mask_count": size,
        "changed_count": int((random != clean).sum()),
        "label_counts": torch.bincount(random, minlength=10).tolist(),
        "first16_clean": clean[:16].tolist(),
        "first16_noisy": random[:16].tolist(),
    }
    metadata = {
        "dataset": "CIFAR10",
        "split": "train",
        "subset_policy": "first_n",
        "num_samples": size,
        "source": "data/fixed_6k_labels.csv",
        "added_14k_label_seed": 20260629 if size == 20000 else None,
        "sha256": {name: digest(tensor) for name, tensor in tensors.items()},
    }
    return summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, choices=(6000, 20000), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tensors = build_artifacts(args.size, args.data_root, args.download)
    summary, metadata = describe_artifacts(args.size, tensors)
    if args.output.exists():
        for name, expected in tensors.items():
            path = args.output / name
            if not path.is_file() or not torch.equal(
                torch.load(path, map_location="cpu", weights_only=True), expected
            ):
                raise ValueError(f"existing artifact differs: {path}")
        for relative_path, expected in (
            ("noise1/summary.json", summary),
            ("metadata.json", metadata),
        ):
            path = args.output / relative_path
            if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != expected:
                raise ValueError(f"existing artifact differs: {path}")
        print(f"Existing artifacts verified: {args.output}")
        return

    (args.output / "noise1").mkdir(parents=True)
    for name, tensor in tensors.items():
        torch.save(tensor, args.output / name)
    (args.output / "noise1" / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Created {args.size} fixed-label artifacts at {args.output}")


if __name__ == "__main__":
    main()
