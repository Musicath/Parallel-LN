"""Run one cell of the separate 108-cell Musicath-base VGG-Bias study.

This is a minimal extraction of the intended protocol, not a recovered binary
or a reproduction of the manuscript's distinct PLN[8] VGG heatmap.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import ACTIVATIONS, NORMS, VGGBiasInit, apply_parametric_init


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
MILESTONES = (20, 40, 60, 80)
PROFILES = ("committed-parser", "yaml-intended")


def learning_rate_for_epoch(epoch, *, profile="committed-parser",
                            base=0.03, total_epochs=100, warmup=10, gamma=0.1):
    """Return the LR used during one training epoch for the selected profile.

    ``committed-parser`` follows the old parser's effective cosine default.
    ``yaml-intended`` follows the YAML's steps/warmup fields, which that parser
    did not read. In both profiles the scheduler was stepped after training.
    """
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")
    if profile == "committed-parser":
        if total_epochs < 1:
            raise ValueError("total_epochs must be positive")
        return base * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
    if profile != "yaml-intended":
        raise ValueError(f"unknown profile: {profile}")
    if epoch == 0:
        return base
    if epoch <= warmup:
        return base * (0.1 + (epoch / warmup) * 0.9)
    scheduler_epoch = epoch - warmup - 1
    return base * gamma ** sum(m <= scheduler_epoch for m in MILESTONES)


def make_loaders(data_root, *, batch_size=512, workers=4, download=False):
    import torchvision
    from torchvision import transforms

    root = Path(data_root).expanduser()
    candidates = (root,) if root.name == "cifar10" else (root, root / "cifar10")
    match = next((path for path in candidates if (path / "cifar-10-batches-py").is_dir()), None)
    if match is None:
        if not download:
            raise FileNotFoundError(f"CIFAR-10 not found under {candidates}; use --download")
        match = root if root.name == "cifar10" else root / "cifar10"
    match.mkdir(parents=True, exist_ok=True)

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomCrop((32, 32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=str(match), train=True, transform=train_transform, download=download
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(match), train=False, transform=test_transform, download=False
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        torch.utils.data.DataLoader(train_set, shuffle=True, **loader_kwargs),
        torch.utils.data.DataLoader(test_set, shuffle=False, **loader_kwargs),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm", choices=NORMS, default="No")
    parser.add_argument("--activation", choices=ACTIVATIONS, default="relu")
    parser.add_argument("--scalar", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=PROFILES, default="committed-parser")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="smoke testing only; nonzero makes results noncomparable")
    args = parser.parse_args(argv)
    if args.batch_size is None:
        args.batch_size = 64 if args.profile == "committed-parser" else 512
    if args.workers is None:
        args.workers = 2 if args.profile == "committed-parser" else 4
    if args.scalar <= 0 or not 0 <= args.beta <= 1:
        parser.error("scalar must be positive and beta must lie in [0, 1]")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.max_train_batches < 0:
        parser.error("epochs/batch size must be positive; workers/max batches nonnegative")
    return args


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            count = labels.numel()
            total_loss += criterion(logits, labels).item() * count
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_count += count
    if total_count == 0:
        raise ValueError("test loader is empty")
    return total_loss / total_count, 100 * total_correct / total_count


def run(args):
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_loader, test_loader = make_loaders(
        args.data_root, batch_size=args.batch_size,
        workers=args.workers, download=args.download,
    )
    model = VGGBiasInit(norm=args.norm, activation=args.activation).to(device)
    apply_parametric_init(model, scalar=args.scalar, beta=args.beta)
    # The original 2026-05 snapshot decayed every trainable parameter,
    # including normalization affine parameters and biases.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03,
                                momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    args.output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["data_root"], config["output"] = str(args.data_root), str(args.output)
    config.update({"optimizer": "SGD", "base_lr": 0.03, "momentum": 0.9,
                   "weight_decay": 1e-4,
                   "lr_schedule": "cosine" if args.profile == "committed-parser"
                                  else "warmup_multistep",
                   "lr_milestones": () if args.profile == "committed-parser" else MILESTONES,
                   "warmup_epochs": 0 if args.profile == "committed-parser" else 10,
                   "warmup_ratio": None if args.profile == "committed-parser" else 0.1,
                   "metric_split": "CIFAR-10 standard test split",
                   "source_commit": "6647574352bbbb3e9b76449097647c0d21b91fe0"})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    best_accuracy = float("-inf")
    with (args.output / "epochs.jsonl").open("x", encoding="utf-8") as stream:
        for epoch in range(args.epochs):
            lr = learning_rate_for_epoch(epoch, profile=args.profile,
                                         total_epochs=args.epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            total_loss = total_correct = total_count = 0
            for batch_index, (images, labels) in enumerate(train_loader):
                if args.max_train_batches and batch_index >= args.max_train_batches:
                    break
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                count = labels.numel()
                total_loss += loss.item() * count
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_count += count
            if total_count == 0:
                raise ValueError("training loader is empty")
            test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
            best_accuracy = max(best_accuracy, test_accuracy)
            record = {
                "epoch": epoch, "lr": lr,
                "train_loss": total_loss / total_count,
                "train_accuracy": 100 * total_correct / total_count,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "best_test_accuracy": best_accuracy,
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(json.dumps(record), flush=True)
    return best_accuracy


if __name__ == "__main__":
    run(parse_args())
