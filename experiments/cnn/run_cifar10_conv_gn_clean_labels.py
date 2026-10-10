#!/usr/bin/env python3
"""Clean-label companion to the normalization-domain fitting experiment.

The architecture and normalization operators are imported from the random-label
runner so that the clean- and random-label heatmaps differ only in their labels
and the addition of held-out evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision.datasets import CIFAR10

if __package__:
    from .run_output import check_existing, metadata, save_exclusive
    from .run_cifar10_conv_gn_random_labels import (
        ConvNormNet,
        make_optimizer,
        parse_csv,
        set_seed,
    )
else:
    from run_output import check_existing, metadata, save_exclusive
    from run_cifar10_conv_gn_random_labels import (
        ConvNormNet,
        make_optimizer,
        parse_csv,
        set_seed,
    )


def _normalize(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device=device, dtype=torch.float32).div_(255.0)
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
    return images.sub_(mean).div_(std)


def make_clean_label_data(
    data_root: Path,
    n_train: int,
    data_seed: int,
    device: torch.device,
    download: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_set = CIFAR10(str(data_root), train=True, download=download)
    test_set = CIFAR10(str(data_root), train=False, download=download)
    if not 1 <= n_train <= len(train_set):
        raise ValueError(f"n_train must be in [1, {len(train_set)}]")

    generator = torch.Generator().manual_seed(data_seed)
    indices = torch.randperm(len(train_set), generator=generator)[:n_train]
    train_images = torch.from_numpy(train_set.data.copy()).permute(0, 3, 1, 2)[indices]
    train_labels = torch.tensor(train_set.targets, dtype=torch.long)[indices]
    test_images = torch.from_numpy(test_set.data.copy()).permute(0, 3, 1, 2)
    test_labels = torch.tensor(test_set.targets, dtype=torch.long)
    return (
        _normalize(train_images, device),
        train_labels.to(device),
        _normalize(test_images, device),
        test_labels.to(device),
    )


@dataclass(frozen=True)
class Config:
    n_train: int
    channels: int
    operator: str
    lr: float
    model_seed: int
    data_seed: int
    epochs: int
    batch_size: int
    pool_size: int
    depth: int
    optimizer: str = "adam"
    weight_decay: float = 0.0
    momentum: float = 0.9
    nesterov: bool = False
    affine: bool = False
    augmentation: bool = False

    @property
    def key(self) -> str:
        lr_tag = f"{self.lr:g}".replace(".", "p").replace("-", "m")
        optimizer_tag = ""
        if self.optimizer != "adam":
            momentum_tag = f"{self.momentum:g}".replace(".", "p")
            optimizer_tag = f"_opt{self.optimizer}_mom{momentum_tag}"
        return (
            f"cifar10_clean_N{self.n_train}_D{self.depth}_c{self.channels}_"
            f"{self.operator}_lr{lr_tag}{optimizer_tag}_mseed{self.model_seed}_dseed{self.data_seed}"
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 1024,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    for start in range(0, labels.numel(), chunk_size):
        stop = min(start + chunk_size, labels.numel())
        logits = model(images[start:stop])
        loss_sum += nn.functional.cross_entropy(
            logits, labels[start:stop], reduction="sum"
        ).item()
        correct += (logits.argmax(dim=1) == labels[start:stop]).sum().item()
    return loss_sum / labels.numel(), correct / labels.numel()


def train_one(
    cfg: Config,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    device: torch.device,
    eval_every: int,
) -> dict:
    set_seed(cfg.model_seed)
    model = ConvNormNet(
        cfg.channels, cfg.operator, cfg.pool_size, depth=cfg.depth
    ).to(device)
    optimizer = make_optimizer(cfg, model)
    criterion = nn.CrossEntropyLoss()
    shuffle_generator = torch.Generator(device=device).manual_seed(
        30_000 + cfg.model_seed
    )
    best_train_acc = 0.0
    best_test_acc = 0.0
    curve: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        permutation = torch.randperm(
            train_labels.numel(), generator=shuffle_generator, device=device
        )
        for start in range(0, train_labels.numel(), cfg.batch_size):
            batch_indices = permutation[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_images[batch_indices]), train_labels[batch_indices])
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % eval_every == 0 or epoch == cfg.epochs:
            train_loss, train_acc = evaluate(model, train_images, train_labels)
            test_loss, test_acc = evaluate(model, test_images, test_labels)
            best_train_acc = max(best_train_acc, train_acc)
            best_test_acc = max(best_test_acc, test_acc)
            curve.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
            print(
                f"[{cfg.key}] epoch={epoch:04d}/{cfg.epochs} "
                f"train_loss={train_loss:.6f} train_acc={train_acc:.4%} "
                f"test_loss={test_loss:.6f} test_acc={test_acc:.4%}",
                flush=True,
            )

    elapsed_sec = time.perf_counter() - start_time
    final_train_loss, final_train_acc = evaluate(model, train_images, train_labels)
    final_test_loss, final_test_acc = evaluate(model, test_images, test_labels)
    return {
        **asdict(cfg),
        "key": cfg.key,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_train_acc": best_train_acc,
        "best_test_acc": best_test_acc,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_test_loss": final_test_loss,
        "final_test_acc": final_test_acc,
        "elapsed_sec": elapsed_sec,
        "curve": curve,
    }


def aggregate(output: Path) -> None:
    rows = []
    for path in sorted(output.glob("cifar10_clean_*.json")):
        payload = json.loads(path.read_text())
        rows.append({key: value for key, value in payload.items() if key != "curve"})
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true",
                        help="download CIFAR-10 if absent; disabled by default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=50_000)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--operators", required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=20260902)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    operators = parse_csv(args.operators, str)
    if args.smoke:
        args.n_train = 256
        args.epochs = 1
        operators = ["bgn8k4"]

    pending = []
    planned_paths = set()
    for operator in operators:
        cfg = Config(
            n_train=args.n_train,
            channels=args.channels,
            operator=operator,
            lr=args.lr,
            model_seed=args.model_seed,
            data_seed=args.data_seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            pool_size=args.pool_size,
            depth=args.depth,
            optimizer=args.optimizer,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=args.nesterov,
        )
        result_path = args.output / f"{cfg.key}.json"
        if result_path in planned_paths:
            raise ValueError(f"duplicate requested run key: {cfg.key}")
        planned_paths.add(result_path)
        expected = metadata(cfg, eval_every=args.eval_every,
                            device=str(device), data_root=args.data_root)
        if check_existing(result_path, expected, skip_existing=args.skip_existing):
            print(f"skip matching existing: {result_path.name}", flush=True)
            continue
        pending.append((cfg, result_path, expected))
    if not pending:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    train_images, train_labels, test_images, test_labels = make_clean_label_data(
        args.data_root, args.n_train, args.data_seed, device,
        download=args.download,
    )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "n_train": args.n_train,
                "n_test": test_labels.numel(),
                "channels": args.channels,
                "depth": args.depth,
                "operators": operators,
                "lr": args.lr,
                "model_seed": args.model_seed,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "momentum": args.momentum,
                "nesterov": args.nesterov,
                "data_seed": args.data_seed,
                "augmentation": False,
                "affine": False,
            },
            indent=2,
        ),
        flush=True,
    )

    for cfg, result_path, expected in pending:
        result = train_one(
            cfg,
            train_images,
            train_labels,
            test_images,
            test_labels,
            device,
            args.eval_every,
        )
        result.update(expected)
        save_exclusive(result_path, result)
        aggregate(args.output)
        print(
            f"completed {cfg.key}: train={result['final_train_acc']:.4%} "
            f"test={result['final_test_acc']:.4%} "
            f"elapsed={result['elapsed_sec'] / 60:.1f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()
