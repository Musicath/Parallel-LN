#!/usr/bin/env python3
"""Controlled shallow-ConvNet study of normalization-induced nonlinearity."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.datasets import CIFAR10


class PositionGroupNorm(nn.Module):
    """Normalize channel groups independently at every spatial position."""

    def __init__(self, channels: int, num_groups: int, eps: float = 1e-5):
        super().__init__()
        if num_groups < 1 or channels % num_groups:
            raise ValueError(
                f"channels={channels} must be divisible by num_groups={num_groups}"
            )
        self.channels = channels
        self.num_groups = num_groups
        self.group_size = channels // num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        grouped = x.reshape(
            batch, self.num_groups, self.group_size, height, width
        )
        centered = grouped - grouped.mean(dim=2, keepdim=True)
        variance = centered.square().mean(dim=2, keepdim=True)
        return (centered * torch.rsqrt(variance + self.eps)).reshape_as(x)


class BlockGroupNorm(nn.Module):
    """Normalize channel groups within non-overlapping spatial blocks.

    ``spatial_splits=k`` divides both H and W into k contiguous pieces, hence
    creates k^2 spatial blocks. Statistics are computed independently over
    (channels within a channel group) x (pixels within one spatial block).
    """

    def __init__(
        self,
        channels: int,
        num_channel_groups: int,
        spatial_splits: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        if num_channel_groups < 1 or channels % num_channel_groups:
            raise ValueError(
                f"channels={channels} must be divisible by "
                f"num_channel_groups={num_channel_groups}"
            )
        if spatial_splits < 1:
            raise ValueError("spatial_splits must be positive")
        self.channels = channels
        self.num_channel_groups = num_channel_groups
        self.group_size = channels // num_channel_groups
        self.spatial_splits = spatial_splits
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        k = self.spatial_splits
        if height % k or width % k:
            raise ValueError(
                f"spatial shape {(height, width)} must be divisible by "
                f"spatial_splits={k}"
            )
        block_h, block_w = height // k, width // k
        grouped = x.reshape(
            batch,
            self.num_channel_groups,
            self.group_size,
            k,
            block_h,
            k,
            block_w,
        )
        centered = grouped - grouped.mean(dim=(2, 4, 6), keepdim=True)
        variance = centered.square().mean(dim=(2, 4, 6), keepdim=True)
        return (centered * torch.rsqrt(variance + self.eps)).reshape_as(x)


def make_operator(name: str, channels: int) -> nn.Module:
    key = name.lower()
    if key == "identity":
        return nn.Identity()
    if key == "relu":
        return nn.ReLU()
    if key == "posln":
        return PositionGroupNorm(channels, num_groups=1)
    if key.startswith("pln"):
        group_size = int(key.removeprefix("pln"))
        if channels % group_size:
            raise ValueError(f"channels={channels} is not divisible by PLN-{group_size}")
        return PositionGroupNorm(channels, num_groups=channels // group_size)
    if key.startswith("bgn"):
        # bgn<G>k<K>: G channel groups and K splits along each spatial axis.
        group_text, split_text = key.removeprefix("bgn").split("k", maxsplit=1)
        return BlockGroupNorm(
            channels,
            num_channel_groups=int(group_text),
            spatial_splits=int(split_text),
        )
    if key.startswith("gn"):
        num_groups = int(key.removeprefix("gn"))
        if channels % num_groups:
            raise ValueError(
                f"channels={channels} is not divisible by GN groups={num_groups}"
            )
        return nn.GroupNorm(num_groups, channels, eps=1e-5, affine=False)
    raise ValueError(f"unknown operator: {name}")


class ConvNormNet(nn.Module):
    """[Conv -> operator]^depth -> linear average pooling -> affine readout."""

    def __init__(
        self, channels: int, operator: str, pool_size: int = 4, depth: int = 1
    ):
        super().__init__()
        if 32 % pool_size:
            raise ValueError("pool_size must divide 32")
        if depth < 1:
            raise ValueError("depth must be positive")
        spatial = 32 // pool_size
        blocks: list[nn.Module] = []
        in_channels = 3
        for _ in range(depth):
            blocks.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    make_operator(operator, channels),
                ]
            )
            in_channels = channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        self.readout = nn.Linear(channels * spatial * spatial, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.pool(x)
        return self.readout(x.flatten(1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_random_label_data(
    data_root: Path,
    n_train: int,
    data_seed: int,
    label_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    dataset = CIFAR10(str(data_root), train=True, download=True)
    if not 1 <= n_train <= len(dataset):
        raise ValueError(f"n_train must be in [1, {len(dataset)}]")

    data_generator = torch.Generator().manual_seed(data_seed)
    indices = torch.randperm(len(dataset), generator=data_generator)[:n_train]
    images = torch.from_numpy(dataset.data.copy()).permute(0, 3, 1, 2)[indices]
    original_labels = torch.tensor(dataset.targets, dtype=torch.long)[indices]

    label_generator = torch.Generator().manual_seed(label_seed)
    label_permutation = torch.randperm(n_train, generator=label_generator)
    random_labels = original_labels[label_permutation]
    match_fraction = (random_labels == original_labels).float().mean().item()

    images = images.to(device=device, dtype=torch.float32).div_(255.0)
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device).view(1, 3, 1, 1)
    images.sub_(mean).div_(std)
    return images, random_labels.to(device), match_fraction


@dataclass(frozen=True)
class Config:
    n_train: int
    channels: int
    operator: str
    lr: float
    model_seed: int
    data_seed: int
    label_seed: int
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
    hidden_operator_count: int = 1

    @property
    def key(self) -> str:
        lr_tag = f"{self.lr:g}".replace(".", "p").replace("-", "m")
        optimizer_tag = ""
        if self.optimizer != "adam":
            momentum_tag = f"{self.momentum:g}".replace(".", "p")
            optimizer_tag = f"_opt{self.optimizer}_mom{momentum_tag}"
        return (
            f"cifar10_rl_N{self.n_train}_D{self.depth}_c{self.channels}_{self.operator}_"
            f"lr{lr_tag}{optimizer_tag}_mseed{self.model_seed}_dseed{self.data_seed}_"
            f"lseed{self.label_seed}"
        )


def make_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        )
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")


def train_one(
    cfg: Config,
    images: torch.Tensor,
    labels: torch.Tensor,
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
    n_train = labels.numel()
    best_acc = 0.0
    curve: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    @torch.no_grad()
    def evaluate() -> tuple[float, float]:
        model.eval()
        loss_sum = 0.0
        correct = 0
        for start in range(0, n_train, 1024):
            stop = min(start + 1024, n_train)
            logits = model(images[start:stop])
            loss_sum += nn.functional.cross_entropy(
                logits, labels[start:stop], reduction="sum"
            ).item()
            correct += (logits.argmax(dim=1) == labels[start:stop]).sum().item()
        return loss_sum / n_train, correct / n_train

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        permutation = torch.randperm(
            n_train, generator=shuffle_generator, device=device
        )
        for start in range(0, n_train, cfg.batch_size):
            batch_indices = permutation[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images[batch_indices]), labels[batch_indices])
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % eval_every == 0 or epoch == cfg.epochs:
            train_loss, train_acc = evaluate()
            best_acc = max(best_acc, train_acc)
            curve.append(
                {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}
            )
            print(
                f"[{cfg.key}] epoch={epoch:04d}/{cfg.epochs} "
                f"loss={train_loss:.6f} acc={train_acc:.4%}",
                flush=True,
            )

    elapsed_sec = time.perf_counter() - start_time
    final_loss, final_acc = evaluate()
    return {
        **asdict(cfg),
        "key": cfg.key,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_train_acc": best_acc,
        "final_train_loss": final_loss,
        "final_train_acc": final_acc,
        "elapsed_sec": elapsed_sec,
        "curve": curve,
    }


def parse_csv(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def aggregate(output: Path) -> None:
    rows = []
    for path in sorted(output.glob("cifar10_rl_*.json")):
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=6000)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--depths", default="1")
    parser.add_argument("--operators", default="identity,relu,posln,pln4,pln8,gn1,gn4,gn8,gn16,gn32,gn64,gn128")
    parser.add_argument("--lrs", default="1e-4,3e-4,1e-3")
    parser.add_argument("--model-seeds", default="0")
    parser.add_argument("--data-seed", type=int, default=20260902)
    parser.add_argument("--label-seed", type=int, default=20260903)
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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    operators = parse_csv(args.operators, str)
    lrs = parse_csv(args.lrs, float)
    model_seeds = parse_csv(args.model_seeds, int)
    depths = parse_csv(args.depths, int)
    if args.smoke:
        args.n_train = min(args.n_train, 256)
        args.epochs = 1
        operators = ["pln4"]
        lrs = [1e-3]
        model_seeds = [0]
        depths = [2]

    images, labels, match_fraction = make_random_label_data(
        args.data_root,
        args.n_train,
        args.data_seed,
        args.label_seed,
        device,
    )
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "n_train": args.n_train,
                "channels": args.channels,
                "pool_size": args.pool_size,
                "depths": depths,
                "operators": operators,
                "lrs": lrs,
                "model_seeds": model_seeds,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "momentum": args.momentum,
                "nesterov": args.nesterov,
                "data_seed": args.data_seed,
                "label_seed": args.label_seed,
                "random_label_match_fraction": match_fraction,
            },
            indent=2,
        ),
        flush=True,
    )

    for model_seed in model_seeds:
        for depth in depths:
            for operator in operators:
                for lr in lrs:
                    cfg = Config(
                    n_train=args.n_train,
                    channels=args.channels,
                    operator=operator,
                    lr=lr,
                    model_seed=model_seed,
                    data_seed=args.data_seed,
                    label_seed=args.label_seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    pool_size=args.pool_size,
                    depth=depth,
                    optimizer=args.optimizer,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum,
                    nesterov=args.nesterov,
                    hidden_operator_count=depth,
                    )
                    result_path = args.output / f"{cfg.key}.json"
                    if args.skip_existing and result_path.exists():
                        print(f"skip existing: {result_path.name}", flush=True)
                        continue
                    result = train_one(cfg, images, labels, device, args.eval_every)
                    temporary = result_path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(result, indent=2))
                    os.replace(temporary, result_path)
                    aggregate(args.output)
                    print(
                        f"completed {cfg.key}: best={result['best_train_acc']:.4%} "
                        f"final={result['final_train_acc']:.4%} "
                        f"elapsed={result['elapsed_sec'] / 60:.1f} min",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
