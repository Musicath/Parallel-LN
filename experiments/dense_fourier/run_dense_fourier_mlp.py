#!/usr/bin/env python3
"""Controlled MLP approximation of sparse-to-dense Fourier targets."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SAT(nn.Module):
    """SAT(x) = sin(arctan(x)) = x / sqrt(1 + x^2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(1.0 + x.square())


class ParallelLayerNorm(nn.Module):
    """Apply LayerNorm independently to consecutive fixed-size feature groups."""

    def __init__(
        self,
        width: int,
        group_size: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if group_size < 1 or width % group_size:
            raise ValueError(
                f"width={width} must be divisible by group_size={group_size}"
            )
        self.width = width
        self.group_size = group_size
        self.num_groups = width // group_size
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(width))
            self.bias = nn.Parameter(torch.zeros(width))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grouped = x.reshape(*x.shape[:-1], self.num_groups, self.group_size)
        mean = grouped.mean(dim=-1, keepdim=True)
        variance = (grouped - mean).square().mean(dim=-1, keepdim=True)
        output = (grouped - mean) * torch.rsqrt(variance + self.eps)
        output = output.reshape_as(x)
        if self.weight is not None:
            output = output * self.weight + self.bias
        return output


def make_activation(name: str, width: int, affine: bool) -> nn.Module:
    key = name.lower()
    if key == "identity":
        return nn.Identity()
    if key == "ln":
        return nn.LayerNorm(width, eps=1e-5, elementwise_affine=affine)
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    if key == "sat":
        return SAT()
    if key.startswith("pln"):
        return ParallelLayerNorm(
            width=width,
            group_size=int(key.removeprefix("pln")),
            affine=affine,
        )
    raise ValueError(f"unknown activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        activation: str,
        affine: bool,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(make_activation(activation, width, affine))
            in_features = width
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(x)).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_inputs(n: int, input_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 2.0 * torch.rand(n, input_dim, generator=generator) - 1.0


@dataclass(frozen=True)
class FourierTarget:
    omega: torch.Tensor
    amplitude: torch.Tensor
    phase: torch.Tensor
    support_size: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        angles = (
            2.0
            * math.pi
            * (x @ self.omega.T)
            / math.sqrt(self.support_size)
            + self.phase
        )
        return (torch.sin(angles) * self.amplitude).sum(dim=-1) / math.sqrt(
            self.omega.shape[0]
        )


def make_fourier_target(
    input_dim: int,
    modes: int,
    support_size: int,
    seed: int,
) -> FourierTarget:
    if not 1 <= support_size <= input_dim:
        raise ValueError("support_size must be between 1 and input_dim")
    generator = torch.Generator().manual_seed(seed + 1009 * support_size)
    omega = torch.zeros(modes, input_dim)
    for mode in range(modes):
        support = torch.randperm(input_dim, generator=generator)[:support_size]
        signs = 2 * torch.randint(
            0, 2, (support_size,), generator=generator, dtype=torch.int64
        ) - 1
        omega[mode, support] = signs.to(torch.float32)
    amplitude = (
        2 * torch.randint(0, 2, (modes,), generator=generator, dtype=torch.int64) - 1
    ).to(torch.float32)
    phase = 2.0 * math.pi * torch.rand(modes, generator=generator)
    return FourierTarget(omega, amplitude, phase, support_size)


@dataclass(frozen=True)
class Config:
    input_dim: int
    modes: int
    support_size: int
    width: int
    depth: int
    activation: str
    affine: bool
    lr: float
    lr_schedule: str
    warmup_steps: int
    steps: int
    batch_size: int
    n_train: int
    n_validation: int
    n_test: int
    target_seed: int
    data_seed: int
    model_seed: int
    weight_decay: float = 0.0

    @property
    def key(self) -> str:
        lr_tag = f"{self.lr:g}".replace(".", "p").replace("-", "m")
        affine_tag = "affine" if self.affine else "noaffine"
        schedule_tag = ""
        if self.lr_schedule != "constant":
            schedule_tag = f"_{self.lr_schedule}_wu{self.warmup_steps}"
        return (
            f"fourier_d{self.input_dim}_m{self.modes}_s{self.support_size}_"
            f"w{self.width}_L{self.depth}_{self.activation}_{affine_tag}_lr{lr_tag}"
            f"{schedule_tag}_"
            f"tseed{self.target_seed}_dseed{self.data_seed}_mseed{self.model_seed}"
        )


@torch.no_grad()
def relative_l2(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    predictions = []
    for start in range(0, y.numel(), 4096):
        predictions.append(model(x[start : start + 4096]))
    prediction = torch.cat(predictions)
    return (torch.linalg.vector_norm(prediction - y) / torch.linalg.vector_norm(y)).item()


def learning_rate_at_step(cfg: Config, step: int) -> float:
    """Return the learning rate used for a one-indexed optimization step."""
    if cfg.lr_schedule == "constant":
        return cfg.lr
    if cfg.lr_schedule != "warmup-cosine":
        raise ValueError(f"unknown learning-rate schedule: {cfg.lr_schedule}")
    if not 0 <= cfg.warmup_steps < cfg.steps:
        raise ValueError("warmup_steps must lie in [0, steps)")
    if cfg.warmup_steps and step <= cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    cosine_steps = cfg.steps - cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / cosine_steps
    progress = min(1.0, max(0.0, progress))
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: Config, device: torch.device, eval_every: int) -> dict:
    set_seed(cfg.model_seed)
    target = make_fourier_target(
        cfg.input_dim, cfg.modes, cfg.support_size, cfg.target_seed
    )
    x_train = sample_inputs(cfg.n_train, cfg.input_dim, cfg.data_seed)
    x_validation = sample_inputs(cfg.n_validation, cfg.input_dim, cfg.data_seed + 1)
    x_test = sample_inputs(cfg.n_test, cfg.input_dim, cfg.data_seed + 2)
    y_train = target(x_train)
    y_validation = target(x_validation)
    y_test = target(x_test)

    target_mean = y_train.mean()
    target_std = y_train.std(unbiased=False).clamp_min(1e-8)
    y_train = (y_train - target_mean) / target_std
    y_validation = (y_validation - target_mean) / target_std
    y_test = (y_test - target_mean) / target_std

    x_train = x_train.to(device)
    x_validation = x_validation.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_validation = y_validation.to(device)
    y_test = y_test.to(device)

    model = MLP(
        cfg.input_dim, cfg.width, cfg.depth, cfg.activation, cfg.affine
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    batch_generator = torch.Generator(device=device).manual_seed(
        30_000 + cfg.model_seed
    )

    best_validation_relative_l2 = math.inf
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    curve: list[dict[str, float | int]] = []
    start_time = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        model.train()
        current_lr = learning_rate_at_step(cfg, step)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr
        indices = torch.randint(
            cfg.n_train,
            (cfg.batch_size,),
            generator=batch_generator,
            device=device,
        )
        prediction = model(x_train[indices])
        loss = nn.functional.mse_loss(prediction, y_train[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == cfg.steps:
            train_error = relative_l2(model, x_train, y_train)
            validation_error = relative_l2(model, x_validation, y_validation)
            curve.append(
                {
                    "step": step,
                    "lr": current_lr,
                    "train_relative_l2": train_error,
                    "validation_relative_l2": validation_error,
                }
            )
            if validation_error < best_validation_relative_l2:
                best_validation_relative_l2 = validation_error
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            print(
                f"[{cfg.key}] step={step:05d}/{cfg.steps} "
                f"lr={current_lr:.6e} "
                f"train_rel_l2={train_error:.6e} "
                f"val_rel_l2={validation_error:.6e}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("no validation checkpoint was recorded")
    model.load_state_dict(best_state)
    elapsed_sec = time.perf_counter() - start_time
    train_relative_l2 = relative_l2(model, x_train, y_train)
    validation_relative_l2 = relative_l2(model, x_validation, y_validation)
    test_relative_l2 = relative_l2(model, x_test, y_test)
    train_target_second_moment = y_train.square().mean().item()
    validation_target_second_moment = y_validation.square().mean().item()
    test_target_second_moment = y_test.square().mean().item()
    return {
        **asdict(cfg),
        "key": cfg.key,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "target_mean": target_mean.item(),
        "target_std": target_std.item(),
        "best_step": best_step,
        "train_relative_l2": train_relative_l2,
        "validation_relative_l2": validation_relative_l2,
        "test_relative_l2": test_relative_l2,
        "train_mse": train_relative_l2**2 * train_target_second_moment,
        "validation_mse": validation_relative_l2**2
        * validation_target_second_moment,
        "test_mse": test_relative_l2**2 * test_target_second_moment,
        "elapsed_sec": elapsed_sec,
        "curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--support-size", type=int, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--activation", required=True)
    parser.add_argument("--affine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup-cosine"),
        default="constant",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-validation", type=int, default=2_000)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--target-seed", type=int, default=20260903)
    parser.add_argument("--data-seed", type=int, default=20260904)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. 'auto' preserves the original CUDA-first behavior.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    cfg = Config(
        input_dim=args.input_dim,
        modes=args.modes,
        support_size=args.support_size,
        width=args.width,
        depth=args.depth,
        activation=args.activation,
        affine=args.affine,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_validation=args.n_validation,
        n_test=args.n_test,
        target_seed=args.target_seed,
        data_seed=args.data_seed,
        model_seed=args.model_seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / f"{cfg.key}.json"
    if result_path.exists():
        if not args.skip_existing:
            raise FileExistsError(f"refusing to overwrite an existing run: {result_path}")
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        expected = {**asdict(cfg), "eval_every": args.eval_every,
                    "device": str(device)}
        if any(saved.get(name) != value for name, value in expected.items()):
            raise ValueError(
                f"existing run does not match every requested parameter: {result_path}"
            )
        print(f"skip matching existing run {result_path}", flush=True)
        return
    result = train(cfg, device, args.eval_every)
    result["device"] = str(device)
    result["eval_every"] = args.eval_every
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        f"completed {cfg.key}: test_rel_l2={result['test_relative_l2']:.6e} "
        f"train_mse={result['train_mse']:.6e} test_mse={result['test_mse']:.6e} "
        f"best_step={result['best_step']} elapsed={result['elapsed_sec'] / 60:.2f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
