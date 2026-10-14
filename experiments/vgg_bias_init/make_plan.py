"""Enumerate, but never launch, the historical 108-cell VGG-Bias grid."""

import argparse
import itertools
import json
from pathlib import Path

from .model import ACTIVATIONS, NORMS


SCALARS = (0.5, 1.0, 2.0)
BETAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def build_plan(data_root, output_root, *, seed=42, device="cuda:0",
               profile="committed-parser"):
    data_root, output_root = Path(data_root), Path(output_root)
    for activation, norm, scalar, beta in itertools.product(
        ACTIVATIONS, NORMS, SCALARS, BETAS
    ):
        run_id = f"VGG_{norm}_{activation}_b{beta}_s{scalar}"
        yield {
            "run_id": run_id,
            "activation": activation,
            "norm": norm,
            "scalar": scalar,
            "beta": beta,
            "seed": seed,
            "profile": profile,
            "argv": [
                "python", "-m", "experiments.vgg_bias_init.train",
                "--data-root", str(data_root),
                "--output", str(output_root / run_id),
                "--activation", activation, "--norm", norm,
                "--scalar", str(scalar), "--beta", str(beta),
                "--seed", str(seed), "--profile", profile,
                "--device", device,
            ],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/vgg_bias_init"))
    parser.add_argument("--plan", type=Path, default=Path("outputs/vgg_bias_init/plan.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", choices=("committed-parser", "yaml-intended"),
                        default="committed-parser")
    args = parser.parse_args(argv)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    with args.plan.open("x", encoding="utf-8") as stream:
        count = 0
        for record in build_plan(args.data_root, args.output_root,
                                 seed=args.seed, device=args.device,
                                 profile=args.profile):
            stream.write(json.dumps(record) + "\n")
            count += 1
    print(f"Wrote {count} commands to {args.plan}; no training was started.")


if __name__ == "__main__":
    main()
