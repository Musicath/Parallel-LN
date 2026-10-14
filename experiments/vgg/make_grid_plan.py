"""Write a prospective VGG-16 experiment plan without starting training.

The historical figure's third seed identifier and per-run records are not
available. Seeds must therefore be supplied explicitly; this script must not
be used to claim that a new plan recovers the original runs.
"""

import argparse
import json
from pathlib import Path


ACTIVATIONS = ("pln", "relu", "silu", "tanh", "sat")
LEARNING_RATES = ("0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1")
INIT_SCALES = ("0.25", "0.35", "0.5", "0.7", "1.0", "1.4", "2.0", "2.8", "4.0")


def build_plan(seeds, data_root, output_root, device="cuda:0", download=False):
    """Yield one JSON-serializable command record per grid cell and seed."""
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide at least one seed, without duplicates")
    for activation in ACTIVATIONS:
        for lr in LEARNING_RATES:
            for init_scale in INIT_SCALES:
                for seed in seeds:
                    run_id = f"{activation}_lr{lr}_init{init_scale}_seed{seed}"
                    output = output_root / run_id
                    argv = [
                        "python", "-m", "experiments.vgg.train_cifar10",
                        "--data-root", str(data_root),
                        "--output", str(output),
                        "--activation", activation,
                        "--lr", lr,
                        "--init-scale", init_scale,
                        "--seed", str(seed),
                        "--epochs", "200",
                        "--warmup-epochs", "10",
                        "--batch-size", "256",
                        "--weight-decay", "0.0001",
                        "--device", device,
                    ]
                    if download:
                        argv.append("--download")
                    yield {
                        "run_id": run_id,
                        "activation": activation,
                        "initial_learning_rate": lr,
                        "initialization_multiplier": init_scale,
                        "seed": seed,
                        "output": str(output),
                        "argv": argv,
                    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="explicit seeds for a new run; the historical third seed is unknown")
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/vgg/runs"))
    parser.add_argument("--plan", type=Path, default=Path("outputs/vgg/grid_plan.jsonl"),
                        help="new JSONL file; an existing file is never overwritten")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    with args.plan.open("x", encoding="utf-8") as stream:
        count = 0
        for record in build_plan(args.seeds, args.data_root, args.output_root,
                                 args.device, args.download):
            stream.write(json.dumps(record) + "\n")
            count += 1
    print(f"Wrote {count} planned runs to {args.plan}; no training was started.")


if __name__ == "__main__":
    main()
