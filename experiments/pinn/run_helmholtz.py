"""Standalone Helmholtz PINN runner for the paper table and NTK follow-up.

The two protocols deliberately remain separate. This file does not import the
private ``Norm-exp-base`` framework or contact W&B.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pinn.model import HelmholtzMLP  # noqa: E402
from experiments.pinn.output_ntk import output_ntk_spectrum  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("table2", "ntk"), default="table2")
    parser.add_argument("--method", choices=("PLN4", "PLN8", "PLN16", "PLN32", "ReLU", "Tanh", "SAT", "LN"), default="PLN4")
    parser.add_argument("--width", type=int, choices=(128, 256), default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss-weights", type=float, nargs=2, default=(1.0, 10.0), metavar=("PDE", "BC"))
    parser.add_argument("--steps", type=int, default=None, help="Adam/SGD steps before optional L-BFGS")
    parser.add_argument("--lbfgs-steps", type=int, default=0, help="NTK protocol only; original staged runs used 1200")
    parser.add_argument("--display-every", type=int, default=None,
                        help="evaluation/log interval; defaults to 100 for Table II or 10 for NTK")
    parser.add_argument("--ntk-points", type=int, default=None, help="interior output-NTK points; 0 disables")
    parser.add_argument("--save-eigvals", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pinn")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.protocol == "table2" and args.method == "LN":
        parser.error("LN is a diagnostic-only baseline, not a Table II method")
    if args.protocol == "ntk" and args.optimizer != "adam":
        parser.error("the archived NTK matrix used Adam before optional L-BFGS")
    if args.protocol == "table2" and args.lbfgs_steps:
        parser.error("Table II did not include an L-BFGS stage")
    if args.steps is None:
        args.steps = 2800 if args.lbfgs_steps else (6000 if args.protocol == "table2" else 4000)
    if args.lr is None:
        args.lr = 1e-4 if args.optimizer == "sgd" else 3e-4
    if args.ntk_points is None:
        args.ntk_points = 500 if args.protocol == "ntk" else 0
    if args.display_every is None:
        args.display_every = 100 if args.protocol == "table2" else 10
    if min(args.steps, args.display_every) < 1 or args.lbfgs_steps < 0 or args.ntk_points < 0:
        parser.error("steps and display interval must be positive; NTK/L-BFGS counts nonnegative")
    if args.lr <= 0 or any(weight <= 0 for weight in args.loss_weights):
        parser.error("learning rate and loss weights must be positive")
    return args


def run_name(args):
    weight_tag = "-".join(f"{weight:g}" for weight in args.loss_weights)
    return (f"{args.protocol}_{args.method}_w{args.width}_lw{weight_tag}_"
            f"{args.optimizer}_lr{args.lr:g}_seed{args.seed}_"
            f"steps{args.steps}_lbfgs{args.lbfgs_steps}")


def make_data(dde, protocol, loss_weights):
    geometry = dde.geometry.Interval(-1, 1)

    def exact(x):
        if torch.is_tensor(x):
            return torch.sin(torch.pi * x[:, 0:1])
        return np.sin(np.pi * x[:, 0:1])

    def residual(x, y):
        second = dde.grad.hessian(y, x, component=0, i=0, j=0)
        return -second + torch.pi**2 * y - 2 * torch.pi**2 * torch.sin(torch.pi * x[:, 0:1])

    boundary = dde.DirichletBC(geometry, exact, lambda _x, on_boundary: on_boundary)
    boundary_count = 500 if protocol == "table2" else 100
    data = dde.data.PDE(geometry, residual, boundary, num_domain=5000,
                        num_boundary=boundary_count, solution=exact)
    return data, geometry, boundary_count


def ntk_record(net, geometry, count, output_dir, phase, save_eigvals):
    points = geometry.uniform_points(count, boundary=False)
    param = next(net.parameters())
    tensor = torch.as_tensor(points, dtype=param.dtype, device=param.device)
    record = output_ntk_spectrum(net, tensor)
    eigenvalues = record.pop("eigenvalues")
    if save_eigvals:
        np.save(output_dir / f"{phase}_domain_eigenvalues.npy", eigenvalues)
    return record


def final_relative_l2(history):
    metrics = history.metrics_test[-1]
    if not metrics or not np.isfinite(float(metrics[0])):
        raise RuntimeError("DeepXDE did not report a finite final relative L2 metric")
    return float(metrics[0])


def main(argv=None):
    args = parse_args(argv)
    config = {
        "protocol": args.protocol, "method": args.method, "width": args.width,
        "depth": 6, "seed": args.seed, "optimizer": args.optimizer,
        "initial_lr": args.lr, "loss_weights": list(args.loss_weights),
        "steps": args.steps, "lbfgs_steps": args.lbfgs_steps,
        "display_every": args.display_every, "ntk_domain_points": args.ntk_points,
        "dtype": "float64", "batch_size": 128,
        "boundary_training_points": 500 if args.protocol == "table2" else 100,
        "scheduler": "none (the historical trainer constructed but did not step it)",
    }
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return

    os.environ["DDE_BACKEND"] = "pytorch"
    import deepxde as dde

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dde.config.set_default_float("float64")
    torch.set_default_dtype(torch.float64)

    output_dir = args.output_dir / run_name(args)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {output_dir}")
    output_dir.mkdir(parents=True)
    net = HelmholtzMLP(args.method, args.width, depth=6)
    if torch.cuda.is_available():
        net = net.cuda()
    data, geometry, _ = make_data(dde, args.protocol, args.loss_weights)
    model = dde.Model(data, net)

    metrics = ["l2 relative error", "MSE"]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    model.compile(optimizer=optimizer, metrics=metrics, loss_weights=args.loss_weights)

    result = {"config": config, "source_status": "standalone extraction; not the archived training binary"}
    if args.ntk_points:
        result["ntk_initial"] = ntk_record(net, geometry, args.ntk_points, output_dir, "initial", args.save_eigvals)

    history, _ = model.train(iterations=args.steps, batch_size=128, display_every=args.display_every)
    result["relative_l2_adam_or_sgd"] = final_relative_l2(history)
    if args.lbfgs_steps:
        dde.optimizers.set_LBFGS_options(maxiter=20)
        model.compile("L-BFGS", metrics=metrics, loss_weights=args.loss_weights)
        history, _ = model.train(iterations=args.lbfgs_steps, batch_size=128,
                                 display_every=args.display_every)
        result["relative_l2_after_lbfgs"] = final_relative_l2(history)
    if args.ntk_points:
        result["ntk_final"] = ntk_record(net, geometry, args.ntk_points, output_dir, "final", args.save_eigvals)
    if args.save_checkpoint:
        torch.save(net.state_dict(), output_dir / "model_state.pt")
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
