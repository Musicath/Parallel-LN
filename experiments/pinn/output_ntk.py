"""Finite-model output NTK diagnostics, not a PDE-residual NTK."""

import numpy as np
import torch


def output_jacobian(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    """Return rows d u_theta(x_i) / d theta for scalar-output models."""
    params = tuple(param for param in model.parameters() if param.requires_grad)
    if not params:
        raise ValueError("model has no trainable parameters")
    was_training = model.training
    model.eval()
    rows = []
    try:
        for point in points:
            output = model(point.reshape(1, -1)).reshape(-1)
            if output.numel() != 1:
                raise ValueError("the PINN output NTK expects one scalar per point")
            gradients = torch.autograd.grad(output[0], params, create_graph=False)
            rows.append(torch.cat([gradient.reshape(-1) for gradient in gradients]).detach())
    finally:
        model.train(was_training)
    return torch.stack(rows)


def spectrum_from_jacobian(jacobian: torch.Tensor) -> dict:
    kernel = jacobian @ jacobian.T
    kernel = (kernel + kernel.T) / 2
    values = torch.linalg.eigvalsh(kernel).detach().cpu().numpy()[::-1]
    values = np.clip(values, 0, None)
    trace = float(values.sum())
    maximum = float(values[0])
    minimum = float(values[-1])
    energy_rank = int(np.searchsorted(np.cumsum(values) / trace, 0.9) + 1) if trace > 0 else 0
    return {
        "eff_rank_90": energy_rank,
        "stable_rank": trace / max(maximum, 1e-12),
        "cond_clipped_1e-12": maximum / max(minimum, 1e-12),
        "lambda_max": maximum,
        "lambda_min_clipped_zero": minimum,
        "trace": trace,
        "eigenvalues": values,
    }


def output_ntk_spectrum(model: torch.nn.Module, points: torch.Tensor) -> dict:
    return spectrum_from_jacobian(output_jacobian(model, points))
