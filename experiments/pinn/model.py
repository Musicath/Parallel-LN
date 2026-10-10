"""Paper-specific MLP extracted from the historical PINN configuration.

There are ``depth`` hidden affine layers. Each is followed by one selected
nonlinear operator; no learnable normalization affine parameters are used.
"""

import torch
from torch import nn

from pln import ParallelLN


class SinArctan(nn.Module):
    """The historical SAT activation, x / sqrt(1 + x^2)."""

    def forward(self, x):
        return x / torch.sqrt(1 + x.square())


def nonlinear_layer(method: str, width: int) -> nn.Module:
    key = method.lower()
    if key.startswith("pln") and key[3:].isdigit():
        group_size = int(key[3:])
        return ParallelLN(width, group_size, dim=2, affine=False)
    if key == "ln":
        return nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "sat":
        return SinArctan()
    raise ValueError(f"unknown method: {method}")


class HelmholtzMLP(nn.Module):
    """Historical depth-6 MLP layout with a scalar output."""

    def __init__(self, method: str, width: int = 128, depth: int = 6):
        super().__init__()
        # DeepXDE's PyTorch backend inspects this attribute during compile.
        self.regularizer = None
        if depth < 1 or width < 1:
            raise ValueError("depth and width must be positive")
        layers = [nn.Identity()]
        incoming = 1
        for _ in range(depth):
            layers.append(nn.Linear(incoming, width))
            # Keep the historical no-op slots for parameter names and layout.
            if method.lower().startswith("pln") or method.lower() == "ln":
                layers.append(nonlinear_layer(method, width))
                layers.append(nn.Identity())
            else:
                layers.append(nn.Identity())
                layers.append(nonlinear_layer(method, width))
            incoming = width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
