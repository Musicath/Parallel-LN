"""Position-wise Parallel Layer Normalization (PLN) and Layer Scaling (PLS).

Each contiguous group of ``num_per_group`` feature coordinates is normalized
independently. For 4D inputs, normalization is performed at each spatial
position; it does not pool statistics across positions like GroupNorm.
``centering=False`` gives the RMS-style PLS variant. Practical ``eps`` and
affine parameters are not part of the paper's idealized mathematical operator.
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ParallelLN(nn.Module):
    """Apply fixed-size group normalization to feature coordinates.

    ``dim=2`` expects ``[batch, features]``; ``dim=3`` expects
    ``[batch, tokens, features]``; ``dim=4`` expects
    ``[batch, features, height, width]``. The historical class name and
    constructor signature are retained for existing users.
    """

    def __init__(
        self,
        num_features,
        num_per_group,
        eps=1e-5,
        centering=True,
        affine=True,
        dim=4,
        *args,
        **kwargs
    ):
        super().__init__()
        if not isinstance(num_features, int) or num_features <= 0:
            raise ValueError("num_features must be a positive integer")
        if not isinstance(num_per_group, int) or num_per_group <= 0:
            raise ValueError("num_per_group must be a positive integer")
        if num_features % num_per_group:
            raise ValueError("num_features must be divisible by num_per_group")
        if dim not in (2, 3, 4):
            raise ValueError("dim must be 2, 3, or 4")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        self.num_features = num_features
        self.num_per_group = num_per_group
        self.eps = eps
        self.dim = dim
        self.affine = affine
        self.centering = centering
        if self.affine:
            if self.dim == 4:
                self.shape = [1, self.num_features, 1, 1]
            elif self.dim == 3:
                self.shape = [1, 1, self.num_features]
            elif self.dim == 2:
                self.shape = [1, self.num_features]
            self.weight = Parameter(torch.Tensor(*self.shape))
            self.bias = Parameter(torch.Tensor(*self.shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        if x.ndim != self.dim:
            raise ValueError(f"expected {self.dim}D input, got {x.ndim}D")
        feature_axis = 1 if self.dim in (2, 4) else 2
        if x.shape[feature_axis] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} features on axis {feature_axis}, "
                f"got {x.shape[feature_axis]}"
            )
        if self.dim == 4:
            x = x.permute(0, 2, 3, 1)
        size = x.size()
        x = x.reshape(-1, self.num_features // self.num_per_group, self.num_per_group)
        if self.centering:
            mean = x.mean(dim=2, keepdim=True)
            xc = x - mean
        else:
            xc = x
        second_moment = torch.mean(xc * xc, dim=2, keepdim=True)
        norm = torch.rsqrt(second_moment + self.eps)
        xn = xc * norm
        xn = xn.reshape(size)
        if self.dim == 4:
            xn = xn.permute(0, 3, 1, 2)
        if self.affine:
            xn = xn * self.weight + self.bias
        return xn

    def extra_repr(self):
        return "{num_features}, {num_per_group}, eps={eps}, dim={dim}, affine={affine}, centering={centering}".format(
            **self.__dict__
        )


if __name__ == "__main__":
    x = torch.randn((1, 9, 3, 3))
    f = ParallelLN(num_features=9, num_per_group=9, dim=4)
    fx = f(x)
    print(fx.mean(dim=(1)))
    print((fx**2).mean(dim=(1)))
    print("-----------------------------------------------")
    x = torch.randn((1, 9, 3, 3))
    f = ParallelLN(num_features=9, num_per_group=3, dim=4)
    fx = f(x)
    fx = fx.view(1, 3, 3, 3, 3)
    print(fx.mean(dim=(2)))
    print((fx**2).mean(dim=(2)))
    print("-----------------------------------------------")
    x = torch.randn((3, 3, 8))
    f = ParallelLN(num_features=8, num_per_group=4, dim=3)
    fx = f(x)
    fx = fx.view(3, 3, 2, 4)
    print(fx.mean(dim=(3)))
    print((fx**2).mean(dim=(3)))
    print("-----------------------------------------------")
    x = torch.randn((4, 16))
    f = ParallelLN(num_features=16, num_per_group=8, dim=2)
    fx = f(x)
    fx = fx.view(4, 2, 8)
    print(fx.mean(dim=(2)))
    print((fx**2).mean(dim=(2)))
    print("-----------------------------------------------")
