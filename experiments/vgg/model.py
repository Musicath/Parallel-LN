# The VGG-D configuration and initializer follow a torchvision-style pattern;
# see THIRD_PARTY_NOTICES.md and licenses/torchvision-BSD-3-Clause.txt.
"""Standalone extraction of the historical CIFAR-10 VGG-16 architecture.

Only the five activation families used in the paper's VGG grid are retained.
The original experiment used ``norm=No``, i.e. an identity normalization
module between each convolution and activation. The broad research framework
is intentionally not a dependency of this module.
"""

import torch
from torch import nn

from pln import ParallelLN


VGG16_FEATURES = (
    64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
    512, 512, 512, "M", 512, 512, 512,
)
ACTIVATIONS = ("pln", "relu", "silu", "tanh", "sat")


class SaturatingActivation(nn.Module):
    """The historical ``sat`` nonlinearity, x / sqrt(x² + 1)."""

    def forward(self, x):
        return x / torch.sqrt(x.square() + 1)


def make_activation(name, channels):
    if name == "pln":
        return ParallelLN(channels, num_per_group=8, dim=4)
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sat":
        return SaturatingActivation()
    raise ValueError(f"unknown VGG activation: {name!r}")


class PaperVGG16(nn.Module):
    """CIFAR-10 VGG-16 used by the private grid-launch scripts.

    Four 2×2 max pools reduce a 32×32 image to 2×2. The historical 2×2
    average pool then yields 512 features for a single linear classifier.
    Convolution and classifier weights use fan-out Kaiming normal followed
    by the experiment's initialization scale; biases are zero.
    """

    def __init__(self, activation="pln", init_scale=1.0, num_classes=10):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if init_scale <= 0:
            raise ValueError("init_scale must be positive")
        layers = []
        in_channels = 3
        for spec in VGG16_FEATURES:
            if spec == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue
            layers.extend((
                nn.Conv2d(in_channels, spec, kernel_size=3, padding=1, bias=False),
                nn.Identity(),
                make_activation(activation, spec),
            ))
            in_channels = spec
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AvgPool2d(2, 2)
        self.classifier = nn.Sequential(nn.Linear(512, num_classes))
        self.init_scale = float(init_scale)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                module.weight.data.mul_(self.init_scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x.flatten(1))
