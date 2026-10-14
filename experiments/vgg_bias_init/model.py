"""VGG-Bias model and parametric initializer for the 108-cell study.

The architecture and initialization formula are reduced from the author-side
Musicath-base snapshot at commit 6647574352bbbb3e9b76449097647c0d21b91fe0.
This is a separate experiment from the manuscript's PLN[8] VGG heatmap.
"""

import math

from torch import nn


FEATURES = (
    64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
    512, 512, 512, "M", 512, 512, 512,
)
NORMS = ("No", "LN", "BN")
ACTIVATIONS = ("relu", "tanh")


def make_norm(name, channels, *, spatial):
    if name == "No":
        return nn.Identity()
    if name == "LN":
        # Musicath-base's LN factory maps an integer feature count to GN(1).
        return nn.GroupNorm(1, channels)
    if name == "BN":
        return nn.BatchNorm2d(channels) if spatial else nn.BatchNorm1d(channels)
    raise ValueError(f"unknown normalization: {name}")


def make_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation: {name}")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, activation):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True)
        self.norm = make_norm(norm, out_channels, spatial=True)
        self.act = make_activation(activation)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FCBlock(nn.Module):
    def __init__(self, norm, activation):
        super().__init__()
        self.fc = nn.Linear(512, 512, bias=True)
        self.norm = make_norm(norm, 512, spatial=False)
        self.act = make_activation(activation)

    def forward(self, x):
        return self.act(self.norm(self.fc(x)))


class VGGBiasInit(nn.Module):
    """Thirteen convolution blocks, one 512-wide FC block, and a classifier."""

    def __init__(self, norm="No", activation="relu"):
        super().__init__()
        if norm not in NORMS or activation not in ACTIVATIONS:
            raise ValueError(f"expected norm in {NORMS} and activation in {ACTIVATIONS}")
        features = []
        in_channels = 3
        for width in FEATURES:
            if width == "M":
                features.append(nn.MaxPool2d(2, 2))
            else:
                features.append(ConvBlock(in_channels, width, norm, activation))
                in_channels = width
        features.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            FCBlock(norm, activation), nn.Linear(512, 10, bias=True)
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def apply_parametric_init(model, *, scalar, beta):
    """Apply the archived normal initializer to every Conv2d/Linear layer.

    Weight variance is (1-beta)*scalar/fan_in; bias variance is beta*scalar.
    All affine parameters of normalization modules keep PyTorch defaults.
    """
    if scalar <= 0 or not 0 <= beta <= 1:
        raise ValueError("scalar must be positive and beta must lie in [0, 1]")
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            fan_in = layer.weight[0].numel()
            nn.init.normal_(layer.weight, std=math.sqrt((1 - beta) * scalar / fan_in))
            if layer.bias is not None:
                nn.init.normal_(layer.bias, std=math.sqrt(beta * scalar))
