"""Minimal ViT backbone adapted from the paper's Musicath-base experiment.

Only the depth-4 CIFAR-10 path used by the released grouping study is kept:
standard multi-head attention, pre-normalization, a class token, and an FFN.
Parameter/module names intentionally match the historical implementation so
that checkpoints can be compared directly. This is not a general ViT library.

The historical backbone follows the classic timm-style ViT implementation,
including its truncated-normal initialization pattern. This extracted file
has been modified for the paper's standalone experiment. See
THIRD_PARTY_NOTICES.md and LICENSE.txt for attribution and license details.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


def gelu_factory(_: int) -> nn.Module:
    return nn.GELU()


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """Historical truncated-normal initializer, bounded at +/- 2."""
    low = (1.0 + math.erf((-2.0 - mean) / (std * math.sqrt(2.0)))) / 2.0
    high = (1.0 + math.erf((2.0 - mean) / (std * math.sqrt(2.0)))) / 2.0
    with torch.no_grad():
        tensor.uniform_(2 * low - 1, 2 * high - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=-2.0, max=2.0)
    return tensor


class DefaultAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("embedding dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.att_mat: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        self.att_mat = attention
        attention = self.attn_drop(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, act_layer: Callable[[int], nn.Module]
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        norm1_layer: Callable[[int], nn.Module],
        norm2_layer: Callable[[int], nn.Module],
        act_layer: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        self.norm1 = norm1_layer(dim)
        self.attn = DefaultAttention(dim, num_heads=num_heads, qkv_bias=True)
        self.drop_path = nn.Identity()
        self.norm2 = norm2_layer(dim)
        self.mlp = Mlp(dim, dim * 4, act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int, image_size: int = 224, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 192,
        patch_size: int = 16,
        depth: int = 4,
        num_heads: int = 3,
        img_size: tuple[int, int] | list[int] = (224, 224),
        norm_layer: Callable[[int], nn.Module] = nn.LayerNorm,
        block_norm1_layer: Callable[[int], nn.Module] | None = None,
        block_norm2_layer: Callable[[int], nn.Module] | None = None,
        final_norm_layer: Callable[[int], nn.Module] | None = None,
        act_layer: Callable[[int], nn.Module] = gelu_factory,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(embed_dim, img_size[0], patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList(
            Block(
                embed_dim,
                num_heads,
                block_norm1_layer or norm_layer,
                block_norm2_layer or norm_layer,
                act_layer,
            )
            for _ in range(depth)
        )
        self.norm = (final_norm_layer or norm_layer)(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.fc(self.norm(x)[:, 0])
