#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from experiments.vit.model import VisionTransformer, gelu_factory
from pln import ParallelLN

REPO_ROOT = Path(__file__).resolve().parents[2]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def noise_tag(noise_rate):
    return f"noise{noise_rate:g}".replace(".", "p")


def git_status_summary():
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}
    count = sum(bool(line.strip()) for line in status.splitlines())
    return {"available": True, "revision": revision,
            "dirty": bool(count), "num_entries": count}


class FixedNoisyCIFAR10(torch.utils.data.Dataset):
    def __init__(self, base, subset_indices, targets, clean_labels):
        self.base = base
        self.subset_indices = subset_indices.long()
        self.targets = targets.long()
        self.clean_targets = clean_labels.long()
        if len(self.subset_indices) != len(self.targets):
            raise ValueError("subset_indices and targets length mismatch")
        if len(self.clean_targets) != len(self.targets):
            raise ValueError("clean_labels and targets length mismatch")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, _ = self.base[int(self.subset_indices[idx])]
        return image, int(self.targets[idx])


class CleanSubsetCIFAR10(torch.utils.data.Dataset):
    def __init__(self, base, subset_indices, clean_labels):
        self.base = base
        self.subset_indices = subset_indices.long()
        self.targets = clean_labels.long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, _ = self.base[int(self.subset_indices[idx])]
        return image, int(self.targets[idx])


class RelabeledCIFAR10(torch.utils.data.Dataset):
    """Attach fixed targets to either CIFAR-10 or a concatenation of splits."""

    def __init__(self, base, targets, clean_labels):
        self.base = base
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.clean_targets = torch.as_tensor(clean_labels, dtype=torch.long)
        if len(self.base) != len(self.targets):
            raise ValueError("base and targets length mismatch")
        if len(self.clean_targets) != len(self.targets):
            raise ValueError("clean_labels and targets length mismatch")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image, _ = self.base[idx]
        return image, int(self.targets[idx])


def build_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_fixed_artifacts(artifact_dir, noise_rate):
    tag = noise_tag(noise_rate)
    subset_path = os.path.join(artifact_dir, "subset_indices.pt")
    clean_path = os.path.join(artifact_dir, "clean_labels.pt")
    label_path = os.path.join(artifact_dir, tag, "noisy_labels.pt")
    mask_path = os.path.join(artifact_dir, tag, "noise_mask.pt")
    summary_path = os.path.join(artifact_dir, tag, "summary.json")
    for path in [subset_path, clean_path, label_path, mask_path, summary_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    return {
        "subset_indices": torch.load(subset_path, map_location="cpu", weights_only=True),
        "clean_labels": torch.load(clean_path, map_location="cpu", weights_only=True),
        "noisy_labels": torch.load(label_path, map_location="cpu", weights_only=True),
        "noise_mask": torch.load(mask_path, map_location="cpu", weights_only=True),
        "summary_path": summary_path,
        "subset_artifact_path": subset_path,
        "label_artifact_path": label_path,
        "mask_artifact_path": mask_path,
    }


def build_loaders(args):
    root = args.data_root
    tfm = build_transforms()
    train_base = torchvision.datasets.CIFAR10(root=root, train=True, transform=tfm, download=args.download)
    test_base = torchvision.datasets.CIFAR10(root=root, train=False, transform=tfm, download=args.download)
    if args.data_mode == "fixed-subset":
        if not args.artifact_dir:
            raise ValueError("--artifact-dir is required for fixed-subset mode")
        artifacts = load_fixed_artifacts(args.artifact_dir, args.noise_rate)
        dataset = FixedNoisyCIFAR10(
            train_base,
            artifacts["subset_indices"],
            artifacts["noisy_labels"],
            artifacts["clean_labels"],
        )
        clean_subset = CleanSubsetCIFAR10(
            train_base,
            artifacts["subset_indices"],
            artifacts["clean_labels"],
        )
    else:
        clean_train = torch.as_tensor(train_base.targets, dtype=torch.long)
        clean_test = torch.as_tensor(test_base.targets, dtype=torch.long)
        if args.data_mode == "full-random":
            base = torch.utils.data.ConcatDataset([train_base, test_base])
            clean_labels = torch.cat([clean_train, clean_test])
            generator = torch.Generator().manual_seed(args.label_seed)
            targets = torch.randint(0, 10, (len(base),), generator=generator)
            noise_mask = torch.ones(len(base), dtype=torch.bool)
            effective_noise_rate = 1.0
        elif args.data_mode == "full-clean":
            base = train_base
            clean_labels = clean_train
            targets = clean_labels.clone()
            noise_mask = torch.zeros(len(base), dtype=torch.bool)
            effective_noise_rate = 0.0
        else:
            raise ValueError(f"unsupported data mode: {args.data_mode}")
        dataset = RelabeledCIFAR10(base, targets, clean_labels)
        clean_subset = RelabeledCIFAR10(base, clean_labels, clean_labels)
        artifacts = {
            "noise_mask": noise_mask,
            "summary_path": f"generated:{args.data_mode}",
            "subset_artifact_path": f"generated:{args.data_mode}:{len(base)}",
            "label_artifact_path": f"generated:seed={args.label_seed}",
            "mask_artifact_path": f"generated:noise_rate={effective_noise_rate}",
        }
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    clean_train_loader = torch.utils.data.DataLoader(
        clean_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    clean_test_loader = torch.utils.data.DataLoader(
        test_base,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader, clean_train_loader, clean_test_loader, artifacts


def verify_fixed_label_setup(train_loader, eval_loader, artifacts, noise_rate):
    train_ds = train_loader.dataset
    eval_ds = eval_loader.dataset
    if train_ds is not eval_ds:
        raise RuntimeError("train/eval loaders do not share the same dataset object")
    labels = train_ds.targets
    clean = train_ds.clean_targets
    mask = artifacts["noise_mask"].bool()
    counts = torch.bincount(labels, minlength=10)
    mask_count = int(mask.sum().item())
    changed_count = int(labels.ne(clean).sum().item())
    expected_mask_count = int(round(len(labels) * float(noise_rate)))
    if mask_count != expected_mask_count:
        raise RuntimeError(f"mask_count={mask_count}, expected={expected_mask_count}")
    same = True
    for idx in range(min(32, len(train_ds))):
        same = same and train_ds[idx][1] == eval_ds[idx][1]
    print(
        "[FixedLabelCheck] "
        f"num_samples={len(train_ds)} "
        f"noise_rate={noise_rate} "
        f"masked={mask_count} "
        f"changed={changed_count} "
        f"counts={counts.tolist()} "
        f"first16={labels[:16].tolist()} "
        f"clean_first16={clean[:16].tolist()} "
        f"train_eval_same_first32={same}",
        flush=True,
    )
    return {
        "num_samples": len(train_ds),
        "noise_rate": float(noise_rate),
        "masked": mask_count,
        "changed": changed_count,
        "counts": counts.tolist(),
        "first16": labels[:16].tolist(),
        "clean_first16": clean[:16].tolist(),
        "train_eval_same_first32": bool(same),
    }


def build_model(args):
    from functools import partial

    if args.norm2 == "LN":
        norm2_layer = nn.LayerNorm
    elif args.norm2 == "PLN":
        norm2_layer = partial(
            ParallelLN,
            num_per_group=args.npg,
            dim=3,
            centering=True,
            affine=True,
        )
    elif args.norm2 == "TGN":
        norm2_layer = partial(
            TokenGroupNorm,
            num_groups=args.token_groups,
            affine=True,
        )
    elif args.norm2 == "BTCN":
        norm2_layer = partial(
            BlockTokenChannelNorm,
            num_per_group=args.npg,
            patch_block_side=args.patch_block_side,
            patch_grid_side=args.patch_grid_side,
            affine=True,
        )
    elif args.norm2 == "GCTN":
        norm2_layer = partial(
            GroupCountTokenChannelNorm,
            spatial_groups_per_axis=args.spatial_groups_per_axis,
            num_channel_groups=args.channel_groups,
            patch_grid_side=args.patch_grid_side,
            affine=True,
        )
    elif args.norm2 == "TCGCN":
        norm2_layer = partial(
            TokenChannelGroupCountNorm,
            num_tokens=args.num_tokens,
            num_token_groups=args.token_groups,
            num_channel_groups=args.channel_groups,
            affine=True,
        )
    else:
        raise ValueError(f"Unsupported norm2 kind: {args.norm2}")

    if args.norm_scope == "all":
        norm_layer = norm2_layer
        block_norm1_layer = norm2_layer
        block_norm2_layer = norm2_layer
        final_norm_layer = norm2_layer
    else:
        norm_layer = nn.LayerNorm
        block_norm1_layer = nn.LayerNorm
        block_norm2_layer = norm2_layer
        final_norm_layer = nn.LayerNorm

    if args.activation_kind == "gelu":
        act_layer = gelu_factory
    elif args.activation_kind == "tcgcn":
        # The activation is applied after the first FFN projection, hence its
        # feature dimension is mlp_ratio * width (768 for W192).  The callable
        # receives that hidden dimension from Mlp and constructs the grouped
        # nonlinearity over the complete [T, 4C] representation.
        act_layer = partial(
            TokenChannelGroupCountNorm,
            num_tokens=args.num_tokens,
            num_token_groups=args.token_groups,
            num_channel_groups=args.channel_groups,
            affine=True,
        )
    else:
        raise ValueError(f"Unsupported activation kind: {args.activation_kind}")

    model = VisionTransformer(
        num_classes=10,
        embed_dim=args.width,
        patch_size=16,
        depth=args.depth,
        num_heads=3,
        img_size=[224, 224],
        norm_layer=norm_layer,
        block_norm1_layer=block_norm1_layer,
        block_norm2_layer=block_norm2_layer,
        final_norm_layer=final_norm_layer,
        act_layer=act_layer,
    )
    if args.norm_scope == "all" and args.norm2 == "TCGCN":
        remaining_layer_norms = [
            name for name, module in model.named_modules() if isinstance(module, nn.LayerNorm)
        ]
        if remaining_layer_norms:
            raise RuntimeError(
                "--norm-scope all left nn.LayerNorm modules in the model: "
                + ", ".join(remaining_layer_norms)
            )
    return model


class TokenGroupNorm(nn.Module):
    """Per-sample GroupNorm on [B, T, C] that includes tokens in its domain.

    Channels are partitioned into ``num_groups`` groups.  Each group's moments
    are computed jointly over its C / num_groups channels and every token T.
    No statistics are shared across the batch dimension.
    """

    def __init__(self, num_features, num_groups=1, eps=1e-5, affine=True, **kwargs):
        super().__init__()
        if num_features % num_groups != 0:
            raise ValueError(
                f"num_features={num_features} must be divisible by num_groups={num_groups}"
            )
        self.num_features = num_features
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.num_features:
            raise ValueError(
                f"TokenGroupNorm expects [B, T, {self.num_features}], got {tuple(x.shape)}"
            )
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TokenChannelGroupCountNorm(nn.Module):
    """Normalize rectangular groups of the full ``[T, C]`` representation.

    The token axis is split into ``num_token_groups`` balanced contiguous
    pieces and the channel axis is independently split into
    ``num_channel_groups`` balanced contiguous pieces.  Their Cartesian
    products form ``num_token_groups * num_channel_groups`` normalization
    domains.  Unlike the patch-grid normalizers below, this layer treats the
    complete sequence -- including the class token -- as a ``T x C`` matrix
    and does not use the two-dimensional image topology.
    """

    def __init__(
        self,
        num_features,
        num_tokens=197,
        num_token_groups=1,
        num_channel_groups=1,
        eps=1e-5,
        affine=True,
        **kwargs,
    ):
        super().__init__()
        if not 1 <= num_token_groups <= num_tokens:
            raise ValueError(
                f"num_token_groups={num_token_groups} must be between 1 and "
                f"num_tokens={num_tokens}"
            )
        if not 1 <= num_channel_groups <= num_features:
            raise ValueError(
                f"num_channel_groups={num_channel_groups} must be between 1 "
                f"and num_features={num_features}"
            )

        self.num_features = num_features
        self.num_tokens = num_tokens
        self.num_token_groups = num_token_groups
        self.num_channel_groups = num_channel_groups
        self.eps = eps

        token_sizes = self._balanced_sizes(num_tokens, num_token_groups)
        channel_sizes = self._balanced_sizes(num_features, num_channel_groups)
        token_starts = self._starts(token_sizes)
        channel_starts = self._starts(channel_sizes)
        max_tokens = max(token_sizes)
        max_channels = max(channel_sizes)

        sentinel = num_tokens * num_features
        slot_to_value = torch.full(
            (
                num_token_groups,
                max_tokens,
                num_channel_groups,
                max_channels,
            ),
            sentinel,
            dtype=torch.long,
        )
        for token_group, (token_start, token_size) in enumerate(
            zip(token_starts, token_sizes)
        ):
            for channel_group, (channel_start, channel_size) in enumerate(
                zip(channel_starts, channel_sizes)
            ):
                for token_offset in range(token_size):
                    token = token_start + token_offset
                    first = token * num_features + channel_start
                    slot_to_value[
                        token_group,
                        token_offset,
                        channel_group,
                        :channel_size,
                    ] = torch.arange(first, first + channel_size)

        flat_slots = slot_to_value.flatten()
        value_to_slot = torch.empty(sentinel, dtype=torch.long)
        valid_slot_indices = torch.nonzero(flat_slots != sentinel).flatten()
        value_to_slot[flat_slots[valid_slot_indices]] = valid_slot_indices
        self.register_buffer("slot_to_value", slot_to_value, persistent=False)
        self.register_buffer(
            "slot_mask", slot_to_value != sentinel, persistent=False
        )
        self.register_buffer("value_to_slot", value_to_slot, persistent=False)

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @staticmethod
    def _balanced_sizes(length, groups):
        return [
            length // groups + int(index < length % groups)
            for index in range(groups)
        ]

    @staticmethod
    def _starts(sizes):
        starts = [0]
        for size in sizes[:-1]:
            starts.append(starts[-1] + size)
        return starts

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (
            self.num_tokens,
            self.num_features,
        ):
            raise ValueError(
                "TokenChannelGroupCountNorm expects "
                f"[B, {self.num_tokens}, {self.num_features}], got {tuple(x.shape)}"
            )

        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        padded = torch.cat((flat, flat.new_zeros(batch, 1)), dim=1)
        grouped = padded[:, self.slot_to_value]
        mask = self.slot_mask[None]
        counts = self.slot_mask.sum(dim=(1, 3))[None, :, None, :, None]
        mean = grouped.sum(dim=(2, 4), keepdim=True) / counts
        centered = (grouped - mean) * mask
        variance = centered.square().sum(dim=(2, 4), keepdim=True) / counts
        normalized = centered * torch.rsqrt(variance + self.eps)
        output = normalized.reshape(batch, -1)[:, self.value_to_slot].reshape_as(x)

        if self.weight is not None:
            output = output * self.weight + self.bias
        return output


class BlockTokenChannelNorm(nn.Module):
    """Normalize joint spatial-patch and channel blocks in a ViT sequence.

    The input is ``[B, 1 + H*W, C]`` with the class token first. Patch tokens
    are restored to their ``H x W`` grid. Non-overlapping square blocks of
    ``patch_block_side x patch_block_side`` patches and consecutive channel
    groups of ``num_per_group`` features form the normalization domains. The
    class token is normalized separately with LayerNorm.
    """

    def __init__(
        self,
        num_features,
        num_per_group=16,
        patch_block_side=1,
        patch_grid_side=14,
        eps=1e-5,
        affine=True,
        **kwargs,
    ):
        super().__init__()
        if num_features % num_per_group != 0:
            raise ValueError(
                f"num_features={num_features} must be divisible by "
                f"num_per_group={num_per_group}"
            )
        if patch_grid_side % patch_block_side != 0:
            raise ValueError(
                f"patch_grid_side={patch_grid_side} must be divisible by "
                f"patch_block_side={patch_block_side}"
            )
        self.num_features = num_features
        self.num_per_group = num_per_group
        self.num_channel_groups = num_features // num_per_group
        self.patch_block_side = patch_block_side
        self.patch_grid_side = patch_grid_side
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        # Reuse the same channel-wise affine parameters for class and patch
        # tokens, matching standard LayerNorm at block=1, group_size=C.
        self.class_token_norm = nn.LayerNorm(
            num_features, eps=eps, elementwise_affine=False
        )

    def forward(self, x):
        expected_tokens = 1 + self.patch_grid_side**2
        if x.ndim != 3 or x.shape[-1] != self.num_features:
            raise ValueError(
                f"BlockTokenChannelNorm expects [B, T, {self.num_features}], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} tokens (one class token plus a "
                f"{self.patch_grid_side}x{self.patch_grid_side} patch grid), "
                f"got {x.shape[1]}"
            )

        batch = x.shape[0]
        grid = self.patch_grid_side
        block = self.patch_block_side
        num_blocks = grid // block
        cls, patches = x[:, :1], x[:, 1:]
        grouped = patches.reshape(
            batch,
            num_blocks,
            block,
            num_blocks,
            block,
            self.num_channel_groups,
            self.num_per_group,
        )
        mean = grouped.mean(dim=(2, 4, 6), keepdim=True)
        variance = (grouped - mean).square().mean(dim=(2, 4, 6), keepdim=True)
        normalized = (grouped - mean) * torch.rsqrt(variance + self.eps)
        patches = normalized.reshape(batch, grid * grid, self.num_features)
        output = torch.cat((self.class_token_norm(cls), patches), dim=1)
        if self.weight is not None:
            output = output * self.weight + self.bias
        return output


class GroupCountTokenChannelNorm(nn.Module):
    """Normalize balanced two-dimensional patch groups and channel groups.

    Unlike :class:`BlockTokenChannelNorm`, this layer is parameterized by the
    *number* of groups rather than by the size of each group.  A patch-grid axis
    that is not divisible by ``spatial_groups_per_axis`` is split into
    contiguous pieces whose sizes differ by at most one.  The class token is
    normalized separately with LayerNorm.
    """

    def __init__(
        self,
        num_features,
        spatial_groups_per_axis=1,
        num_channel_groups=1,
        patch_grid_side=14,
        eps=1e-5,
        affine=True,
        **kwargs,
    ):
        super().__init__()
        if not 1 <= spatial_groups_per_axis <= patch_grid_side:
            raise ValueError(
                "spatial_groups_per_axis must be between 1 and patch_grid_side"
            )
        if num_features % num_channel_groups != 0:
            raise ValueError(
                f"num_features={num_features} must be divisible by "
                f"num_channel_groups={num_channel_groups}"
            )
        self.num_features = num_features
        self.spatial_groups_per_axis = spatial_groups_per_axis
        self.num_channel_groups = num_channel_groups
        self.num_per_group = num_features // num_channel_groups
        self.patch_grid_side = patch_grid_side
        self.eps = eps

        sizes = [
            patch_grid_side // spatial_groups_per_axis
            + int(index < patch_grid_side % spatial_groups_per_axis)
            for index in range(spatial_groups_per_axis)
        ]
        max_size = max(sizes)
        starts = [0]
        for size in sizes[:-1]:
            starts.append(starts[-1] + size)

        sentinel = patch_grid_side * patch_grid_side
        slot_to_patch = torch.full(
            (
                spatial_groups_per_axis,
                max_size,
                spatial_groups_per_axis,
                max_size,
            ),
            sentinel,
            dtype=torch.long,
        )
        for row_group, (row_start, row_size) in enumerate(zip(starts, sizes)):
            for col_group, (col_start, col_size) in enumerate(zip(starts, sizes)):
                for row_offset in range(row_size):
                    for col_offset in range(col_size):
                        row = row_start + row_offset
                        col = col_start + col_offset
                        slot_to_patch[
                            row_group, row_offset, col_group, col_offset
                        ] = row * patch_grid_side + col

        flat_slots = slot_to_patch.flatten()
        patch_to_slot = torch.empty(sentinel, dtype=torch.long)
        valid_slot_indices = torch.nonzero(flat_slots != sentinel).flatten()
        patch_to_slot[flat_slots[valid_slot_indices]] = valid_slot_indices
        self.register_buffer("slot_to_patch", slot_to_patch, persistent=False)
        self.register_buffer(
            "slot_mask", slot_to_patch != sentinel, persistent=False
        )
        self.register_buffer("patch_to_slot", patch_to_slot, persistent=False)

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.class_token_norm = nn.LayerNorm(
            num_features, eps=eps, elementwise_affine=False
        )

    def forward(self, x):
        expected_tokens = 1 + self.patch_grid_side**2
        if x.ndim != 3 or x.shape[-1] != self.num_features:
            raise ValueError(
                f"GroupCountTokenChannelNorm expects [B, T, {self.num_features}], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} tokens (one class token plus a "
                f"{self.patch_grid_side}x{self.patch_grid_side} patch grid), "
                f"got {x.shape[1]}"
            )

        batch = x.shape[0]
        groups = self.spatial_groups_per_axis
        block = self.slot_to_patch.shape[1]
        cls, patches = x[:, :1], x[:, 1:]
        padded = torch.cat(
            (patches, patches.new_zeros(batch, 1, self.num_features)), dim=1
        )
        grouped = padded[:, self.slot_to_patch].reshape(
            batch,
            groups,
            block,
            groups,
            block,
            self.num_channel_groups,
            self.num_per_group,
        )
        mask = self.slot_mask[None, :, :, :, :, None, None]
        counts = (
            self.slot_mask.sum(dim=(1, 3))[None, :, None, :, None, None, None]
            * self.num_per_group
        )
        mean = grouped.sum(dim=(2, 4, 6), keepdim=True) / counts
        centered = (grouped - mean) * mask
        variance = centered.square().sum(dim=(2, 4, 6), keepdim=True) / counts
        normalized = centered * torch.rsqrt(variance + self.eps)
        normalized_slots = normalized.reshape(batch, -1, self.num_features)
        patches = normalized_slots[:, self.patch_to_slot]

        output = torch.cat((self.class_token_norm(cls), patches), dim=1)
        if self.weight is not None:
            output = output * self.weight + self.bias
        return output


NORM_AFFINE_CLASS_NAMES = frozenset(
    {
        "LayerNorm",
        "GroupNorm",
        "ParallelLN",
        "TokenGroupNorm",
        "TokenChannelGroupCountNorm",
        "BlockTokenChannelNorm",
        "GroupCountTokenChannelNorm",
    }
)


def split_norm_affine_parameters(model):
    """Separate normalization affine parameters from all other parameters.

    Normalization modules in this experiment use channel-wise ``weight`` and
    ``bias`` parameters.  The custom modules and PyTorch LayerNorm/GroupNorm
    are identified by class name so this also covers ParallelLN, which is
    imported dynamically from Musicath-base.
    """

    norm_parameter_ids = set()
    norm_parameter_names = []
    for module_name, module in model.named_modules():
        if module.__class__.__name__ not in NORM_AFFINE_CLASS_NAMES:
            continue
        for local_name, parameter in module.named_parameters(recurse=False):
            if local_name not in {"weight", "bias"} or not parameter.requires_grad:
                continue
            norm_parameter_ids.add(id(parameter))
            qualified_name = f"{module_name}.{local_name}" if module_name else local_name
            norm_parameter_names.append(qualified_name)

    decay_parameters = []
    no_decay_parameters = []
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in norm_parameter_ids:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    if len(decay_parameters) + len(no_decay_parameters) != sum(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise RuntimeError("optimizer parameter split is incomplete")
    return decay_parameters, no_decay_parameters, sorted(norm_parameter_names)


def build_adamw_optimizer(model, lr, weight_decay, exclude_norm_weight_decay=False):
    if not exclude_norm_weight_decay:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    decay_parameters, no_decay_parameters, norm_parameter_names = (
        split_norm_affine_parameters(model)
    )
    if not no_decay_parameters:
        raise RuntimeError("no normalization affine parameters found for zero decay")
    print(
        "[Optimizer] excluding normalization affine parameters from weight decay: "
        f"{len(norm_parameter_names)} tensors",
        flush=True,
    )
    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def cosine_warmup_lr(epoch, step, steps_per_epoch, epochs, warmup_epochs, base_lr):
    progress_step = epoch * steps_per_epoch + step
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    if warmup_steps > 0 and progress_step < warmup_steps:
        return base_lr * float(progress_step + 1) / float(warmup_steps)
    denom = max(1, total_steps - warmup_steps)
    t = (progress_step - warmup_steps) / denom
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))


def run_epoch(model, loader, optimizer, device, epoch, args):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    steps = len(loader)
    for step, (images, targets) in enumerate(loader):
        if args.lr_schedule == "constant":
            lr = args.lr
        else:
            lr = cosine_warmup_lr(epoch, step, steps, args.epochs, args.warmup_epochs, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.numel()
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total += targets.numel()
    return {
        "train/loss": total_loss / total,
        "train/acc": 100.0 * total_correct / total,
        "lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def evaluate(model, loader, device, prefix):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, targets)
        total_loss += loss.item() * targets.numel()
        total_correct += logits.argmax(dim=1).eq(targets).sum().item()
        total += targets.numel()
    return {
        f"{prefix}/loss": total_loss / total,
        f"{prefix}/acc": 100.0 * total_correct / total,
    }


def save_checkpoint(path, model, optimizer, epoch, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def init_swanlab(args, config):
    if not args.enable_swanlab:
        class LocalOnlyLogger:
            @staticmethod
            def log(*unused_args, **unused_kwargs):
                return None

            @staticmethod
            def finish():
                return None

        return LocalOnlyLogger()
    if not args.swanlab_workspace or not args.swanlab_project:
        raise ValueError("--enable-swanlab requires --swanlab-workspace and --swanlab-project")
    api_key = os.environ.get("SWANLAB_API_KEY", "")
    if not api_key:
        raise RuntimeError("--enable-swanlab requires SWANLAB_API_KEY")
    try:
        import swanlab
        from swanlab.swanlab_settings import Settings
    except ImportError as exc:
        raise RuntimeError("swanlab is not installed") from exc

    settings = Settings(
        requirements_collect=False,
        conda_collect=False,
        hardware_monitor=False,
        collect_hardware=False,
    )
    last_exc = None
    for attempt in range(1, args.swanlab_init_retries + 1):
        try:
            swanlab.login(api_key=api_key)
            swanlab.init(
                project=args.swanlab_project,
                workspace=args.swanlab_workspace,
                experiment_name=args.exp_name,
                description=args.swanlab_description,
                config=config,
                logdir=os.path.join(args.log_dir, "swanlab"),
                log_level="critical",
                settings=settings,
            )
            return swanlab
        except Exception as exc:
            last_exc = exc
            if attempt >= args.swanlab_init_retries:
                break
            delay = args.swanlab_retry_wait * attempt
            print(
                f"[SwanLabInitRetry] attempt={attempt} failed: {exc}; "
                f"sleep={delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"swanlab init failed after {args.swanlab_init_retries} attempts") from last_exc


def build_config(args, fixed_check, artifacts):
    cfg = vars(args).copy()
    cfg["optimizer"] = "AdamW"
    cfg["weight_decay"] = args.weight_decay
    cfg["norm_tag"] = f"{args.norm_scope}:{args.norm2}"
    cfg["subset_artifact_path"] = artifacts["subset_artifact_path"]
    cfg["label_artifact_path"] = artifacts["label_artifact_path"]
    cfg["mask_artifact_path"] = artifacts["mask_artifact_path"]
    cfg["fixed_label_check"] = fixed_check
    cfg["git_status_summary"] = git_status_summary()
    return cfg


def prepare_output_dirs(output, exp_name):
    if not exp_name or exp_name in {".", ".."} or "/" in exp_name or "\\" in exp_name:
        raise ValueError("--exp-name must be a single nonempty directory name")
    out_dir = Path(output) / exp_name
    out_dir.mkdir(parents=True, exist_ok=False)
    log_dir = out_dir / "log_results"
    model_dir = out_dir / "model_results"
    log_dir.mkdir()
    model_dir.mkdir()
    return str(out_dir), str(log_dir), str(model_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default="data", help="torchvision CIFAR-10 cache directory"
    )
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 if absent")
    parser.add_argument(
        "--norm2",
        choices=["LN", "PLN", "TGN", "BTCN", "GCTN", "TCGCN"],
        required=True,
    )
    parser.add_argument(
        "--norm-scope",
        choices=["norm2", "all"],
        default="norm2",
        help=(
            "norm2 replaces only the FFN-side normalization (legacy behavior); "
            "all uses the selected normalization for every block norm1/norm2 "
            "and the final sequence normalization"
        ),
    )
    parser.add_argument(
        "--activation-kind",
        choices=["gelu", "tcgcn"],
        default="gelu",
        help=(
            "gelu keeps the standard element-wise FFN activation; tcgcn "
            "replaces GELU by token-channel grouped normalization while "
            "leaving the requested normalization layers unchanged"
        ),
    )
    parser.add_argument("--npg", type=int, default=16)
    parser.add_argument("--token-groups", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=197)
    parser.add_argument("--patch-block-side", type=int, default=1)
    parser.add_argument("--patch-grid-side", type=int, default=14)
    parser.add_argument("--spatial-groups-per-axis", type=int, default=1)
    parser.add_argument("--channel-groups", type=int, default=1)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-seed", type=int, default=20260629)
    parser.add_argument("--noise-rate", type=float, required=True)
    parser.add_argument(
        "--data-mode",
        choices=["fixed-subset", "full-random", "full-clean"],
        default="fixed-subset",
        help=(
            "fixed-subset uses saved 6k or 20k artifacts; full-random trains on all "
            "60k CIFAR-10 images with fixed random labels; full-clean uses the "
            "standard 50k train / 10k test split"
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr-schedule", default="constant", choices=["cosine", "constant"])
    parser.add_argument("--weight-decay", type=float, default=5e-6)
    parser.add_argument(
        "--exclude-norm-weight-decay",
        action="store_true",
        help="keep normalization affine weight/bias in a zero-weight-decay group",
    )
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--disable-checkpoints",
        action="store_true",
        help="keep configs and metrics but do not write best/last model checkpoints",
    )
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--swanlab-workspace")
    parser.add_argument("--swanlab-project")
    parser.add_argument("--swanlab-description", default="")
    parser.add_argument(
        "--enable-swanlab", action="store_true",
        help="opt in to SwanLab logging using SWANLAB_API_KEY from the environment",
    )
    parser.add_argument("--swanlab-init-retries", type=int, default=4)
    parser.add_argument("--swanlab-retry-wait", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.noise_rate <= 1.0:
        raise ValueError("--noise-rate must be in [0, 1]")
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, eval_loader, clean_train_loader, clean_test_loader, artifacts = build_loaders(args)
    effective_noise_rate = 1.0 if args.data_mode == "full-random" else 0.0 if args.data_mode == "full-clean" else args.noise_rate
    fixed_check = verify_fixed_label_setup(train_loader, eval_loader, artifacts, effective_noise_rate)
    out_dir, log_dir, model_dir = prepare_output_dirs(args.output, args.exp_name)
    args.log_dir = log_dir
    config = build_config(args, fixed_check, artifacts)
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    swanlab = init_swanlab(args, config)
    model = build_model(args).to(device)
    optimizer = build_adamw_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        exclude_norm_weight_decay=args.exclude_norm_weight_decay,
    )

    metrics_path = os.path.join(log_dir, "metrics.jsonl")
    best_acc = -1.0
    best_epoch = -1
    start_time = time.time()
    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            metrics = run_epoch(model, train_loader, optimizer, device, epoch, args)
            should_eval = (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
            if should_eval:
                metrics.update(evaluate(model, eval_loader, device, "eval_same"))
                if args.data_mode == "full-clean":
                    metrics.update(evaluate(model, clean_test_loader, device, "clean_test"))
                if metrics["eval_same/acc"] > best_acc:
                    best_acc = metrics["eval_same/acc"]
                    best_epoch = epoch
                    if not args.disable_checkpoints:
                        save_checkpoint(
                            os.path.join(model_dir, "best.pth"),
                            model,
                            optimizer,
                            epoch,
                            metrics,
                        )
            metrics["epoch"] = epoch
            metrics["best_eval_same/acc"] = best_acc
            metrics["best_epoch"] = best_epoch
            metrics["epoch_time"] = time.time() - epoch_start
            metrics["elapsed"] = time.time() - start_time
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, sort_keys=True) + "\n")
            swanlab.log(metrics, step=epoch)
            print(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"train_acc={metrics['train/acc']:.2f} train_loss={metrics['train/loss']:.4f} "
                f"eval_acc={metrics.get('eval_same/acc', float('nan')):.2f} "
                f"best={best_acc:.2f} lr={metrics['lr']:.2e} "
                f"time={metrics['epoch_time']:.1f}s",
                flush=True,
            )
            if not args.disable_checkpoints:
                save_checkpoint(
                    os.path.join(model_dir, "last.pth"),
                    model,
                    optimizer,
                    epoch,
                    metrics,
                )

        clean_train_prefix = f"clean_train{len(clean_train_loader.dataset)}" if args.data_mode == "fixed-subset" else "clean_train"
        final_metrics = evaluate(model, clean_train_loader, device, clean_train_prefix)
        final_metrics.update(evaluate(model, clean_test_loader, device, "clean_test"))
        final_metrics["epoch"] = args.epochs - 1
        final_metrics["best_eval_same/acc"] = best_acc
        final_metrics["best_epoch"] = best_epoch
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(final_metrics, sort_keys=True) + "\n")
        swanlab.log(final_metrics, step=args.epochs)
        print(
            f"done {args.exp_name} best_eval_same_acc={best_acc:.2f} "
            f"best_epoch={best_epoch} clean_test_acc={final_metrics['clean_test/acc']:.2f}",
            flush=True,
        )
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
