"""CPU checks for the extracted, framework-independent ViT path."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from experiments.vit.model import VisionTransformer
from experiments.vit.prepare_fixed_labels import digest, read_manifest
from experiments.vit.train_vit_token_grouping import (
    TokenChannelGroupCountNorm,
    build_model,
    prepare_output_dirs,
)


class ViTReleaseTests(unittest.TestCase):
    def test_output_directory_is_new_and_experiment_name_is_local(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "runs"
            out_dir, log_dir, model_dir = prepare_output_dirs(output, "first_run")
            self.assertEqual(Path(out_dir), output / "first_run")
            self.assertTrue(Path(log_dir).is_dir())
            self.assertTrue(Path(model_dir).is_dir())
            with self.assertRaises(FileExistsError):
                prepare_output_dirs(output, "first_run")
            for name in ("", ".", "..", "../other", "sub/run", r"sub\run"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    prepare_output_dirs(output, name)

    def test_minimal_backbone_forward_and_backward(self):
        torch.manual_seed(42)
        model = VisionTransformer(
            num_classes=10, embed_dim=24, depth=1, num_heads=3,
            img_size=(32, 32), patch_size=16,
        )
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 10))
        output.square().mean().backward()
        self.assertIsNotNone(model.patch_embed.proj.weight.grad)

    def test_token_channel_groups_normalize_independently(self):
        layer = TokenChannelGroupCountNorm(
            12, num_tokens=5, num_token_groups=3, num_channel_groups=4,
            affine=False,
        )
        output = layer(torch.randn(2, 5, 12))
        self.assertEqual(tuple(output.shape), (2, 5, 12))
        self.assertTrue(torch.isfinite(output).all())
        for token_slice in (slice(0, 2), slice(2, 4), slice(4, 5)):
            for channel_start in (0, 3, 6, 9):
                group = output[:, token_slice, channel_start:channel_start + 3]
                self.assertTrue(torch.allclose(
                    group.mean(dim=(1, 2)), torch.zeros(2), atol=2e-6,
                ))

    def test_both_paper_replacement_modes_construct_without_musicath_base(self):
        args = SimpleNamespace(
            norm2="TCGCN", norm_scope="all", activation_kind="gelu",
            num_tokens=197, token_groups=32, channel_groups=192,
            npg=16, patch_block_side=1, patch_grid_side=14,
            spatial_groups_per_axis=1, width=192, depth=4,
        )
        torch.manual_seed(42)
        normalization_model = build_model(args)
        self.assertEqual(len(normalization_model.state_dict()), 56)
        args.norm2 = "LN"
        args.activation_kind = "tcgcn"
        torch.manual_seed(42)
        activation_model = build_model(args)
        self.assertEqual(len(activation_model.state_dict()), 64)

    def test_archived_6k_manifest_digest(self):
        indices, clean, random = read_manifest()
        self.assertEqual(len(indices), 6000)
        self.assertEqual(len(clean), 6000)
        self.assertEqual(
            digest(random),
            "a88577fdb640f2d0b8832119d58be96bc6d9af9a9bf481e53cec16f7d000fd6a",
        )


if __name__ == "__main__":
    unittest.main()
