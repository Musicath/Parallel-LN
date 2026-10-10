"""Check the CNN normalization-domain endpoints without training."""

import unittest

import torch
from torch import nn

from experiments.cnn.run_cifar10_conv_gn_random_labels import (
    BlockGroupNorm,
    PositionGroupNorm,
)


class CNNDomainTests(unittest.TestCase):
    def test_full_map_endpoint_equals_group_norm(self):
        x = torch.randn(2, 32, 8, 8)
        actual = BlockGroupNorm(32, num_channel_groups=4, spatial_splits=1)(x)
        expected = nn.GroupNorm(4, 32, eps=1e-5, affine=False)(x)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))

    def test_single_position_endpoint_equals_position_wise_pln(self):
        x = torch.randn(2, 32, 8, 8)
        actual = BlockGroupNorm(32, num_channel_groups=4, spatial_splits=8)(x)
        expected = PositionGroupNorm(32, num_groups=4)(x)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))

    def test_nondivisible_spatial_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            BlockGroupNorm(32, num_channel_groups=4, spatial_splits=3)(
                torch.randn(1, 32, 8, 8)
            )


if __name__ == "__main__":
    unittest.main()
