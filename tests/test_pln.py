"""Numerical checks for the public, position-wise PLN operator."""

import unittest

import torch
import torch.nn.functional as F

from pln import ParallelLN


class ParallelLNTests(unittest.TestCase):
    def test_matches_independent_layer_norm_on_vector_sequence_and_image(self):
        torch.manual_seed(7)
        cases = (
            (torch.randn(3, 12, dtype=torch.float64), 2),
            (torch.randn(2, 5, 12, dtype=torch.float64), 3),
            (torch.randn(2, 12, 3, 4, dtype=torch.float64), 4),
        )
        for x, dim in cases:
            with self.subTest(dim=dim):
                layer = ParallelLN(12, 4, dim=dim, affine=False).double()
                actual = layer(x)
                channel_last = x.permute(0, 2, 3, 1) if dim == 4 else x
                expected = F.layer_norm(
                    channel_last.reshape(*channel_last.shape[:-1], 3, 4),
                    (4,),
                    eps=1e-5,
                ).reshape_as(channel_last)
                if dim == 4:
                    expected = expected.permute(0, 3, 1, 2)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_affine_and_gradients(self):
        torch.manual_seed(9)
        x = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
        layer = ParallelLN(8, 4, dim=3).double()
        with torch.no_grad():
            layer.weight.copy_(torch.linspace(0.5, 1.2, 8))
            layer.bias.copy_(torch.linspace(-0.2, 0.2, 8))
        actual = layer(x)
        expected = F.layer_norm(x.reshape(2, 3, 2, 4), (4,), eps=1e-5)
        expected = expected.reshape_as(x) * layer.weight + layer.bias
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertTrue(torch.isfinite(layer.bias.grad).all())

    def test_pls_uses_groupwise_rms_without_centering(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
        actual = ParallelLN(4, 2, dim=2, centering=False, affine=False)(x)
        groups = x.reshape(1, 2, 2)
        expected = groups * torch.rsqrt(groups.square().mean(-1, keepdim=True) + 1e-5)
        torch.testing.assert_close(actual, expected.reshape_as(x))

    def test_invalid_configuration_and_shape_fail_early(self):
        for kwargs in (
            {"num_features": 0, "num_per_group": 4},
            {"num_features": 8, "num_per_group": 0},
            {"num_features": 8, "num_per_group": 3},
            {"num_features": 8, "num_per_group": 4, "dim": 5},
            {"num_features": 8, "num_per_group": 4, "eps": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ParallelLN(**kwargs)
        layer = ParallelLN(8, 4, dim=3)
        with self.assertRaises(ValueError):
            layer(torch.randn(2, 8))
        with self.assertRaises(ValueError):
            layer(torch.randn(2, 3, 9))


if __name__ == "__main__":
    unittest.main()
