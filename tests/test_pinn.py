"""CPU checks for the standalone PINN extraction and sanitized data."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.pinn.model import HelmholtzMLP, SinArctan
from experiments.pinn.output_ntk import output_jacobian, spectrum_from_jacobian
from experiments.pinn.run_helmholtz import parse_args
from experiments.pinn.verify_data import read_rows, verify_ntk, verify_table2


class PINNTests(unittest.TestCase):
    def test_model_shape_and_depth(self):
        for method in ("PLN4", "PLN8", "ReLU", "Tanh", "SAT", "LN"):
            with self.subTest(method=method):
                model = HelmholtzMLP(method, width=128, depth=6)
                self.assertIsNone(model.regularizer)
                self.assertEqual(tuple(model(torch.randn(3, 1)).shape), (3, 1))
                linear = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
                self.assertEqual(len(linear), 7)
                self.assertEqual(linear[0].in_features, 1)
                self.assertEqual(linear[-1].out_features, 1)
                self.assertEqual(list(model.state_dict())[0], "net.1.weight")

    def test_sat_formula(self):
        x = torch.tensor([-2.0, 0.0, 2.0])
        self.assertTrue(torch.allclose(SinArctan()(x), x / torch.sqrt(1 + x * x)))

    def test_output_ntk_linear_example(self):
        model = torch.nn.Linear(1, 1).double()
        points = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        jacobian = output_jacobian(model, points)
        expected = torch.tensor([[1.0, 1.0], [2.0, 1.0]], dtype=torch.float64)
        self.assertTrue(torch.allclose(jacobian, expected))
        result = spectrum_from_jacobian(jacobian)
        self.assertTrue(np.isclose(result["trace"], 7.0))
        self.assertEqual(result["eff_rank_90"], 1)

    def test_archived_lr_defaults_and_protocols(self):
        sgd = parse_args(["--optimizer", "sgd", "--dry-run"])
        self.assertEqual((sgd.lr, sgd.steps, sgd.ntk_points, sgd.display_every),
                         (1e-4, 6000, 0, 100))
        ntk = parse_args(["--protocol", "ntk", "--lbfgs-steps", "1200", "--dry-run"])
        self.assertEqual((ntk.lr, ntk.steps, ntk.ntk_points, ntk.display_every),
                         (3e-4, 2800, 500, 10))

    def test_sanitized_data_reconstructs_manuscript(self):
        self.assertEqual(len(verify_table2()), 42)
        summary = verify_ntk()
        self.assertEqual(summary["PLN4"]["initial_eff_rank_90_median"], 12)
        self.assertLess(summary["SAT"]["adam_lbfgs_relative_l2_median"],
                        summary["PLN4"]["adam_lbfgs_relative_l2_median"])

    def test_ntk_verifier_rejects_mismatched_or_nonfinite_rows(self):
        original = read_rows("ntk_matched_240.csv")
        changed = [dict(row) for row in original]
        changed[0]["init_domain_eff_rank_90"] = "999"
        with patch("experiments.pinn.verify_data.read_rows", return_value=changed):
            with self.assertRaisesRegex(ValueError, "different initial metrics"):
                verify_ntk()
        changed = [dict(row) for row in original]
        changed[0]["l2_relative_error"] = "nan"
        with patch("experiments.pinn.verify_data.read_rows", return_value=changed):
            with self.assertRaisesRegex(ValueError, "invalid NTK numeric field"):
                verify_ntk()
        changed = [dict(row) for row in original]
        for row in changed:
            if row["method"] == "PLN4" and row["stage"] == "adam_only":
                row["l2_relative_error"] = "0.2"
        with patch("experiments.pinn.verify_data.read_rows", return_value=changed):
            with self.assertRaisesRegex(ValueError, "differs from manuscript"):
                verify_ntk()

    def test_table2_verifier_rejects_changed_archived_metric(self):
        changed = [dict(row) for row in read_rows("table2_selected_runs.csv")]
        changed[0]["paper_l2_6dp"] = "999.000000"
        with patch("experiments.pinn.verify_data.read_rows", return_value=changed):
            with self.assertRaisesRegex(ValueError, "differs from manuscript"):
                verify_table2()
        changed = [dict(row) for row in read_rows("table2_selected_runs.csv")]
        changed[0]["paper_l2_6dp"] = "0.500000"
        changed[0]["wandb_final_relative_l2"] = "0.5"
        with patch("experiments.pinn.verify_data.read_rows", return_value=changed):
            with self.assertRaisesRegex(ValueError, "differs from manuscript"):
                verify_table2()
        changed = [dict(row) for row in read_rows("table2_selected_runs.csv")]
        changed[0]["wandb_final_relative_l2"] = "0.5"
        with patch("experiments.pinn.verify_data.read_rows", return_value=changed):
            with self.assertRaisesRegex(ValueError, "does not round to the manuscript cell"):
                verify_table2()


if __name__ == "__main__":
    unittest.main()
