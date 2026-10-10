"""Reject coherent data/summary edits that no longer match the manuscript."""

import unittest
from unittest import mock

from figures import verify_fourier_data as verifier


class FourierVerificationTests(unittest.TestCase):
    def test_manuscript_constant_rejects_coherent_csv_change(self):
        runs = [dict(row) for row in verifier.read_csv("dense_fourier_d4_runs.csv")]
        table = [dict(row) for row in verifier.read_csv("dense_fourier_d4_table.csv")]
        selected = next(row for row in table
                        if row["activation"] == "identity" and row["depth"] == "1")
        for row in runs:
            if (row["activation"] == "identity" and row["depth"] == "1"
                    and float(row["lr"]) == float(selected["selected_lr"])):
                row["test_mse"] = str(float(row["test_mse"]) + 0.01)
        selected["test_mse_mean"] = str(float(selected["test_mse_mean"]) + 0.01)
        with mock.patch.object(verifier, "read_csv", side_effect=(runs, table)):
            with self.assertRaisesRegex(ValueError, "manuscript prints"):
                verifier.verify_dimension(4)

    def test_duplicate_summary_row_rejected(self):
        runs = verifier.read_csv("dense_fourier_d4_runs.csv")
        table = verifier.read_csv("dense_fourier_d4_table.csv")
        table[1] = table[0]
        with mock.patch.object(verifier, "read_csv", side_effect=(runs, table)):
            with self.assertRaisesRegex(ValueError, "duplicate summary rows"):
                verifier.verify_dimension(4)


if __name__ == "__main__":
    unittest.main()
