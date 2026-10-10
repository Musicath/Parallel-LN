"""Ensure the compact, privacy-screened release data have not changed silently."""

import ast
import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "DATA_MANIFEST.sha256"
DATA_DIRS = (ROOT / "experiments" / "pinn" / "data",
             ROOT / "experiments" / "vit" / "data",
             ROOT / "figures" / "data")


class ReleaseDataManifestTests(unittest.TestCase):
    def test_verifiers_keep_checks_under_python_optimized_mode(self):
        verifiers = (
            ROOT / "figures" / "verify_data.py",
            ROOT / "figures" / "verify_fourier_data.py",
            ROOT / "experiments" / "pinn" / "verify_data.py",
        )
        for path in verifiers:
            with self.subTest(path=path.relative_to(ROOT)):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)),
                                 "data checks must not disappear under python -O")

    def test_manifest_covers_each_shipped_csv_and_matches_bytes(self):
        listed = {}
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            digest, separator, relative_path = line.partition("  ")
            self.assertEqual(separator, "  ")
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(char in "0123456789abcdef" for char in digest))
            self.assertNotIn(relative_path, listed)
            listed[relative_path] = digest

        actual = {
            path.relative_to(ROOT).as_posix()
            for directory in DATA_DIRS
            for path in directory.rglob("*.csv")
        }
        self.assertEqual(set(listed), actual)
        for relative_path, expected in listed.items():
            with self.subTest(path=relative_path):
                data = (ROOT / relative_path).read_bytes()
                self.assertNotIn(b"\r\n", data)
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
