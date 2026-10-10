"""Guard the release path against reintroducing private framework imports."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (ROOT / "experiments", ROOT / "figures")
PRIVATE_IMPORT_ROOTS = {"extension", "musicath_base", "soft_norm"}


class SourceIndependenceTests(unittest.TestCase):
    def test_no_private_framework_imports(self):
        paths = [ROOT / "pln.py", ROOT / "example.py"]
        paths.extend(path for directory in SOURCE_DIRS for path in directory.rglob("*.py"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = (alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        names = (node.module or "",)
                    else:
                        continue
                    for name in names:
                        self.assertNotIn(name.split(".", 1)[0], PRIVATE_IMPORT_ROOTS)


if __name__ == "__main__":
    unittest.main()
