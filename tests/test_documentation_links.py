"""Keep repository-local links in release documentation usable."""

from pathlib import Path
import re
import unittest
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)]+)\)")


class DocumentationLinkTests(unittest.TestCase):
    def test_local_links_resolve_inside_repository(self) -> None:
        documents = [ROOT / "README.md", ROOT / "THIRD_PARTY_NOTICES.md"]
        documents.extend(sorted((ROOT / "docs").glob("*.md")))
        documents.extend(sorted((ROOT / "experiments").glob("*/README.md")))

        checked = 0
        for document in documents:
            with self.subTest(document=document.relative_to(ROOT)):
                self.assertTrue(document.is_file())
            for match in LINK.finditer(document.read_text(encoding="utf-8")):
                reference = match.group(1).split("#", 1)[0]
                if not reference or reference.startswith(("https://", "http://", "mailto:")):
                    continue
                target = (document.parent / unquote(reference)).resolve()
                with self.subTest(document=document.relative_to(ROOT), link=reference):
                    self.assertTrue(target.is_relative_to(ROOT))
                    self.assertTrue(target.is_file())
                checked += 1

        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
