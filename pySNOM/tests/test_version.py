import re
import unittest
from pathlib import Path

import pySNOM


class TestVersionMetadata(unittest.TestCase):
    def test_conda_version_matches_package_version(self):
        meta = (
            Path(pySNOM.__file__).resolve().parents[1] / "conda" / "meta.yaml"
        ).read_text()
        match = re.search(r'{% set version = "([^"]+)" %}', meta)

        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), pySNOM.__version__)
