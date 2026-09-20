"""Release asset lookup must not silently substitute a different base model."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from opengazelink_pc.personal_binocular_training import _public_files


class PublicModelAssetTests(unittest.TestCase):
    def test_explicit_directory_is_authoritative_even_when_assets_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"OPENGAZELINK_PUBLIC_MODEL_DIR": directory}):
                checkpoint, metadata = _public_files()
            self.assertEqual(checkpoint, Path(directory).resolve() / "best.pt")
            self.assertEqual(metadata, Path(directory).resolve() / "result.json")
            self.assertFalse(checkpoint.exists())

    def test_packaged_pair_is_used_without_a_development_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory) / "models" / "public-conditioned"
            assets.mkdir(parents=True)
            (assets / "best.pt").write_bytes(b"test-placeholder")
            (assets / "result.json").write_text("{}")
            with patch.dict(os.environ, {"OPENGAZELINK_PUBLIC_MODEL_DIR": ""}), \
                    patch("opengazelink_pc.personal_binocular_training.RESOURCE_ROOT", Path(directory)):
                self.assertEqual(_public_files(), (assets / "best.pt", assets / "result.json"))
