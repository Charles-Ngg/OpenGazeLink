from __future__ import annotations

import unittest

from opengazelink_pc.model_registry import ModelRegistry


class ModelRegistryStatusTest(unittest.TestCase):
    def test_status_is_cached_until_registry_is_cleared(self) -> None:
        registry = ModelRegistry()
        first = registry.status()
        second = registry.status()
        self.assertIs(first, second)

        registry.clear()
        third = registry.status()
        self.assertIsNot(first, third)
        self.assertEqual(first["datasets"]["tasks"]["samples"], third["datasets"]["tasks"]["samples"])


if __name__ == "__main__":
    unittest.main()
