"""Tests package imports and removed legacy API cleanup."""

import importlib
from unittest import TestCase


class PackageImportTests(TestCase):
    """Covers package import smoke tests and removed exports."""

    def test_src_import_surface_loads_without_legacy_module(self) -> None:
        """Importing the top-level package succeeds after legacy removal."""

        src = importlib.import_module("src")

        self.assertIsNotNone(src.load_image)
        self.assertIsNotNone(src.population)
        self.assertIsNotNone(src.rendering)
        self.assertIsNotNone(src.ga)

    def test_ga_package_no_longer_exports_legacy(self) -> None:
        """The GA package surface no longer exposes deleted legacy symbols."""

        ga = importlib.import_module("src.ga")

        self.assertFalse(hasattr(ga, "legacy"))
        self.assertFalse(hasattr(ga, "LegacyPipelineConfig"))
        self.assertFalse(hasattr(ga, "LegacyRunResult"))
        self.assertFalse(hasattr(ga, "run_legacy_pipeline"))
