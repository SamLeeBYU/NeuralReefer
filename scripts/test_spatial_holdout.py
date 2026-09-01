"""Tests for depth-based FoV spatial hold-out helpers."""

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

# Stub torch before train.py imports it (scipy inspects torch.Tensor via issubclass).
_fake_torch = types.ModuleType("torch")
_fake_torch.Tensor = type("Tensor", (), {})
sys.modules["torch"] = _fake_torch

for _mod in ("data", "filter", "segmenter", "transforms", "skopt", "skopt.space"):
    sys.modules.setdefault(_mod, mock.MagicMock())

sys.path.insert(0, os.path.dirname(__file__))

from train import (
    assign_components_to_split,
    build_overlap_components,
    fov_radius_m,
    hold_out,
)


class TestFovRadius(unittest.TestCase):
    def test_depth_two_meters(self):
        r = fov_radius_m(2.0, hfov_deg=89.0)
        expected = 2.0 * np.tan(np.radians(89.0 / 2))
        self.assertAlmostEqual(r, expected, places=5)

    def test_negative_depth_uses_absolute(self):
        self.assertAlmostEqual(fov_radius_m(-2.0), fov_radius_m(2.0))

    def test_missing_depth(self):
        self.assertTrue(np.isnan(fov_radius_m(float("nan"))))
        self.assertTrue(np.isnan(fov_radius_m(None)))


class TestOverlapComponents(unittest.TestCase):
    def test_overlap_when_within_sum_of_radii(self):
        coords = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=float)
        radii = np.array([2.0, 2.0], dtype=float)
        comps = build_overlap_components(coords, radii)
        self.assertEqual(len(comps), 1)
        self.assertEqual(sorted(comps[0]), [0, 1])

    def test_no_overlap_when_beyond_sum_of_radii(self):
        coords = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=float)
        radii = np.array([2.0, 2.0], dtype=float)
        comps = build_overlap_components(coords, radii)
        self.assertEqual(len(comps), 2)
        sizes = sorted(len(c) for c in comps)
        self.assertEqual(sizes, [1, 1])


class TestAssignComponents(unittest.TestCase):
    def test_ten_singletons_target_two(self):
        components = [[i] for i in range(10)]
        rng = np.random.default_rng(42)
        test_set = assign_components_to_split(components, n_target_test=2, rng=rng)
        self.assertEqual(len(test_set), 2)


class TestHoldOutSanity(unittest.TestCase):
    def test_real_metadata_reaches_target_fraction(self):
        from config import METADATA, VAL_SIZE

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        metadata_path = os.path.join(repo_root, METADATA)
        split_path = os.path.join(repo_root, "data/performance/train_test_split_metadata.csv")
        if not os.path.exists(metadata_path) or not os.path.exists(split_path):
            self.skipTest("metadata or split snapshot not present")

        import pandas as pd

        split_df = pd.read_csv(split_path)
        images = [
            os.path.join(repo_root, p.replace("\\", os.sep))
            for p in split_df["filepath"]
        ]

        train, test = hold_out(
            images,
            val_size=VAL_SIZE,
            seed=42,
            metadata_path=metadata_path,
        )
        n = len(images)
        target = int(round(n * VAL_SIZE))
        self.assertEqual(len(train) + len(test), n)
        self.assertEqual(len(test), target)


if __name__ == "__main__":
    unittest.main()
