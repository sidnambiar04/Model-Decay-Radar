"""
Stage 4 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. Wasserstein distance computation on calibration scores.
  2. Per-feature Kolmogorov-Smirnov (KS) test detection of drifted sensors.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np

from calibration import (
    permutation_drift_test,
    compute_feature_ks_tests,
    wasserstein_distance,
)


class TestStage4WassersteinKS(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.feature_names = ["LIT101", "DPIT301", "P402", "FIT101", "PIT201"]

    def test_wasserstein_distance_calculation(self):
        ref_scores = np.random.uniform(0, 0.3, size=200)
        cur_scores = np.random.uniform(0.6, 1.0, size=200)

        wd = wasserstein_distance(ref_scores, cur_scores)
        self.assertGreater(wd, 0.3)

    def test_feature_ks_tests_drift_detection(self):
        ref_features = np.random.normal(0, 1, size=(200, 5))
        cur_features = ref_features.copy()

        # Inject intentional drift into LIT101 (index 0) and DPIT301 (index 1)
        cur_features[:, 0] += 3.5
        cur_features[:, 1] += 2.8

        results = compute_feature_ks_tests(
            reference_features=ref_features,
            current_features=cur_features,
            feature_names=self.feature_names,
            alpha=0.05,
        )

        self.assertEqual(len(results), 5)
        top_two = [r["feature"] for r in results[:2]]
        self.assertIn("LIT101", top_two)
        self.assertIn("DPIT301", top_two)
        
        lit_res = next(r for r in results if r["feature"] == "LIT101")
        self.assertTrue(lit_res["drift_confirmed"])
        self.assertLess(lit_res["p_value"], 0.01)


if __name__ == "__main__":
    unittest.main()
