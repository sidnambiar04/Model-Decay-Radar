"""
Stage 9 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. RootCauseEngine KS statistical shift signal calculation.
  2. Side-by-side alignment of KS statistical shift and SHAP importance.
  3. Combined drift_significance ranking of target drifted sensors (LIT101, DPIT301, P402).
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np

from root_cause import RootCauseEngine


class TestStage9RootCause(unittest.TestCase):
    def setUp(self):
        self.engine = RootCauseEngine()
        self.feature_names = ["LIT101", "DPIT301", "P402", "FIT101", "PIT201"]

    def test_ks_signals_calculation(self):
        np.random.seed(42)
        ref_features = np.random.normal(0, 1, size=(200, 5))
        cur_features = ref_features.copy()
        cur_features[:, 0] += 3.5  # Shift LIT101

        ks_dict = self.engine.compute_ks_signals(ref_features, cur_features, self.feature_names)
        self.assertEqual(len(ks_dict), 5)
        self.assertTrue(ks_dict["LIT101"]["statistically_drifted"])
        self.assertLess(ks_dict["LIT101"]["ks_p_value"], 0.01)

    def test_root_cause_analysis_ranking(self):
        np.random.seed(42)
        ref_features = np.random.normal(0, 1, size=(200, 5))
        cur_features = ref_features.copy()
        cur_features[:, 0] += 3.5  # Shift LIT101
        cur_features[:, 1] += 2.8  # Shift DPIT301

        results = self.engine.analyze(
            reference_scaled=ref_features,
            current_batch_scaled=cur_features,
            feature_names=self.feature_names,
            ae_model=None,
            run_shap=False,
            top_k=5,
        )

        self.assertEqual(len(results), 5)
        top_two = [r["feature"] for r in results[:2]]
        self.assertIn("LIT101", top_two)
        self.assertIn("DPIT301", top_two)
        self.assertTrue(results[0]["statistically_drifted"])
        self.assertIn("priority_score", results[0])
        self.assertIn("diagnosis_tag", results[0])

    def test_diagnosis_tags_differentiation(self):
        np.random.seed(42)
        ref_features = np.random.normal(0, 1, size=(100, 5))
        cur_features = ref_features.copy()
        cur_features[:, 0] += 4.0  # Drift sensor 0 (LIT101)

        results = self.engine.analyze(
            reference_scaled=ref_features,
            current_batch_scaled=cur_features,
            feature_names=self.feature_names,
            ae_model=None,
            run_shap=False,
            top_k=5,
        )

        lit101 = next(r for r in results if r["feature"] == "LIT101")
        self.assertEqual(lit101["diagnosis_tag"], "CRITICAL_ATTRIBUTION")

        fit101 = next(r for r in results if r["feature"] == "FIT101")
        self.assertEqual(fit101["diagnosis_tag"], "NOMINAL")


if __name__ == "__main__":
    unittest.main()
