"""
Stage 7 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. DriftSignalFusionEngine multi-detector weighted voting.
  2. Automated drift taxonomy classification (gradual, sudden, concept_drift, covariate_drift).
  3. Isolated false alarm suppression.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np

from drift_fusion import DriftSignalFusionEngine


class TestStage7FusionEngine(unittest.TestCase):
    def setUp(self):
        self.fusion = DriftSignalFusionEngine()

    def test_nominal_state_no_drift(self):
        res = self.fusion.evaluate_fusion(
            mean_reconstruction_error=0.1,
            dynamic_threshold=0.5,
            observed_kl=0.01,
            p_value=0.50,
            wasserstein_distance=0.02,
            score_ks_p_value=0.50,
            feature_ks_results=[],
        )
        self.assertFalse(res.drift_detected)
        self.assertEqual(res.drift_type, "none")
        self.assertLess(res.fusion_score, 0.35)

    def test_concept_drift_classification(self):
        labels = np.array([0, 1, 0, 1, 0, 1] * 20)
        preds = np.array([1, 0, 1, 0, 1, 0] * 20)  # High error rate (0% accuracy)

        res = self.fusion.evaluate_fusion(
            mean_reconstruction_error=0.8,
            dynamic_threshold=0.3,
            observed_kl=0.5,
            p_value=0.001,
            wasserstein_distance=0.25,
            score_ks_p_value=0.001,
            feature_ks_results=[{"feature": "LIT101", "drift_confirmed": True}],
            batch_labels=labels,
            batch_predictions=preds,
        )
        self.assertTrue(res.drift_detected)
        self.assertEqual(res.drift_type, "concept_drift")
        self.assertGreaterEqual(res.drift_severity, 0.5)

    def test_sudden_drift_classification(self):
        res = self.fusion.evaluate_fusion(
            mean_reconstruction_error=2.5,
            dynamic_threshold=0.3,
            observed_kl=1.2,
            p_value=0.0001,
            wasserstein_distance=0.45,
            score_ks_p_value=0.0001,
            feature_ks_results=[{"feature": f"f_{i}", "drift_confirmed": True} for i in range(10)],
        )
        self.assertTrue(res.drift_detected)
        self.assertEqual(res.drift_type, "sudden")

    def test_covariate_drift_classification(self):
        # Feature shift occurred (high KS shift ratio, Wasserstein), but accuracy is 100% stable
        labels = np.array([0, 1, 0, 1] * 20)
        preds = np.array([0, 1, 0, 1] * 20)  # 100% accuracy, no error degradation

        res = self.fusion.evaluate_fusion(
            mean_reconstruction_error=0.4,
            dynamic_threshold=0.3,
            observed_kl=0.05,
            p_value=0.04,
            wasserstein_distance=0.30,
            score_ks_p_value=0.001,
            feature_ks_results=[{"feature": f"sensor_{i}", "drift_confirmed": True} for i in range(5)],
            batch_labels=labels,
            batch_predictions=preds,
        )
        self.assertTrue(res.drift_detected)
        self.assertEqual(res.drift_type, "covariate_drift")
        self.assertIn("classification_evidence", res.detector_signals)
        self.assertEqual(res.classification_evidence["condition_matched"], "feature_shift_stable_labels")

    def test_novelty_warning_classification(self):
        # No distribution drift on VAE/KL, but high epistemic uncertainty
        res = self.fusion.evaluate_fusion(
            mean_reconstruction_error=0.1,
            dynamic_threshold=0.5,
            observed_kl=0.01,
            p_value=0.50,
            wasserstein_distance=0.02,
            score_ks_p_value=0.50,
            feature_ks_results=[],
            mean_uncertainty=0.85,
        )
        self.assertTrue(res.drift_detected)
        self.assertEqual(res.drift_type, "novelty_warning")
        self.assertEqual(res.classification_evidence["condition_matched"], "epistemic_uncertainty_spike")


if __name__ == "__main__":
    unittest.main()
