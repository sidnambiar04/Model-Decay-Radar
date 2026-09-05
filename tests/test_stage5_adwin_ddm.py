"""
Stage 5 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. ADWINDetector sliding window drift detection on error rates.
  2. DDMDetector state transitions (in_control -> warning -> drift).
  3. Supervised drift detectors non-blocking execution when labels are absent.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np

from drift_fusion import ADWINDetector, DDMDetector, DriftSignalFusionEngine


class TestStage5ADWINDDM(unittest.TestCase):
    def test_adwin_detector_error_shift(self):
        adwin = ADWINDetector(delta=0.01)
        # Nominal stream (low errors)
        for _ in range(50):
            adwin.update(0.0)

        # Shifted stream (high errors)
        drift_found = False
        for _ in range(50):
            if adwin.update(1.0):
                drift_found = True
                break
        self.assertTrue(drift_found)

    def test_ddm_detector_state_transitions(self):
        ddm = DDMDetector(min_num_instances=10)
        # Stable phase
        for _ in range(30):
            ddm.update(is_error=False)
        self.assertEqual(ddm.status, "in_control")

        # Escalating error phase
        for _ in range(30):
            ddm.update(is_error=True)
        self.assertIn(ddm.status, ["warning", "drift"])

    def test_fusion_engine_unsupervised_without_labels(self):
        fusion = DriftSignalFusionEngine()
        res = fusion.evaluate_fusion(
            mean_reconstruction_error=0.8,
            dynamic_threshold=0.3,
            observed_kl=0.5,
            p_value=0.001,
            wasserstein_distance=0.2,
            score_ks_p_value=0.001,
            feature_ks_results=[],
            batch_labels=None,
            batch_predictions=None,
        )
        self.assertTrue(res.drift_detected)
        self.assertFalse(res.detector_signals["supervised_drift"])


if __name__ == "__main__":
    unittest.main()
