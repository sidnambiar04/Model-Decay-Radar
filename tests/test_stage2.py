"""
Stage 2 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. ADWIN & DDM streaming drift detectors.
  2. DriftSignalFusionEngine signal evaluation and weighted fusion scoring.
  3. Classification of drift types (gradual, sudden, concept_drift, covariate_drift).
  4. Integration within RadarOrchestrator monitoring cycle.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np
import pandas as pd

from drift_fusion import ADWINDetector, DDMDetector, DriftSignalFusionEngine, FusionResult
from orchestrator import RadarOrchestrator
from data_pipeline import DataWindowManager
from config import config


class TestStage2Detectors(unittest.TestCase):
    def test_adwin_detector(self):
        adwin = ADWINDetector()
        # Stable 0 errors
        for _ in range(50):
            adwin.update(0.0)
        self.assertFalse(adwin.drift_detected)

        # Sudden jump to errors
        drift_found = False
        for _ in range(100):
            if adwin.update(1.0):
                drift_found = True
                break
        self.assertTrue(drift_found)

    def test_ddm_detector(self):
        ddm = DDMDetector()
        # Stable sequence
        for _ in range(50):
            ddm.update(is_error=False)
        self.assertEqual(ddm.status, "in_control")

        # Ingest stream of errors
        for _ in range(50):
            ddm.update(is_error=True)
        self.assertIn(ddm.status, ["warning", "drift"])


class TestStage2FusionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DriftSignalFusionEngine()

    def test_fusion_no_drift(self):
        res = self.engine.evaluate_fusion(
            mean_reconstruction_error=0.1,
            dynamic_threshold=0.5,
            observed_kl=0.01,
            p_value=0.5,
            wasserstein_distance=0.02,
            score_ks_p_value=0.6,
            feature_ks_results=[{"feature": "f1", "drift_confirmed": False}],
        )
        self.assertFalse(res.drift_detected)
        self.assertEqual(res.drift_type, "none")
        self.assertLess(res.fusion_score, 0.35)

    def test_fusion_sudden_drift(self):
        res = self.engine.evaluate_fusion(
            mean_reconstruction_error=1.5,
            dynamic_threshold=0.5,
            observed_kl=0.8,
            p_value=0.001,
            wasserstein_distance=0.35,
            score_ks_p_value=0.001,
            feature_ks_results=[{"feature": f"f{i}", "drift_confirmed": True} for i in range(10)],
        )
        self.assertTrue(res.drift_detected)
        self.assertIn(res.drift_type, ["sudden", "gradual", "concept_drift"])
        self.assertGreater(res.drift_severity, 0.5)

    def test_fusion_concept_drift(self):
        labels = np.array([0, 1] * 50)
        preds  = np.array([1, 0] * 50) # 100% misclassification
        res = self.engine.evaluate_fusion(
            mean_reconstruction_error=1.2,
            dynamic_threshold=0.5,
            observed_kl=0.6,
            p_value=0.001,
            wasserstein_distance=0.25,
            score_ks_p_value=0.001,
            feature_ks_results=[{"feature": "f1", "drift_confirmed": True}],
            batch_labels=labels,
            batch_predictions=preds,
        )
        self.assertTrue(res.drift_detected)
        self.assertEqual(res.drift_type, "concept_drift")


class TestStage2OrchestratorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        feature_cols = [f"sensor_{i:02d}" for i in range(10)]
        ref_data = np.random.normal(0, 1, size=(500, 10))
        ref_df = pd.DataFrame(ref_data, columns=feature_cols)
        ref_df["Label"] = np.random.choice([0, 1], size=500, p=[0.95, 0.05])
        ref_df["_window"] = "reference"

        cls.wm = DataWindowManager(ref_df, feature_cols, label_col="Label")
        cls.wm.set_reference_window(ref_df)

        cls.orchestrator = RadarOrchestrator(feature_names=feature_cols)
        cls.orchestrator.setup(
            cls.wm.reference_scaled,
            ae_epochs=2,
            rnn_epochs=2,
        )

    def test_orchestrator_returns_fusion_result(self):
        np.random.seed(99)
        drift_batch = np.random.normal(2.5, 1.5, size=(100, 10))
        result = self.orchestrator.run_monitoring_cycle(drift_batch)

        self.assertIsNotNone(result.drift_type)
        self.assertIsInstance(result.fusion_score, float)
        self.assertIn("vae_loss_drift", result.detector_signals)
        self.assertIn("kl_permutation_drift", result.detector_signals)


if __name__ == "__main__":
    unittest.main()
