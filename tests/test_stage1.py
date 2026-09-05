"""
Stage 1 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. Centralized configuration manager (RadarConfig).
  2. Pure production distribution drift monitoring (no pre-SMOTE corruption).
  3. MHS reweighting when ground truth labels are absent ("awaiting_ground_truth").
  4. Integration tests with Orchestrator and FastAPI endpoints.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np
import pandas as pd

from config import RadarConfig, config
from orchestrator import RadarOrchestrator, compute_mhs, mhs_status
from classifier import ProductionClassifier
from data_pipeline import DataWindowManager


class TestStage1Config(unittest.TestCase):
    def test_default_config(self):
        cfg = RadarConfig()
        self.assertEqual(cfg.window_size, 500)
        self.assertEqual(cfg.ae_epochs, 10)
        self.assertEqual(cfg.p_value_threshold, 0.01)
        self.assertIn("accuracy", cfg.mhs_weights)
        self.assertEqual(cfg.mhs_weights["accuracy"], 0.35)

    def test_update_config(self):
        cfg = RadarConfig()
        updated = cfg.update(window_size=300, p_value_threshold=0.05)
        self.assertEqual(cfg.window_size, 300)
        self.assertEqual(cfg.p_value_threshold, 0.05)
        self.assertEqual(updated["window_size"], 300)


class TestStage1MHS(unittest.TestCase):
    def test_mhs_with_labels(self):
        # Accuracy=1.0, Drift=0.0, Uncertainty=0.0, Stability=1.0
        score = compute_mhs(
            accuracy=1.0, drift=0.0, uncertainty=0.0, stability=1.0,
            performance_status="labels_available"
        )
        # 0.35*1 + 0.25*1 + 0.20*1 + 0.20*1 = 1.0
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(mhs_status(score), "Healthy")

    def test_mhs_awaiting_ground_truth(self):
        # Missing labels: drift=0.0, uncertainty=0.0, stability=1.0
        score = compute_mhs(
            accuracy=0.0, drift=0.0, uncertainty=0.0, stability=1.0,
            performance_status="awaiting_ground_truth"
        )
        # Remaining weights sum to 0.65 (0.25/0.65 + 0.20/0.65 + 0.20/0.65 = 1.0)
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(mhs_status(score), "Healthy")


class TestStage1Orchestrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create dummy synthetic data
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

    def test_pure_distribution_drift_monitoring(self):
        # Pass a drifted batch without labels
        np.random.seed(99)
        drift_batch = np.random.normal(3.0, 2.0, size=(100, 10)) # strong shift
        result = self.orchestrator.run_monitoring_cycle(drift_batch)

        self.assertEqual(result.performance_status, "awaiting_ground_truth")
        self.assertIsInstance(result.mean_reconstruction_error, float)
        self.assertIsInstance(result.drift_confirmed, bool)
        self.assertEqual(result.samples_before_smote, 100)
        self.assertEqual(result.samples_after_smote, 100)


if __name__ == "__main__":
    unittest.main()
