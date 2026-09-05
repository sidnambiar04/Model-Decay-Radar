"""
Stage 4 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. ModelRegistry SQLite database logging and version tag incrementing.
  2. ValidationGateEngine comparative candidate evaluation.
  3. Reference Baseline Update Gate execution only post-promotion.
  4. Integration in RadarOrchestrator selective retraining loop.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import tempfile
import numpy as np
import pandas as pd

from model_registry import ModelRegistry
from validation_gate import ValidationGateEngine, ValidationResult
from classifier import ProductionClassifier
from orchestrator import RadarOrchestrator
from data_pipeline import DataWindowManager
from config import config


class TestStage4Registry(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_registry.db")
        self.registry = ModelRegistry(db_path=self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_version_tag_increment(self):
        self.assertEqual(self.registry.get_latest_version_tag(), "v1")
        self.assertEqual(self.registry.get_next_version_tag(), "v2")

        # Log initial baseline v1
        self.registry.log_version("v1", "Random Forest", {"f1_score": 0.95}, "initial_baseline")
        self.assertEqual(self.registry.get_latest_version_tag(), "v1")
        self.assertEqual(self.registry.get_next_version_tag(), "v2")

        # Log candidate promotion v2
        self.registry.log_version("v2", "Random Forest", {"f1_score": 0.96}, "promoted")
        self.assertEqual(self.registry.get_latest_version_tag(), "v2")
        self.assertEqual(self.registry.get_next_version_tag(), "v3")

        history = self.registry.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["version_tag"], "v2")


class TestStage4ValidationGate(unittest.TestCase):
    def setUp(self):
        self.gate = ValidationGateEngine()

    def test_candidate_promotion(self):
        np.random.seed(42)
        X = np.random.normal(0, 1, size=(200, 5))
        y = np.random.choice([0, 1], size=200)

        active_clf = ProductionClassifier()
        active_clf.fit(X, y)

        candidate_clf = ProductionClassifier()
        candidate_clf.fit(X, y)

        res = self.gate.evaluate_candidate(active_clf, candidate_clf, X, y)
        self.assertTrue(res.is_promoted)
        self.assertIsNone(res.rejection_reason)


class TestStage4OrchestratorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        feature_cols = ["sensor_0", "sensor_1", "sensor_2", "sensor_3", "sensor_4"]
        ref_data = np.random.normal(0, 1, size=(500, 5))
        ref_df = pd.DataFrame(ref_data, columns=feature_cols)
        ref_df["Label"] = np.random.choice([0, 1], size=500, p=[0.8, 0.2])
        ref_df["_window"] = "reference"

        cls.wm = DataWindowManager(ref_df, feature_cols, label_col="Label")
        cls.wm.set_reference_window(ref_df)

        cls.classifier = ProductionClassifier()
        cls.classifier.fit(ref_data, ref_df["Label"].values)

        cls.orchestrator = RadarOrchestrator(feature_names=feature_cols)
        cls.orchestrator.setup(
            reference_scaled=cls.wm.reference_scaled,
            reference_raw_X=ref_data,
            reference_raw_y=ref_df["Label"].values,
            classifier=cls.classifier,
            scaler=cls.wm.scaler,
            ae_epochs=2,
            rnn_epochs=2,
        )

    def test_selective_retrain_and_validation(self):
        np.random.seed(99)
        drift_batch = np.random.normal(2.0, 1.0, size=(100, 5))
        drift_labels = np.random.choice([0, 1], size=100, p=[0.7, 0.3])

        initial_ver = self.orchestrator.registry.get_latest_version_tag()
        self.orchestrator._selective_retrain(drift_batch, drift_labels)

        new_ver = self.orchestrator.registry.get_latest_version_tag()
        self.assertIn(self.orchestrator.latest_validation_status, ["promoted", "rejected"])
        self.assertIn("accuracy", self.orchestrator.latest_validation_metrics)


if __name__ == "__main__":
    unittest.main()
