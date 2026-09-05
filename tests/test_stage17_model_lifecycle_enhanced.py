"""
Stage 17 Test Suite: Enhanced Model Lifecycle, Artifact Management, and Rollback
===================================================================================
Tests Member 2 enhancements:
  - ProductionClassifier binary serialization (.joblib) and metadata extraction
  - ModelRegistry artifact path auto-migration & logging
  - ModelRegistry version rollback capability
  - ValidationGateEngine delta calculation and direct registry integration
  - Imbalance handler metadata logging
"""

import os
import sys
import tempfile
import unittest
import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))

from classifier import ProductionClassifier
from model_registry import ModelRegistry
from validation_gate import ValidationGateEngine, ValidationResult
from imbalance_handler import get_imbalance_metadata, apply_borderline_smote_if_needed


class TestStage17ModelLifecycleEnhanced(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_registry.db")
        self.artifact_dir = os.path.join(self.temp_dir.name, "artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)

        # Generate synthetic 2-class dataset
        np.random.seed(42)
        self.X_train = np.random.normal(0, 1, size=(200, 10)).astype(np.float32)
        self.y_train = np.random.choice([0, 1], size=200, p=[0.7, 0.3]).astype(np.int32)
        
        self.X_val = np.random.normal(0, 1, size=(100, 10)).astype(np.float32)
        self.y_val = np.random.choice([0, 1], size=100, p=[0.7, 0.3]).astype(np.int32)

        self.classifier = ProductionClassifier()
        self.classifier.fit(self.X_train, self.y_train, model_name="Random Forest")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_classifier_serialization_and_metadata(self):
        """Test binary saving, loading, and metadata extraction of ProductionClassifier."""
        meta = self.classifier.get_model_metadata("Random Forest")
        self.assertEqual(meta["model_name"], "Random Forest")
        self.assertTrue(meta["is_fitted"])
        self.assertIn("hyperparameters", meta)

        artifact_file = os.path.join(self.artifact_dir, "rf_v1.joblib")
        saved_path = self.classifier.save_model_artifact(artifact_file, "Random Forest")
        self.assertTrue(os.path.exists(saved_path))

        # Re-instantiate fresh classifier and restore artifact
        new_classifier = ProductionClassifier()
        loaded_name = new_classifier.load_model_artifact(saved_path)
        self.assertEqual(loaded_name, "Random Forest")
        self.assertEqual(new_classifier.active_model_name, "Random Forest")

        # Test prediction equality
        preds_orig = self.classifier.predict_batch(self.X_val)
        preds_loaded = new_classifier.predict_batch(self.X_val)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_model_registry_artifact_and_rollback(self):
        """Test ModelRegistry artifact schema, version logging, and version rollback."""
        registry = ModelRegistry(db_path=self.db_path)
        self.assertEqual(registry.get_latest_version_tag(), "v1")

        artifact_file = os.path.join(self.artifact_dir, "v1.joblib")
        self.classifier.save_model_artifact(artifact_file)

        metrics_v1 = {"accuracy": 0.90, "precision": 0.88, "recall": 0.85, "f1_score": 0.865}
        registry.log_version(
            version_tag="v1",
            classifier_type="Random Forest",
            metrics=metrics_v1,
            promotion_status="initial_baseline",
            artifact_path=artifact_file,
        )

        details = registry.get_version_details("v1")
        self.assertIsNotNone(details)
        self.assertEqual(details["version_tag"], "v1")
        self.assertEqual(details["artifact_path"], artifact_file)

        # Log a degraded v2 candidate
        metrics_v2 = {"accuracy": 0.70, "precision": 0.65, "recall": 0.60, "f1_score": 0.624}
        registry.log_version(
            version_tag="v2",
            classifier_type="Gradient Boosting",
            metrics=metrics_v2,
            promotion_status="promoted",
            artifact_path=None,
        )
        self.assertEqual(registry.get_latest_version_tag(), "v2")

        # Rollback to v1
        rollback_res = registry.rollback_to_version("v1", classifier_engine=self.classifier)
        self.assertEqual(rollback_res["promotion_status"], "rollback_promoted")
        self.assertEqual(self.classifier.active_model_name, "Random Forest")

    def test_validation_gate_enhanced_deltas(self):
        """Test ValidationGateEngine delta calculations and dictionary serialization."""
        gate = ValidationGateEngine()
        res = gate.evaluate_candidate(self.classifier, self.classifier, self.X_val, self.y_val)

        self.assertTrue(res.is_promoted)
        self.assertIn("accuracy_delta", res.deltas)
        self.assertEqual(res.deltas["f1_delta"], 0.0)

        res_dict = res.to_dict()
        self.assertEqual(res_dict["promotion_status"], "promoted")
        self.assertIn("deltas", res_dict)

    def test_validation_gate_and_registry_integration(self):
        """Test evaluate_and_log_registry workflow."""
        registry = ModelRegistry(db_path=self.db_path)
        gate = ValidationGateEngine()

        candidate = ProductionClassifier()
        candidate.fit(self.X_train, self.y_train, model_name="Gradient Boosting")

        val_res = gate.evaluate_and_log_registry(
            active_classifier=self.classifier,
            candidate_classifier=candidate,
            val_X=self.X_val,
            val_y=self.y_val,
            registry=registry,
        )
        self.assertIsInstance(val_res, ValidationResult)
        history = registry.get_history(limit=5)
        self.assertGreater(len(history), 0)

    def test_imbalance_metadata_audit(self):
        """Test get_imbalance_metadata statistics."""
        # Create highly imbalanced labels
        y_imbalanced = np.array([0] * 95 + [1] * 5, dtype=np.int32)
        X_dummy = np.random.normal(0, 1, size=(100, 10))

        meta = get_imbalance_metadata(X_dummy, y_imbalanced, imbalance_threshold=0.2)
        self.assertTrue(meta["is_imbalanced"])
        self.assertLess(meta["imbalance_ratio"], 0.20)
        self.assertEqual(meta["class_distribution"][1], 5)


if __name__ == "__main__":
    unittest.main()
