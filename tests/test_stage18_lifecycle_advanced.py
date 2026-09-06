"""
Stage 18 Test Suite: Advanced Model Lifecycle — Experiment Tracking, Training Lineage, and Retraining Cooldown
==============================================================================================================
Tests Member 2 Round 2 enhancements:
  - ExperimentTracker structured logging and summary aggregation
  - ModelRegistry training lineage table recording and retrieval
  - RetrainingCooldownManager enforcement and expiry
  - Full retrain→experiment→lineage integration flow
"""

import os
import sys
import tempfile
import time
import unittest
import numpy as np

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))

from experiment_tracker import ExperimentTracker
from model_registry import ModelRegistry
from validation_gate import RetrainingCooldownManager
from classifier import ProductionClassifier
from validation_gate import ValidationGateEngine


class TestStage18ExperimentTracker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_experiments.db")
        self.tracker = ExperimentTracker(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_experiment_tracker_logging(self):
        """Test that a retrain experiment is correctly recorded and retrievable."""
        ctx = self.tracker.start_experiment(
            trigger_reason="MHS Critical at batch 5",
            classifier_type="Random Forest",
            batch_id=5,
        )
        self.assertIn("experiment_id", ctx)
        self.assertIn("_start_time_ns", ctx)

        record = self.tracker.log_experiment(
            context=ctx,
            sample_count=2000,
            reference_samples=1600,
            drift_samples=400,
            smote_applied=True,
            class_balance_before={0: 380, 1: 20},
            class_balance_after={0: 380, 1: 380},
            before_metrics={"accuracy": 0.92, "f1_score": 0.88, "precision": 0.90, "recall": 0.86},
            after_metrics={"accuracy": 0.95, "f1_score": 0.93, "precision": 0.94, "recall": 0.92},
            promotion_decision="promoted",
            version_tag="v2",
        )

        self.assertEqual(record["promotion_decision"], "promoted")
        self.assertEqual(record["sample_count"], 2000)
        self.assertTrue(record["smote_applied"])
        self.assertIn("accuracy_delta", record["delta_metrics"])
        self.assertAlmostEqual(record["delta_metrics"]["accuracy_delta"], 0.03, places=3)
        self.assertGreater(record["duration_ms"], 0)

        # Verify retrieval
        history = self.tracker.get_experiment_history(limit=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["experiment_id"], record["experiment_id"])

    def test_experiment_tracker_summary(self):
        """Test aggregation summary across multiple experiments."""
        # Log a promoted experiment
        ctx1 = self.tracker.start_experiment("MHS Critical", "Random Forest", batch_id=1)
        self.tracker.log_experiment(
            context=ctx1, sample_count=1000, reference_samples=800, drift_samples=200,
            smote_applied=False,
            class_balance_before={0: 100, 1: 100}, class_balance_after={0: 100, 1: 100},
            before_metrics={"accuracy": 0.90, "f1_score": 0.85},
            after_metrics={"accuracy": 0.93, "f1_score": 0.91},
            promotion_decision="promoted", version_tag="v2",
        )

        # Log a rejected experiment
        ctx2 = self.tracker.start_experiment("MHS Critical", "Gradient Boosting", batch_id=2)
        self.tracker.log_experiment(
            context=ctx2, sample_count=800, reference_samples=640, drift_samples=160,
            smote_applied=True,
            class_balance_before={0: 150, 1: 10}, class_balance_after={0: 150, 1: 150},
            before_metrics={"accuracy": 0.90, "f1_score": 0.85},
            after_metrics={"accuracy": 0.70, "f1_score": 0.60},
            promotion_decision="rejected", rejection_reason="F1 drop below threshold",
        )

        summary = self.tracker.get_experiment_summary()
        self.assertEqual(summary["total_experiments"], 2)
        self.assertEqual(summary["promoted_count"], 1)
        self.assertEqual(summary["rejected_count"], 1)
        self.assertAlmostEqual(summary["promotion_rate"], 0.5, places=2)
        self.assertGreater(summary["avg_duration_ms"], 0)


class TestStage18TrainingLineage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_registry.db")
        self.registry = ModelRegistry(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_training_lineage_recording(self):
        """Test that training lineage is correctly stored and retrieved."""
        lineage = self.registry.log_training_lineage(
            version_tag="v2",
            reference_samples=1600,
            drift_samples=400,
            smote_applied=True,
            imbalance_ratio_before=0.05,
            imbalance_ratio_after=1.0,
            training_duration_ms=1250.5,
            drift_event_batch_id=5,
            notes="Retrain triggered by Critical MHS at batch 5",
        )

        self.assertEqual(lineage["version_tag"], "v2")
        self.assertEqual(lineage["reference_samples"], 1600)
        self.assertEqual(lineage["drift_samples"], 400)
        self.assertEqual(lineage["total_samples"], 2000)
        self.assertTrue(lineage["smote_applied"])
        self.assertAlmostEqual(lineage["imbalance_ratio_before"], 0.05)
        self.assertAlmostEqual(lineage["imbalance_ratio_after"], 1.0)

        # Verify retrieval
        retrieved = self.registry.get_training_lineage("v2")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["version_tag"], "v2")
        self.assertTrue(retrieved["smote_applied"])
        self.assertEqual(retrieved["total_samples"], 2000)

    def test_training_lineage_not_found(self):
        """Test that querying a non-existent version returns None."""
        result = self.registry.get_training_lineage("v999")
        self.assertIsNone(result)


class TestStage18RetrainingCooldown(unittest.TestCase):
    def test_retraining_cooldown_enforcement(self):
        """Test that the cooldown blocks rapid-fire retraining."""
        manager = RetrainingCooldownManager(cooldown_seconds=5, max_per_hour=3)

        # First retrain should be allowed
        allowed, reason = manager.can_retrain()
        self.assertTrue(allowed)
        manager.record_retrain()

        # Immediately after, should be blocked by cooldown
        allowed, reason = manager.can_retrain()
        self.assertFalse(allowed)
        self.assertIn("Cooldown active", reason)

    def test_retraining_cooldown_expiry(self):
        """Test that cooldown expires after the configured period."""
        manager = RetrainingCooldownManager(cooldown_seconds=1, max_per_hour=100)

        allowed, _ = manager.can_retrain()
        self.assertTrue(allowed)
        manager.record_retrain()

        # Should be blocked immediately
        allowed, _ = manager.can_retrain()
        self.assertFalse(allowed)

        # Wait for cooldown to expire
        time.sleep(1.1)

        # Should be allowed again
        allowed, reason = manager.can_retrain()
        self.assertTrue(allowed)
        self.assertIn("allowed", reason.lower())

    def test_retraining_hourly_rate_limit(self):
        """Test that the hourly rate limit is enforced."""
        manager = RetrainingCooldownManager(cooldown_seconds=0, max_per_hour=2)

        # First two should be allowed
        allowed, _ = manager.can_retrain()
        self.assertTrue(allowed)
        manager.record_retrain()

        allowed, _ = manager.can_retrain()
        self.assertTrue(allowed)
        manager.record_retrain()

        # Third should be blocked
        allowed, reason = manager.can_retrain()
        self.assertFalse(allowed)
        self.assertIn("rate limit", reason.lower())

    def test_cooldown_status_reporting(self):
        """Test that get_status returns correct monitoring data."""
        manager = RetrainingCooldownManager(cooldown_seconds=60, max_per_hour=5)
        status = manager.get_status()

        self.assertIn("cooldown_remaining_seconds", status)
        self.assertIn("retrains_last_hour", status)
        self.assertEqual(status["retrains_last_hour"], 0)
        self.assertEqual(status["max_per_hour"], 5)


class TestStage18FullIntegrationFlow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.exp_db = os.path.join(self.temp_dir.name, "experiments.db")
        self.reg_db = os.path.join(self.temp_dir.name, "registry.db")

        # Generate synthetic separable 2-class dataset
        np.random.seed(42)
        self.X_train = np.random.normal(0, 1, size=(200, 10)).astype(np.float32)
        self.y_train = np.random.choice([0, 1], size=200, p=[0.5, 0.5]).astype(np.int32)
        self.X_train[self.y_train == 1] += 3.0

        self.X_val = np.random.normal(0, 1, size=(100, 10)).astype(np.float32)
        self.y_val = np.random.choice([0, 1], size=100, p=[0.5, 0.5]).astype(np.int32)
        self.X_val[self.y_val == 1] += 3.0

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_retrain_experiment_flow(self):
        """Test complete flow: train → experiment track → validate → log lineage."""
        tracker = ExperimentTracker(db_path=self.exp_db)
        registry = ModelRegistry(db_path=self.reg_db)
        gate = ValidationGateEngine()

        # Train active model
        active = ProductionClassifier()
        active.fit(self.X_train, self.y_train, model_name="Random Forest")

        # Log baseline
        registry.log_version("v1", "Random Forest",
                             {"accuracy": 0.95, "f1_score": 0.93, "precision": 0.94, "recall": 0.92},
                             "initial_baseline")

        # Start experiment
        ctx = tracker.start_experiment("MHS Critical at batch 3", "Random Forest", batch_id=3)

        # Get before metrics
        before_metrics = gate.evaluate_model_metrics(active, self.X_val, self.y_val)

        # Train candidate (same data, should produce similar results)
        candidate = ProductionClassifier()
        candidate.fit(self.X_train, self.y_train, model_name="Random Forest")

        # Validate
        val_result = gate.evaluate_candidate(active, candidate, self.X_val, self.y_val)

        # Log to registry
        next_tag = registry.get_next_version_tag()
        status = "promoted" if val_result.is_promoted else "rejected"
        registry.log_version(
            version_tag=next_tag,
            classifier_type="Random Forest",
            metrics=val_result.candidate_metrics,
            promotion_status=status,
        )

        # Log lineage
        if val_result.is_promoted:
            registry.log_training_lineage(
                version_tag=next_tag,
                reference_samples=160,
                drift_samples=40,
                smote_applied=False,
                imbalance_ratio_before=1.0,
                imbalance_ratio_after=1.0,
                drift_event_batch_id=3,
            )

        # Log experiment
        record = tracker.log_experiment(
            context=ctx,
            sample_count=200,
            reference_samples=160,
            drift_samples=40,
            smote_applied=False,
            class_balance_before={0: 100, 1: 100},
            class_balance_after={0: 100, 1: 100},
            before_metrics=before_metrics,
            after_metrics=val_result.candidate_metrics,
            promotion_decision=status,
            version_tag=next_tag if val_result.is_promoted else None,
        )

        # Verify everything connected
        self.assertEqual(record["promotion_decision"], status)
        self.assertGreater(record["duration_ms"], 0)

        exp_history = tracker.get_experiment_history()
        self.assertEqual(len(exp_history), 1)

        reg_history = registry.get_history()
        self.assertGreaterEqual(len(reg_history), 2)  # v1 baseline + retrain result

        summary = tracker.get_experiment_summary()
        self.assertEqual(summary["total_experiments"], 1)


if __name__ == "__main__":
    unittest.main()
