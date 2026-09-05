"""
Stage 13 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. ModelRegistry SQLite database initialization.
  2. Model version tag incrementing (v1 -> v2 -> v3).
  3. Audit logging of promoted and rejected models.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import tempfile

from model_registry import ModelRegistry


class TestStage13VersioningSQLite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_registry.db")
        self.registry = ModelRegistry(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_version_tag_increment(self):
        # Default when empty
        self.assertEqual(self.registry.get_latest_version_tag(), "v1")
        self.assertEqual(self.registry.get_next_version_tag(), "v2")

        # Log v1 baseline
        self.registry.log_version(
            version_tag="v1",
            classifier_type="Random Forest",
            metrics={"accuracy": 0.95, "f1_score": 0.94},
            promotion_status="initial_baseline",
        )

        self.assertEqual(self.registry.get_latest_version_tag(), "v1")
        self.assertEqual(self.registry.get_next_version_tag(), "v2")

        # Log v2 promoted
        self.registry.log_version(
            version_tag="v2",
            classifier_type="Random Forest",
            metrics={"accuracy": 0.96, "f1_score": 0.95},
            promotion_status="promoted",
        )

        self.assertEqual(self.registry.get_latest_version_tag(), "v2")
        self.assertEqual(self.registry.get_next_version_tag(), "v3")

    def test_audit_logging_history(self):
        self.registry.log_version(
            version_tag="v1",
            classifier_type="Random Forest",
            metrics={"accuracy": 0.90, "f1_score": 0.89},
            promotion_status="initial_baseline",
        )
        self.registry.log_version(
            version_tag="v2",
            classifier_type="Random Forest",
            metrics={"accuracy": 0.60, "f1_score": 0.55},
            promotion_status="rejected",
            rejection_reason="Candidate F1 dropped below threshold.",
        )

        history = self.registry.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["version_tag"], "v2")
        self.assertEqual(history[0]["promotion_status"], "rejected")
        self.assertIn("dropped", history[0]["rejection_reason"])


if __name__ == "__main__":
    unittest.main()
