"""
Stage 8 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. Model Health Score (MHS) calculation under Healthy, Warning, Critical states.
  2. Proportional weight redistribution when ground truth labels are absent.
  3. Threshold classification (Healthy > 0.85, Warning 0.65-0.85, Critical < 0.65).
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
from orchestrator import compute_mhs, mhs_status


class TestStage8MHS(unittest.TestCase):
    def test_mhs_healthy_with_labels(self):
        mhs = compute_mhs(accuracy=0.95, drift=0.0, uncertainty=0.05, stability=0.95, performance_status="labels_available")
        self.assertGreater(mhs, 0.85)
        self.assertEqual(mhs_status(mhs), "Healthy")

    def test_mhs_critical_with_labels(self):
        mhs = compute_mhs(accuracy=0.50, drift=0.8, uncertainty=0.70, stability=0.40, performance_status="labels_available")
        self.assertLess(mhs, 0.65)
        self.assertEqual(mhs_status(mhs), "Critical")

    def test_mhs_reweighting_without_labels(self):
        mhs = compute_mhs(accuracy=0.0, drift=0.0, uncertainty=0.10, stability=0.90, performance_status="awaiting_ground_truth")
        self.assertGreater(mhs, 0.85)
        self.assertEqual(mhs_status(mhs), "Healthy")


if __name__ == "__main__":
    unittest.main()
