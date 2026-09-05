"""
Stage 12 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. ValidationGateEngine metric calculation.
  2. Candidate model promotion when criteria are satisfied.
  3. Candidate model rejection with reason logging when metrics degrade.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np

from validation_gate import ValidationGateEngine
from classifier import ProductionClassifier


class TestStage12ValidationGate(unittest.TestCase):
    def setUp(self):
        self.gate = ValidationGateEngine()
        np.random.seed(42)
        self.X = np.random.normal(0, 1, size=(200, 5))
        self.y = np.random.choice([0, 1], size=200, p=[0.8, 0.2])

    def test_candidate_promotion_success(self):
        active_clf = ProductionClassifier()
        active_clf.fit(self.X, self.y)

        candidate_clf = ProductionClassifier()
        candidate_clf.fit(self.X, self.y)

        res = self.gate.evaluate_candidate(active_clf, candidate_clf, self.X, self.y)
        self.assertTrue(res.is_promoted)
        self.assertIsNone(res.rejection_reason)

    def test_candidate_rejection_degraded(self):
        active_clf = ProductionClassifier()
        active_clf.fit(self.X, self.y)

        # Create dummy candidate model with random labels to force degradation
        from sklearn.dummy import DummyClassifier
        candidate_clf = DummyClassifier(strategy="uniform")
        candidate_clf.fit(self.X, self.y)

        res = self.gate.evaluate_candidate(active_clf, candidate_clf, self.X, self.y)
        self.assertFalse(res.is_promoted)
        self.assertIsNotNone(res.rejection_reason)


if __name__ == "__main__":
    unittest.main()
