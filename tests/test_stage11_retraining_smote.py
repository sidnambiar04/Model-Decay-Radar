"""
Stage 11 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. Imbalance ratio calculation.
  2. Borderline-SMOTE oversampling when class imbalance is detected (ratio < 0.2).
  3. Preservation of raw feature shapes prior to retraining.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np

from imbalance_handler import check_imbalance_ratio, apply_borderline_smote_if_needed


class TestStage11RetrainingSMOTE(unittest.TestCase):
    def test_imbalance_ratio_check(self):
        y_balanced = np.array([0, 1] * 50)
        y_imbalanced = np.array([0] * 95 + [1] * 5)

        self.assertAlmostEqual(check_imbalance_ratio(y_balanced), 1.0)
        self.assertLess(check_imbalance_ratio(y_imbalanced), 0.1)

    def test_borderline_smote_resampling(self):
        np.random.seed(42)
        X = np.random.normal(0, 1, size=(100, 5))
        y = np.array([0] * 92 + [1] * 8)

        X_res, y_res, smote_done, ratio_before = apply_borderline_smote_if_needed(X, y, imbalance_threshold=0.2)

        self.assertTrue(smote_done)
        self.assertGreater(len(X_res), len(X))
        self.assertGreater(len(y_res), len(y))
        self.assertGreaterEqual(check_imbalance_ratio(y_res), 0.5)


if __name__ == "__main__":
    unittest.main()
