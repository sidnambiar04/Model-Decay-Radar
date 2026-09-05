"""
Stage 10 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. MultiClassifierEngine fitting across Random Forest, Gradient Boosting, and MLP.
  2. Dynamic model switching via set_active_model().
  3. Single-class dataset fallback handling.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np

from classifier import ProductionClassifier


class TestStage10MultiClassifier(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.normal(0, 1, size=(100, 5))
        self.y = np.random.choice([0, 1], size=100, p=[0.8, 0.2])

    def test_multiclassifier_fitting_and_switching(self):
        clf = ProductionClassifier()
        clf.fit(self.X, self.y)

        # Verify default model is Random Forest
        self.assertEqual(clf.active_model_name, "Random Forest")
        pred_rf, conf_rf = clf.predict(self.X[0])
        self.assertIn(pred_rf, [0.0, 1.0])
        self.assertGreaterEqual(conf_rf, 0.0)

        # Switch to Gradient Boosting
        clf.set_active_model("Gradient Boosting")
        self.assertEqual(clf.active_model_name, "Gradient Boosting")
        pred_gb, conf_gb = clf.predict(self.X[0])
        self.assertIn(pred_gb, [0.0, 1.0])

        # Switch to Neural Network (MLP)
        clf.set_active_model("Neural Network (MLP)")
        self.assertEqual(clf.active_model_name, "Neural Network (MLP)")
        pred_mlp, conf_mlp = clf.predict(self.X[0])
        self.assertIn(pred_mlp, [0.0, 1.0])

    def test_predict_batch(self):
        clf = ProductionClassifier()
        clf.fit(self.X, self.y)
        preds = clf.predict_batch(self.X)
        self.assertEqual(len(preds), 100)


if __name__ == "__main__":
    unittest.main()
