"""
Stage 2 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. MultiClassifierEngine switching (RF, GB, MLP).
  2. POST /predict inference payload schema.
  3. Buffer accumulation and prediction_id generation.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np
from fastapi.testclient import TestClient

from classifier import ProductionClassifier
from api.server import app, orchestrator, FEATURE_COLS


class TestStage2ClassifierEngine(unittest.TestCase):
    def test_multiclassifier_switching(self):
        clf = ProductionClassifier()
        X = np.random.normal(0, 1, size=(100, 5))
        y = np.random.choice([0, 1], size=100)

        clf.fit(X, y)
        self.assertEqual(clf.active_model_name, "Random Forest")
        
        clf.set_active_model("Gradient Boosting")
        self.assertEqual(clf.active_model_name, "Gradient Boosting")
        
        clf.set_active_model("Neural Network (MLP)")
        self.assertEqual(clf.active_model_name, "Neural Network (MLP)")

        pred, conf = clf.predict(X[0])
        self.assertIn(pred, [0.0, 1.0])
        self.assertGreaterEqual(conf, 0.0)


class TestStage2PredictAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        orchestrator.is_ready = True
        cls.client = TestClient(app)

    def test_predict_endpoint_response_schema(self):
        sample_features = {col: float(np.random.normal()) for col in FEATURE_COLS}
        payload = {"features": sample_features, "label": 0}

        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("prediction_id", data)
        self.assertTrue(data["prediction_id"].startswith("pred_"))
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("model_version", data)
        self.assertIn("timestamp", data)
        self.assertIn("model_status", data)
        self.assertIn("batch_buffered", data)


if __name__ == "__main__":
    unittest.main()
