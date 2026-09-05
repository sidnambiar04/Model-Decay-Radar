"""
Stage 1 API Endpoint Unit Tests using FastAPI TestClient
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
from fastapi.testclient import TestClient

from api.server import app, FEATURE_COLS
from config import config


class TestStage1API(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("window_size", data)
        self.assertEqual(data["window_size"], config.window_size)

    def test_config_endpoint(self):
        # GET config
        response = self.client.get("/admin/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["window_size"], config.window_size)
        self.assertEqual(data["p_value_threshold"], config.p_value_threshold)
        self.assertIn("available_classifiers", data)

        # POST config update
        update_payload = {"window_size": 400, "p_value_threshold": 0.02}
        post_response = self.client.post("/admin/config", json=update_payload)
        self.assertEqual(post_response.status_code, 200)
        updated_data = post_response.json()["config"]
        self.assertEqual(updated_data["window_size"], 400)
        self.assertEqual(config.window_size, 400)

        # Revert back to 500
        self.client.post("/admin/config", json={"window_size": 500, "p_value_threshold": 0.01})

    def test_predict_endpoint(self):
        sample_features = {col: 10.0 for col in FEATURE_COLS[:51]}
        payload = {"features": sample_features, "label": None}
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("confidence", data)
        self.assertIn("model_status", data)
        self.assertIn("batch_buffered", data)


if __name__ == "__main__":
    unittest.main()
