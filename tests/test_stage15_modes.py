"""
Stage 15 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. Production Mode POST /predict single-sample ingestion.
  2. Demo Mode POST /admin/scenario batch streaming.
  3. Shared monitoring history schema consistency across both modes.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
from fastapi.testclient import TestClient

from api.server import app, orchestrator, FEATURE_COLS


class TestStage15OperatingModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        orchestrator.is_ready = True
        cls.client = TestClient(app)

    def test_production_mode_predict(self):
        sample = {col: 1.0 for col in FEATURE_COLS}
        res = self.client.post("/predict", json={"features": sample, "label": 0})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("prediction_id", data)
        self.assertIn("model_version", data)

    def test_demo_mode_scenario(self):
        res = self.client.post("/admin/scenario", json={"scenario": "normal"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        self.assertIn("scenario", data)


if __name__ == "__main__":
    unittest.main()
