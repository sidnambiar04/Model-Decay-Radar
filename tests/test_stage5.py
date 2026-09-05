"""
Stage 5 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. ScenarioSimulator batch generation across 5 demo scenarios.
  2. Prediction ID tracking for delayed ground-truth association.
  3. Integration with FastAPI POST /admin/scenario endpoint.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np
from fastapi.testclient import TestClient

from simulator import ScenarioSimulator
from api.server import app


class TestStage5Simulator(unittest.TestCase):
    def setUp(self):
        self.sim = ScenarioSimulator()

    def test_generate_all_scenarios(self):
        scenarios = ["normal", "gradual", "sudden", "imbalance", "recovery"]
        for sc in scenarios:
            raw_X, labels, pred_ids = self.sim.get_scenario_batch(scenario=sc, n_samples=100)
            self.assertEqual(len(raw_X), 100)
            self.assertEqual(len(labels), 100)
            self.assertEqual(len(pred_ids), 100)
            self.assertTrue(pred_ids[0].startswith("pred_"))

    def test_sudden_drift_magnitude(self):
        norm_X, _, _ = self.sim.get_scenario_batch(scenario="normal", n_samples=100)
        sudd_X, _, _ = self.sim.get_scenario_batch(scenario="sudden", n_samples=100, drift_strength=1.5)

        # Check mean offset in LIT101 (index 0 or feature position)
        norm_mean = norm_X.mean()
        sudd_mean = sudd_X.mean()
        self.assertGreater(sudd_mean, norm_mean)


class TestStage5API(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.server import orchestrator
        orchestrator.is_ready = True
        cls.client = TestClient(app)

    def test_scenario_endpoint(self):
        payload = {
            "scenario": "normal",
            "n_samples": 100,
            "drift_strength": 1.0
        }
        response = self.client.post("/admin/scenario", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["scenario"], "normal")


if __name__ == "__main__":
    unittest.main()
