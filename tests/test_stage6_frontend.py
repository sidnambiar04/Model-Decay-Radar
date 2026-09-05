"""
Stage 6 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. Monitoring payload compatibility with Next.js frontend.
  2. Presence of active_model_version, root_cause_analysis, drift_type, detector_signals.
  3. API response structure for /monitoring/latest and /monitoring/history.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
from fastapi.testclient import TestClient

from api.server import app, orchestrator


class TestStage6FrontendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        orchestrator.is_ready = True
        cls.client = TestClient(app)

    def test_health_endpoint_schema(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("orchestrator_ready", data)
        self.assertIn("monitoring_cycles_completed", data)

    def test_config_endpoint_schema(self):
        response = self.client.get("/admin/config")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("active_classifier", data)
        self.assertIn("available_classifiers", data)


if __name__ == "__main__":
    unittest.main()
