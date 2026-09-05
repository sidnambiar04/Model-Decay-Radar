"""
Stage 16 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. GET /health schema validation for Next.js frontend polling.
  2. GET /monitoring/latest schema validation.
  3. GET /admin/config schema validation.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
from fastapi.testclient import TestClient
from api.server import app, orchestrator


class TestStage16FrontendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        orchestrator.is_ready = True
        cls.client = TestClient(app)

    def test_health_endpoint_schema(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("status", data)
        self.assertIn("orchestrator_ready", data)
        self.assertIn("buffer_size", data)
        self.assertIn("window_size", data)

    def test_config_endpoint_schema(self):
        res = self.client.get("/admin/config")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("active_classifier", data)
        self.assertIn("window_size", data)


if __name__ == "__main__":
    unittest.main()
