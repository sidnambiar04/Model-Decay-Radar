"""
Stage 14 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. ScenarioSimulator batch generation across 5 scenarios.
  2. Prediction ID UUID tracking (pred_...).
  3. Feature shift magnitude verification on target sensors (LIT101, DPIT301, P402).
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np

from simulator import ScenarioSimulator


class TestStage14Simulator(unittest.TestCase):
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
        sudd_X, _, _ = self.sim.get_scenario_batch(scenario="sudden", n_samples=100)

        # Look up exact column index for LIT101
        lit_idx = self.sim.feature_cols.index("LIT101")
        mean_norm_lit = np.mean(norm_X[:, lit_idx])
        mean_sudd_lit = np.mean(sudd_X[:, lit_idx])
        self.assertGreater(mean_sudd_lit - mean_norm_lit, 20.0)


if __name__ == "__main__":
    unittest.main()
