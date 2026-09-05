"""
Stage 6 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. EnsembleRNNUncertainty training on reconstruction error series.
  2. Epistemic uncertainty estimation with T=50 Monte Carlo Dropout passes.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np

from uncertainty import EnsembleRNNUncertainty


class TestStage6Uncertainty(unittest.TestCase):
    def test_mc_dropout_t50_passes(self):
        np.random.seed(42)
        ensemble = EnsembleRNNUncertainty(n_members=2, dropout_rate=0.3)
        series = np.random.normal(0.2, 0.05, size=80).astype(np.float32)
        
        ensemble.train(series, seq_len=10, epochs=2, verbose=False)

        mean_pred, var_pred = ensemble.mc_dropout_predict(series, seq_len=10, T=50)

        self.assertEqual(len(mean_pred), len(series) - 10)
        self.assertEqual(len(var_pred), len(series) - 10)
        self.assertTrue(np.all(var_pred >= 0.0))

        unc_scores = EnsembleRNNUncertainty.uncertainty_score(var_pred)
        self.assertEqual(len(unc_scores), len(series) - 10)


if __name__ == "__main__":
    unittest.main()
