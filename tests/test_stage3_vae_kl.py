"""
Stage 3 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. VAE Autoencoder loss and reconstruction error computation.
  2. Dynamic reconstruction thresholding.
  3. KL divergence score calibration and permutation p-value test.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np

from autoencoder import (
    AutoencoderDriftDetector,
    train_autoencoder,
    compute_reconstruction_errors,
    compute_dynamic_threshold,
)
from calibration import DriftCalibrator, permutation_drift_test


class TestStage3VAEKL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.input_dim = 10
        cls.ref_data = np.random.normal(0, 1, size=(200, cls.input_dim)).astype(np.float32)
        
        cls.ae_model = AutoencoderDriftDetector(input_dim=cls.input_dim, latent_dim=4, hidden_dim=16)
        train_autoencoder(cls.ae_model, cls.ref_data, epochs=2, batch_size=32, verbose=False)
        
        cls.ref_errors = compute_reconstruction_errors(cls.ae_model, cls.ref_data)
        cls.dynamic_threshold = compute_dynamic_threshold(cls.ref_errors)
        cls.calibrator = DriftCalibrator().fit(cls.ref_errors)
        cls.ref_scores = cls.calibrator.transform(cls.ref_errors)

    def test_dynamic_threshold_positive(self):
        self.assertGreater(self.dynamic_threshold, 0.0)
        self.assertGreater(self.dynamic_threshold, np.mean(self.ref_errors))

    def test_drifted_batch_kl_permutation(self):
        np.random.seed(99)
        # Shift distribution significantly
        drifted_data = np.random.normal(3.5, 1.5, size=(100, self.input_dim)).astype(np.float32)
        drift_errors = compute_reconstruction_errors(self.ae_model, drifted_data)
        drift_scores = self.calibrator.transform(drift_errors)

        res = permutation_drift_test(self.ref_scores, drift_scores, n_bins=15, n_permutations=100, seed=42)
        
        self.assertIn("observed_kl", res)
        self.assertIn("p_value", res)
        self.assertGreater(res["observed_kl"], 0.0)
        self.assertLess(res["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
