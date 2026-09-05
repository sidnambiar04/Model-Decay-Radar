"""
Stage 3 Unit & Integration Tests — Model Decay Radar
======================================================
Tests:
  1. Epistemic uncertainty estimation with T=50 forward passes.
  2. RootCauseEngine alignment of statistical shift (KS) and model attribution (SHAP).
  3. Combined drift_significance ranking for drifted features.
  4. Integration within RadarOrchestrator.
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, _ROOT)

import unittest
import numpy as np
import pandas as pd

from root_cause import RootCauseEngine
from uncertainty import EnsembleRNNUncertainty
from orchestrator import RadarOrchestrator
from data_pipeline import DataWindowManager
from config import config


class TestStage3Uncertainty(unittest.TestCase):
    def test_mc_dropout_t50(self):
        ensemble = EnsembleRNNUncertainty(n_members=2, dropout_rate=0.3)
        series = np.random.normal(0, 1, size=100)
        ensemble.train(series, seq_len=10, epochs=1, verbose=False)

        mean_pred, var_pred = ensemble.mc_dropout_predict(series, seq_len=10, T=50)
        self.assertEqual(len(mean_pred), len(series) - 10)
        self.assertEqual(len(var_pred), len(series) - 10)
        self.assertTrue(np.all(var_pred >= 0.0))


class TestStage3RootCause(unittest.TestCase):
    def setUp(self):
        self.engine = RootCauseEngine()
        self.feature_names = ["LIT101", "DPIT301", "P402", "FIT101", "PIT201"]

    def test_root_cause_analysis_ranking(self):
        np.random.seed(42)
        ref_features = np.random.normal(0, 1, size=(200, 5))
        cur_features = ref_features.copy()
        # Inject intentional drift into LIT101 (index 0) and DPIT301 (index 1)
        cur_features[:, 0] += 3.5
        cur_features[:, 1] += 2.8

        results = self.engine.analyze(
            reference_scaled=ref_features,
            current_batch_scaled=cur_features,
            feature_names=self.feature_names,
            ae_model=None,
            run_shap=False,
            top_k=5,
        )

        self.assertEqual(len(results), 5)
        top_feature = results[0]["feature"]
        self.assertIn(top_feature, ["LIT101", "DPIT301"])
        self.assertTrue(results[0]["statistically_drifted"])
        self.assertGreater(results[0]["drift_significance"], 0.4)


class TestStage3OrchestratorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        feature_cols = ["LIT101", "DPIT301", "P402", "FIT101", "PIT201"]
        ref_data = np.random.normal(0, 1, size=(500, 5))
        ref_df = pd.DataFrame(ref_data, columns=feature_cols)
        ref_df["Label"] = np.random.choice([0, 1], size=500, p=[0.95, 0.05])
        ref_df["_window"] = "reference"

        cls.wm = DataWindowManager(ref_df, feature_cols, label_col="Label")
        cls.wm.set_reference_window(ref_df)

        cls.orchestrator = RadarOrchestrator(feature_names=feature_cols)
        cls.orchestrator.setup(
            cls.wm.reference_scaled,
            ae_epochs=2,
            rnn_epochs=2,
        )

    def test_monitoring_result_contains_root_cause(self):
        np.random.seed(99)
        drift_batch = np.random.normal(2.5, 1.5, size=(100, 5))
        result = self.orchestrator.run_monitoring_cycle(drift_batch)

        self.assertIsNotNone(result.root_cause_analysis)
        self.assertGreater(len(result.root_cause_analysis), 0)
        first_item = result.root_cause_analysis[0]
        self.assertIn("feature", first_item)
        self.assertIn("ks_statistic", first_item)
        self.assertIn("shap_importance", first_item)
        self.assertIn("drift_significance", first_item)


if __name__ == "__main__":
    unittest.main()
