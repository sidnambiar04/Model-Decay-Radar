"""
Root Cause Attribution Engine — Model Decay Radar
==================================================
Combines Kolmogorov-Smirnov (KS) statistical data drift testing with SHAP
model attribution (on VAE reconstruction loss) to produce a unified,
side-by-side root cause feature analysis matrix.
"""

import numpy as np
from scipy.stats import ks_2samp
from typing import List, Dict, Any, Optional
import torch

from autoencoder import MSE_WEIGHT, KL_WEIGHT
from config import config


# Root Cause Analysis Fields: feature, ks_statistic, ks_p_value, shap_importance, drift_significance, statistically_drifted


class RootCauseEngine:
    """
    Root Cause Attribution Engine providing complementary signals:
    1. Statistical Shift (KS Test: statistic & p-value)
    2. Model Attribution (SHAP: impact on VAE reconstruction error)
    """

    @staticmethod
    def compute_ks_signals(
        reference_features: np.ndarray,
        current_features: np.ndarray,
        feature_names: List[str],
        alpha: float = 0.05,
    ) -> Dict[str, Dict[str, Any]]:
        ks_dict = {}
        for idx, name in enumerate(feature_names):
            ref_col = reference_features[:, idx]
            cur_col = current_features[:, idx]
            res = ks_2samp(ref_col, cur_col)
            ks_dict[name] = {
                "ks_statistic": float(res.statistic),
                "ks_p_value": float(res.pvalue),
                "statistically_drifted": bool(res.pvalue < alpha),
            }
        return ks_dict

    @staticmethod
    def compute_shap_signals(
        ae_model: torch.nn.Module,
        batch_scaled: np.ndarray,
        feature_names: List[str],
        max_bg: int = 30,
        max_test: int = 30,
    ) -> Dict[str, float]:
        shap_dict = {f: 0.0 for f in feature_names}
        try:
            import shap
            rng = np.random.default_rng(42)

            bg_idx = rng.integers(0, len(batch_scaled), size=min(max_bg, len(batch_scaled)))
            test_idx = rng.integers(0, len(batch_scaled), size=min(max_test, len(batch_scaled)))
            background = batch_scaled[bg_idx]
            test_data = batch_scaled[test_idx]

            ae_model.eval()
            def loss_fn(X):
                with torch.no_grad():
                    X_t = torch.tensor(X, dtype=torch.float32)
                    x_hat, mu, logvar = ae_model(X_t)
                    mse = torch.mean((X_t - x_hat) ** 2, dim=1)
                    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    return (MSE_WEIGHT * mse + KL_WEIGHT * kl).numpy()

            explainer = shap.KernelExplainer(loss_fn, background)
            shap_values = explainer.shap_values(test_data, nsamples=50, silent=True)
            mean_abs = np.abs(shap_values).mean(axis=0)

            for idx, name in enumerate(feature_names):
                shap_dict[name] = float(mean_abs[idx])
        except Exception as e:
            print(f"[RootCauseEngine] SHAP computation notice: {e}")
        return shap_dict

    def analyze(
        self,
        reference_scaled: np.ndarray,
        current_batch_scaled: np.ndarray,
        feature_names: List[str],
        ae_model: Optional[torch.nn.Module] = None,
        run_shap: bool = True,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        top_k = top_k or getattr(config, "root_cause_top_k", 10)
        alpha = config.ks_alpha

        # 1. Compute KS Statistical Signals
        ks_dict = self.compute_ks_signals(
            reference_scaled, current_batch_scaled, feature_names, alpha=alpha
        )

        # 2. Compute SHAP Model Attribution Signals
        shap_dict = {}
        if run_shap and ae_model is not None:
            shap_dict = self.compute_shap_signals(
                ae_model, current_batch_scaled, feature_names
            )
        else:
            shap_dict = {f: 0.0 for f in feature_names}

        # Normalize statistics for combined drift_significance
        max_ks = max([v["ks_statistic"] for v in ks_dict.values()] or [1.0])
        max_shap = max(shap_dict.values() or [1.0])

        max_ks = max(max_ks, 1e-6)
        max_shap = max(max_shap, 1e-6)

        results = []
        for name in feature_names:
            ks_info = ks_dict[name]
            shap_val = shap_dict.get(name, 0.0)

            norm_ks = ks_info["ks_statistic"] / max_ks
            norm_shap = shap_val / max_shap
            significance = round(float(0.5 * norm_ks + 0.5 * norm_shap), 4)

            results.append({
                "feature": name,
                "ks_statistic": round(ks_info["ks_statistic"], 4),
                "ks_p_value": round(ks_info["ks_p_value"], 6),
                "shap_importance": round(shap_val, 4),
                "drift_significance": significance,
                "statistically_drifted": ks_info["statistically_drifted"],
            })

        # Sort by drift_significance descending
        results = sorted(results, key=lambda x: x["drift_significance"], reverse=True)
        return results[:top_k]
