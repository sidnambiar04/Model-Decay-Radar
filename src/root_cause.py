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
        background_reference: Optional[np.ndarray] = None,
        max_bg: int = 30,
        max_test: int = 20,
        nsamples: int = 40,
    ) -> Dict[str, float]:
        """
        Computes KernelExplainer SHAP values with optimized subsampling
        to complete within 1.5s per batch without latency bottlenecks.
        """
        shap_dict = {f: 0.0 for f in feature_names}
        try:
            import shap
            rng = np.random.default_rng(42)

            bg_source = background_reference if (background_reference is not None and len(background_reference) > 0) else batch_scaled
            bg_idx = rng.integers(0, len(bg_source), size=min(max_bg, len(bg_source)))
            test_idx = rng.integers(0, len(batch_scaled), size=min(max_test, len(batch_scaled)))
            background = bg_source[bg_idx]
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
            shap_values = explainer.shap_values(test_data, nsamples=nsamples, silent=True)
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
        """
        Produces the unified SHAP + KS Root Cause Priority Matrix with
        multi-signal priority scoring and explicit diagnosis tags:
          - CRITICAL_ATTRIBUTION: High SHAP impact and statistically confirmed shift (p < 0.01)
          - STATISTICAL_SHIFT_ONLY: Significant shift (p < 0.05) with minor model impact
          - MODEL_SENSITIVE_ONLY: High SHAP sensitivity without feature distribution shift
          - NOMINAL: Normal feature behavior
        """
        top_k = top_k or getattr(config, "root_cause_top_k", 10)
        alpha = config.ks_alpha

        # 1. Compute KS Statistical Signals
        ks_dict = self.compute_ks_signals(
            reference_scaled, current_batch_scaled, feature_names, alpha=alpha
        )

        # 2. Compute SHAP Model Attribution Signals
        shap_dict = {}
        has_shap = False
        if run_shap and ae_model is not None:
            shap_dict = self.compute_shap_signals(
                ae_model,
                current_batch_scaled,
                feature_names,
                background_reference=reference_scaled,
                max_bg=30,
                max_test=20,
                nsamples=40,
            )
            has_shap = any(v > 1e-6 for v in shap_dict.values())
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
            ks_p = ks_info["ks_p_value"]

            norm_ks = float(ks_info["ks_statistic"] / max_ks)
            norm_shap = float(shap_val / max_shap) if has_shap else 0.0
            ks_conf = float(1.0 - np.clip(ks_p, 0.0, 1.0))

            # Multi-signal priority attribution score:
            # Priority Score = 0.5 * Normalized SHAP + 0.5 * (1 - p_KS)
            if has_shap:
                priority_score = round(float(0.5 * norm_shap + 0.5 * (0.7 * ks_conf + 0.3 * norm_ks)), 4)
            else:
                priority_score = round(float(0.7 * ks_conf + 0.3 * norm_ks), 4)

            # Assign explicit diagnosis tags
            stat_drift = bool(ks_info["statistically_drifted"])
            is_high_shap = bool(norm_shap >= 0.25)

            if stat_drift and ks_p < 0.01 and (is_high_shap or not has_shap):
                diagnosis_tag = "CRITICAL_ATTRIBUTION" if (is_high_shap or not has_shap) else "STATISTICAL_SHIFT_ONLY"
            elif stat_drift:
                diagnosis_tag = "STATISTICAL_SHIFT_ONLY"
            elif is_high_shap and not stat_drift:
                diagnosis_tag = "MODEL_SENSITIVE_ONLY"
            else:
                diagnosis_tag = "NOMINAL"

            results.append({
                "feature": name,
                "ks_statistic": round(ks_info["ks_statistic"], 4),
                "ks_p_value": round(ks_p, 6),
                "shap_importance": round(shap_val, 4),
                "drift_significance": priority_score,
                "priority_score": priority_score,
                "diagnosis_tag": diagnosis_tag,
                "statistically_drifted": stat_drift,
            })

        # Sort by priority_score descending with secondary sort by ks_statistic
        results = sorted(results, key=lambda x: (x["priority_score"], x["ks_statistic"]), reverse=True)
        return results[:top_k]
