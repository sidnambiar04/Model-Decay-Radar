"""
Centralized Drift Signal Fusion Engine & Streaming Detectors
============================================================
Combines multi-signal evidence from VAE reconstruction error, KL divergence,
Wasserstein distance, KS test statistics, ADWIN, and DDM to evaluate true
drift presence, drift severity, and drift classification type.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from config import config


class ADWINDetector:
    """
    Adaptive Windowing (ADWIN) detector for streaming error monitoring.
    Maintains a sliding window of error signals and detects distribution change.
    """
    def __init__(self, delta: float = 0.002, max_window: int = 1000):
        self.delta = delta
        self.max_window = max_window
        self.window: List[float] = []
        self.drift_detected = False

    def update(self, val: float) -> bool:
        self.window.append(val)
        if len(self.window) > self.max_window:
            self.window.pop(0)

        n = len(self.window)
        if n < 30:
            self.drift_detected = False
            return False

        # Evaluate cut points
        arr = np.array(self.window)
        self.drift_detected = False
        for i in range(10, n - 10, max(1, n // 20)):
            w0 = arr[:i]
            w1 = arr[i:]
            n0, n1 = len(w0), len(w1)
            m0, m1 = w0.mean(), w1.mean()
            diff = abs(m0 - m1)

            m_hat = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = np.sqrt((1.0 / (2.0 * m_hat)) * np.log(4.0 * n / self.delta))
            if diff > epsilon:
                self.drift_detected = True
                self.window = list(w1)  # trim old window
                break

        return self.drift_detected


class DDMDetector:
    """
    Drift Detection Method (DDM) monitoring model error rate p and standard deviation s.
    """
    def __init__(self, min_num_instances: int = 30):
        self.min_num_instances = min_num_instances
        self.n = 0
        self.p = 0.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.ps_min = float("inf")
        self.status = "in_control"  # "in_control" | "warning" | "drift"

    def update(self, is_error: bool):
        self.n += 1
        x = 1.0 if is_error else 0.0
        self.p = self.p + (x - self.p) / self.n
        self.s = np.sqrt(max(1e-9, self.p * (1.0 - self.p) / self.n))

        if self.n < self.min_num_instances:
            self.status = "in_control"
            return

        ps = self.p + self.s
        if ps < self.ps_min:
            self.p_min = self.p
            self.s_min = self.s
            self.ps_min = ps

        if ps >= self.p_min + 3.0 * self.s_min:
            self.status = "drift"
        elif ps >= self.p_min + 2.0 * self.s_min:
            self.status = "warning"
        else:
            self.status = "in_control"

    def reset(self):
        self.n = 0
        self.p = 0.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.ps_min = float("inf")
        self.status = "in_control"


@dataclass
class FusionResult:
    drift_detected: bool
    drift_type: str              # "none" | "gradual" | "sudden" | "concept_drift" | "covariate_drift"
    drift_severity: float        # 0.0 to 1.0
    fusion_score: float          # 0.0 to 1.0
    detector_signals: Dict[str, Any]


class DriftSignalFusionEngine:
    """
    Centralized Drift Signal Fusion Engine.
    Combines VAE reconstruction loss, KL divergence, Wasserstein distance,
    KS statistics, ADWIN, and DDM.
    """
    def __init__(self):
        self.adwin = ADWINDetector()
        self.ddm = DDMDetector()

    def evaluate_fusion(
        self,
        mean_reconstruction_error: float,
        dynamic_threshold: float,
        observed_kl: float,
        p_value: float,
        wasserstein_distance: float,
        score_ks_p_value: float,
        feature_ks_results: List[Dict[str, Any]],
        batch_labels: Optional[np.ndarray] = None,
        batch_predictions: Optional[np.ndarray] = None,
        mean_uncertainty: float = 0.0,
    ) -> FusionResult:
        # 1. Unsupervised Detector Signals
        vae_signal = bool(mean_reconstruction_error > dynamic_threshold)
        kl_signal = bool(p_value < config.p_value_threshold)
        wasserstein_signal = bool(wasserstein_distance > config.wasserstein_threshold)

        # Feature KS shift ratio (proportion of features showing p_value < config.ks_alpha)
        drifted_features = [f for f in feature_ks_results if f.get("drift_confirmed", False)]
        tot_features = max(1, len(feature_ks_results))
        ks_feature_ratio = len(drifted_features) / tot_features
        ks_signal = bool(ks_feature_ratio > 0.15 or score_ks_p_value < config.ks_alpha)

        # 2. Supervised Streaming Signals (when labels are available)
        adwin_signal = False
        ddm_signal = False
        accuracy_drop = False
        if batch_labels is not None and batch_predictions is not None and len(batch_labels) == len(batch_predictions) and len(batch_labels) > 0:
            errors = (batch_labels != batch_predictions).astype(float)
            acc = 1.0 - float(np.mean(errors))
            if acc < 0.80:
                accuracy_drop = True

            for err in errors:
                adwin_signal = self.adwin.update(err) or adwin_signal
                self.ddm.update(bool(err == 1.0))
            ddm_signal = bool(self.ddm.status in ("drift", "warning"))

        supervised_signal = adwin_signal or ddm_signal or accuracy_drop

        # 3. Weighted Fusion Calculation
        w = config.fusion_weights
        vae_val = float(vae_signal) * min(1.0, mean_reconstruction_error / (dynamic_threshold + 1e-6))
        kl_val = float(kl_signal) * min(1.0, observed_kl / (observed_kl + 1.0))
        ks_val = float(ks_signal) * min(1.0, ks_feature_ratio * 2.0)
        wass_val = float(wasserstein_signal) * min(1.0, wasserstein_distance / (config.wasserstein_threshold * 2.0))
        sup_val = float(supervised_signal)

        fusion_score = (
            w["vae"] * vae_val
            + w["kl"] * kl_val
            + w["ks"] * ks_val
            + w["wasserstein"] * wass_val
            + w["supervised"] * sup_val
        )
        fusion_score = float(np.clip(fusion_score, 0.0, 1.0))

        drift_detected = bool(fusion_score >= config.fusion_vote_threshold or kl_signal)
        drift_severity = float(np.clip(fusion_score * 1.2, 0.0, 1.0))

        # 4. Drift Type Classification Logic
        if not drift_detected:
            drift_type = "none"
        elif supervised_signal and (vae_signal or kl_signal):
            drift_type = "concept_drift"
        elif (vae_signal or kl_signal) and drift_severity >= 0.65:
            drift_type = "sudden"
        elif (vae_signal or kl_signal) and drift_severity < 0.65:
            drift_type = "gradual"
        elif ks_signal or wasserstein_signal:
            drift_type = "covariate_drift"
        else:
            drift_type = "gradual"

        detector_signals = {
            "vae_loss_drift": vae_signal,
            "kl_permutation_drift": kl_signal,
            "wasserstein_drift": wasserstein_signal,
            "ks_feature_drift": ks_signal,
            "feature_ks_ratio": round(ks_feature_ratio, 4),
            "adwin_drift": adwin_signal,
            "ddm_drift": ddm_signal,
            "supervised_drift": supervised_signal,
            "uncertainty_level": round(mean_uncertainty, 4),
        }

        return FusionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            drift_severity=drift_severity,
            fusion_score=fusion_score,
            detector_signals=detector_signals,
        )
