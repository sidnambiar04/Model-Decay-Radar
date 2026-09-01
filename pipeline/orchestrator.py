"""
Pipeline Orchestrator
---------------------
This is the brain of Model Decay Radar. It receives a batch of
production samples (collected by the buffer manager in the FastAPI
server), runs the full 7-layer pipeline, and returns a structured
MonitoringResult that the FastAPI server stores and the Streamlit
dashboard reads.

Call order:
  1. Preprocessing + SMOTE (Layer 2)
  2. Autoencoder drift detection + calibration + permutation test (Layer 3)
  3. RNN ensemble uncertainty (Layer 4)
  4. Drift signal fusion (Layer 5)
  5. SHAP interpretation IF drift confirmed (Layer 6)
  6. Selective retraining IF MHS critical (Layer 6)
  7. MHS + alert generation (Layer 7)
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable

import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from autoencoder import (
    AutoencoderDriftDetector,
    MSE_WEIGHT,
    KL_WEIGHT,
    train_autoencoder,
    compute_reconstruction_errors,
    compute_dynamic_threshold,
)
from calibration import DriftCalibrator, permutation_drift_test, N_PERMUTATIONS, compute_feature_ks_tests
from imbalance_handler import apply_borderline_smote_if_needed
from uncertainty import EnsembleRNNUncertainty


# ─────────────────────────────────────────────
# Result dataclass (serialisable → JSON via FastAPI)
# ─────────────────────────────────────────────

@dataclass
class MonitoringResult:
    batch_id: int
    timestamp: float

    # Layer 3 – drift detection
    mean_reconstruction_error: float
    dynamic_threshold: float
    observed_kl: float
    p_value: float
    drift_confirmed: bool
    drift_severity: float          # 0-1
    wasserstein_distance: float
    score_ks_p_value: float

    # Layer 4 – uncertainty
    mean_uncertainty: float        # 0-1 normalised

    # Layer 5 – fusion
    mhs: float                     # Model Health Score 0-1
    mhs_status: str                # "Healthy" | "Warning" | "Critical"
    accuracy: float                # Real batch accuracy (predictions vs labels)
    precision: float
    recall: float
    f1_score: float

    # Layer 6 – interpretation (only when drift_confirmed)
    top_drift_features: list       # [{feature, importance}]
    feature_ks_results: list       # [{feature, importance, p_value, drift_confirmed}]
    smote_applied: bool
    samples_before_smote: int
    samples_after_smote: int

    # Layer 7 – alerting
    alert_level: str               # "none" | "warning" | "critical"
    alert_message: str
    retraining_triggered: bool


# ─────────────────────────────────────────────
# Model Health Score (Eq 3.2 / 3.8)
# ─────────────────────────────────────────────

def compute_mhs(accuracy: float, drift: float,
                uncertainty: float, stability: float) -> float:
    w1, w2, w3, w4 = 0.35, 0.25, 0.20, 0.20
    return (w1 * accuracy
            + w2 * (1.0 - drift)
            + w3 * (1.0 - uncertainty)
            + w4 * stability)


def mhs_status(mhs: float) -> str:
    if mhs > 0.85:
        return "Healthy"
    elif mhs >= 0.65:
        return "Warning"
    else:
        return "Critical"


# ─────────────────────────────────────────────
# Orchestrator class
# ─────────────────────────────────────────────

class RadarOrchestrator:
    """
    Holds all trained components in memory and exposes a single
    `run_monitoring_cycle(batch_features, batch_labels)` method
    that the FastAPI server calls every time a window fills up.
    """

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.input_dim = len(feature_names)
        self.batch_counter = 0

        # Components (initialised in setup())
        self.ae_model: Optional[AutoencoderDriftDetector] = None
        self.calibrator: Optional[DriftCalibrator] = None
        self.ensemble: Optional[EnsembleRNNUncertainty] = None
        self.ref_errors: Optional[np.ndarray] = None
        self.ref_scores: Optional[np.ndarray] = None
        self.dynamic_threshold: float = 0.0

        # Running stats for stability (last 5 MHS values)
        self._mhs_history: list[float] = []

        self.is_ready = False

    # ── Setup: called once at startup with reference data ──────────────

    def setup(self, reference_scaled: np.ndarray,
              reference_raw_X: Optional[np.ndarray] = None,
              reference_raw_y: Optional[np.ndarray] = None,
              classifier: Optional[object] = None,
              scaler: Optional[object] = None,
              ae_epochs: int = 40, rnn_epochs: int = 20,
              progress_callback: Optional[Callable[[str], None]] = None):
        """
        Train all components on the reference (historical) window.
        In production this would load pre-trained weights from disk;
        for the demo we train from scratch on startup.
        """
        def _progress(stage: str):
            if progress_callback:
                progress_callback(stage)

        self.reference_scaled = reference_scaled
        self.reference_raw_X = reference_raw_X
        self.reference_raw_y = reference_raw_y
        self.classifier = classifier
        self.scaler = scaler

        _progress("training_ae")
        print("[Orchestrator] Training Autoencoder on reference window...", flush=True)
        self.ae_model = AutoencoderDriftDetector(
            input_dim=self.input_dim, latent_dim=8, hidden_dim=32
        )
        train_autoencoder(
            self.ae_model, reference_scaled,
            epochs=ae_epochs, batch_size=256, lr=1e-3, lam=0.7, verbose=True
        )

        self.ref_errors = compute_reconstruction_errors(
            self.ae_model, reference_scaled, lam=0.7
        )
        self.dynamic_threshold = compute_dynamic_threshold(self.ref_errors)
        self.calibrator = DriftCalibrator().fit(self.ref_errors)
        self.ref_scores = self.calibrator.transform(self.ref_errors)

        _progress("training_rnn")
        print("[Orchestrator] Training RNN Ensemble on reference error series...", flush=True)
        self.ensemble = EnsembleRNNUncertainty(n_members=3, dropout_rate=0.3)
        # Subsample for RNN training to keep startup fast on CPU
        rnn_train_size = min(5000, int(len(self.ref_errors) * 0.75))
        train_series = self.ref_errors[:rnn_train_size]
        self.ensemble.train(
            train_series, seq_len=10, epochs=rnn_epochs,
            batch_size=64, lr=1e-3, verbose=True
        )

        self.is_ready = True
        _progress("ready")
        print("[Orchestrator] Setup complete. Ready to monitor.", flush=True)

    # ── Main monitoring cycle ───────────────────────────────────────────

    def run_monitoring_cycle(
        self,
        batch_scaled: np.ndarray,
        batch_labels: Optional[np.ndarray] = None,
        batch_predictions: Optional[np.ndarray] = None,
    ) -> MonitoringResult:
        """
        Full 7-layer pipeline on one incoming batch.
        Called by the FastAPI buffer manager every time a window fills.
        """
        if not self.is_ready:
            raise RuntimeError("Call setup() before run_monitoring_cycle().")

        self.batch_counter += 1
        t_start = time.time()

        # Save a copy of batch before SMOTE for drift test and per-feature KS tests
        batch_scaled_original = batch_scaled.copy()

        # Calculate Classification Metrics
        batch_precision = 0.0
        batch_recall = 0.0
        batch_f1 = 0.0
        if (
            batch_labels is not None
            and batch_predictions is not None
            and len(batch_labels) == len(batch_predictions) == len(batch_scaled)
        ):
            batch_accuracy = float(np.mean(batch_predictions == batch_labels))
            batch_precision = float(precision_score(batch_labels, batch_predictions, zero_division=0))
            batch_recall = float(recall_score(batch_labels, batch_predictions, zero_division=0))
            batch_f1 = float(f1_score(batch_labels, batch_predictions, zero_division=0))
        else:
            batch_accuracy = 0.0

        # ── Layer 2: SMOTE ──────────────────────────────────────────────
        n_before = len(batch_scaled)
        smote_applied = False
        if batch_labels is not None:
            batch_scaled, batch_labels, smote_applied, _ = \
                apply_borderline_smote_if_needed(
                    batch_scaled, batch_labels, imbalance_threshold=0.2
                )
        n_after = len(batch_scaled)

        # ── Layer 3: Autoencoder + calibration + permutation test ───────
        cur_errors = compute_reconstruction_errors(self.ae_model, batch_scaled, lam=0.7)
        cur_scores = self.calibrator.transform(cur_errors)

        perm_result = permutation_drift_test(
            self.ref_scores, cur_scores,
            n_bins=15, n_permutations=N_PERMUTATIONS, seed=42
        )
        drift_severity = float(
            min(perm_result["observed_kl"] / (perm_result["observed_kl"] + 1.0), 1.0)
        )

        # Per-feature Kolmogorov-Smirnov tests to evaluate statistical shift significance (data-centric)
        feature_ks_results = compute_feature_ks_tests(
            self.reference_scaled, batch_scaled_original, self.feature_names
        )

        # ── Layer 4: RNN uncertainty ────────────────────────────────────
        if len(cur_errors) > 10:
            _, unc = self.ensemble.mc_dropout_predict(cur_errors, seq_len=10, T=10)
            raw_unc = EnsembleRNNUncertainty.uncertainty_score(unc).mean()
            # normalise against reference uncertainty for a 0-1 score
            ref_unc_sample = self.ref_errors[:len(cur_errors)]
            _, ref_unc_mc = self.ensemble.mc_dropout_predict(ref_unc_sample, seq_len=10, T=10)
            ref_unc_val = EnsembleRNNUncertainty.uncertainty_score(ref_unc_mc).mean()
            norm_unc = float(np.clip(raw_unc / (ref_unc_val + 1e-9), 0.0, 1.0))
        else:
            norm_unc = 0.0

        # ── Layer 5: MHS + fusion ───────────────────────────────────────
        stability = float(np.std(self._mhs_history[-5:]) if len(self._mhs_history) >= 2 else 1.0)
        stability = float(np.clip(1.0 - stability, 0.0, 1.0))

        mhs = compute_mhs(
            accuracy=batch_accuracy,
            drift=float(perm_result["drift_confirmed"]),
            uncertainty=min(norm_unc, 1.0),
            stability=stability,
        )
        mhs = float(np.clip(mhs, 0.0, 1.0))
        status = mhs_status(mhs)
        self._mhs_history.append(mhs)

        # ── Layer 6: SHAP interpretation (only when drift confirmed) ────
        top_features = []
        if perm_result["drift_confirmed"]:
            top_features = self._run_shap(batch_scaled)

        # ── Layer 6: Selective retraining (only when Critical) ──────────
        retraining_triggered = False
        if status == "Critical":
            retraining_triggered = True
            # Pass batch labels (potentially SMOTE augmented) for classifier training
            self._selective_retrain(batch_scaled, batch_labels)

        # ── Layer 7: Alert ──────────────────────────────────────────────
        if status == "Critical":
            alert_level = "critical"
            feat_str = ", ".join([f["feature"] for f in top_features[:3]])
            alert_message = (
                f"CRITICAL: MHS={mhs:.2f}. Drift confirmed (p={perm_result['p_value']:.4f}, "
                f"KL={perm_result['observed_kl']:.4f}). "
                f"Top features: {feat_str or 'N/A'}. Retraining triggered."
            )
        elif status == "Warning":
            alert_level = "warning"
            alert_message = (
                f"WARNING: MHS={mhs:.2f}. "
                f"{'Drift confirmed' if perm_result['drift_confirmed'] else 'Drift not confirmed'}. "
                f"Monitor closely."
            )
        else:
            alert_level = "none"
            alert_message = f"OK: MHS={mhs:.2f}. No drift detected."

        return MonitoringResult(
            batch_id=self.batch_counter,
            timestamp=t_start,
            mean_reconstruction_error=float(cur_errors.mean()),
            dynamic_threshold=self.dynamic_threshold,
            observed_kl=float(perm_result["observed_kl"]),
            p_value=float(perm_result["p_value"]),
            drift_confirmed=bool(perm_result["drift_confirmed"]),
            drift_severity=drift_severity,
            wasserstein_distance=perm_result["wasserstein_distance"],
            score_ks_p_value=perm_result["score_ks_p_value"],
            mean_uncertainty=float(norm_unc),
            mhs=mhs,
            mhs_status=status,
            accuracy=batch_accuracy,
            precision=batch_precision,
            recall=batch_recall,
            f1_score=batch_f1,
            top_drift_features=top_features,
            feature_ks_results=feature_ks_results,
            smote_applied=smote_applied,
            samples_before_smote=n_before,
            samples_after_smote=n_after,
            alert_level=alert_level,
            alert_message=alert_message,
            retraining_triggered=retraining_triggered,
        )

    # ── SHAP (called internally) ────────────────────────────────────────

    def _run_shap(self, batch_scaled: np.ndarray, max_bg: int = 30,
                  max_test: int = 30) -> list[dict]:
        """Run SHAP and return top 10 features as [{feature, importance}]."""
        try:
            import shap
            rng = np.random.default_rng(42)

            # Background: random sample from the batch itself (as a proxy for the reference features)
            bg_idx = rng.integers(0, len(batch_scaled),
                                  size=min(max_bg, len(batch_scaled)))
            # Build reference scaled array lazily (we stored errors, not features)
            # In production you'd keep a reference feature cache.
            # Here we use batch itself as a proxy for the test set.
            test_idx = rng.integers(0, len(batch_scaled),
                                    size=min(max_test, len(batch_scaled)))
            background = batch_scaled[bg_idx]
            test_data  = batch_scaled[test_idx]

            self.ae_model.eval()
            def loss_fn(X):
                with torch.no_grad():
                    X_t = torch.tensor(X, dtype=torch.float32)
                    x_hat, mu, logvar = self.ae_model(X_t)
                    mse = torch.mean((X_t - x_hat) ** 2, dim=1)
                    kl  = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    return (MSE_WEIGHT * mse + KL_WEIGHT * kl).numpy()

            explainer   = shap.KernelExplainer(loss_fn, background)
            shap_values = explainer.shap_values(test_data, nsamples=50, silent=True)
            mean_abs    = np.abs(shap_values).mean(axis=0)
            pairs = sorted(zip(self.feature_names, mean_abs),
                           key=lambda kv: kv[1], reverse=True)
            return [{"feature": f, "importance": float(v)} for f, v in pairs[:10]]
        except Exception as e:
            print(f"[SHAP] Error: {e}")
            return []

    # ── Selective retraining (called internally) ────────────────────────

    def _selective_retrain(self, drift_batch: np.ndarray, drift_labels: Optional[np.ndarray] = None):
        """
        Quick selective retrain: fine-tune the autoencoder and retrain the downstream model
        on a combined set of historical reference data and drift data.
        """
        try:
            # Retrain Autoencoder
            print(f"[Orchestrator] Selective retraining of Autoencoder on drift batch ({len(drift_batch)} samples)...")
            train_autoencoder(
                self.ae_model, drift_batch,
                epochs=5, batch_size=64, lr=5e-4, lam=0.7, verbose=False
            )
            # Update reference errors and threshold after retraining
            new_ref_errors = compute_reconstruction_errors(
                self.ae_model, drift_batch, lam=0.7
            )
            n_retained = min(500, len(self.ref_errors))
            self.ref_errors   = np.concatenate([self.ref_errors[n_retained:], new_ref_errors])
            self.dynamic_threshold = compute_dynamic_threshold(self.ref_errors)
            self.calibrator   = DriftCalibrator().fit(self.ref_errors)
            self.ref_scores   = self.calibrator.transform(self.ref_errors)
            print("[Orchestrator] Autoencoder retraining complete.")

            # Retrain Classifier
            if (
                self.classifier is not None 
                and self.scaler is not None 
                and drift_labels is not None 
                and self.reference_raw_X is not None
                and self.reference_raw_y is not None
            ):
                # Inverse scale features to get raw features back
                drift_raw_X = self.scaler.inverse_transform(drift_batch)
                
                # Combine reference features and labels with drifted features and labels
                combined_X = np.concatenate([self.reference_raw_X, drift_raw_X], axis=0)
                combined_y = np.concatenate([self.reference_raw_y, drift_labels], axis=0)
                
                print(f"[Orchestrator] Retraining classifier model '{self.classifier.active_model_name}' on {len(combined_X)} samples...")
                self.classifier.fit(combined_X, combined_y, model_name=self.classifier.active_model_name)
                print("[Orchestrator] Classifier retraining complete.")
            else:
                print("[Orchestrator] Skipping classifier retraining due to missing labels/scalers/reference raw data.")
        except Exception as e:
            print(f"[Orchestrator] Retraining error: {e}")
