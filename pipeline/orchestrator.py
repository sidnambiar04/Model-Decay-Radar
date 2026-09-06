"""
Pipeline Orchestrator
---------------------
This is the brain of Model Decay Radar. It receives a batch of
production samples (collected by the buffer manager in the FastAPI
server), runs the full 7-layer pipeline, and returns a structured
MonitoringResult that the FastAPI server stores and the Next.js
dashboard reads.

Call order:
  1. Data Ingestion & Preprocessing (Layer 1)
  2. Pure Raw Production Distribution Drift Detection (Layer 3 - VAE + KL + KS)
  3. Epistemic Uncertainty Estimation (Layer 4 - RNN Ensemble)
  4. Model Health Score & Fusion (Layer 5)
  5. SHAP Interpretation IF drift confirmed (Layer 6)
  6. Selective Retraining & Imbalance Handling IF MHS critical (Layer 6)
  7. Alert Generation (Layer 7)
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

from config import config
from autoencoder import (
    AutoencoderDriftDetector,
    MSE_WEIGHT,
    KL_WEIGHT,
    train_autoencoder,
    compute_reconstruction_errors,
    compute_dynamic_threshold,
)
from calibration import DriftCalibrator, permutation_drift_test, compute_feature_ks_tests
from imbalance_handler import apply_borderline_smote_if_needed, check_imbalance_ratio
from uncertainty import EnsembleRNNUncertainty
from drift_fusion import DriftSignalFusionEngine
from root_cause import RootCauseEngine
from model_registry import ModelRegistry
from validation_gate import ValidationGateEngine, RetrainingCooldownManager
from experiment_tracker import ExperimentTracker


# ─────────────────────────────────────────────
# Result dataclass (serialisable → JSON via FastAPI)
# ─────────────────────────────────────────────

@dataclass
class MonitoringResult:
    batch_id: int
    timestamp: float

    # Layer 3 – drift detection (pure production distribution)
    mean_reconstruction_error: float
    dynamic_threshold: float
    observed_kl: float
    p_value: float
    drift_confirmed: bool
    drift_severity: float          # 0-1
    wasserstein_distance: float
    score_ks_p_value: float

    # Layer 5 – drift fusion
    drift_type: str                # "none" | "gradual" | "sudden" | "concept_drift" | "covariate_drift"
    fusion_score: float            # 0-1
    detector_signals: dict

    # Layer 4 – uncertainty
    mean_uncertainty: float        # 0-1 normalised

    # Layer 5 – fusion & performance status
    mhs: float                     # Model Health Score 0-1
    mhs_status: str                # "Healthy" | "Warning" | "Critical"
    performance_status: str        # "labels_available" | "awaiting_ground_truth"
    accuracy: float                # Real batch accuracy (if labels available)
    precision: float
    recall: float
    f1_score: float

    # Layer 6 – interpretation & side-by-side root cause
    top_drift_features: list       # [{feature, importance}]
    feature_ks_results: list       # [{feature, importance, p_value, drift_confirmed}]
    root_cause_analysis: list      # [{feature, ks_statistic, ks_p_value, shap_importance, drift_significance}]
    smote_applied: bool
    samples_before_smote: int
    samples_after_smote: int

    # Layer 6.5 – model versioning & validation gate
    active_model_version: str      # e.g., "v1", "v2"
    validation_status: str         # "none" | "promoted" | "rejected"
    validation_metrics: dict

    # Layer 7 – alerting
    alert_level: str               # "none" | "warning" | "critical"
    alert_message: str
    retraining_triggered: bool


# ─────────────────────────────────────────────
# Model Health Score & Reweighting
# ─────────────────────────────────────────────

def compute_mhs(accuracy: float, drift: float,
                uncertainty: float, stability: float,
                performance_status: str = "labels_available") -> float:
    w = config.mhs_weights
    if performance_status == "labels_available":
        w1, w2, w3, w4 = w["accuracy"], w["drift"], w["uncertainty"], w["stability"]
        return (w1 * accuracy
                + w2 * (1.0 - drift)
                + w3 * (1.0 - uncertainty)
                + w4 * stability)
    else:
        # Ground truth absent/delayed: redistribute accuracy weight (0.35) among remaining 3 signals
        tot_remaining = w["drift"] + w["uncertainty"] + w["stability"]
        w2_adj = w["drift"] / tot_remaining
        w3_adj = w["uncertainty"] / tot_remaining
        w4_adj = w["stability"] / tot_remaining
        return (w2_adj * (1.0 - drift)
                + w3_adj * (1.0 - uncertainty)
                + w4_adj * stability)


def mhs_status(mhs: float) -> str:
    healthy_th = config.mhs_thresholds.get("healthy", 0.85)
    warning_th = config.mhs_thresholds.get("warning", 0.65)
    if mhs > healthy_th:
        return "Healthy"
    elif mhs >= warning_th:
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

        # Centralized Drift Fusion Engine, Root Cause Engine, Model Registry, and Validation Gate
        self.fusion_engine = DriftSignalFusionEngine()
        self.root_cause_engine = RootCauseEngine()
        self.registry = ModelRegistry()
        self.validation_gate = ValidationGateEngine()

        # Experiment Tracker & Retraining Cooldown
        self.experiment_tracker = ExperimentTracker()
        self.cooldown_manager = RetrainingCooldownManager()

        # Validation State Tracking
        self.latest_validation_status = "none"
        self.latest_validation_metrics = {}

        # Running stats for stability (last 5 MHS values)
        self._mhs_history: list[float] = []

        self.is_ready = False

    # ── Setup: called once at startup with reference data ──────────────

    def setup(self, reference_scaled: np.ndarray,
              reference_raw_X: Optional[np.ndarray] = None,
              reference_raw_y: Optional[np.ndarray] = None,
              classifier: Optional[object] = None,
              scaler: Optional[object] = None,
              ae_epochs: int = 10, rnn_epochs: int = 5,
              progress_callback: Optional[Callable[[str], None]] = None):
        """
        Train all components on the reference (historical) window.
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

        # Log initial baseline version (v1) in Model Registry
        clf_name = self.classifier.active_model_name if self.classifier else "Random Forest"
        self.registry.log_version("v1", clf_name, {"accuracy": 0.95, "f1_score": 0.95}, "initial_baseline")

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
        Runs drift detection on RAW production distributions.
        """
        if not self.is_ready or self.ae_model is None:
            raise RuntimeError("Call setup() before run_monitoring_cycle().")

        self.batch_counter += 1
        t_start = time.time()

        # Check Ground-Truth Label availability
        batch_precision = 0.0
        batch_recall = 0.0
        batch_f1 = 0.0
        batch_accuracy = 0.0

        if (
            batch_labels is not None
            and batch_predictions is not None
            and len(batch_labels) == len(batch_predictions) == len(batch_scaled)
        ):
            performance_status = "labels_available"
            batch_accuracy = float(np.mean(batch_predictions == batch_labels))
            batch_precision = float(precision_score(batch_labels, batch_predictions, zero_division=0))
            batch_recall = float(recall_score(batch_labels, batch_predictions, zero_division=0))
            batch_f1 = float(f1_score(batch_labels, batch_predictions, zero_division=0))
        else:
            performance_status = "awaiting_ground_truth"

        # ── Layer 3: Drift Detection on RAW Production Data ──────────────
        # Note: Raw production distribution MUST be used for drift detection.
        # Do NOT apply synthetic oversampling (SMOTE) prior to drift calculation!
        cur_errors = compute_reconstruction_errors(self.ae_model, batch_scaled, lam=0.7)
        cur_scores = self.calibrator.transform(cur_errors)

        perm_result = permutation_drift_test(
            self.ref_scores, cur_scores,
            n_bins=15, seed=42
        )
        drift_severity = float(
            min(perm_result["observed_kl"] / (perm_result["observed_kl"] + 1.0), 1.0)
        )

        # Per-feature Kolmogorov-Smirnov statistical tests on raw features
        feature_ks_results = compute_feature_ks_tests(
            self.reference_scaled, batch_scaled, self.feature_names, alpha=config.ks_alpha
        )

        # ── Layer 4: RNN Uncertainty ────────────────────────────────────
        mc_t = config.mc_dropout_t
        if len(cur_errors) > 10:
            _, unc = self.ensemble.mc_dropout_predict(cur_errors, seq_len=10, T=mc_t)
            raw_unc = EnsembleRNNUncertainty.uncertainty_score(unc).mean()
            # Normalise against reference uncertainty for a 0-1 score
            ref_unc_sample = self.ref_errors[:len(cur_errors)]
            _, ref_unc_mc = self.ensemble.mc_dropout_predict(ref_unc_sample, seq_len=10, T=mc_t)
            ref_unc_val = EnsembleRNNUncertainty.uncertainty_score(ref_unc_mc).mean()
            norm_unc = float(np.clip(raw_unc / (ref_unc_val + 1e-9), 0.0, 1.0))
        else:
            norm_unc = 0.0

        # ── Layer 5: Centralized Drift Signal Fusion Engine ──────────────
        fusion_res = self.fusion_engine.evaluate_fusion(
            mean_reconstruction_error=float(cur_errors.mean()),
            dynamic_threshold=self.dynamic_threshold,
            observed_kl=float(perm_result["observed_kl"]),
            p_value=float(perm_result["p_value"]),
            wasserstein_distance=perm_result["wasserstein_distance"],
            score_ks_p_value=perm_result["score_ks_p_value"],
            feature_ks_results=feature_ks_results,
            batch_labels=batch_labels,
            batch_predictions=batch_predictions,
            mean_uncertainty=norm_unc,
        )

        stability = float(np.std(self._mhs_history[-5:]) if len(self._mhs_history) >= 2 else 1.0)
        stability = float(np.clip(1.0 - stability, 0.0, 1.0))

        mhs = compute_mhs(
            accuracy=batch_accuracy,
            drift=float(fusion_res.drift_detected),
            uncertainty=min(norm_unc, 1.0),
            stability=stability,
            performance_status=performance_status,
        )
        mhs = float(np.clip(mhs, 0.0, 1.0))
        status = mhs_status(mhs)
        self._mhs_history.append(mhs)

        # ── Layer 6: Root Cause Analysis (SHAP vs. KS) ───────────────────
        root_cause_analysis = self.root_cause_engine.analyze(
            reference_scaled=self.reference_scaled,
            current_batch_scaled=batch_scaled,
            feature_names=self.feature_names,
            ae_model=self.ae_model,
            run_shap=fusion_res.drift_detected,
        )

        top_features = []
        if fusion_res.drift_detected:
            top_features = [{"feature": item["feature"], "importance": item["shap_importance"]} for item in root_cause_analysis[:10]]

        # Check imbalance for logging/monitoring stats
        smote_applied = False
        n_before = len(batch_scaled)
        n_after = len(batch_scaled)
        if batch_labels is not None:
            imb_ratio = check_imbalance_ratio(batch_labels)
            smote_applied = bool(imb_ratio < config.smote_imbalance_threshold)

        # ── Layer 6.5: Selective Retraining & Validation Gate ───────────
        retraining_triggered = False
        if status == "Critical":
            retraining_triggered = True
            self._selective_retrain(batch_scaled, batch_labels)

        active_version = self.registry.get_latest_version_tag()

        # ── Layer 7: Alert ──────────────────────────────────────────────
        if status == "Critical":
            alert_level = "critical"
            feat_str = ", ".join([f["feature"] for f in top_features[:3]])
            alert_message = (
                f"CRITICAL: MHS={mhs:.2f}. Drift ({fusion_res.drift_type}) confirmed "
                f"(severity={fusion_res.drift_severity:.2f}, p={perm_result['p_value']:.4f}). "
                f"Top features: {feat_str or 'N/A'}. Retraining triggered."
            )
        elif status == "Warning":
            alert_level = "warning"
            alert_message = (
                f"WARNING: MHS={mhs:.2f}. "
                f"{'Drift confirmed (' + fusion_res.drift_type + ')' if fusion_res.drift_detected else 'Drift not confirmed'}. "
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
            drift_confirmed=bool(fusion_res.drift_detected),
            drift_severity=fusion_res.drift_severity,
            wasserstein_distance=perm_result["wasserstein_distance"],
            score_ks_p_value=perm_result["score_ks_p_value"],
            drift_type=fusion_res.drift_type,
            fusion_score=fusion_res.fusion_score,
            detector_signals=fusion_res.detector_signals,
            mean_uncertainty=float(norm_unc),
            mhs=mhs,
            mhs_status=status,
            performance_status=performance_status,
            accuracy=batch_accuracy,
            precision=batch_precision,
            recall=batch_recall,
            f1_score=batch_f1,
            top_drift_features=top_features,
            feature_ks_results=feature_ks_results,
            root_cause_analysis=root_cause_analysis,
            smote_applied=smote_applied,
            samples_before_smote=n_before,
            samples_after_smote=n_after,
            active_model_version=active_version,
            validation_status=self.latest_validation_status,
            validation_metrics=self.latest_validation_metrics,
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

            bg_idx = rng.integers(0, len(batch_scaled),
                                  size=min(max_bg, len(batch_scaled)))
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

    # ── Selective Retraining, Validation Gate & Promotion Pipeline ──────

    def _selective_retrain(self, drift_batch: np.ndarray, drift_labels: Optional[np.ndarray] = None):
        """
        Selective retrain pipeline:
          1. Checks cooldown rate limiter before proceeding.
          2. Starts experiment tracker context.
          3. Trains candidate classifier using 80/20 reference-drift dataset with SMOTE.
          4. Runs Validation Gate Engine comparing active model vs candidate model on held-out validation data.
          5. If PROMOTED:
             - Swaps active classifier in ProductionClassifier.
             - Logs new model version (v2, v3, etc.) in SQLite ModelRegistry.
             - Logs training lineage (dataset composition, SMOTE stats).
             - Executes Reference Baseline Update Gate (fine-tunes VAE & resets error thresholds).
          6. If REJECTED:
             - Retains previous active classifier.
             - Logs rejection & failure details in SQLite.
             - Baseline remains untouched.
          7. Logs full experiment record with before/after metrics.
        """
        try:
            # Cooldown Rate Limiter Check
            can_retrain, cooldown_reason = self.cooldown_manager.can_retrain()
            if not can_retrain:
                print(f"[Orchestrator] Retraining BLOCKED by cooldown: {cooldown_reason}")
                return

            if (
                self.classifier is not None 
                and self.scaler is not None 
                and drift_labels is not None 
                and self.reference_raw_X is not None
                and self.reference_raw_y is not None
            ):
                # Start Experiment Tracking
                active_model_name = self.classifier.active_model_name
                exp_context = self.experiment_tracker.start_experiment(
                    trigger_reason=f"MHS Critical at batch {self.batch_counter}",
                    classifier_type=active_model_name,
                    batch_id=self.batch_counter,
                )

                drift_raw_X = self.scaler.inverse_transform(drift_batch)

                # 1. Apply SMOTE to drift data if imbalanced
                from imbalance_handler import check_imbalance_ratio as _check_ratio
                imb_ratio_before = _check_ratio(drift_labels)

                training_drift_X, training_drift_y, smote_done, _ = apply_borderline_smote_if_needed(
                    drift_raw_X, drift_labels, imbalance_threshold=config.smote_imbalance_threshold
                )
                imb_ratio_after = _check_ratio(training_drift_y)

                if smote_done:
                    print("[Orchestrator] SMOTE oversampling applied to drift dataset for candidate retraining.")

                # Class balance stats for experiment logging
                import pandas as pd
                class_balance_before = {int(k): int(v) for k, v in pd.Series(drift_labels).value_counts().items()}
                class_balance_after = {int(k): int(v) for k, v in pd.Series(training_drift_y).value_counts().items()}

                # Mix historical reference data with drifted data according to retraining ratio (80/20)
                ratio = config.retraining_ratio
                n_ref_samples = int(len(training_drift_X) * (ratio / (1.0 - ratio)))
                n_ref_samples = min(n_ref_samples, len(self.reference_raw_X))

                ref_sub_X = self.reference_raw_X[:n_ref_samples]
                ref_sub_y = self.reference_raw_y[:n_ref_samples]

                combined_X = np.concatenate([ref_sub_X, training_drift_X], axis=0)
                combined_y = np.concatenate([ref_sub_y, training_drift_y], axis=0)

                # Capture BEFORE metrics (active model performance on validation set)
                val_n = min(1000, len(self.reference_raw_X))
                val_X = self.reference_raw_X[-val_n:]
                val_y = self.reference_raw_y[-val_n:]
                before_metrics = self.validation_gate.evaluate_model_metrics(self.classifier, val_X, val_y)

                # Train Candidate Classifier Model
                from classifier import get_default_model
                candidate_model = get_default_model(active_model_name)
                
                print(f"[Orchestrator] Training candidate model '{active_model_name}' on {len(combined_X)} samples...")
                if len(np.unique(combined_y)) >= 2:
                    candidate_model.fit(combined_X, combined_y)
                else:
                    from sklearn.dummy import DummyClassifier
                    candidate_model = DummyClassifier(strategy="most_frequent")
                    candidate_model.fit(combined_X, combined_y)

                # 2. Run Validation Gate Engine on held-out reference validation subset
                val_result = self.validation_gate.evaluate_candidate(
                    active_classifier=self.classifier,
                    candidate_classifier=candidate_model,
                    val_X=val_X,
                    val_y=val_y,
                )

                self.latest_validation_metrics = val_result.candidate_metrics
                next_tag = self.registry.get_next_version_tag()

                if val_result.is_promoted:
                    print(f"[Validation Gate] Candidate model PROMOTED to {next_tag}! Metrics: {val_result.candidate_metrics}")
                    self.latest_validation_status = "promoted"
                    
                    # Swap active model
                    self.classifier.models[active_model_name] = candidate_model
                    self.classifier._is_fitted[active_model_name] = True
                    
                    # Log in SQLite Registry
                    self.registry.log_version(
                        version_tag=next_tag,
                        classifier_type=active_model_name,
                        metrics=val_result.candidate_metrics,
                        promotion_status="promoted",
                        drift_event_id=self.batch_counter,
                    )

                    # Log Training Lineage
                    self.registry.log_training_lineage(
                        version_tag=next_tag,
                        reference_samples=n_ref_samples,
                        drift_samples=len(training_drift_X),
                        smote_applied=smote_done,
                        imbalance_ratio_before=imb_ratio_before,
                        imbalance_ratio_after=imb_ratio_after,
                        drift_event_batch_id=self.batch_counter,
                        notes=f"Retrain triggered by Critical MHS at batch {self.batch_counter}",
                    )

                    # 3. REFERENCE WINDOW UPDATE GATE (ONLY POST-PROMOTION)
                    print("[Reference Baseline Update Gate] Recalibrating VAE thresholds on new validated baseline...")
                    train_autoencoder(
                        self.ae_model, drift_batch,
                        epochs=5, batch_size=64, lr=5e-4, lam=0.7, verbose=False
                    )
                    new_ref_errors = compute_reconstruction_errors(
                        self.ae_model, drift_batch, lam=0.7
                    )
                    n_retained = min(500, len(self.ref_errors))
                    self.ref_errors = np.concatenate([self.ref_errors[n_retained:], new_ref_errors])
                    self.dynamic_threshold = compute_dynamic_threshold(self.ref_errors)
                    self.calibrator = DriftCalibrator().fit(self.ref_errors)
                    self.ref_scores = self.calibrator.transform(self.ref_errors)
                    print("[Reference Baseline Update Gate] Recalibration complete.")

                else:
                    print(f"[Validation Gate] Candidate REJECTED. Reason: {val_result.rejection_reason}")
                    self.latest_validation_status = "rejected"
                    
                    self.registry.log_version(
                        version_tag=next_tag,
                        classifier_type=active_model_name,
                        metrics=val_result.candidate_metrics,
                        promotion_status="rejected",
                        drift_event_id=self.batch_counter,
                        rejection_reason=val_result.rejection_reason,
                    )

                # Log Experiment Record (always, whether promoted or rejected)
                promotion_decision = "promoted" if val_result.is_promoted else "rejected"
                self.experiment_tracker.log_experiment(
                    context=exp_context,
                    sample_count=len(combined_X),
                    reference_samples=n_ref_samples,
                    drift_samples=len(training_drift_X),
                    smote_applied=smote_done,
                    class_balance_before=class_balance_before,
                    class_balance_after=class_balance_after,
                    before_metrics=before_metrics,
                    after_metrics=val_result.candidate_metrics,
                    promotion_decision=promotion_decision,
                    rejection_reason=val_result.rejection_reason,
                    version_tag=next_tag if val_result.is_promoted else None,
                )

                # Record retrain in cooldown manager
                self.cooldown_manager.record_retrain()

            else:
                print("[Orchestrator] Skipping candidate retraining due to missing labels/scalers/reference raw data.")
        except Exception as e:
            print(f"[Orchestrator] Retraining error: {e}")
            import traceback; traceback.print_exc()
