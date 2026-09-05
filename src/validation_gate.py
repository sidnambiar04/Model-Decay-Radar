"""
Validation Gate Engine — Model Decay Radar
==========================================
Enforces comparative validation between candidate retrained models
and active production models before allowing model promotion.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import config


@dataclass
class ValidationResult:
    is_promoted: bool
    active_metrics: Dict[str, float]
    candidate_metrics: Dict[str, float]
    deltas: Dict[str, float]
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result into structured dictionary for UI rendering and API serialization."""
        return {
            "is_promoted": self.is_promoted,
            "promotion_status": "promoted" if self.is_promoted else "rejected",
            "active_metrics": self.active_metrics,
            "candidate_metrics": self.candidate_metrics,
            "deltas": self.deltas,
            "rejection_reason": self.rejection_reason,
        }


class ValidationGateEngine:
    """
    Validation Gate Engine preventing automatic deployment of degraded candidate models.
    """

    @staticmethod
    def evaluate_model_metrics(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if hasattr(model, "predict_batch"):
            preds = model.predict_batch(X)
        elif hasattr(model, "predict"):
            preds = model.predict(X)
        else:
            raise ValueError("Model does not implement predict or predict_batch.")

        acc = float(accuracy_score(y, preds))
        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }

    def evaluate_candidate(
        self,
        active_classifier: Any,
        candidate_classifier: Any,
        val_X: np.ndarray,
        val_y: np.ndarray,
    ) -> ValidationResult:
        active_metrics = self.evaluate_model_metrics(active_classifier, val_X, val_y)
        candidate_metrics = self.evaluate_model_metrics(candidate_classifier, val_X, val_y)

        deltas = {
            "accuracy_delta": round(candidate_metrics["accuracy"] - active_metrics["accuracy"], 4),
            "precision_delta": round(candidate_metrics["precision"] - active_metrics["precision"], 4),
            "recall_delta": round(candidate_metrics["recall"] - active_metrics["recall"], 4),
            "f1_delta": round(candidate_metrics["f1_score"] - active_metrics["f1_score"], 4),
        }

        margin = getattr(config, "validation_f1_threshold_margin", 0.02)
        min_acc = getattr(config, "validation_min_accuracy", 0.80)

        cand_f1 = candidate_metrics["f1_score"]
        act_f1 = active_metrics["f1_score"]
        cand_acc = candidate_metrics["accuracy"]

        rejection_reason = None
        is_promoted = True

        if cand_f1 < (act_f1 - margin):
            is_promoted = False
            rejection_reason = f"Candidate F1 ({cand_f1:.4f}) dropped below active F1 threshold ({act_f1 - margin:.4f})."
        elif cand_acc < min_acc:
            is_promoted = False
            rejection_reason = f"Candidate accuracy ({cand_acc:.4f}) below minimum requirement ({min_acc:.4f})."

        return ValidationResult(
            is_promoted=is_promoted,
            active_metrics=active_metrics,
            candidate_metrics=candidate_metrics,
            deltas=deltas,
            rejection_reason=rejection_reason,
        )

    def evaluate_and_log_registry(
        self,
        active_classifier: Any,
        candidate_classifier: Any,
        val_X: np.ndarray,
        val_y: np.ndarray,
        registry: Any,
        drift_event_id: Optional[int] = None,
        artifact_path: Optional[str] = None,
    ) -> ValidationResult:
        """Evaluate candidate against active baseline and log record directly into ModelRegistry."""
        val_result = self.evaluate_candidate(active_classifier, candidate_classifier, val_X, val_y)
        version_tag = registry.get_next_version_tag() if val_result.is_promoted else f"candidate_{registry.get_next_version_tag()}"
        status = "promoted" if val_result.is_promoted else "rejected"
        classifier_type = getattr(candidate_classifier, "active_model_name", "UnknownClassifier")

        registry.log_version(
            version_tag=version_tag,
            classifier_type=classifier_type,
            metrics=val_result.candidate_metrics,
            promotion_status=status,
            drift_event_id=drift_event_id,
            rejection_reason=val_result.rejection_reason,
            artifact_path=artifact_path if val_result.is_promoted else None,
        )

        return val_result

