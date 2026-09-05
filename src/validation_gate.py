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
    rejection_reason: Optional[str] = None


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
            rejection_reason=rejection_reason,
        )
