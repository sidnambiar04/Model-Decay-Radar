"""
Centralized Configuration Manager — Model Decay Radar
=====================================================
Centralized object holding all system configurations, thresholds,
weights, and training parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class RadarConfig:
    # Monitoring Window & Pipeline
    window_size: int = 500
    ae_epochs: int = 10
    rnn_epochs: int = 5
    mc_dropout_t: int = 50
    retraining_ratio: float = 0.8  # 80% reference historical data, 20% drift data

    # Statistical Thresholds
    p_value_threshold: float = 0.01
    ks_alpha: float = 0.05
    smote_imbalance_threshold: float = 0.2

    # Active Classifier
    active_classifier: str = "Random Forest"

    # Model Health Score (MHS) Weights
    mhs_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.35,
        "drift": 0.25,
        "uncertainty": 0.20,
        "stability": 0.20,
    })

    # MHS Status Thresholds
    mhs_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "healthy": 0.85,
        "warning": 0.65,
    })

    # Drift Signal Fusion Config
    wasserstein_threshold: float = 0.15
    fusion_vote_threshold: float = 0.35
    fusion_weights: Dict[str, float] = field(default_factory=lambda: {
        "vae": 0.25,
        "kl": 0.25,
        "ks": 0.20,
        "wasserstein": 0.15,
        "supervised": 0.15,
    })

    # Validation Gate & Registry Config
    validation_f1_threshold_margin: float = 0.02
    validation_min_accuracy: float = 0.80
    db_path: str = "models/model_registry.db"

    # Retraining Cooldown & Rate Limiter
    retrain_cooldown_seconds: int = 60
    max_retrains_per_hour: int = 5

    # Experiment Tracker
    experiment_db_path: str = "logs/experiments.db"

    def update(self, **kwargs: Any) -> Dict[str, Any]:
        """Dynamically update configuration values."""
        updated = {}
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
                updated[key] = getattr(self, key)
        return updated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "ae_epochs": self.ae_epochs,
            "rnn_epochs": self.rnn_epochs,
            "mc_dropout_t": self.mc_dropout_t,
            "retraining_ratio": self.retraining_ratio,
            "p_value_threshold": self.p_value_threshold,
            "ks_alpha": self.ks_alpha,
            "smote_imbalance_threshold": self.smote_imbalance_threshold,
            "wasserstein_threshold": self.wasserstein_threshold,
            "fusion_vote_threshold": self.fusion_vote_threshold,
            "validation_f1_threshold_margin": self.validation_f1_threshold_margin,
            "validation_min_accuracy": self.validation_min_accuracy,
            "db_path": self.db_path,
            "retrain_cooldown_seconds": self.retrain_cooldown_seconds,
            "max_retrains_per_hour": self.max_retrains_per_hour,
            "experiment_db_path": self.experiment_db_path,
            "active_classifier": self.active_classifier,
            "mhs_weights": self.mhs_weights,
            "mhs_thresholds": self.mhs_thresholds,
            "fusion_weights": self.fusion_weights,
        }


# Global configuration singleton
config = RadarConfig()
