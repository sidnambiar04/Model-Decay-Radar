"""
Scenario Simulator Engine — Model Decay Radar
===============================================
Generates realistic, controlled production-like data streams for demonstration:
  1. Normal: Baseline nominal production data.
  2. Gradual Drift: Linearly increasing sensor drift.
  3. Sudden Drift: Immediate step-function sensor shift.
  4. Class Imbalance: Skewed anomaly distribution.
  5. Recovery: Post-retraining nominal data showing MHS recovery.

CRITICAL REQUIREMENT:
The simulator ONLY generates/modifies incoming sensor feature data.
It does NOT mock or fake monitoring results or MHS scores. All monitoring
metrics emerge naturally from running the actual algorithms.
"""

import os
import uuid
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional

_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/synthetic_swat.csv")


class ScenarioSimulator:
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or _DATA_PATH
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            self.feature_cols = [c for c in self.df.columns if c not in ("Label", "_window")]
            self.label_col = "Label"
        else:
            self.df = None
            self.feature_cols = []
            self.label_col = "Label"

    @staticmethod
    def generate_prediction_id() -> str:
        return f"pred_{uuid.uuid4().hex[:10]}"

    def get_scenario_batch(
        self,
        scenario: str = "normal",
        n_samples: int = 500,
        drift_strength: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generates raw features (np.ndarray), labels (np.ndarray), and prediction_ids (list).
        """
        if self.df is None or len(self.feature_cols) == 0:
            # Fallback dummy generation if CSV not found
            features = np.random.normal(50.0, 5.0, size=(n_samples, 51)).astype(np.float32)
            labels = np.zeros(n_samples, dtype=np.int32)
            pred_ids = [self.generate_prediction_id() for _ in range(n_samples)]
            return features, labels, pred_ids

        scenario = scenario.lower()
        pred_ids = [self.generate_prediction_id() for _ in range(n_samples)]

        if scenario == "normal":
            stable_df = self.df[self.df["_window"] == "current_stable"]
            sample_df = stable_df.sample(n=n_samples, replace=True, random_state=42)
            raw_X = sample_df[self.feature_cols].values.astype(np.float32)
            labels = sample_df[self.label_col].values.astype(np.int32)

        elif scenario == "gradual":
            drift_df = self.df[self.df["_window"] == "current_drift"]
            sample_df = drift_df.sample(n=n_samples, replace=True, random_state=42)
            raw_X = sample_df[self.feature_cols].values.astype(np.float32).copy()
            labels = sample_df[self.label_col].values.astype(np.int32)

            # Apply gradual ramp scaling based on drift_strength
            ramp = np.linspace(0.2, 1.0, n_samples).reshape(-1, 1)
            target_cols = ["LIT101", "DPIT301", "P402"]
            for col in target_cols:
                if col in self.feature_cols:
                    idx = self.feature_cols.index(col)
                    raw_X[:, idx] += (ramp.ravel() * 15.0 * drift_strength).astype(np.float32)

        elif scenario == "sudden":
            drift_df = self.df[self.df["_window"] == "current_drift"]
            sample_df = drift_df.sample(n=n_samples, replace=True, random_state=42)
            raw_X = sample_df[self.feature_cols].values.astype(np.float32).copy()
            labels = sample_df[self.label_col].values.astype(np.int32)

            # Apply immediate step shift
            target_cols = ["LIT101", "DPIT301", "P402"]
            for col in target_cols:
                if col in self.feature_cols:
                    idx = self.feature_cols.index(col)
                    raw_X[:, idx] += np.float32(35.0 * drift_strength)

        elif scenario in ("imbalance", "class_imbalance"):
            # Generates highly imbalanced dataset (97% normal, 3% anomaly)
            stable_df = self.df[self.df["_window"] == "current_stable"]
            drift_df = self.df[self.df["_window"] == "current_drift"]

            n_norm = int(n_samples * 0.97)
            n_anom = n_samples - n_norm

            norm_sample = stable_df[stable_df[self.label_col] == 0].sample(n=n_norm, replace=True, random_state=42)
            anom_sample = drift_df[drift_df[self.label_col] == 1].sample(n=n_anom, replace=True, random_state=42)

            combined_df = pd.concat([norm_sample, anom_sample]).sample(frac=1.0, random_state=42)
            raw_X = combined_df[self.feature_cols].values.astype(np.float32)
            labels = combined_df[self.label_col].values.astype(np.int32)

        elif scenario == "recovery":
            # Post-retraining nominal data showing recovery back to healthy MHS
            stable_df = self.df[self.df["_window"] == "current_stable"]
            sample_df = stable_df.sample(n=n_samples, replace=True, random_state=99)
            raw_X = sample_df[self.feature_cols].values.astype(np.float32)
            labels = sample_df[self.label_col].values.astype(np.int32)

        else:
            raise ValueError(f"Unknown scenario: '{scenario}'. Choose from normal, gradual, sudden, imbalance, recovery.")

        return raw_X, labels, pred_ids


class CustomCSVReplayEngine:
    """Batch iterator for custom user-uploaded CSV files in Real Data Monitoring Mode."""
    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"Custom dataset not found at: {csv_filepath}")
        self.df = pd.read_csv(csv_filepath)
        self.feature_cols = [c for c in self.df.columns if c not in ("Label", "label", "target", "Target", "_window", "prediction_id")]
        self.current_idx = 0

    def has_next_batch(self) -> bool:

        return self.current_idx < len(self.df)

    def get_next_batch(self, batch_size: int = 500) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        if not self.has_next_batch():
            self.current_idx = 0  # Loop around if reached end
        
        sub_df = self.df.iloc[self.current_idx : self.current_idx + batch_size]
        self.current_idx += len(sub_df)
        
        raw_X = sub_df[self.feature_cols].values.astype(np.float32)
        
        # Extract label if available
        labels = None
        for lcol in ("Label", "label", "target", "Target"):
            if lcol in sub_df.columns:
                labels = sub_df[lcol].values.astype(np.int32)
                break
                
        pred_ids = [f"custom_{uuid.uuid4().hex[:10]}" for _ in range(len(sub_df))]
        return raw_X, labels, pred_ids

