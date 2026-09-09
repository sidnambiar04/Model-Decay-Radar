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
    """
    Intelligent batch iterator for custom user-uploaded CSV files in Real Data Monitoring Mode.
    Automatically handles arbitrary Kaggle datasets and SWaT test sets:
    - Extracts labels from string/binary/multi-class columns (e.g. 'Normal'/'Attack', 'In'/'Out', 'target', 'class')
    - Parses timestamps/dates into useful numeric features (hour, dayofweek, minute)
    - Drops high-cardinality string identifiers (e.g. IDs, UUIDs)
    - Replaces missing values/NaNs with column medians
    - Converts features to pure float32 arrays
    """
    def __init__(self, csv_filepath: str):
        self.csv_filepath = csv_filepath
        if not os.path.exists(csv_filepath):
            raise FileNotFoundError(f"Custom dataset not found at: {csv_filepath}")
        self.df = pd.read_csv(csv_filepath)
        self.current_idx = 0
        self._process_dataset()

    def _process_dataset(self) -> None:
        clean_df = self.df.copy()

        # 1. Identify label column
        label_col = None
        priority_labels = [
            "label", "target", "class", "normal/attack", "attack",
            "out/in", "y", "status", "is_fraud", "outcome", "anomaly"
        ]
        for pl in priority_labels:
            for c in clean_df.columns:
                if c.lower().strip() == pl:
                    label_col = c
                    break
            if label_col:
                break

        if not label_col:
            for c in clean_df.columns:
                c_low = c.lower().strip()
                if any(k in c_low for k in ["label", "target", "attack", "anomaly"]):
                    label_col = c
                    break

        labels: Optional[np.ndarray] = None
        if label_col:
            raw_l = clean_df[label_col].astype(str).str.strip()
            normal_words = {"normal", "in", "0", "false", "no", "benign", "good", "healthy", "0.0"}
            unique_vals = list(raw_l.unique())
            if len(unique_vals) == 2 and any(v.lower() in normal_words for v in unique_vals):
                norm_val = [v for v in unique_vals if v.lower() in normal_words][0]
                labels = (raw_l != norm_val).astype(np.int32).values
            else:
                codes, _ = pd.factorize(clean_df[label_col])
                labels = codes.astype(np.int32)
            clean_df = clean_df.drop(columns=[label_col])

        # 2. Extract date/time features and drop pure identifier columns
        cols_to_drop = []
        for c in clean_df.columns:
            c_low = c.lower().strip()
            if c_low in ["_window", "prediction_id", "id"] or c_low.endswith("/id") or c_low.startswith("id_") or c_low.endswith("_id"):
                cols_to_drop.append(c)
                continue
            if any(d in c_low for d in ["date", "time", "timestamp"]):
                try:
                    dt_series = pd.to_datetime(clean_df[c], errors="coerce")
                    if dt_series.notna().sum() > 0.5 * len(clean_df):
                        clean_df[f"{c}_hour"] = dt_series.dt.hour.fillna(0).astype(float)
                        clean_df[f"{c}_dayofweek"] = dt_series.dt.dayofweek.fillna(0).astype(float)
                        clean_df[f"{c}_minute"] = dt_series.dt.minute.fillna(0).astype(float)
                except Exception:
                    pass
                cols_to_drop.append(c)

        clean_df = clean_df.drop(columns=cols_to_drop, errors="ignore")

        # 3. Numeric conversion for remaining feature columns
        feature_cols = []
        for c in clean_df.columns:
            num_s = pd.to_numeric(clean_df[c], errors="coerce")
            if num_s.notna().sum() > 0.5 * len(clean_df):
                clean_df[c] = num_s.fillna(num_s.median() if num_s.notna().any() else 0.0)
                feature_cols.append(c)

        if not feature_cols:
            raise ValueError("No valid numeric feature columns found in uploaded dataset.")

        self.feature_cols = feature_cols
        self.raw_X = clean_df[self.feature_cols].values.astype(np.float32)
        self.labels = labels

    def has_next_batch(self) -> bool:
        return self.current_idx < len(self.raw_X)

    def get_next_batch(self, batch_size: int = 500) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        total_rows = len(self.raw_X)
        if total_rows == 0:
            raise ValueError("Dataset contains 0 rows.")

        if self.current_idx >= total_rows:
            self.current_idx = 0  # Loop around

        end_idx = min(self.current_idx + batch_size, total_rows)
        raw_X_batch = self.raw_X[self.current_idx : end_idx]
        labels_batch = self.labels[self.current_idx : end_idx] if self.labels is not None else None
        
        self.current_idx = end_idx
        if self.current_idx >= total_rows:
            self.current_idx = 0

        pred_ids = [f"custom_{uuid.uuid4().hex[:10]}" for _ in range(len(raw_X_batch))]
        return raw_X_batch, labels_batch, pred_ids

