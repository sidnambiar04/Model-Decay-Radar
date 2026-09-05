"""
Data Pipeline & Window Manager — Model Decay Radar
===================================================
Manages reference historical data windows, sliding monitoring buffers,
schema validation, missing value imputation, and StandardScaler preprocessing
locked exclusively to the reference window distribution.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler


class DataWindowManager:
    """
    Data Window Manager ensuring reference preprocessing parameters (mean, std)
    are fitted strictly once on reference data and never re-fitted on incoming batches.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: List[str], label_col: Optional[str] = "Label"):
        self.df = df.reset_index(drop=True)
        self.feature_cols = list(feature_cols)
        self.label_col = label_col
        self.scaler: Optional[StandardScaler] = None
        self.reference_medians: Optional[pd.Series] = None
        
        self.reference_raw_df: Optional[pd.DataFrame] = None
        self.reference_raw_X: Optional[np.ndarray] = None
        self.reference_scaled: Optional[np.ndarray] = None
        self.reference_y: Optional[np.ndarray] = None

    def validate_schema(self, batch_df: pd.DataFrame) -> bool:
        """Verify all required sensor feature columns are present."""
        missing = [c for c in self.feature_cols if c not in batch_df.columns]
        if missing:
            raise ValueError(f"Schema Validation Error: Missing required feature columns {missing[:5]}...")
        return True

    def set_reference_window(self, reference_df: pd.DataFrame) -> None:
        """
        Fit scaler and store reference baseline distribution metrics.
        """
        self.validate_schema(reference_df)
        self.reference_raw_df = reference_df.reset_index(drop=True)
        
        ref_features = self.reference_raw_df[self.feature_cols]
        self.reference_medians = ref_features.median()
        ref_filled = ref_features.fillna(self.reference_medians)
        
        # Fit StandardScaler strictly on reference data
        self.scaler = StandardScaler()
        self.reference_raw_X = ref_filled.values.astype(np.float32)
        self.reference_scaled = np.clip(self.scaler.fit_transform(self.reference_raw_X), -3.0, 3.0).astype(np.float32)
        
        if self.label_col and self.label_col in self.reference_raw_df.columns:
            self.reference_y = self.reference_raw_df[self.label_col].values.astype(np.int32)
        else:
            self.reference_y = None

    def transform(self, batch_df: pd.DataFrame) -> np.ndarray:
        """
        Impute missing values and scale batch dataframe using locked reference scaler.
        """
        if self.scaler is None:
            raise RuntimeError("Call set_reference_window() before transform().")
            
        self.validate_schema(batch_df)
        features = batch_df[self.feature_cols].fillna(self.reference_medians)
        scaled = self.scaler.transform(features.values.astype(np.float32))
        return np.clip(scaled, -3.0, 3.0).astype(np.float32)

    def transform_raw(self, raw_X: np.ndarray) -> np.ndarray:
        """
        Scale raw numpy array directly using locked reference scaler.
        """
        if self.scaler is None:
            raise RuntimeError("Call set_reference_window() before transform_raw().")
            
        scaled = self.scaler.transform(raw_X.astype(np.float32))
        return np.clip(scaled, -3.0, 3.0).astype(np.float32)
