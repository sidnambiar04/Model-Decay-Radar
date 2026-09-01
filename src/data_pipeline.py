import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataWindowManager:
    def __init__(self, df, feature_cols, label_col=None):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.scaler = None
        self.reference_medians = None

    def set_reference_window(self, reference_df):
        ref_features = reference_df[self.feature_cols]
        self.reference_medians = ref_features.median()
        ref_filled = ref_features.fillna(self.reference_medians)
        self.scaler = MinMaxScaler()
        self.scaler.fit(ref_filled.values)
        self.reference_raw = reference_df
        self.reference_scaled = self.transform(reference_df)

    def transform(self, batch_df):
        if self.scaler is None:
            raise RuntimeError("Call set_reference_window() first.")
        features = batch_df[self.feature_cols].fillna(self.reference_medians)
        return self.scaler.transform(features.values)
