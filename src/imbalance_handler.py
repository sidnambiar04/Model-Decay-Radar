import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE

def check_imbalance_ratio(labels):
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        return 1.0
    return float(counts.min() / counts.max())

def apply_borderline_smote_if_needed(X, y, imbalance_threshold=0.2, k_neighbors=5, random_state=42):
    ratio_before = check_imbalance_ratio(y)
    if ratio_before >= imbalance_threshold:
        return X, y, False, ratio_before
    minority_count = int(pd.Series(y).value_counts().min())
    safe_k = min(k_neighbors, max(1, minority_count - 1))
    if minority_count <= 1:
        return X, y, False, ratio_before
    smote = BorderlineSMOTE(k_neighbors=safe_k, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res, True, ratio_before

def get_imbalance_metadata(X, y, imbalance_threshold=0.2):
    """Return detailed class imbalance metadata audit dictionary."""
    counts = pd.Series(y).value_counts().to_dict()
    ratio = check_imbalance_ratio(y)
    is_imbalanced = bool(ratio < imbalance_threshold)
    return {
        "sample_count": len(y),
        "class_distribution": {int(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": round(ratio, 4),
        "is_imbalanced": is_imbalanced,
        "threshold": imbalance_threshold,
    }

