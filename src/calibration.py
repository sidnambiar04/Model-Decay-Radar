import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.stats import wasserstein_distance, ks_2samp

# Permutation test configuration
N_PERMUTATIONS: int = 200
DRIFT_P_VALUE_THRESHOLD: float = 0.01


class DriftCalibrator:
    def __init__(self):
        self.iso = None
    def fit(self, reference_errors):
        sorted_e = np.sort(reference_errors)
        n = len(sorted_e)
        percentiles = np.linspace(0, 1, n)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(sorted_e, percentiles)
        return self
    def transform(self, errors):
        return self.iso.predict(errors)

def compute_histogram(scores, n_bins=15, bin_edges=None):
    if bin_edges is None:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    counts, edges = np.histogram(scores, bins=bin_edges)
    freqs = (counts + 1e-6) / (counts.sum() + 1e-6 * len(counts))
    return freqs, edges

def kl_divergence(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    return float(np.sum(p * np.log(p / q)))

def permutation_drift_test(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    n_bins: int = 15,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    p_hist, _ = compute_histogram(reference_scores, n_bins=n_bins, bin_edges=bin_edges)
    q_hist, _ = compute_histogram(current_scores,   n_bins=n_bins, bin_edges=bin_edges)
    observed_kl = kl_divergence(p_hist, q_hist)
    
    # Calculate Wasserstein Distance
    wd = float(wasserstein_distance(reference_scores, current_scores))
    
    # Calculate Overall KS Test on Scores
    ks_res = ks_2samp(reference_scores, current_scores)
    ks_stat = float(ks_res.statistic)
    ks_p_value = float(ks_res.pvalue)

    pooled = np.concatenate([reference_scores, current_scores])
    n_ref  = len(reference_scores)
    permuted_kls = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(pooled)
        pp, _ = compute_histogram(pooled[:n_ref], n_bins=n_bins, bin_edges=bin_edges)
        qp, _ = compute_histogram(pooled[n_ref:], n_bins=n_bins, bin_edges=bin_edges)
        permuted_kls[i] = kl_divergence(pp, qp)
    p_value = float(np.mean(permuted_kls >= observed_kl))
    
    return {
        "observed_kl": observed_kl, 
        "p_value": p_value,
        "drift_confirmed": bool(p_value < DRIFT_P_VALUE_THRESHOLD),
        "wasserstein_distance": wd,
        "score_ks_p_value": ks_p_value,
        "score_ks_statistic": ks_stat,
        "reference_histogram": p_hist.tolist(), # Convert to list for JSON serialization
        "current_histogram": q_hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "permuted_kls": permuted_kls.tolist()
    }

def compute_feature_ks_tests(
    reference_features: np.ndarray,
    current_features: np.ndarray,
    feature_names: list[str],
    alpha: float = 0.05
) -> list[dict]:
    """Compute per-feature Kolmogorov-Smirnov test to measure data drift significance for each feature."""
    results = []
    for col_idx, name in enumerate(feature_names):
        ref_col = reference_features[:, col_idx]
        cur_col = current_features[:, col_idx]
        res = ks_2samp(ref_col, cur_col)
        p_val = float(res.pvalue)
        results.append({
            "feature": name,
            "importance": float(res.statistic), # use statistic as shift magnitude
            "p_value": p_val,
            "drift_confirmed": bool(p_val < alpha)
        })
    # Sort by p-value (most statistically drifted first)
    results = sorted(results, key=lambda x: x["p_value"])
    return results
