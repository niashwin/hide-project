"""
hide/utils/metrics.py — Evaluation Metrics for All Phases
==========================================================
Bootstrap CIs, Cohen's d, and all task-specific metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats


def accuracy(predictions: List[str], gold: List[str]) -> float:
    """Exact match accuracy, case-insensitive, stripped."""
    correct = sum(
        p.strip().lower() == g.strip().lower()
        for p, g in zip(predictions, gold)
    )
    return correct / len(gold) if gold else 0.0


def precision_at_k(
    retrieved_ids: List[List[int]],
    relevant_ids: List[List[int]],
    k: int,
) -> float:
    """Average precision@k across queries."""
    scores = []
    for ret, rel in zip(retrieved_ids, relevant_ids):
        top_k = ret[:k]
        rel_set = set(rel)
        hits = sum(1 for r in top_k if r in rel_set)
        scores.append(hits / k if k > 0 else 0.0)
    return np.mean(scores)


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval.
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic_fn(sample))
    boot_stats = np.array(boot_stats)
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(statistic_fn(data)), float(ci_lower), float(ci_upper)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² goodness of fit."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def fit_power_law(
    times: np.ndarray, retention: np.ndarray
) -> Dict[str, float]:
    """Fit Ebbinghaus power law R(t) = a * t^(-b)."""
    from scipy.optimize import curve_fit

    def power_law(t, a, b):
        return a * np.power(t + 1e-8, -b)

    try:
        popt, _ = curve_fit(
            power_law, times, retention, p0=[1.0, 0.5], maxfev=10000
        )
        y_pred = power_law(times, *popt)
        return {
            "a": float(popt[0]),
            "b": float(popt[1]),
            "r_squared": r_squared(retention, y_pred),
        }
    except Exception:
        return {"a": 0.0, "b": 0.0, "r_squared": 0.0}


def backward_transfer_matrix(
    task_accuracies: Dict[int, Dict[int, float]]
) -> np.ndarray:
    """
    Compute backward transfer matrix.
    task_accuracies[i][j] = accuracy on task i after learning task j.
    BT[i][j] = acc(i, after j) - acc(i, right after i).
    """
    n_tasks = max(task_accuracies.keys()) + 1
    matrix = np.zeros((n_tasks, n_tasks))
    for i in range(n_tasks):
        baseline = task_accuracies.get(i, {}).get(i, 0.0)
        for j in range(i + 1, n_tasks):
            matrix[i][j] = task_accuracies.get(i, {}).get(j, 0.0) - baseline
    return matrix


def aggregate_seeds(
    seed_results: Dict[int, Dict]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across seeds. Returns mean ± std + CI for each metric.
    """
    first = next(iter(seed_results.values()))
    metrics = {}

    for key, val in first.items():
        if isinstance(val, (int, float)):
            values = np.array([
                seed_results[s].get(key, 0.0) for s in seed_results
            ])
            point, ci_lo, ci_hi = bootstrap_ci(values)
            metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "values": values.tolist(),
            }

    return metrics


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Compute the participation ratio (effective dimensionality).

    d_eff = (sum(lambda_i))^2 / sum(lambda_i^2)
    """
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0.0
    return float((eigenvalues.sum()) ** 2 / (eigenvalues ** 2).sum())
