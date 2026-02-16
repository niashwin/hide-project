"""
Interference experiment utilities.

Implements the core interference protocol: encode target sentences,
add near/far distractors with age-proportional noise, measure retrieval
accuracy as a function of memory age, and fit power-law forgetting curves.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional


def age_proportional_noise(embedding: np.ndarray, age: float, sigma: float,
                           dim: int) -> np.ndarray:
    """Add age-proportional Gaussian noise to an embedding.

    noise = (sigma * sqrt(age + 0.01) / sqrt(dim)) * z, z ~ N(0, I)
    """
    noise_scale = sigma * np.sqrt(age + 0.01) / np.sqrt(dim)
    noisy = embedding + noise_scale * np.random.randn(*embedding.shape)
    return noisy / np.linalg.norm(noisy)


def power_law(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power-law forgetting: R(t) = a * t^(-b)."""
    return a * np.power(t, -b)


def fit_forgetting_curve(ages: np.ndarray, retentions: np.ndarray,
                         bounds=([0.5, 0.0], [2.0, 2.0])) -> Tuple[float, float, float]:
    """Fit a power-law forgetting curve. Returns (a, b, r_squared)."""
    try:
        valid = (ages > 0) & (retentions > 0) & np.isfinite(retentions)
        if valid.sum() < 3:
            return 0.0, 0.0, 0.0
        popt, _ = curve_fit(power_law, ages[valid], retentions[valid],
                            p0=[1.0, 0.5], bounds=bounds, maxfev=5000)
        predicted = power_law(ages[valid], *popt)
        ss_res = np.sum((retentions[valid] - predicted) ** 2)
        ss_tot = np.sum((retentions[valid] - np.mean(retentions[valid])) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return popt[0], popt[1], r_squared
    except (RuntimeError, ValueError):
        return 0.0, 0.0, 0.0


def bootstrap_ci(values: List[float], n_bootstrap: int = 10000,
                 ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap confidence interval."""
    if len(values) == 0:
        return 0.0, 0.0
    arr = np.array(values)
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True))
                  for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), \
           float(np.percentile(boot_means, (1 - alpha) * 100))
