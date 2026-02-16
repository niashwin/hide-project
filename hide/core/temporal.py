"""
Temporal encoding and decay functions for time-aware memory.
"""

import numpy as np


class TemporalEncoding:
    """Multi-scale sinusoidal temporal encoding."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        assert dim % 3 == 0 or dim >= 6
        self.scale_dim = dim // 3
        # Three scales: fine (1 day), medium (30 days), coarse (365 days)
        self.scales = [1.0, 30.0, 365.0]

    def encode(self, time_days: float) -> np.ndarray:
        """Encode a timestamp (in days) as a temporal vector."""
        components = []
        for scale in self.scales:
            t = time_days / scale
            d = self.scale_dim
            positions = np.arange(d // 2)
            freqs = 1.0 / (10000.0 ** (2 * positions / d))
            components.append(np.sin(t * freqs))
            components.append(np.cos(t * freqs))
        vec = np.concatenate(components)[:self.dim]
        return vec


def power_law_decay(t: float, beta: float = 1.0, psi: float = 0.5) -> float:
    """Power-law decay: S(t) = (1 + beta * t)^(-psi)."""
    return (1.0 + beta * t) ** (-psi)


def exponential_decay(t: float, lam: float = 0.1) -> float:
    """Exponential decay: S(t) = exp(-lambda * t)."""
    return np.exp(-lam * t)
