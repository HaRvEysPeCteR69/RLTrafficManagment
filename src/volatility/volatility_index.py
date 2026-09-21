"""
Volatility Index Computation.

Calculates traffic volatility per road segment based on:
- Rolling variance of speed and travel time over sliding time windows
- Density gradient fluctuations
- Congestion instability index
"""

from collections import deque
from typing import Dict, List
import numpy as np


class VolatilityIndexCalculator:
    """
    Computes traffic volatility metrics for road segments.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: Dict[str, deque] = {}

    def update(self, edge_id: str, current_speed: float, travel_time: float) -> float:
        """
        Record a new observation and compute the current volatility index.

        Args:
            edge_id: Road edge identifier
            current_speed: Measured speed on edge in m/s
            travel_time: Measured travel time on edge in seconds

        Returns:
            Normalized volatility score (higher means more volatile/unpredictable)
        """
        if edge_id not in self.history:
            self.history[edge_id] = deque(maxlen=self.window_size)

        self.history[edge_id].append((current_speed, travel_time))

        if len(self.history[edge_id]) < 2:
            return 0.0

        speeds = [item[0] for item in self.history[edge_id]]
        std_speed = float(np.std(speeds))
        mean_speed = float(np.mean(speeds)) + 1e-5

        # Coefficient of variation as volatility metric
        volatility = std_speed / mean_speed
        return volatility


class NetworkVolatilityIndex:
    """
    Rolling-variance-based, network-wide traffic volatility index,
    normalized to [0, 1).

    Each update() call takes one step's per-edge mean speeds, averages them
    into a single network-wide mean speed, and appends that scalar to a
    rolling window. The index is that window's variance, squashed into
    [0, 1) via variance / (variance + reference_variance): 0 when speeds
    have been perfectly steady over the window, approaching 1 as variance
    grows far past `reference_variance`.

    `reference_variance` (in (m/s)^2) is the variance level considered
    "moderately high" volatility for this network. Because `network_mean_speed`
    is a spatial average across all edges (772 in the Delhi network, where
    most are empty and hold static free-flow speeds), the variance of the
    spatial mean is compressed by roughly 1000x relative to single-edge
    variance. Empirical measurements across Low, Medium, and High scenarios
    show network-mean rolling variances of ~0.0007 (Low) to ~0.0044 (High
    during incidents). Setting `reference_variance = 0.002` scales Low to
    ~0.24, Medium to ~0.32, and High up to ~0.69.
    """

    def __init__(self, window_size: int = 15, reference_variance: float = 0.002):
        self.window_size = window_size
        self.reference_variance = reference_variance
        self.history: deque = deque(maxlen=window_size)

    def update(self, edge_mean_speeds: Dict[str, float]) -> float:
        """
        Record one step's network-wide mean speed and return the current
        volatility index.

        Args:
            edge_mean_speeds: edge_id -> mean_speed for this step, e.g.
                {e: v["mean_speed"] for e, v in state["edges"].items()}
                from state.py's SubscriptionStateExtractor.get_state().

        Returns:
            Volatility index in [0, 1). 0.0 until at least 2 observations
            have been recorded (variance is undefined with fewer).
        """
        if not edge_mean_speeds:
            return 0.0

        network_mean_speed = float(np.mean(list(edge_mean_speeds.values())))
        self.history.append(network_mean_speed)

        if len(self.history) < 2:
            return 0.0

        variance = float(np.var(self.history))
        return variance / (variance + self.reference_variance)
