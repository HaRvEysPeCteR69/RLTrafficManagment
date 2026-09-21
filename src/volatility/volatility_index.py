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
    "highly volatile" for this network -- tune it to the network's typical
    free-flow speed (e.g. a busier, higher-speed corridor should use a
    larger reference so ordinary noise doesn't read as already-saturated
    volatility).
    """

    def __init__(self, window_size: int = 15, reference_variance: float = 4.0):
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
