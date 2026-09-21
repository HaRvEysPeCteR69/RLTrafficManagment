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
