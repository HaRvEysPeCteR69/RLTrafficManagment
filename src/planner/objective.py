"""
Routing Objective / Fitness functions.

Combines:
- Total expected travel time
- Volatility / risk penalty
- Delay / deadline penalty
- Energy / distance cost
"""

import numpy as np
from typing import Dict, List, Any


class RoutingObjective:
    """
    Evaluates fitness of a route or permutation under dynamic network weights.
    """

    def __init__(
        self,
        weight_travel_time: float = 1.0,
        weight_volatility: float = 0.5,
        weight_distance: float = 0.1,
    ):
        self.w_time = weight_travel_time
        self.w_vol = weight_volatility
        self.w_dist = weight_distance

    def evaluate(
        self,
        route_edges: List[str],
        edge_travel_times: Dict[str, float],
        edge_volatilities: Dict[str, float],
        edge_lengths: Dict[str, float],
    ) -> float:
        """Calculate aggregate fitness (lower is better)."""
        total_time = sum(edge_travel_times.get(e, 30.0) for e in route_edges)
        total_volatility = sum(edge_volatilities.get(e, 0.0) for e in route_edges)
        total_dist = sum(edge_lengths.get(e, 100.0) for e in route_edges)

        cost = (
            self.w_time * total_time
            + self.w_vol * total_volatility
            + self.w_dist * total_dist
        )
        return cost
