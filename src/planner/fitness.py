"""
Route fitness scoring: travel time, distance, and congestion cost.

score_route() combines three weighted terms into a single scalar (lower is
better) — the intended QPSO fitness function once a candidate particle
position has been decoded to a stop order (see qpso_encoding.decode_order /
qpso_optimizer.optimize_tour):

    score = w1*T + w2*D + w3*C

- T: total travel time between consecutive stops in `order`, read from
  `distance_matrix`. That matrix MUST be built from live/current edge
  weights (e.g. a state.py subscription snapshot fed through
  qpso_encoding.compute_distance_matrix), not static free-flow times —
  otherwise T does not reflect current congestion at all.
- D: total distance between consecutive stops, read from the SAME
  `distance_matrix` as T. This project's pipeline currently has only one
  precomputed stop-to-stop matrix, so T and D are numerically identical by
  construction; they are kept as separate terms so a future second matrix
  (pure physical distance, decoupled from live travel time) can be dropped
  in without changing this function's contract or weighting scheme.
- C: congestion cost. For every road-network edge touched by the route —
  every edge on every leg (order[k] -> order[k+1]) — add
  (occupancy / capacity) ** 2. Squaring means a near-saturated edge
  (ratio close to 1) contributes disproportionately more than a lightly
  used one, rather than penalizing congestion linearly. A route that
  traverses the same physical edge on two different legs is charged twice,
  since it really does add load to that edge twice.

`weights` is normalized so w1 + w2 + w3 == 1 before use, regardless of what
scale the caller passes in.
"""

from typing import Dict, List, Tuple

import numpy as np

# Maps a route leg -- the pair of stop indices (order[k], order[k+1]) exactly
# as they appear consecutively in `order` -- to every road-network edge that
# leg passes through, each as {"occupancy": float, "capacity": float}.
CongestionLookup = Dict[Tuple[int, int], List[Dict[str, float]]]


def route_components(
    order: np.ndarray,
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
) -> Tuple[float, float, float]:
    """
    Compute the raw (T, D, C) components for a route, before weighting.

    Args:
        order: Visit-order permutation (indices into distance_matrix), e.g.
            from qpso_encoding.decode_order().
        distance_matrix: (n, n) live-weighted travel-time/distance matrix,
            e.g. from qpso_encoding.compute_distance_matrix().
        congestion_lookup: See CongestionLookup above. A leg absent from the
            lookup contributes 0 to C (treated as uncongested).

    Returns:
        (T, D, C) -- see module docstring for definitions.
    """
    legs = list(zip(order[:-1], order[1:]))

    T = 0.0
    D = 0.0
    for i, j in legs:
        d = float(distance_matrix[i, j])
        T += d
        D += d

    C = 0.0
    for i, j in legs:
        for edge in congestion_lookup.get((int(i), int(j)), ()):
            ratio = edge["occupancy"] / edge["capacity"]
            C += ratio ** 2

    return T, D, C


def score_route(
    order: np.ndarray,
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Weighted route fitness: w1*T + w2*D + w3*C, with weights normalized to
    sum to 1 before use. Lower is better.

    Args:
        order: Visit-order permutation (indices into distance_matrix).
        distance_matrix: (n, n) live-weighted travel-time/distance matrix.
        congestion_lookup: See CongestionLookup above.
        weights: (w1, w2, w3) for (T, D, C). Need not already sum to 1 --
            they are rescaled by their sum before use.

    Returns:
        Scalar fitness score.
    """
    w1, w2, w3 = weights
    total_weight = w1 + w2 + w3
    if total_weight <= 0:
        raise ValueError(f"weights must sum to a positive value, got {weights}")
    w1, w2, w3 = w1 / total_weight, w2 / total_weight, w3 / total_weight

    T, D, C = route_components(order, distance_matrix, congestion_lookup)
    return w1 * T + w2 * D + w3 * C
