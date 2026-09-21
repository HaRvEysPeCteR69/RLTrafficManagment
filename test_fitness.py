"""
Unit tests for src/planner/fitness.py, with a hand-checkable 4-stop case.

Route: order = [0, 1, 2, 3] (open path, 3 legs: 0->1, 1->2, 2->3).

distance_matrix leg values (everything else is a deliberately huge decoy
value, so a test failure here would mean the wrong cells are being read):
    leg 0->1: 10
    leg 1->2: 20
    leg 2->3: 15
  => T = D = 10 + 20 + 15 = 45

congestion_lookup (occupancy/capacity per edge, squared, summed):
    leg 0->1: one edge,  occ=5,  cap=10 -> (0.5)^2 = 0.25
    leg 1->2: one edge,  occ=8,  cap=10 -> (0.8)^2 = 0.64
    leg 2->3: two edges, occ=2,  cap=10 -> (0.2)^2 = 0.04
                         occ=9,  cap=10 -> (0.9)^2 = 0.81
  => C = 0.25 + 0.64 + 0.04 + 0.81 = 1.74

weights = (0.5, 0.3, 0.2) (already sums to 1)
  => score = 0.5*45 + 0.3*45 + 0.2*1.74 = 22.5 + 13.5 + 0.348 = 36.348
"""

import numpy as np

from src.planner.fitness import route_components, score_route

ORDER = np.array([0, 1, 2, 3])

# Decoy value on every unused cell: if the implementation ever reads the
# wrong (i, j) pair, T/D would come out wildly wrong instead of silently
# matching by coincidence.
DECOY = 999.0
DISTANCE_MATRIX = np.array([
    [0.0,   10.0,  DECOY, DECOY],
    [10.0,  0.0,   20.0,  DECOY],
    [DECOY, 20.0,  0.0,   15.0],
    [DECOY, DECOY, 15.0,  0.0],
])

CONGESTION_LOOKUP = {
    (0, 1): [{"occupancy": 5.0, "capacity": 10.0}],
    (1, 2): [{"occupancy": 8.0, "capacity": 10.0}],
    (2, 3): [
        {"occupancy": 2.0, "capacity": 10.0},
        {"occupancy": 9.0, "capacity": 10.0},
    ],
}

EXPECTED_T = 45.0
EXPECTED_D = 45.0
EXPECTED_C = 1.74


def test_route_components_hand_checked():
    T, D, C = route_components(ORDER, DISTANCE_MATRIX, CONGESTION_LOOKUP)
    assert np.isclose(T, EXPECTED_T)
    assert np.isclose(D, EXPECTED_D)
    assert np.isclose(C, EXPECTED_C)


def test_score_route_hand_checked():
    weights = (0.5, 0.3, 0.2)  # already sums to 1
    expected = 0.5 * EXPECTED_T + 0.3 * EXPECTED_D + 0.2 * EXPECTED_C
    assert np.isclose(expected, 36.348)

    score = score_route(ORDER, DISTANCE_MATRIX, CONGESTION_LOOKUP, weights)
    assert np.isclose(score, expected)


def test_score_route_normalizes_unnormalized_weights():
    # (5, 3, 2) has the same ratios as (0.5, 0.3, 0.2) but sums to 10, not 1.
    raw_weights = (5.0, 3.0, 2.0)
    normalized_score = score_route(ORDER, DISTANCE_MATRIX, CONGESTION_LOOKUP, (0.5, 0.3, 0.2))
    raw_score = score_route(ORDER, DISTANCE_MATRIX, CONGESTION_LOOKUP, raw_weights)
    assert np.isclose(raw_score, normalized_score)


def test_missing_leg_in_congestion_lookup_contributes_zero():
    lookup_without_last_leg = {
        (0, 1): CONGESTION_LOOKUP[(0, 1)],
        (1, 2): CONGESTION_LOOKUP[(1, 2)],
        # (2, 3) intentionally omitted
    }
    T, D, C = route_components(ORDER, DISTANCE_MATRIX, lookup_without_last_leg)
    assert np.isclose(T, EXPECTED_T)  # T/D unaffected by congestion data
    assert np.isclose(D, EXPECTED_D)
    assert np.isclose(C, 0.25 + 0.64)  # last leg's 0.85 dropped


def test_zero_sum_weights_raise():
    try:
        score_route(ORDER, DISTANCE_MATRIX, CONGESTION_LOOKUP, (0.0, 0.0, 0.0))
        assert False, "expected ValueError for zero-sum weights"
    except ValueError:
        pass


if __name__ == "__main__":
    test_route_components_hand_checked()
    test_score_route_hand_checked()
    test_score_route_normalizes_unnormalized_weights()
    test_missing_leg_in_congestion_lookup_contributes_zero()
    test_zero_sum_weights_raise()
    print("OK: all fitness.py unit tests passed.")
