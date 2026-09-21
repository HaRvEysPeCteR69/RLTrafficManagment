"""
Unit tests for src/reactive/reactive.py, with hand-checkable synthetic cases.

Synthetic junction layout (a diamond: two parallel edges from J1 to J2):
    J1 --(e_congested, length=100)--> J2
    J1 --(e_alt_good,   length=110)--> J2
    J1 --(e_alt_worse,  length=200)--> J2   (too long: fails distance budget)
    J1 --(e_alt_close,  length=100)--> J2   (occupancy too close: fails improvement bar)
    J2 --(e_downstream, length=100)--> J3   (different from-node: never a candidate)
"""

import numpy as np

from src.reactive.reactive import evaluate_vehicle_reroute, find_alternative_edge


class FakeNetworkGraph:
    """Minimal stand-in for NetworkGraph -- just .edges and .adjacency."""

    def __init__(self, edges, adjacency):
        self.edges = edges
        self.adjacency = adjacency


EDGES = {
    "e_congested": {"from": "J1", "to": "J2", "length": 100.0, "speed": 13.89},
    "e_alt_good":  {"from": "J1", "to": "J2", "length": 110.0, "speed": 13.89},
    "e_alt_worse": {"from": "J1", "to": "J2", "length": 200.0, "speed": 13.89},
    "e_alt_close": {"from": "J1", "to": "J2", "length": 100.0, "speed": 13.89},
    "e_downstream": {"from": "J2", "to": "J3", "length": 100.0, "speed": 13.89},
}
ADJACENCY = {
    "J1": ["e_congested", "e_alt_good", "e_alt_worse", "e_alt_close"],
    "J2": ["e_downstream"],
}
NETWORK_GRAPH = FakeNetworkGraph(EDGES, ADJACENCY)

OCCUPANCY_THRESHOLD = 0.8
MIN_IMPROVEMENT = 0.15
MAX_EXTRA_DISTANCE_RATIO = 0.3  # e_alt_good (110) <= 100*1.3=130: passes; e_alt_worse (200): fails


def _state(occupancies):
    return {"edges": {e: {"occupancy": o} for e, o in occupancies.items()}}


def test_no_reroute_when_below_threshold():
    state = _state({
        "e_congested": 0.5,  # below 0.8 threshold -- not congested enough
        "e_alt_good": 0.1,
    })
    assert find_alternative_edge(
        state, "e_congested", NETWORK_GRAPH,
        occupancy_threshold=OCCUPANCY_THRESHOLD,
        min_occupancy_improvement=MIN_IMPROVEMENT,
        max_extra_distance_ratio=MAX_EXTRA_DISTANCE_RATIO,
    ) is None


def test_reroutes_to_meaningfully_better_comparable_alternative():
    state = _state({
        "e_congested": 0.90,
        "e_alt_good": 0.50,    # 0.90 - 0.50 = 0.40 >= 0.15 improvement; length 110 <= 130: OK
        "e_alt_worse": 0.10,   # even better occupancy, but length 200 > 130: fails distance budget
        "e_alt_close": 0.80,   # 0.90 - 0.80 = 0.10 < 0.15: fails improvement bar
    })
    result = find_alternative_edge(
        state, "e_congested", NETWORK_GRAPH,
        occupancy_threshold=OCCUPANCY_THRESHOLD,
        min_occupancy_improvement=MIN_IMPROVEMENT,
        max_extra_distance_ratio=MAX_EXTRA_DISTANCE_RATIO,
    )
    assert result == "e_alt_good"


def test_no_viable_alternative_returns_none():
    state = _state({
        "e_congested": 0.90,
        "e_alt_good": 0.85,    # improvement only 0.05: fails
        "e_alt_worse": 0.10,   # improvement is huge, but distance fails
        "e_alt_close": 0.80,   # improvement only 0.10: fails
    })
    assert find_alternative_edge(
        state, "e_congested", NETWORK_GRAPH,
        occupancy_threshold=OCCUPANCY_THRESHOLD,
        min_occupancy_improvement=MIN_IMPROVEMENT,
        max_extra_distance_ratio=MAX_EXTRA_DISTANCE_RATIO,
    ) is None


def test_downstream_edge_from_different_junction_never_a_candidate():
    # e_downstream starts at J2, not J1 -- must never be considered even if
    # it looks attractive on paper (it can't reconnect: different from-node).
    state = _state({
        "e_congested": 0.95,
        "e_downstream": 0.0,
    })
    result = find_alternative_edge(
        state, "e_congested", NETWORK_GRAPH,
        occupancy_threshold=OCCUPANCY_THRESHOLD,
        min_occupancy_improvement=MIN_IMPROVEMENT,
        max_extra_distance_ratio=MAX_EXTRA_DISTANCE_RATIO,
    )
    assert result is None


def test_missing_edge_state_is_treated_as_no_reroute():
    # planned_next_edge not present in state["edges"] at all.
    assert find_alternative_edge({"edges": {}}, "e_congested", NETWORK_GRAPH) is None


def test_evaluate_vehicle_reroute_packages_decision():
    state = _state({"e_congested": 0.90, "e_alt_good": 0.50})
    decision = evaluate_vehicle_reroute(
        "veh1", state, "e_congested", NETWORK_GRAPH,
        occupancy_threshold=OCCUPANCY_THRESHOLD,
        min_occupancy_improvement=MIN_IMPROVEMENT,
        max_extra_distance_ratio=MAX_EXTRA_DISTANCE_RATIO,
    )
    assert decision is not None
    assert decision.vehicle_id == "veh1"
    assert decision.from_edge == "e_congested"
    assert decision.to_edge == "e_alt_good"
    assert np.isclose(decision.occupancy_before, 0.90)
    assert np.isclose(decision.occupancy_after, 0.50)


def test_evaluate_vehicle_reroute_none_when_no_reroute_warranted():
    state = _state({"e_congested": 0.5, "e_alt_good": 0.1})
    assert evaluate_vehicle_reroute("veh1", state, "e_congested", NETWORK_GRAPH) is None


if __name__ == "__main__":
    test_no_reroute_when_below_threshold()
    test_reroutes_to_meaningfully_better_comparable_alternative()
    test_no_viable_alternative_returns_none()
    test_downstream_edge_from_different_junction_never_a_candidate()
    test_missing_edge_state_is_treated_as_no_reroute()
    test_evaluate_vehicle_reroute_packages_decision()
    test_evaluate_vehicle_reroute_none_when_no_reroute_warranted()
    print("OK: all reactive.py unit tests passed.")
