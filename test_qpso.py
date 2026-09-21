"""
Test script comparing fixed_beta_qpso and va_qpso (src/planner/qpso.py) on
the real Delhi network.

Both variants share the exact same core swarm loop (Sun, Feng & Xu, 2004)
and differ only in how beta is computed (see qpso.py's module docstring).
This exercises both against the same live-weighted distance matrix and
checks:
  - both return a valid, finite-cost tour at least as good as the naive
    identity ordering
  - va_qpso's beta actually responds to volatility_index -- a volatility of
    0.0 vs 1.0 (beta_min vs beta_max, held constant for the whole run)
    changes the swarm's trajectory under an otherwise identical seed.
    Note this is asserted on the gbest position, not the score: different
    trajectories can still converge to the same local optimum, so equal
    final scores are not evidence that beta was ignored.
  - replan() is equivalent to calling va_qpso directly with a
    score_route-based fitness function
"""

import numpy as np

from src.planner.fitness import score_route
from src.planner.qpso import fixed_beta_qpso, va_qpso, replan
from src.planner.qpso_encoding import (
    adjacency_from_network_graph,
    compute_distance_matrix,
    decode_order,
    pick_mutually_reachable_stops,
    tour_length,
)
from src.state_extraction.network_graph import NetworkGraph

NET_FILE = "networks/delhi/delhi_intersection.net.xml"
NUM_STOPS = 8

_network_graph = NetworkGraph(NET_FILE)
_adjacency = adjacency_from_network_graph(_network_graph, edge_weights={})
STOPS = pick_mutually_reachable_stops(_adjacency, NUM_STOPS)
DISTANCE_MATRIX = compute_distance_matrix(_adjacency, STOPS)
assert np.all(np.isfinite(DISTANCE_MATRIX)), "expected all stops to be mutually reachable"

IDENTITY_DISTANCE = tour_length(np.arange(NUM_STOPS), DISTANCE_MATRIX)


def _tour_fitness_fn(x: np.ndarray) -> float:
    return tour_length(decode_order(x), DISTANCE_MATRIX)


def test_fixed_beta_qpso_beats_identity():
    best_position, best_score = fixed_beta_qpso(
        dim=NUM_STOPS,
        fitness_fn=_tour_fitness_fn,
        num_particles=30,
        max_iterations=80,
        seed=42,
    )
    best_order = decode_order(best_position)

    print(f"fixed_beta_qpso: order={best_order.tolist()} score={best_score:.2f} "
          f"(identity={IDENTITY_DISTANCE:.2f})")

    assert sorted(best_order.tolist()) == list(range(NUM_STOPS))
    assert np.isfinite(best_score)
    assert best_score <= IDENTITY_DISTANCE


def test_va_qpso_beats_identity():
    best_position, best_score = va_qpso(
        dim=NUM_STOPS,
        fitness_fn=_tour_fitness_fn,
        volatility_index=0.5,
        num_particles=30,
        max_iterations=80,
        seed=42,
    )
    best_order = decode_order(best_position)

    print(f"va_qpso (volatility=0.5): order={best_order.tolist()} score={best_score:.2f} "
          f"(identity={IDENTITY_DISTANCE:.2f})")

    assert sorted(best_order.tolist()) == list(range(NUM_STOPS))
    assert np.isfinite(best_score)
    assert best_score <= IDENTITY_DISTANCE


def test_va_qpso_beta_responds_to_volatility_index():
    """
    va_qpso holds beta constant for the whole run at
    beta_min + (beta_max - beta_min) * volatility_index. With the same seed
    and fitness landscape, volatility_index=0.0 (beta=beta_min=0.5) and
    volatility_index=1.0 (beta=beta_max=1.0) scale every position update
    differently, so the swarm must traverse a different trajectory.

    The invariant asserted here is on the returned gbest POSITION, not the
    score. Two different trajectories can legitimately converge to the same
    local optimum -- measured over seeds 0-9, the two betas reach an equal
    final score on ~1 seed in 10 (e.g. seed 6, both 87.34) while the gbest
    position differs on every single one. Asserting equal-score-is-a-bug
    would therefore be flaky and would be testing a property QPSO does not
    guarantee.
    """
    position_low, score_low = va_qpso(
        dim=NUM_STOPS,
        fitness_fn=_tour_fitness_fn,
        volatility_index=0.0,
        num_particles=30,
        max_iterations=80,
        seed=123,
    )
    position_high, score_high = va_qpso(
        dim=NUM_STOPS,
        fitness_fn=_tour_fitness_fn,
        volatility_index=1.0,
        num_particles=30,
        max_iterations=80,
        seed=123,
    )

    print(f"va_qpso volatility=0.0 -> score={score_low:.2f}; "
          f"volatility=1.0 -> score={score_high:.2f} "
          f"(same local optimum: {np.isclose(score_low, score_high)})")

    assert not np.allclose(position_low, position_high), (
        "expected different beta (from different volatility_index) to change "
        "the swarm's trajectory under an identical seed"
    )


def test_va_qpso_volatility_changes_outcome_across_seeds():
    """
    Across a spread of seeds, the two beta extremes should reach genuinely
    different solutions most of the time -- a stronger, less seed-dependent
    statement than the single-seed trajectory check above. Asserted loosely
    (at least one seed differs) so this never turns flaky; the printed count
    is the informative part.
    """
    seeds = range(10)
    differing = 0
    for seed in seeds:
        _, score_low = va_qpso(
            dim=NUM_STOPS, fitness_fn=_tour_fitness_fn, volatility_index=0.0,
            num_particles=30, max_iterations=80, seed=seed,
        )
        _, score_high = va_qpso(
            dim=NUM_STOPS, fitness_fn=_tour_fitness_fn, volatility_index=1.0,
            num_particles=30, max_iterations=80, seed=seed,
        )
        if not np.isclose(score_low, score_high):
            differing += 1

    print(f"volatility 0.0 vs 1.0 reached different scores on {differing}/{len(list(seeds))} seeds")
    assert differing > 0


def test_va_qpso_rejects_out_of_range_volatility_index():
    try:
        va_qpso(
            dim=NUM_STOPS,
            fitness_fn=_tour_fitness_fn,
            volatility_index=1.5,
            num_particles=5,
            max_iterations=5,
            seed=1,
        )
        assert False, "expected ValueError for out-of-range volatility_index"
    except ValueError:
        pass


def test_replan_matches_va_qpso_via_score_route():
    """
    replan() should be equivalent to calling va_qpso directly with a
    score_route-based fitness function (empty congestion_lookup, default
    equal weights) -- same seed, same result.
    """
    congestion_lookup = {}

    order_a, score_a = replan(
        STOPS,
        DISTANCE_MATRIX,
        congestion_lookup,
        volatility_index=0.5,
        num_particles=30,
        max_iterations=80,
        seed=99,
    )

    def fitness_fn(x: np.ndarray) -> float:
        return score_route(decode_order(x), DISTANCE_MATRIX, congestion_lookup, (1.0, 1.0, 1.0))

    best_position, score_b = va_qpso(
        dim=NUM_STOPS,
        fitness_fn=fitness_fn,
        volatility_index=0.5,
        num_particles=30,
        max_iterations=80,
        seed=99,
    )
    order_b = decode_order(best_position)

    print(f"replan() score={score_a:.2f} vs direct va_qpso+score_route score={score_b:.2f}")

    assert order_a.tolist() == order_b.tolist()
    assert np.isclose(score_a, score_b)


if __name__ == "__main__":
    test_fixed_beta_qpso_beats_identity()
    test_va_qpso_beats_identity()
    test_va_qpso_beta_responds_to_volatility_index()
    test_va_qpso_volatility_changes_outcome_across_seeds()
    test_va_qpso_rejects_out_of_range_volatility_index()
    test_replan_matches_va_qpso_via_score_route()
    print("OK: all qpso.py tests passed.")
