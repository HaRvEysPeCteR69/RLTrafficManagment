"""
Test script for QPSOOptimizer.optimize_tour() (src/planner/qpso_optimizer.py)
on the real Delhi road network.

Builds an adjacency graph from the Delhi .net.xml (free-flow travel times,
i.e. no live state.py snapshot needed for this test), picks a handful of
junctions that are mutually reachable (so every pairwise distance is finite),
precomputes their distance matrix once, then runs QPSO to find a good visit
order and sanity-checks the result.
"""

import numpy as np

from src.planner.qpso_encoding import (
    adjacency_from_network_graph,
    compute_distance_matrix,
    tour_length,
)
from src.planner.qpso_optimizer import QPSOOptimizer
from src.state_extraction.network_graph import NetworkGraph

NET_FILE = "networks/delhi/delhi_intersection.net.xml"
NUM_STOPS = 8


def _reachable_from(adjacency, source):
    """Nodes reachable from `source` by following adjacency edges forward."""
    seen = {source}
    stack = [source]
    while stack:
        node = stack.pop()
        for neighbor, _ in adjacency.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _reverse(adjacency):
    reverse = {}
    for node, edges in adjacency.items():
        for neighbor, weight in edges:
            reverse.setdefault(neighbor, []).append((node, weight))
    return reverse


def _pick_mutually_reachable_stops(adjacency, num_stops):
    """
    Pick `num_stops` nodes that are all mutually reachable, so the resulting
    distance matrix has no np.inf entries.

    A node x is mutually reachable with `source` iff x is forward-reachable
    from source AND source is forward-reachable from x (i.e. x is
    reverse-reachable from source on the reverse graph). The intersection of
    those two sets is exactly the strongly connected component containing
    `source`.
    """
    reverse_adjacency = _reverse(adjacency)
    for source in adjacency:
        if not adjacency.get(source):
            continue
        forward = _reachable_from(adjacency, source)
        backward = _reachable_from(reverse_adjacency, source)
        scc = forward & backward
        if len(scc) >= num_stops:
            return sorted(scc)[:num_stops]
    raise RuntimeError(
        f"Could not find {num_stops} mutually-reachable nodes in this network."
    )


def test_optimize_tour():
    network_graph = NetworkGraph(NET_FILE)
    adjacency = adjacency_from_network_graph(network_graph, edge_weights={})

    stops = _pick_mutually_reachable_stops(adjacency, NUM_STOPS)
    print(f"Stops ({len(stops)}): {stops}")

    distance_matrix = compute_distance_matrix(adjacency, stops)
    assert distance_matrix.shape == (NUM_STOPS, NUM_STOPS)
    assert np.all(np.isfinite(distance_matrix)), "expected all stops to be mutually reachable"
    assert np.allclose(np.diag(distance_matrix), 0.0)

    optimizer = QPSOOptimizer(
        num_particles=30,
        max_iterations=60,
        dim=NUM_STOPS,
        bounds=(0.0, 1.0),
        seed=42,
    )
    # No congestion data here, so optimize_tour's default (w1, w2, w3) =
    # (1, 1, 1) normalizes to (1/3, 1/3, 1/3) and, since T == D and C == 0,
    # best_score collapses to (2/3) * tour_length(best_order).
    best_order, best_score = optimizer.optimize_tour(distance_matrix)

    # best_order must be a valid permutation of the stop indices.
    assert sorted(best_order.tolist()) == list(range(NUM_STOPS))
    assert np.isfinite(best_score)
    assert best_score > 0.0

    best_distance = tour_length(best_order, distance_matrix)
    assert np.isclose(best_score, (2.0 / 3.0) * best_distance)

    # QPSO searched hundreds of candidate orderings; it should be at least as
    # good as the naive identity ordering (stops in the order they were picked).
    identity_order = np.arange(NUM_STOPS)
    identity_distance = tour_length(identity_order, distance_matrix)
    print(f"Identity-order distance: {identity_distance:.2f}")
    print(f"QPSO best order: {best_order.tolist()} -> stops: {[stops[i] for i in best_order]}")
    print(f"QPSO best distance: {best_distance:.2f} (score_route score: {best_score:.2f})")
    assert best_distance <= identity_distance

    print("OK: optimize_tour() returned a valid, finite-cost permutation.")


if __name__ == "__main__":
    test_optimize_tour()
