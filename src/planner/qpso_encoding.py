"""
Random-Key Encoding and Distance-Matrix Precomputation for QPSO Route Planning.

QPSO (see qpso.py) searches a continuous real-valued space, but stop
sequencing is a discrete permutation problem. Bean's random-key encoding
bridges the two: each particle position is a real-valued vector x in R^n, and
the visit order is decoded as decode_order(x) = argsort(x). Because argsort is
stable under small perturbations of x (nudging one coordinate a little only
swaps it past neighbors whose keys are close in value), small moves in
continuous QPSO-space correspond to small changes in visit order. That
locality is what makes a continuous swarm search work on a combinatorial
sequencing problem at all — without it, arbitrary encodings would make the
fitness landscape discontinuous and the swarm's velocity/position updates
meaningless.

Reference:
    Bean, J.C. (1994). "Genetic algorithms and random keys for sequencing
    and optimization." ORSA Journal on Computing, 6(2), 154-160.
"""

import heapq
from typing import Dict, List, Tuple

import numpy as np

from src.state_extraction.network_graph import NetworkGraph


def decode_order(x: np.ndarray) -> np.ndarray:
    """
    Decode a random-key vector into a visit-order permutation (Bean, 1994).

    Args:
        x: Real-valued particle position, shape (n,). Values need not be
           sorted, bounded to a specific range, or unique — only their
           relative order matters.

    Returns:
        Integer array of shape (n,): the indices of x in ascending order,
        i.e. the order in which stops should be visited.
    """
    return np.argsort(x, kind="stable")


def _dijkstra_single_source(
    adjacency: Dict[str, List[Tuple[str, float]]],
    source: str,
) -> Dict[str, float]:
    """
    Single-source shortest path distances via Dijkstra's algorithm.

    O(E + V log V) with a binary heap. Returns distances to every node
    reachable from `source`; unreached nodes are simply absent (treated as
    +inf by the caller).
    """
    distances: Dict[str, float] = {source: 0.0}
    visited = set()
    pq: List[Tuple[float, str]] = [(0.0, source)]

    while pq:
        dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        for neighbor, edge_cost in adjacency.get(node, []):
            if neighbor in visited:
                continue
            new_dist = dist + edge_cost
            if new_dist < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances


def compute_distance_matrix(
    adjacency: Dict[str, List[Tuple[str, float]]],
    stops: List[str],
) -> np.ndarray:
    """
    Precompute the full stop-to-stop shortest-path distance matrix.

    Runs one Dijkstra per stop (O(n) runs, each O(E + V log V) on the live-
    weighted graph) and caches every pairwise result in an n x n matrix.
    Call this ONCE per re-plan tick, on a frozen snapshot of edge weights —
    never inside the QPSO fitness function. A single re-plan evaluates
    hundreds of candidate orderings; if each fitness evaluation re-ran
    Dijkstra instead of doing an O(n) matrix lookup, a re-plan tick that
    should take milliseconds would instead scale with
    (num_particles * num_iterations) full graph searches, which is
    prohibitively slow.

    Staleness trade-off: this matrix reflects edge weights at the moment it
    was built (a frozen snapshot from state.py, e.g. subscription-derived
    mean speeds), not the live network. It goes stale the instant traffic
    conditions shift after that snapshot, and it is only ever as fresh as
    the last re-plan tick. The re-plan cadence (how often a new snapshot is
    taken and this matrix rebuilt) is therefore the knob that trades staleness
    against recomputation cost — a tighter cadence keeps this matrix fresher
    at the price of recomputing it more often.

    Args:
        adjacency: Directed graph as node -> [(neighbor, edge_weight), ...],
            e.g. built from NetworkGraph plus a live edge-weight snapshot.
        stops: Node ids to compute pairwise distances between, in the order
            they should map to matrix rows/columns.

    Returns:
        (n, n) float matrix where entry [i, j] is the shortest-path distance
        from stops[i] to stops[j]. Unreachable pairs are np.inf.
    """
    n = len(stops)
    matrix = np.full((n, n), np.inf, dtype=float)

    for i, source in enumerate(stops):
        distances = _dijkstra_single_source(adjacency, source)
        for j, target in enumerate(stops):
            if target in distances:
                matrix[i, j] = distances[target]

    return matrix


def adjacency_from_network_graph(
    network_graph: NetworkGraph,
    edge_weights: Dict[str, float],
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build a node -> [(neighbor, weight), ...] adjacency dict from a parsed
    NetworkGraph, using a caller-supplied edge-weight snapshot (e.g. the
    frozen travel times from state.py's get_state()) instead of static edge
    length/speed.

    Falls back to the edge's static free-flow travel time (length / speed)
    for any edge missing from `edge_weights`, so a partial or stale snapshot
    still yields a usable, fully-connected adjacency.
    """
    adjacency: Dict[str, List[Tuple[str, float]]] = {}

    for edge_id, edge in network_graph.edges.items():
        from_node, to_node = edge["from"], edge["to"]
        weight = edge_weights.get(edge_id)
        if weight is None:
            weight = edge["length"] / edge["speed"] if edge["speed"] > 0 else float("inf")
        adjacency.setdefault(from_node, []).append((to_node, weight))

    return adjacency


def _reachable_from(
    adjacency: Dict[str, List[Tuple[str, float]]],
    source: str,
) -> set:
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


def _reverse_adjacency(
    adjacency: Dict[str, List[Tuple[str, float]]],
) -> Dict[str, List[Tuple[str, float]]]:
    reverse: Dict[str, List[Tuple[str, float]]] = {}
    for node, edges in adjacency.items():
        for neighbor, weight in edges:
            reverse.setdefault(neighbor, []).append((node, weight))
    return reverse


def pick_mutually_reachable_stops(
    adjacency: Dict[str, List[Tuple[str, float]]],
    num_stops: int,
) -> List[str]:
    """
    Pick `num_stops` nodes that are all mutually reachable, so the distance
    matrix built from them has no np.inf entries.

    Real road networks are directed (one-ways), so an arbitrary set of nodes
    is usually NOT mutually reachable and would yield an unusable matrix. A
    node x is mutually reachable with `source` iff x is forward-reachable
    from source AND source is forward-reachable from x (i.e. x is reachable
    from source on the reversed graph); the intersection of those two sets is
    exactly the strongly connected component containing `source`.

    Raises:
        RuntimeError: if no strongly connected component is large enough.
    """
    reverse = _reverse_adjacency(adjacency)
    for source in adjacency:
        if not adjacency.get(source):
            continue
        scc = _reachable_from(adjacency, source) & _reachable_from(reverse, source)
        if len(scc) >= num_stops:
            return sorted(scc)[:num_stops]
    raise RuntimeError(
        f"Could not find {num_stops} mutually-reachable nodes in this network."
    )


def tour_length(order: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Total distance of visiting stops in `order` (as produced by
    decode_order), using O(n) lookups into a precomputed distance_matrix
    rather than any graph search.
    """
    total = 0.0
    for a, b in zip(order[:-1], order[1:]):
        total += distance_matrix[a, b]
    return total
