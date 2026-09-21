"""
Reactive per-hop rerouting.

A fast, local check, meant to run every simulation step for every active
vehicle: is the vehicle's immediate next edge congested, and if so, is
there a genuinely better sibling edge that reconnects at the EXACT SAME
downstream junction? It only ever touches that one upcoming edge -- never
the rest of the vehicle's planned route, and never the QPSO stop order
(that's qpso.replan's job, run far less often).

Why "exact same downstream junction": SUMO requires a route to be a
contiguous chain of connected edges. Splicing an alternative edge into an
already-assigned route is only guaranteed valid if the alternative starts
where the vehicle currently is heading FROM and ends at the same node the
original edge would have -- everything after that point in the route is
untouched and still connects correctly. This also keeps the check itself
cheap: it looks at exactly the outgoing edges of one junction (via
NetworkGraph.adjacency), never re-searches the graph toward some distant
goal.

Measured limitation on the Delhi network (see validate_brute_force.py-style
diagnostics elsewhere in this project for why measuring beats assuming):
of 768 distinct (from, to) junction pairs, only 4 have more than one edge
between them. This reroute can only ever fire on those 4 pairs on THIS map
-- it is correct and cheap, but rarely applicable here. A network with more
divided/multi-lane roads modeled as parallel edges would exercise it far
more.
"""

from typing import Any, Dict, NamedTuple, Optional

from src.state_extraction.network_graph import NetworkGraph


class RerouteDecision(NamedTuple):
    vehicle_id: str
    from_edge: str
    to_edge: str
    occupancy_before: float
    occupancy_after: float


def find_alternative_edge(
    state: Dict[str, Any],
    planned_next_edge: str,
    network_graph: NetworkGraph,
    occupancy_threshold: float = 0.8,
    min_occupancy_improvement: float = 0.15,
    max_extra_distance_ratio: float = 0.3,
) -> Optional[str]:
    """
    Args:
        state: The current state.get_state() snapshot -- state["edges"][e]
            must have an "occupancy" field in [0, 1] (as produced by
            state.SubscriptionStateExtractor).
        planned_next_edge: The edge id the vehicle is about to enter.
        network_graph: Parsed road network, for edge lengths and the
            from-node -> outgoing-edge-ids adjacency used to find siblings.
        occupancy_threshold: Only consider rerouting if planned_next_edge's
            occupancy is at or above this (fraction of edge length occupied).
        min_occupancy_improvement: A candidate only counts as "meaningfully
            better" if its occupancy is at least this much lower (absolute)
            than planned_next_edge's.
        max_extra_distance_ratio: A candidate's length must not exceed
            planned_next_edge's length by more than this fraction (e.g. 0.3
            allows up to 30% extra distance for this one hop).

    Returns:
        The best alternative edge id, or None if planned_next_edge isn't
        congested enough to warrant a look, or no candidate is both
        meaningfully better and within the distance budget.
    """
    edges = state.get("edges", {})
    planned_state = edges.get(planned_next_edge)
    if planned_state is None:
        return None

    occupancy = planned_state["occupancy"]
    if occupancy < occupancy_threshold:
        return None

    planned_edge = network_graph.edges.get(planned_next_edge)
    if planned_edge is None:
        return None
    from_node, to_node = planned_edge["from"], planned_edge["to"]
    planned_length = planned_edge["length"]

    best_edge_id = None
    best_occupancy = occupancy

    for candidate_id in network_graph.adjacency.get(from_node, []):
        if candidate_id == planned_next_edge:
            continue
        candidate = network_graph.edges.get(candidate_id)
        if candidate is None or candidate["to"] != to_node:
            continue  # must reconnect at the exact same junction

        candidate_state = edges.get(candidate_id)
        if candidate_state is None:
            continue
        candidate_occupancy = candidate_state["occupancy"]

        meaningfully_better = candidate_occupancy <= occupancy - min_occupancy_improvement
        comparable_distance = candidate["length"] <= planned_length * (1 + max_extra_distance_ratio)

        if meaningfully_better and comparable_distance and candidate_occupancy < best_occupancy:
            best_edge_id, best_occupancy = candidate_id, candidate_occupancy

    return best_edge_id


def evaluate_vehicle_reroute(
    vehicle_id: str,
    state: Dict[str, Any],
    planned_next_edge: str,
    network_graph: NetworkGraph,
    **kwargs,
) -> Optional[RerouteDecision]:
    """
    Same check as find_alternative_edge, packaged with the vehicle id and
    before/after occupancy -- the shape run_hybrid.py needs to both apply
    the reroute and log it.

    kwargs are passed through to find_alternative_edge (occupancy_threshold,
    min_occupancy_improvement, max_extra_distance_ratio).
    """
    alternative = find_alternative_edge(state, planned_next_edge, network_graph, **kwargs)
    if alternative is None:
        return None

    return RerouteDecision(
        vehicle_id=vehicle_id,
        from_edge=planned_next_edge,
        to_edge=alternative,
        occupancy_before=state["edges"][planned_next_edge]["occupancy"],
        occupancy_after=state["edges"][alternative]["occupancy"],
    )
