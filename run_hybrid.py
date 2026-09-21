"""
Hybrid QPSO + reactive control main loop.

Runs SUMO/TraCI and, once per simulated second:
  1. advances the simulation one step
  2. refreshes network + vehicle state via state.py's subscription-based
     SubscriptionStateExtractor (Day 2's module) -- one batched call, no
     per-object polling
  3. feeds the network-wide mean edge speed into
     volatility.NetworkVolatilityIndex for a rolling volatility_index
  4. runs reactive.py's per-vehicle next-hop check, and feeds every reroute
     it fires into arbiter.py's rolling reroute-rate tracker
  5. runs qpso.replan() on a variable cadence: N seconds since the last
     replan, where N is derived from the CURRENT volatility_index (see
     replan_interval() below) -- or immediately, if the arbiter's
     reroute-rate threshold was crossed first

Every reroute and replan event is appended to a structured JSON Lines log
at LOG_PATH (one JSON object per line -- see log_event()'s docstring for
the schema). This is the raw data later statistics/experiments run on, so
the schema is considered part of this file's contract, not an incidental
detail.

KNOWN SIMPLIFICATIONS (flagged rather than hidden):
  - There is no fleet/dispatch module in this codebase yet, so "the stops"
    are a fixed placeholder set (NUM_STOPS mutually-reachable junctions,
    picked once at startup), not real delivery destinations. Swap
    `pick_stops()` for a real assignment once that module exists.
  - `qpso.replan()`'s congestion_lookup term (the C in score_route) needs a
    per-leg EDGE PATH, not just a distance, to look up occupancy/capacity
    along the way. compute_distance_matrix() only returns distances, not
    the paths that produced them, so this loop currently passes an empty
    congestion_lookup and replans on travel-time/distance only (C == 0 for
    every replan). Path reconstruction is a natural follow-up, not done here.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.planner.fitness import CongestionLookup
from src.planner.qpso import replan as qpso_replan
from src.planner.qpso_encoding import (
    adjacency_from_network_graph,
    compute_distance_matrix,
    pick_mutually_reachable_stops,
)
from src.reactive.arbiter import ReplanArbiter
from src.reactive.reactive import evaluate_vehicle_reroute
from src.state_extraction.network_graph import NetworkGraph
from src.state_extraction.state import SubscriptionStateExtractor
from src.volatility import NetworkVolatilityIndex

# Exception classes are backend-agnostic (plain Python classes, no live
# connection needed to import them), so this is importable unconditionally
# regardless of whether the run ends up using traci or libsumo -- mirrors
# state.py's unconditional `import traci.constants as tc`.
from traci.exceptions import FatalTraCIError

# --- Tunable constants ------------------------------------------------

NET_FILE = "networks/delhi/delhi_intersection.net.xml"
SUMO_CFG = "networks/delhi/delhi_intersection.sumocfg"

NUM_STOPS = 8      # placeholder delivery-stop count -- see module docstring
NUM_SECONDS = 3600  # total simulated seconds to run

# Replan cadence: N seconds between qpso.replan() calls, derived from the
# CURRENT volatility_index rather than held constant --
#     N = N_MAX - (N_MAX - N_MIN) * volatility_index
# so chaotic traffic (volatility_index -> 1) replans as often as every
# N_MIN seconds, and calm traffic (volatility_index -> 0) coasts on a plan
# for up to N_MAX seconds before the next scheduled refresh.
N_MAX = 120.0  # seconds between replans when volatility_index == 0 (calm)
N_MIN = 20.0   # seconds between replans when volatility_index == 1 (chaotic)

# Reactive per-hop reroute thresholds (see reactive.find_alternative_edge).
OCCUPANCY_THRESHOLD = 0.8
MIN_OCCUPANCY_IMPROVEMENT = 0.15
MAX_EXTRA_DISTANCE_RATIO = 0.3

# Arbiter: an early replan is triggered if this many reactive reroutes fire
# within this many simulated seconds (see arbiter.ReplanArbiter).
ARBITER_WINDOW_SECONDS = 60.0
ARBITER_REROUTE_THRESHOLD = 5

VOLATILITY_WINDOW = 15  # rolling-window size for NetworkVolatilityIndex

LOG_PATH = "logs/hybrid_run.jsonl"


def replan_interval(volatility_index: float) -> float:
    """N = N_MAX - (N_MAX - N_MIN) * volatility_index."""
    return N_MAX - (N_MAX - N_MIN) * volatility_index


def log_event(log_file, event: Dict[str, Any]) -> None:
    """
    Append one JSON object per line (JSON Lines, not a single JSON array):
    an append-only writer never needs to rewrite the whole file, and Day
    6/7 analysis scripts can stream it line by line without parsing
    partial/in-progress runs.

    Schema (two event types, sharing "event"/"sim_time"/"volatility_index"):

    Reroute event (one per reactive.py trigger):
        {
            "event": "reroute",
            "sim_time": float,           # simulated seconds since sim start
            "volatility_index": float,   # in [0, 1] at this instant
            "vehicle_id": str,
            "from_edge": str,            # the congested planned next edge
            "to_edge": str,              # the alternative edge substituted
            "occupancy_before": float,   # from_edge's occupancy, [0, 1]
            "occupancy_after": float,    # to_edge's occupancy, [0, 1]
        }

    Replan event (one per qpso.replan() call):
        {
            "event": "replan",
            "sim_time": float,
            "volatility_index": float,
            "trigger": "scheduled" | "arbiter",  # why this replan ran now
            "num_stops": int,
            "stops": [str, ...],          # stop ids, index-aligned with best_order
            "best_order": [int, ...],     # visit-order permutation (indices into stops)
            "fitness": float,             # score_route() fitness of best_order
            "next_interval_seconds": float,  # N used to schedule the NEXT
                                              # scheduled replan from here
        }
    """
    log_file.write(json.dumps(event) + "\n")
    log_file.flush()


def pick_stops(network_graph: NetworkGraph, num_stops: int) -> List[str]:
    """
    Placeholder stop selection (see module docstring) -- picks `num_stops`
    mutually-reachable junctions from the static network so replan() always
    has a solvable problem, independent of live traffic.
    """
    free_flow_adjacency = adjacency_from_network_graph(network_graph, edge_weights={})
    return pick_mutually_reachable_stops(free_flow_adjacency, num_stops)


def build_live_distance_matrix(network_graph: NetworkGraph, state: Dict[str, Any], stops: List[str]):
    """
    Freeze the current state snapshot into a live-weighted distance matrix
    for this replan tick (see qpso_encoding.compute_distance_matrix's
    staleness-tradeoff docstring: this is deliberately NOT recomputed
    between replans).
    """
    edge_weights: Dict[str, float] = {}
    for edge_id, metrics in state["edges"].items():
        speed = metrics["mean_speed"]
        if speed > 0.1:  # avoid a near-zero speed blowing up to a huge travel time
            edge_weights[edge_id] = network_graph.edges[edge_id]["length"] / speed
        # else: leave unset -- adjacency_from_network_graph falls back to
        # this edge's static free-flow travel time.

    adjacency = adjacency_from_network_graph(network_graph, edge_weights)
    return compute_distance_matrix(adjacency, stops)


def main():
    parser = argparse.ArgumentParser(description="Hybrid QPSO + reactive control loop")
    parser.add_argument("--seconds", type=float, default=NUM_SECONDS, help="Simulated seconds to run")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO with GUI")
    parser.add_argument("--use-libsumo", action="store_true", help="Drive the sim in-process via libsumo")
    parser.add_argument("--log", type=str, default=LOG_PATH, help="Path to the JSONL event log")
    parser.add_argument("--cfg", type=str, default=SUMO_CFG, help="Path to SUMO configuration file")
    args = parser.parse_args()

    network_graph = NetworkGraph(NET_FILE)
    edge_ids = list(network_graph.edges.keys())
    stops = pick_stops(network_graph, NUM_STOPS)
    print(f"Stops ({NUM_STOPS}): {stops}")

    extractor = SubscriptionStateExtractor(edge_ids, use_libsumo=args.use_libsumo)
    extractor.connect(args.cfg, use_gui=args.gui)

    volatility_calc = NetworkVolatilityIndex(window_size=VOLATILITY_WINDOW)
    arbiter = ReplanArbiter(ARBITER_WINDOW_SECONDS, ARBITER_REROUTE_THRESHOLD)

    # veh_id -> cached full route (edge id list). Populated once per vehicle
    # at departure via a single getRoute() call, updated in place whenever
    # a reactive reroute splices in an alternative edge -- never re-fetched
    # every step. See state.py's VAR_ROUTE_INDEX comment for why this cache
    # exists at all (the full route isn't itself subscribable).
    vehicle_routes: Dict[str, List[str]] = {}
    known_vehicle_ids: set = set()

    next_replan_time = 0.0  # force an immediate first replan at sim_time=0

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    reroute_count = 0
    replan_count = 0
    sim_time = 0.0

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            while sim_time < args.seconds:
                try:
                    sim_time = extractor.step()
                except FatalTraCIError as exc:
                    # SUMO itself aborted the run (observed cause on this
                    # network: routes.rou.xml has a stale, now-disconnected
                    # route baked in for one vehicle -- a scenario-data
                    # problem, not something this loop did). Preserve
                    # whatever was already logged rather than losing it to
                    # an unhandled crash.
                    print(f"SUMO ended the simulation early at sim_time={sim_time:.0f}s: {exc}")
                    break
                state = extractor.get_state()

                edge_mean_speeds = {e: m["mean_speed"] for e, m in state["edges"].items()}
                volatility_index = volatility_calc.update(edge_mean_speeds)

                # --- reactive layer: per-vehicle next-hop check ---
                current_vehicle_ids = set(state["vehicles"].keys())
                for veh_id in current_vehicle_ids - known_vehicle_ids:
                    vehicle_routes[veh_id] = list(extractor.traci.vehicle.getRoute(veh_id))
                for veh_id in known_vehicle_ids - current_vehicle_ids:
                    vehicle_routes.pop(veh_id, None)  # arrived; drop cached route
                known_vehicle_ids = current_vehicle_ids

                for veh_id, route in vehicle_routes.items():
                    route_index = state["vehicles"][veh_id]["route_index"]
                    if route_index is None or route_index + 1 >= len(route):
                        continue  # on its last edge already -- nothing ahead to reroute
                    planned_next_edge = route[route_index + 1]

                    decision = evaluate_vehicle_reroute(
                        veh_id, state, planned_next_edge, network_graph,
                        occupancy_threshold=OCCUPANCY_THRESHOLD,
                        min_occupancy_improvement=MIN_OCCUPANCY_IMPROVEMENT,
                        max_extra_distance_ratio=MAX_EXTRA_DISTANCE_RATIO,
                    )
                    if decision is None:
                        continue

                    new_route = list(route)
                    new_route[route_index + 1] = decision.to_edge
                    extractor.traci.vehicle.setRoute(veh_id, new_route)
                    vehicle_routes[veh_id] = new_route

                    arbiter.record_reroute(sim_time)
                    reroute_count += 1
                    log_event(log_file, {
                        "event": "reroute",
                        "sim_time": sim_time,
                        "volatility_index": volatility_index,
                        "vehicle_id": decision.vehicle_id,
                        "from_edge": decision.from_edge,
                        "to_edge": decision.to_edge,
                        "occupancy_before": decision.occupancy_before,
                        "occupancy_after": decision.occupancy_after,
                    })

                # --- replan cadence: scheduled (volatility-derived N) or arbiter-triggered ---
                is_scheduled = sim_time >= next_replan_time
                is_arbiter_triggered = arbiter.should_trigger_early_replan(sim_time)

                if is_scheduled or is_arbiter_triggered:
                    distance_matrix = build_live_distance_matrix(network_graph, state, stops)
                    congestion_lookup: CongestionLookup = {}  # see module docstring

                    best_order, best_score = qpso_replan(
                        stops, distance_matrix, congestion_lookup,
                        volatility_index=volatility_index,
                    )

                    interval = replan_interval(volatility_index)
                    next_replan_time = sim_time + interval
                    arbiter.notify_replanned(sim_time)
                    replan_count += 1

                    log_event(log_file, {
                        "event": "replan",
                        "sim_time": sim_time,
                        "volatility_index": volatility_index,
                        "trigger": "arbiter" if is_arbiter_triggered else "scheduled",
                        "num_stops": NUM_STOPS,
                        "stops": stops,
                        "best_order": best_order.tolist(),
                        "fitness": best_score,
                        "next_interval_seconds": interval,
                    })
    finally:
        extractor.close()

    print(f"Done: {sim_time:.0f}s simulated, {replan_count} replans, {reroute_count} reactive reroutes.")
    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
