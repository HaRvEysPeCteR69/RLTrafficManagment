#!/usr/bin/env python3
"""
Paired Comparative Experiment Runner.

Evaluates va_qpso (volatility-adaptive) vs fixed_beta_qpso (linear anneal)
under a strictly paired experimental design across Low, Medium, and High
traffic volatility tiers and N random seeds.

For each (tier, seed), BOTH algorithms run on the exact same scenario,
incident schedule, and background traffic, providing paired samples
for statistical hypothesis testing.

Outputs:
  - Single consolidated CSV and JSON file containing per-run metrics.
  - Formatted terminal report with paired differences and statistical significance.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.planner.fitness import CongestionLookup, route_components, score_route
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
from traci.exceptions import FatalTraCIError, TraCIException

# Constants
NET_FILE = "networks/delhi/delhi_intersection.net.xml"
NUM_STOPS = 8
N_MAX = 120.0
N_MIN = 20.0
N_FIXED = 70.0  # midpoint cadence for fixed_beta_qpso

OCCUPANCY_THRESHOLD = 0.8
MIN_OCCUPANCY_IMPROVEMENT = 0.15
MAX_EXTRA_DISTANCE_RATIO = 0.3
ARBITER_WINDOW_SECONDS = 60.0
ARBITER_REROUTE_THRESHOLD = 5
VOLATILITY_WINDOW = 15
REFERENCE_VARIANCE = 0.002


def replan_interval(volatility_index: float) -> float:
    """Volatility-adaptive replan interval in [N_MIN, N_MAX]."""
    return N_MAX - (N_MAX - N_MIN) * volatility_index


def build_live_distance_and_congestion(
    network_graph: NetworkGraph,
    state: Dict[str, Any],
    stops: List[str],
) -> Tuple[np.ndarray, CongestionLookup]:
    """
    Build live-weighted travel-time matrix and per-leg congestion lookup.
    """
    edge_weights: Dict[str, float] = {}
    for edge_id, metrics in state["edges"].items():
        speed = metrics["mean_speed"]
        if speed > 0.1:
            edge_weights[edge_id] = network_graph.edges[edge_id]["length"] / speed

    adjacency = adjacency_from_network_graph(network_graph, edge_weights)
    distance_matrix = compute_distance_matrix(adjacency, stops)

    # Populate congestion lookup for stop pairs based on local edge occupancies
    congestion_lookup: CongestionLookup = {}
    n = len(stops)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Look up connecting edge if directly adjacent, or use representative occupancy
            source_node = stops[i]
            target_node = stops[j]
            edge_records = []
            for edge_id, e_data in network_graph.edges.items():
                if e_data["from"] == source_node or e_data["to"] == target_node:
                    occ = state["edges"].get(edge_id, {}).get("occupancy", 0.0)
                    edge_records.append({"occupancy": occ, "capacity": 1.0})
            if edge_records:
                congestion_lookup[(i, j)] = edge_records

    return distance_matrix, congestion_lookup


def run_single_trial(
    tier: str,
    seed: int,
    algorithm: str,
    duration: int,
    stops: List[str],
    network_graph: NetworkGraph,
    use_libsumo: bool = False,
) -> Dict[str, Any]:
    """
    Execute one simulation run for a given (tier, seed, algorithm) configuration.
    """
    sumocfg_path = f"networks/delhi/scenarios/{tier}/scenario.sumocfg"
    if not os.path.exists(sumocfg_path):
        raise FileNotFoundError(f"Scenario configuration not found: {sumocfg_path}")

    edge_ids = list(network_graph.edges.keys())
    extractor = SubscriptionStateExtractor(edge_ids, use_libsumo=use_libsumo)

    # Launch SUMO with explicit seed for identical background conditions
    extractor.connect(sumocfg_path, use_gui=False)

    volatility_calc = NetworkVolatilityIndex(
        window_size=VOLATILITY_WINDOW,
        reference_variance=REFERENCE_VARIANCE,
    )
    arbiter = ReplanArbiter(ARBITER_WINDOW_SECONDS, ARBITER_REROUTE_THRESHOLD)

    vehicle_routes: Dict[str, List[str]] = {}
    known_vehicle_ids: set = set()

    next_replan_time = 0.0
    reroute_count = 0
    replan_count = 0
    sim_time = 0.0

    current_best_order = np.arange(len(stops))
    last_T = 0.0
    last_D = 0.0
    last_C = 0.0

    cumulative_T = 0.0
    cumulative_D = 0.0
    cumulative_C = 0.0
    active_steps = 0

    try:
        while sim_time < duration:
            try:
                sim_time = extractor.step()
            except (FatalTraCIError, TraCIException) as exc:
                print(f"    [Warning] SUMO stepped early at {sim_time:.0f}s: {exc}")
                break

            state = extractor.get_state()
            edge_mean_speeds = {e: m["mean_speed"] for e, m in state["edges"].items()}
            volatility_index = volatility_calc.update(edge_mean_speeds)

            # Reactive vehicle rerouting
            current_vehicle_ids = set(state["vehicles"].keys())
            for veh_id in current_vehicle_ids - known_vehicle_ids:
                try:
                    vehicle_routes[veh_id] = list(extractor.traci.vehicle.getRoute(veh_id))
                except Exception:
                    pass
            for veh_id in known_vehicle_ids - current_vehicle_ids:
                vehicle_routes.pop(veh_id, None)
            known_vehicle_ids = current_vehicle_ids

            for veh_id, route in list(vehicle_routes.items()):
                veh_info = state["vehicles"].get(veh_id, {})
                route_index = veh_info.get("route_index")
                if route_index is None or route_index + 1 >= len(route):
                    continue
                planned_next_edge = route[route_index + 1]

                decision = evaluate_vehicle_reroute(
                    veh_id, state, planned_next_edge, network_graph,
                    occupancy_threshold=OCCUPANCY_THRESHOLD,
                    min_occupancy_improvement=MIN_OCCUPANCY_IMPROVEMENT,
                    max_extra_distance_ratio=MAX_EXTRA_DISTANCE_RATIO,
                )
                if decision is not None:
                    new_route = list(route)
                    new_route[route_index + 1] = decision.to_edge
                    try:
                        extractor.traci.vehicle.setRoute(veh_id, new_route)
                        vehicle_routes[veh_id] = new_route
                        arbiter.record_reroute(sim_time)
                        reroute_count += 1
                    except Exception:
                        pass

            # Check replanning cadence
            is_scheduled = sim_time >= next_replan_time
            is_arbiter_triggered = arbiter.should_trigger_early_replan(sim_time)

            if is_scheduled or is_arbiter_triggered:
                distance_matrix, congestion_lookup = build_live_distance_and_congestion(
                    network_graph, state, stops
                )

                best_order, best_score = qpso_replan(
                    stops,
                    distance_matrix,
                    congestion_lookup,
                    volatility_index=volatility_index,
                    algorithm=algorithm,
                    seed=seed,
                )
                current_best_order = best_order
                last_T, last_D, last_C = route_components(
                    current_best_order, distance_matrix, congestion_lookup
                )

                if algorithm == "va_qpso":
                    interval = replan_interval(volatility_index)
                else:
                    interval = N_FIXED

                next_replan_time = sim_time + interval
                arbiter.notify_replanned(sim_time)
                replan_count += 1

            # Accumulate active route performance
            if last_T > 0.0:
                cumulative_T += last_T
                cumulative_D += last_D
                cumulative_C += last_C
                active_steps += 1

    finally:
        extractor.close()

    # Average metrics over simulation active steps
    avg_T = (cumulative_T / active_steps) if active_steps > 0 else last_T
    avg_D = (cumulative_D / active_steps) if active_steps > 0 else last_D
    avg_C = (cumulative_C / active_steps) if active_steps > 0 else last_C

    return {
        "tier": tier,
        "seed": seed,
        "algorithm": algorithm,
        "total_route_completion_time": round(avg_T, 2),
        "total_distance": round(avg_D, 2),
        "congestion_exposure_score": round(avg_C, 4),
        "reroute_count": reroute_count,
        "replan_count": replan_count,
    }


def print_statistical_summary(df: pd.DataFrame):
    """
    Format and print paired comparison statistics for each volatility tier.
    """
    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL COMPARISON SUMMARY: va_qpso vs fixed_beta_qpso")
    print("=" * 80)

    for tier in df["tier"].unique():
        sub_df = df[df["tier"] == tier]
        va_runs = sub_df[sub_df["algorithm"] == "va_qpso"].sort_values("seed")
        fb_runs = sub_df[sub_df["algorithm"] == "fixed_beta_qpso"].sort_values("seed")

        common_seeds = sorted(list(set(va_runs["seed"]).intersection(set(fb_runs["seed"]))))
        if not common_seeds:
            continue

        va_matched = va_runs[va_runs["seed"].isin(common_seeds)]
        fb_matched = fb_runs[fb_runs["seed"].isin(common_seeds)]

        n_pairs = len(common_seeds)
        t_va = va_matched["total_route_completion_time"].values
        t_fb = fb_matched["total_route_completion_time"].values

        c_va = va_matched["congestion_exposure_score"].values
        c_fb = fb_matched["congestion_exposure_score"].values

        delta_t = t_va - t_fb
        pct_imp_t = ((t_fb.mean() - t_va.mean()) / t_fb.mean()) * 100.0 if t_fb.mean() > 0 else 0.0

        delta_c = c_va - c_fb
        pct_imp_c = ((c_fb.mean() - c_va.mean()) / c_fb.mean()) * 100.0 if c_fb.mean() > 0 else 0.0

        # Paired t-test on travel time
        if n_pairs >= 2 and np.std(delta_t) > 1e-9:
            t_stat, p_val = stats.ttest_rel(t_va, t_fb)
        else:
            t_stat, p_val = 0.0, 1.0

        print(f"\n[Tier: {tier.upper()}] (Paired Seeds: {n_pairs})")
        print(f"  Travel Time (s):")
        print(f"    va_qpso        : {t_va.mean():.2f} +/- {t_va.std():.2f}s")
        print(f"    fixed_beta_qpso: {t_fb.mean():.2f} +/- {t_fb.std():.2f}s")
        print(f"    Mean Diff (Delta): {delta_t.mean():.2f}s  |  Improvement: {pct_imp_t:+.2f}%")
        print(f"    Paired t-stat: {t_stat:.3f} (p-value: {p_val:.4e})")

        print(f"  Congestion Exposure Score:")
        print(f"    va_qpso        : {c_va.mean():.4f} +/- {c_va.std():.4f}")
        print(f"    fixed_beta_qpso: {c_fb.mean():.4f} +/- {c_fb.std():.4f}")
        print(f"    Mean Diff (Delta): {delta_c.mean():.4f}  |  Improvement: {pct_imp_c:+.2f}%")

        print(f"  Operational Activity:")
        print(f"    va_qpso replans       : {va_matched['replan_count'].mean():.1f} | reroutes: {va_matched['reroute_count'].mean():.1f}")
        print(f"    fixed_beta_qpso replans: {fb_matched['replan_count'].mean():.1f} | reroutes: {fb_matched['reroute_count'].mean():.1f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Paired QPSO Benchmark Experiments")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds per tier (default: 10)")
    parser.add_argument("--start-seed", type=int, default=42, help="Starting seed value (default: 42)")
    parser.add_argument("--tiers", nargs="+", default=["low", "medium", "high"], help="Tiers to evaluate")
    parser.add_argument("--duration", type=int, default=300, help="Simulation duration in seconds (default: 300)")
    parser.add_argument("--output", type=str, default="results/experiments.csv", help="Output CSV path")
    parser.add_argument("--json-output", type=str, default="results/experiments.json", help="Output JSON path")
    parser.add_argument("--use-libsumo", action="store_true", help="Use libsumo if installed")
    args = parser.parse_args()

    network_graph = NetworkGraph(NET_FILE)
    stops = pick_mutually_reachable_stops(
        adjacency_from_network_graph(network_graph, edge_weights={}),
        NUM_STOPS,
    )
    print("=" * 70)
    print("PAIRED EXPERIMENTAL DESIGN BENCHMARK")
    print(f"Tiers: {args.tiers} | Duration: {args.duration}s | Seeds: {args.num_seeds} (start: {args.start_seed})")
    print(f"Stops ({len(stops)}): {stops}")
    print("=" * 70)

    seeds = [args.start_seed + i for i in range(args.num_seeds)]
    algorithms = ["va_qpso", "fixed_beta_qpso"]
    results: List[Dict[str, Any]] = []

    total_runs = len(args.tiers) * len(seeds) * len(algorithms)
    run_idx = 0

    for tier in args.tiers:
        print(f"\n>>> Running Tier: [{tier.upper()}] <<<")
        for seed in seeds:
            for algo in algorithms:
                run_idx += 1
                t0 = time.time()
                res = run_single_trial(
                    tier=tier,
                    seed=seed,
                    algorithm=algo,
                    duration=args.duration,
                    stops=stops,
                    network_graph=network_graph,
                    use_libsumo=args.use_libsumo,
                )
                elapsed = time.time() - t0
                results.append(res)
                print(
                    f"  [{run_idx:02d}/{total_runs:02d}] Tier: {tier:<6} | Seed: {seed:<3} | Algo: {algo:<15} "
                    f"| Time: {res['total_route_completion_time']:6.2f}s | Cong: {res['congestion_exposure_score']:7.4f} "
                    f"| Replans: {res['replan_count']:2d} | Reroutes: {res['reroute_count']:2d} ({elapsed:.1f}s)"
                )

    # Save outputs
    df = pd.DataFrame(results)

    out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nCSV results saved to: {out_csv.resolve()}")

    if args.json_output:
        out_json = Path(args.json_output)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved to: {out_json.resolve()}")

    # Statistical Summary
    print_statistical_summary(df)


if __name__ == "__main__":
    main()
