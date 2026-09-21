"""
Brute-force validation of va_qpso on a 6-stop case.

With 6 stops there are 6! = 720 possible orderings, so the true optimum can
be found exhaustively and compared against what the swarm actually finds.

Experimental setup: the SHIPPED defaults from qpso.default_budget(dim), so
this validates what callers actually get rather than a hand-picked budget.
At dim=6 that resolves to 24 particles / 450 iterations / 30 restarts, held
constant across all 30 runs, with PATIENCE=15 (qpso.py's stagnation
threshold). The exact numbers are printed at run time.

For scale, 24 x 450 is up to 10800 fitness evaluations against a
720-ordering search space. A swarm given many times the search space in
evaluations and still missing the optimum is not under-budgeted -- that was
the original finding here, and it is what the diagnostics below exist to
distinguish.

MATCH TOLERANCE: a run counts as hitting the optimum if its score is within
TOLERANCE_RTOL = 1e-6 *relative* of the brute-force optimum. This is a
floating-point equality tolerance, NOT a quality allowance -- it exists so
that the same permutation scored twice compares equal, not to let
near-optimal answers pass. Near-optimal results (within 1%) are counted and
printed separately, as information only; they do not count toward the
threshold.

PASS THRESHOLD: 28 of 30 runs must hit the optimum.

The fitness function handed to va_qpso and the objective brute-forced here
are the SAME call into fitness.score_route with the same congestion_lookup
and weights -- if those ever diverge, the comparison is meaningless, so the
diagnostics re-verify that identity explicitly.
"""

import itertools
import math
import sys

import numpy as np

from src.planner.fitness import route_components, score_route
from src.planner.qpso import default_budget, va_qpso
from src.planner.qpso_encoding import (
    adjacency_from_network_graph,
    compute_distance_matrix,
    decode_order,
    pick_mutually_reachable_stops,
)
from src.state_extraction.network_graph import NetworkGraph

NET_FILE = "networks/delhi/delhi_intersection.net.xml"
NUM_STOPS = 6

NUM_PARTICLES, MAX_ITERATIONS, MAX_RESTARTS = default_budget(NUM_STOPS)
PATIENCE = 15
NUM_RUNS = 30
PASS_THRESHOLD = 28

TOLERANCE_RTOL = 1e-6
NEAR_OPTIMAL_RTOL = 0.01  # informational only

VOLATILITY_INDEX = 0.5
WEIGHTS = (1.0, 1.0, 1.0)
CONGESTION_LOOKUP = {}


def build_problem():
    network_graph = NetworkGraph(NET_FILE)
    adjacency = adjacency_from_network_graph(network_graph, edge_weights={})
    stops = pick_mutually_reachable_stops(adjacency, NUM_STOPS)
    distance_matrix = compute_distance_matrix(adjacency, stops)
    assert np.all(np.isfinite(distance_matrix)), "expected all stops to be mutually reachable"
    return stops, distance_matrix


def objective(order, distance_matrix):
    """The single source of truth for fitness, used by BOTH brute force and the swarm."""
    return score_route(np.asarray(order), distance_matrix, CONGESTION_LOOKUP, WEIGHTS)


def brute_force_optimum(distance_matrix):
    """Exhaustively score all 720 orderings. Returns (best_order, best_score, all_scores)."""
    all_scores = []
    best_order, best_score = None, np.inf
    for order in itertools.permutations(range(NUM_STOPS)):
        score = objective(order, distance_matrix)
        all_scores.append(score)
        if score < best_score:
            best_score, best_order = score, np.array(order)
    return best_order, best_score, np.array(all_scores)


def run_swarm(distance_matrix, seed):
    def fitness_fn(x):
        return objective(decode_order(x), distance_matrix)

    position, score = va_qpso(
        dim=NUM_STOPS,
        fitness_fn=fitness_fn,
        volatility_index=VOLATILITY_INDEX,
        num_particles=NUM_PARTICLES,
        max_iterations=MAX_ITERATIONS,
        max_restarts=MAX_RESTARTS,
        patience=PATIENCE,
        seed=seed,
    )
    return decode_order(position), score


def diagnose(stops, distance_matrix, optimal_order, optimal_score, failures):
    """
    Failure diagnostics, in the order an encoding/fitness bug should be ruled
    out BEFORE suspecting the QPSO update equations -- an encoding or fitness
    bug is by far the more common failure mode.
    """
    print()
    print("=" * 78)
    print("DIAGNOSTICS -- ruling out encoding/fitness bugs before search logic")
    print("=" * 78)

    print("\n[1] Failing seeds: brute-force optimum vs QPSO result")
    print(f"    {'seed':>6} | {'brute-force':>12} | {'qpso':>12} | {'gap':>10} | {'gap %':>7} | order")
    for seed, order, score in failures:
        gap = score - optimal_score
        print(f"    {seed:>6} | {optimal_score:>12.6f} | {score:>12.6f} | {gap:>10.6f} | "
              f"{100 * gap / optimal_score:>6.2f}% | {order.tolist()}")

    print("\n[2] decode_order() -- does it produce valid permutations?")
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(1000):
        decoded = decode_order(rng.uniform(0, 1, NUM_STOPS))
        if sorted(decoded.tolist()) != list(range(NUM_STOPS)):
            bad += 1
    print(f"    1000 random key vectors -> {1000 - bad} valid permutations, {bad} invalid")
    print(f"    {'OK' if bad == 0 else 'BUG: decode_order is not returning permutations'}")

    print("\n[3] decode_order() -- is the optimum REPRESENTABLE in random-key space?")
    # Build the key vector that must decode to optimal_order: assign ascending
    # keys in the order the optimum visits stops.
    keys = np.empty(NUM_STOPS)
    for rank, stop_index in enumerate(optimal_order):
        keys[stop_index] = rank / NUM_STOPS
    round_trip = decode_order(keys)
    representable = round_trip.tolist() == optimal_order.tolist()
    print(f"    keys {np.round(keys, 3).tolist()} -> decode_order -> {round_trip.tolist()}")
    print(f"    optimum {optimal_order.tolist()} representable: {representable}")
    print(f"    {'OK' if representable else 'BUG: optimum cannot be encoded -- search can never find it'}")

    print("\n[4] score_route() -- does it reproduce the brute-force optimum score?")
    rescored = objective(optimal_order, distance_matrix)
    via_encoding = objective(decode_order(keys), distance_matrix)
    print(f"    brute-force optimum score        : {optimal_score:.9f}")
    print(f"    score_route(optimal_order)       : {rescored:.9f}")
    print(f"    score_route(decode_order(keys))  : {via_encoding:.9f}")
    consistent = np.isclose(rescored, optimal_score) and np.isclose(via_encoding, optimal_score)
    print(f"    {'OK -- fitness is self-consistent' if consistent else 'BUG: score_route disagrees with itself'}")

    print("\n[5] score_route() -- is the swarm optimizing the SAME objective that is scored?")
    check_seed = failures[0][0]
    order, reported = run_swarm(distance_matrix, check_seed)
    independent = objective(order, distance_matrix)
    print(f"    seed {check_seed}: va_qpso reported {reported:.9f}")
    print(f"    independent score_route of returned order: {independent:.9f}")
    same = np.isclose(reported, independent)
    print(f"    {'OK -- optimized and measured objectives match' if same else 'BUG: swarm optimizes something other than what is scored'}")

    print("\n[6] score_route() -- component breakdown, optimum vs a failing result")
    T_opt, D_opt, C_opt = route_components(optimal_order, distance_matrix, CONGESTION_LOOKUP)
    T_bad, D_bad, C_bad = route_components(failures[0][1], distance_matrix, CONGESTION_LOOKUP)
    print(f"    optimum      T={T_opt:.4f} D={D_opt:.4f} C={C_opt:.4f}")
    print(f"    seed {failures[0][0]:<7} T={T_bad:.4f} D={D_bad:.4f} C={C_bad:.4f}")
    print("    (T and D are equal by construction -- see fitness.py module docstring)")

    print("\n[7] Encoding and fitness are sound, so measure the search itself:")
    print("    how much of permutation space does the swarm actually reach?")
    total_orderings = math.factorial(NUM_STOPS)
    budget = NUM_PARTICLES * MAX_ITERATIONS

    for seed, _, _ in failures[:2]:
        visited, final = _track_search_coverage(distance_matrix, seed)
        print(f"    seed {seed:<3} final={final:8.4f}  visited {len(visited):>3}/{total_orderings} "
              f"orderings ({100 * len(visited) / total_orderings:.1f}% of space)")

    rng = np.random.default_rng(0)
    random_visited = {tuple(decode_order(rng.uniform(0, 1, NUM_STOPS))) for _ in range(budget)}
    print(f"    {'uniform random':<12} baseline, same {budget}-evaluation budget: "
          f"{len(random_visited)}/{total_orderings} "
          f"({100 * len(random_visited) / total_orderings:.1f}% of space)")

    print("\n    If the swarm's coverage is far below the random baseline, the search is")
    print("    collapsing onto a subregion of permutation space rather than running out")
    print("    of budget -- confirm by re-running with early stopping disabled")
    print("    (patience >= max_iterations); if the result does not change, the swarm is")
    print("    trapped, not truncated. Likely causes, in order:")
    print("      a. contraction toward mbest/gbest concentrating the random keys, so")
    print("         argsort keeps decoding the same family of orderings")
    print("      b. position clipping to bounds creating argsort ties at 0.0/1.0")
    print("      c. the beta schedule / contraction rate")
    print("      d. the update equations themselves (least likely)")


def _track_search_coverage(distance_matrix, seed, iterations=300):
    """
    Re-run the swarm loop while recording every distinct ordering evaluated.
    Mirrors qpso._run_qpso's update equations exactly, with early stopping
    disabled, purely to measure how much of permutation space is reached.
    """
    rng = np.random.default_rng(seed)
    bounds = (0.0, 1.0)
    beta = 0.5 + (1.0 - 0.5) * VOLATILITY_INDEX  # va_qpso's constant beta

    positions = rng.uniform(bounds[0], bounds[1], (NUM_PARTICLES, NUM_STOPS))
    personal_best = np.copy(positions)
    personal_best_scores = np.full(NUM_PARTICLES, np.inf)
    global_best_score = np.inf
    global_best = np.zeros(NUM_STOPS)
    visited = set()

    for _ in range(iterations):
        for i in range(NUM_PARTICLES):
            order = decode_order(positions[i])
            visited.add(tuple(order))
            score = objective(order, distance_matrix)
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = np.copy(positions[i])
            if score < global_best_score:
                global_best_score = score
                global_best = np.copy(positions[i])

        mbest = np.mean(personal_best, axis=0)
        phi = rng.uniform(0.0, 1.0, (NUM_PARTICLES, NUM_STOPS))
        p = phi * personal_best + (1 - phi) * global_best
        u = rng.uniform(1e-12, 1.0, (NUM_PARTICLES, NUM_STOPS))
        k = rng.uniform(0.0, 1.0, (NUM_PARTICLES, NUM_STOPS))
        positions = p + np.where(k >= 0.5, 1.0, -1.0) * beta * np.abs(mbest - positions) * np.log(1.0 / u)
        positions = np.clip(positions, bounds[0], bounds[1])

    return visited, global_best_score


def main():
    stops, distance_matrix = build_problem()
    print(f"Stops ({NUM_STOPS}): {stops}")
    print(f"Setup (from qpso.default_budget({NUM_STOPS})): {NUM_PARTICLES} particles, "
          f"{MAX_ITERATIONS} max iterations, {MAX_RESTARTS} max restarts, patience {PATIENCE}, "
          f"{NUM_RUNS} runs, volatility_index={VOLATILITY_INDEX}")

    optimal_order, optimal_score, all_scores = brute_force_optimum(distance_matrix)
    print(f"\nBrute force over {len(all_scores)} orderings ({NUM_STOPS}! = "
          f"{math.factorial(NUM_STOPS)}):")
    print(f"  true optimum : {optimal_order.tolist()}  score={optimal_score:.6f}")
    print(f"  worst        : score={all_scores.max():.6f}")
    print(f"  mean         : score={all_scores.mean():.6f}")

    hits, near, failures = 0, 0, []
    for seed in range(NUM_RUNS):
        order, score = run_swarm(distance_matrix, seed)
        if np.isclose(score, optimal_score, rtol=TOLERANCE_RTOL):
            hits += 1
        else:
            failures.append((seed, order, score))
            if np.isclose(score, optimal_score, rtol=NEAR_OPTIMAL_RTOL):
                near += 1

    print(f"\nResults over {NUM_RUNS} runs (seeds 0-{NUM_RUNS - 1}):")
    print(f"  hit the true optimum (rtol={TOLERANCE_RTOL}): {hits}/{NUM_RUNS}")
    print(f"  near-optimal within {NEAR_OPTIMAL_RTOL:.0%} (informational, does not count): {near}")
    print(f"  missed: {len(failures)}")

    passed = hits >= PASS_THRESHOLD
    print()
    print("=" * 78)
    if passed:
        print(f"PASS: {hits}/{NUM_RUNS} hit the optimum (threshold {PASS_THRESHOLD}/{NUM_RUNS})")
        print("=" * 78)
        return 0

    print(f"FAIL: {hits}/{NUM_RUNS} hit the optimum (threshold {PASS_THRESHOLD}/{NUM_RUNS})")
    print("=" * 78)
    diagnose(stops, distance_matrix, optimal_order, optimal_score, failures)
    print("\nDo not proceed to further experiments until this passes.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
