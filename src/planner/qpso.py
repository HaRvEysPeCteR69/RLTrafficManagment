"""
QPSO per Sun, Feng & Xu (2004)'s delta-potential-well formulation.

Sun, J., Feng, B., & Xu, W. (2004). "Particle swarm optimization with
particles having quantum behavior." Proceedings of the 2004 Congress on
Evolutionary Computation, 325-331.

Exact update equations used below (particle i, dimension d, iteration t):

    mbest_d = (1/M) * sum_i personal_best[i][d]
    phi ~ U(0, 1)
    p_id = phi * personal_best[i][d] + (1 - phi) * global_best[d]
    u ~ U(0, 1);  k ~ U(0, 1)
    x[i][d] = p_id + beta * |mbest_d - x[i][d]| * ln(1/u)   if k >= 0.5
            = p_id - beta * |mbest_d - x[i][d]| * ln(1/u)   otherwise

This is the standard form cited throughout the QPSO literature -- not a
simplified variant. Both variants below (fixed_beta_qpso, va_qpso) share this
exact core loop; they differ ONLY in how `beta` is computed each iteration.

Convergence / stopping criterion: fitness-plateau early stopping. The swarm
always runs at least `patience` iterations, then stops as soon as
global_best has failed to improve by more than `tol` for `patience`
consecutive iterations, or after `max_iterations`, whichever comes first.
This is chosen over a fixed iteration count so replan() doesn't keep
spinning once the swarm has converged, while `max_iterations` still bounds
the worst case.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .fitness import CongestionLookup, score_route
from .qpso_encoding import decode_order

FitnessFn = Callable[[np.ndarray], float]
BetaFn = Callable[[int], float]  # iteration t -> beta(t)


def _run_qpso(
    dim: int,
    fitness_fn: FitnessFn,
    beta_fn: BetaFn,
    num_particles: int,
    max_iterations: int,
    bounds: Tuple[float, float],
    seed: Optional[int],
    patience: int,
    tol: float,
) -> Tuple[np.ndarray, float]:
    """
    Shared QPSO core loop (Sun, Feng & Xu, 2004) -- see module docstring for
    the exact update equations. Both fixed_beta_qpso and va_qpso call this
    with different `beta_fn` implementations and nothing else differs.

    Returns:
        (global_best position, global_best fitness).
    """
    rng = np.random.default_rng(seed)

    positions = rng.uniform(bounds[0], bounds[1], (num_particles, dim))
    personal_best = np.copy(positions)
    personal_best_scores = np.full(num_particles, np.inf)
    global_best = np.zeros(dim)
    global_best_score = np.inf

    iterations_since_improvement = 0

    for t in range(max_iterations):
        # 1. Evaluate fitness, update personal_best / global_best.
        for i in range(num_particles):
            score = fitness_fn(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = np.copy(positions[i])
            if score < global_best_score - tol:
                global_best_score = score
                global_best = np.copy(positions[i])
                iterations_since_improvement = 0

        # Fitness-plateau stopping criterion (see module docstring).
        if t >= patience and iterations_since_improvement >= patience:
            break
        iterations_since_improvement += 1

        # 2. mbest_d = mean personal best across the swarm, per dimension.
        mbest = np.mean(personal_best, axis=0)

        # 3. beta(t) -- the only thing that differs between variants.
        beta = beta_fn(t)

        # 4. Quantum position update, per particle per dimension.
        phi = rng.uniform(0.0, 1.0, (num_particles, dim))
        p = phi * personal_best + (1 - phi) * global_best

        u = rng.uniform(1e-12, 1.0, (num_particles, dim))
        k = rng.uniform(0.0, 1.0, (num_particles, dim))
        sign = np.where(k >= 0.5, 1.0, -1.0)

        positions = p + sign * beta * np.abs(mbest - positions) * np.log(1.0 / u)
        positions = np.clip(positions, bounds[0], bounds[1])

    return global_best, global_best_score


def fixed_beta_qpso(
    dim: int,
    fitness_fn: FitnessFn,
    num_particles: int = 30,
    max_iterations: int = 100,
    beta_max: float = 1.0,
    beta_min: float = 0.5,
    bounds: Tuple[float, float] = (0.0, 1.0),
    seed: Optional[int] = None,
    patience: int = 15,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Standard linear-anneal beta baseline, used throughout the QPSO literature
    to compare novel variants against (this is that baseline, not a
    strawman):

        beta(t) = beta_max - (beta_max - beta_min) * (t / max_iterations)

    beta depends only on iteration progress -- a property of the swarm's own
    schedule, not of anything external. See va_qpso below for the contrast.
    """
    def beta_fn(t: int) -> float:
        return beta_max - (beta_max - beta_min) * (t / max_iterations)

    return _run_qpso(dim, fitness_fn, beta_fn, num_particles, max_iterations, bounds, seed, patience, tol)


# --- va_qpso: volatility-adaptive beta (this project's contribution) ------
#
# Prior adaptive-beta strategies in the QPSO literature -- iteration-count
# schedules (fixed_beta_qpso above), fitness-stagnation triggers, population-
# diversity measures -- all derive beta from the swarm's OWN internal search
# state: how far along the run is, whether gbest has stopped improving, how
# spread out the particles currently are. Every one of those quantities is
# computable from the optimizer alone, with no reference to the problem it
# is solving.
#
# va_qpso instead derives beta from a measurement of the EXTERNAL
# environment being optimized over: live traffic volatility on the road
# network (see volatility.NetworkVolatilityIndex), taken at the moment
# replan() is invoked. It has nothing to do with this swarm run's iteration
# count, fitness history, or particle spread -- a converged, static swarm
# and a freshly-initialized one get the same beta if network conditions are
# the same. That is the paper's actual contribution, so it must stay
# unambiguous in code and comments alike: no iteration/fitness/diversity
# term is allowed to leak into how beta is computed here.
def va_qpso(
    dim: int,
    fitness_fn: FitnessFn,
    volatility_index: float,
    num_particles: int = 30,
    max_iterations: int = 100,
    beta_max: float = 1.0,
    beta_min: float = 0.5,
    bounds: Tuple[float, float] = (0.0, 1.0),
    seed: Optional[int] = None,
    patience: int = 15,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    beta = beta_min + (beta_max - beta_min) * volatility_index

    volatility_index must already be normalized to [0, 1] (e.g. from
    volatility.NetworkVolatilityIndex.update()) and is held fixed for the
    whole run: it is a snapshot of network conditions at replan time, not a
    per-iteration swarm-state signal.
    """
    if not 0.0 <= volatility_index <= 1.0:
        raise ValueError(f"volatility_index must be in [0, 1], got {volatility_index}")

    beta = beta_min + (beta_max - beta_min) * volatility_index

    def beta_fn(t: int) -> float:
        return beta

    return _run_qpso(dim, fitness_fn, beta_fn, num_particles, max_iterations, bounds, seed, patience, tol)


def replan(
    stops: List[str],
    distance_matrix: np.ndarray,
    congestion_lookup: CongestionLookup,
    volatility_index: float,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_particles: int = 30,
    max_iterations: int = 100,
    beta_max: float = 1.0,
    beta_min: float = 0.5,
    seed: Optional[int] = None,
    patience: int = 15,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Run va_qpso to convergence (see module docstring for the stopping
    criterion) on a frozen re-plan snapshot, and return the best stop order
    found plus its fitness score.

    Args:
        stops: Stop identifiers, in the order they map to distance_matrix's
            rows/columns. Not used directly in the search -- only its length
            (the encoding dimension) matters here; map the returned order
            back to ids yourself via [stops[i] for i in best_order].
        distance_matrix: (n, n) live-weighted travel-time matrix for this
            snapshot, e.g. from qpso_encoding.compute_distance_matrix().
        congestion_lookup: Per-leg edge occupancy/capacity data (see
            fitness.CongestionLookup).
        volatility_index: Current network volatility in [0, 1], e.g. from
            volatility.NetworkVolatilityIndex.update(), measured on the same
            snapshot as distance_matrix / congestion_lookup.
        weights: (w1, w2, w3) passed through to score_route for (T, D, C).

    Returns:
        (best_order, best_score): best_order is the decoded visit-order
        permutation (indices into distance_matrix / stops), best_score is
        its score_route() fitness.
    """
    n = len(stops)
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match len(stops)={n}."
        )

    def fitness_fn(x: np.ndarray) -> float:
        order = decode_order(x)
        return score_route(order, distance_matrix, congestion_lookup, weights)

    best_position, best_score = va_qpso(
        dim=n,
        fitness_fn=fitness_fn,
        volatility_index=volatility_index,
        num_particles=num_particles,
        max_iterations=max_iterations,
        beta_max=beta_max,
        beta_min=beta_min,
        seed=seed,
        patience=patience,
        tol=tol,
    )
    best_order = decode_order(best_position)
    return best_order, best_score
