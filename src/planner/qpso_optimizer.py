"""
Quantum-behaved Particle Swarm Optimization (QPSO) for Delivery Route Planning.

Formulates route selection / waypoint sequencing under uncertainty.
Particles update positions via delta-potential well wave function:
    mbest = (1/M) * sum(pbest_i)
    p_i = phi * pbest_i + (1 - phi) * gbest
    x_i(t+1) = p_i +/- beta * |mbest - x_i(t)| * ln(1/u)
"""

import numpy as np
from typing import List, Tuple, Callable, Optional

from .qpso_encoding import decode_order
from .fitness import score_route, CongestionLookup


class QPSOOptimizer:
    """
    Quantum-behaved Particle Swarm Optimizer.
    """

    def __init__(
        self,
        num_particles: int = 30,
        max_iterations: int = 100,
        beta: float = 0.7,
        dim: int = 10,
        bounds: Tuple[float, float] = (0.0, 1.0),
        seed: Optional[int] = None,
    ):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.beta = beta
        self.dim = dim
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)

        # Initialize particles
        self.positions = self.rng.uniform(bounds[0], bounds[1], (num_particles, dim))
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(num_particles, float("inf"))
        self.gbest_position = np.zeros(dim)
        self.gbest_score = float("inf")

    def optimize(self, fitness_fn: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
        """
        Run QPSO optimization loop.

        Args:
            fitness_fn: Function mapping particle position to scalar cost

        Returns:
            (best_position, best_fitness)
        """
        for it in range(self.max_iterations):
            # Evaluate fitness
            for i in range(self.num_particles):
                score = fitness_fn(self.positions[i])
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = np.copy(self.positions[i])
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = np.copy(self.positions[i])

            # Calculate Mean Best (mbest)
            mbest = np.mean(self.pbest_positions, axis=0)

            # Contraction-Expansion coefficient decay (optional annealing)
            current_beta = self.beta * (1.0 - 0.5 * it / self.max_iterations)

            # Quantum position updates
            for i in range(self.num_particles):
                phi = self.rng.uniform(0, 1, self.dim)
                p = phi * self.pbest_positions[i] + (1 - phi) * self.gbest_position
                u = self.rng.uniform(1e-6, 1.0, self.dim)
                sign = self.rng.choice([-1.0, 1.0], size=self.dim)
                self.positions[i] = p + sign * current_beta * np.abs(mbest - self.positions[i]) * np.log(1.0 / u)
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

        return self.gbest_position, self.gbest_score

    def optimize_tour(
        self,
        distance_matrix: np.ndarray,
        congestion_lookup: Optional[CongestionLookup] = None,
        weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Tuple[np.ndarray, float]:
        """
        Optimize a stop visit order against a precomputed distance matrix and
        (optionally) live congestion data.

        Wires the swarm to the discrete sequencing problem via Bean's (1994)
        random-key encoding (see qpso_encoding.py): each particle position is
        just treated as a random-key vector, decoded to a permutation with
        decode_order() (argsort), and scored with fitness.score_route() —
        O(n) distance_matrix/congestion_lookup lookups per evaluation, no
        graph search inside the swarm loop.

        Args:
            distance_matrix: (n, n) precomputed live-weighted travel-time
                matrix, e.g. from qpso_encoding.compute_distance_matrix().
                n must equal self.dim (one random key per stop).
            congestion_lookup: Per-leg edge occupancy/capacity data for the
                congestion term (see fitness.CongestionLookup). Defaults to
                no congestion data (every leg contributes 0 to that term),
                so the fitness reduces to the distance/time terms only.
            weights: (w1, w2, w3) passed through to score_route for
                (T, D, C); normalized internally to sum to 1.

        Returns:
            (best_order, best_score): best_order is the decoded visit-order
            permutation (indices into distance_matrix's rows/columns), and
            best_score is its weighted score_route() fitness — not a raw
            distance. Call fitness.route_components(best_order,
            distance_matrix, congestion_lookup) if you need the underlying
            (T, D, C) breakdown.
        """
        n = distance_matrix.shape[0]
        if n != self.dim:
            raise ValueError(
                f"distance_matrix has {n} stops but optimizer was constructed with dim={self.dim}; "
                "construct QPSOOptimizer with dim=n (one random key per stop)."
            )
        if congestion_lookup is None:
            congestion_lookup = {}

        def fitness_fn(x: np.ndarray) -> float:
            order = decode_order(x)
            return score_route(order, distance_matrix, congestion_lookup, weights)

        best_position, best_score = self.optimize(fitness_fn)
        best_order = decode_order(best_position)
        return best_order, best_score
