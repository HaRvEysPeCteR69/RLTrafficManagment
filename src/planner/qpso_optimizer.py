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
