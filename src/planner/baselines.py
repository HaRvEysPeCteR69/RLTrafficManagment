"""
Routing Baselines for comparison against QPSO.

Includes:
- Dijkstra / A* Shortest Path
- Standard Particle Swarm Optimization (PSO)
- Greedy Next-Stop Heuristic
"""

from typing import List, Dict, Tuple
import heapq


class DijkstraBaseline:
    """Standard Dijkstra's algorithm on road graph."""

    def __init__(self, adjacency: Dict[str, List[Tuple[str, float]]]):
        self.adj = adjacency

    def find_shortest_path(self, start: str, end: str) -> Tuple[List[str], float]:
        """Compute shortest path between start and end nodes."""
        pq = [(0.0, start, [start])]
        visited = set()

        while pq:
            cost, current, path = heapq.heappop(pq)
            if current == end:
                return path, cost
            if current in visited:
                continue
            visited.add(current)

            for neighbor, edge_cost in self.adj.get(current, []):
                if neighbor not in visited:
                    heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))

        return [], float("inf")


class StandardPSOBaseline:
    """Standard velocity-displacement Particle Swarm Optimization."""
    pass
