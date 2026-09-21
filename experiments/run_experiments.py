"""
Experiment Runner.

Runs multi-scenario benchmarks comparing:
- QPSO Route Optimizer
- Reactive QPSO (QPSO + Reactive Detours)
- Static Shortest Path (Dijkstra)
- Standard PSO

Collects:
- Total delivery travel time
- Service delay / lateness
- Average vehicle speed
- Detour counts
- Volatility exposure score
"""

import argparse
from typing import Dict, List, Any


class ExperimentRunner:
    """
    Orchestrates comparative benchmarking across delivery scenarios.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path

    def run_benchmark(self, scenarios: List[str]) -> Dict[str, Any]:
        """Execute benchmark suite across scenarios."""
        print(f"Running benchmarks for scenarios: {scenarios}")
        results = {}
        # Scaffolding for benchmark loop
        return results


def main():
    parser = argparse.ArgumentParser(description="Run delivery route optimization experiments")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--scenarios", nargs="+", default=["baseline_delhi"])
    args = parser.parse_args()

    runner = ExperimentRunner(config_path=args.config)
    runner.run_benchmark(args.scenarios)


if __name__ == "__main__":
    main()
