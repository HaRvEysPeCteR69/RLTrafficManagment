"""
Main CLI entrypoint to run QPSO Route Planner.

Usage:
    python run_planner.py --scenario config/scenarios.yaml --gui
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.volatility import VolatilityIndexCalculator
from src.state_extraction import TrafficStateExtractor
from src.reactive import ReactiveRuleEngine
from src.simulation import SumoSimulationManager


def main():
    parser = argparse.ArgumentParser(description="QPSO Delivery Route Planner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--gui", action="store_true", help="Launch SUMO with GUI")
    parser.add_argument("--particles", type=int, default=30, help="Swarm particle count")
    parser.add_argument("--iterations", type=int, default=50, help="QPSO iteration count")
    args = parser.parse_args()

    print("=" * 60)
    print("QPSO Delivery Route Optimizer - Delhi Network")
    print(f"Config: {args.config} | GUI: {args.gui}")
    print(f"Particles: {args.particles} | Iterations: {args.iterations}")
    print("=" * 60)

    # Instantiate modules. Route planning itself goes through
    # src.planner.qpso.replan(), which is called per re-plan tick with a
    # frozen state snapshot rather than held as a long-lived optimizer object.
    volatility_calc = VolatilityIndexCalculator()
    reactive_engine = ReactiveRuleEngine()

    print("Modules initialized successfully.")


if __name__ == "__main__":
    main()
