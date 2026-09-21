"""
Test script for SubscriptionStateExtractor (src/state_extraction/state.py).

Connects, steps the simulation 60 times, calls get_state() every step, and
asserts the result is well-formed (no NaNs, occupancy in [0, 1]). Also times
60 back-to-back get_state() calls as a baseline for later performance
comparisons (e.g. against the old per-object getXxx() polling approach, or
against a libsumo backend).
"""

import math
import time

from src.state_extraction.network_graph import NetworkGraph
from src.state_extraction.state import SubscriptionStateExtractor

NET_FILE = "networks/delhi/delhi_intersection.net.xml"
SUMO_CFG = "networks/delhi/delhi_intersection.sumocfg"
NUM_STEPS = 60


def test_get_state():
    """Step 60 times, validating get_state() output at every step."""
    edge_ids = list(NetworkGraph(NET_FILE).edges.keys())
    print(f"Tracking {len(edge_ids)} edges.")

    extractor = SubscriptionStateExtractor(edge_ids, use_libsumo=False)
    extractor.connect(SUMO_CFG, use_gui=False, step_length=1.0)

    try:
        for i in range(NUM_STEPS):
            extractor.step()
            state = extractor.get_state()

            assert "edges" in state and "vehicles" in state
            assert len(state["edges"]) == len(edge_ids)

            for edge_id, metrics in state["edges"].items():
                for key, value in metrics.items():
                    assert not math.isnan(value), f"NaN in edges[{edge_id}][{key}] at step {i}"
                occupancy = metrics["occupancy"]
                assert 0.0 <= occupancy <= 1.0, (
                    f"occupancy {occupancy} out of [0, 1] for edge {edge_id} at step {i}"
                )

            for veh_id, metrics in state["vehicles"].items():
                for key, value in metrics.items():
                    if isinstance(value, float):
                        assert not math.isnan(value), f"NaN in vehicles[{veh_id}][{key}] at step {i}"

        print(f"OK: {NUM_STEPS} steps, no NaNs, occupancy always in [0, 1].")

        # Baseline timing: 60 get_state() calls with no simulationStep() in
        # between, i.e. pure subscription-result reshaping cost.
        start = time.perf_counter()
        for _ in range(NUM_STEPS):
            extractor.get_state()
        elapsed = time.perf_counter() - start
        print(f"get_state() x{NUM_STEPS}: {elapsed:.4f}s total ({elapsed / NUM_STEPS * 1000:.3f} ms/call)")

    finally:
        extractor.close()


if __name__ == "__main__":
    test_get_state()
