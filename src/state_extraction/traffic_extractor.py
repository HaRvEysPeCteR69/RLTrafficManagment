"""
Traffic State Extractor.

Extracts real-time traffic features per edge/lane:
- Mean speed
- Travel time
- Halting vehicles / queue length
- Vehicle occupancy and density
- Vehicle composition (two-wheelers, autos, buses, etc.)
"""

from typing import Dict, Any, List
import traci


class TrafficStateExtractor:
    """
    Extracts edge-level dynamic traffic states from running TraCI simulation.
    """

    def __init__(self, monitored_edges: List[str]):
        self.monitored_edges = monitored_edges

    def extract_edge_states(self) -> Dict[str, Dict[str, float]]:
        """
        Extract dynamic metrics for all monitored edges.

        Returns:
            Dict mapping edge_id -> {
                'mean_speed': float,
                'travel_time': float,
                'halting_count': int,
                'occupancy': float,
                'vehicle_count': int
            }
        """
        states = {}
        for edge_id in self.monitored_edges:
            states[edge_id] = {
                "mean_speed": traci.edge.getLastStepMeanSpeed(edge_id),
                "travel_time": traci.edge.getTraveltime(edge_id),
                "halting_count": traci.edge.getLastStepHaltingNumber(edge_id),
                "occupancy": traci.edge.getLastStepOccupancy(edge_id),
                "vehicle_count": traci.edge.getLastStepVehicleNumber(edge_id),
            }
        return states
