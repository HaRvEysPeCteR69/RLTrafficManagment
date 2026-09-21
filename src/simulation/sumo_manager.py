"""
Simulation Manager for SUMO & TraCI.

Controls simulation lifecycle, stepping, delivery vehicle dispatch,
and route updating in the real-time simulation.
"""

import os
import sys
from typing import Optional, Dict, Any, List

# Locate SUMO_HOME
if 'SUMO_HOME' not in os.environ:
    possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
        "/usr/share/sumo",
        "/opt/sumo",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            break

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools not in sys.path:
        sys.path.append(tools)

try:
    import traci
except ImportError:
    traci = None


class SumoSimulationManager:
    """
    Manages TraCI connection and simulation stepping for route optimization.
    """

    def __init__(
        self,
        sumo_cfg: str,
        use_gui: bool = False,
        step_length: float = 1.0,
        port: Optional[int] = None,
    ):
        self.sumo_cfg = sumo_cfg
        self.use_gui = use_gui
        self.step_length = step_length
        self.port = port
        self.is_running = False

    def start(self):
        """Start SUMO / TraCI process."""
        if traci is None:
            raise RuntimeError("TraCI module is not available. Ensure SUMO is installed.")

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumo_cfg,
            "--step-length", str(self.step_length),
            "--start",
            "--quit-on-end", "false"
        ]
        traci.start(sumo_cmd)
        self.is_running = True

    def step(self) -> float:
        """Advance the simulation by one step and return current sim time."""
        if not self.is_running:
            raise RuntimeError("Simulation not started.")
        traci.simulationStep()
        return traci.simulation.getTime()

    def stop(self):
        """Close TraCI simulation."""
        if self.is_running:
            traci.close()
            self.is_running = False
