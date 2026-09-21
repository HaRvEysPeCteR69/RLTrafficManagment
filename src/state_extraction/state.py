"""
Subscription-based traffic state extraction over TraCI/libsumo.

TraCI's own documentation flags per-object polling (calling traci.edge.getXxx()
or traci.vehicle.getXxx() for every object, every step) as the standard
performance mistake: each such call is a network round-trip to the running
SUMO process. The fix is subscriptions: register interest in a set of
variables for an object ONCE, and thereafter retrieve everything that changed
in a single batched call per step (traci.edge.getAllSubscriptionResults(),
traci.vehicle.getAllSubscriptionResults()) instead of looping per-object.

This module subscribes:
- Every edge, once at connect time, to mean speed / occupancy / halting count.
- Every vehicle, once when it departs (vehicles can't be subscribed before
  they exist), to speed / current edge / waiting time. Departures and
  arrivals are themselves discovered via a single simulation-level
  subscription rather than by scanning traci.vehicle.getIDList() every step.

get_state() only reshapes whatever the last subscription round already
delivered — it issues no new TraCI queries of its own.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Set

if "SUMO_HOME" not in os.environ:
    _possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
        "/usr/share/sumo",
        "/opt/sumo",
    ]
    for _path in _possible_paths:
        if os.path.exists(_path):
            os.environ["SUMO_HOME"] = _path
            break

if "SUMO_HOME" in os.environ:
    _tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if _tools not in sys.path:
        sys.path.append(_tools)

# Subscription/result variable ids are plain protocol constants shared by
# traci and libsumo, so they can be imported without an active connection
# regardless of which backend get_state() ends up using.
import traci.constants as tc

EDGE_SUBSCRIBE_VARS = (
    tc.LAST_STEP_MEAN_SPEED,
    tc.LAST_STEP_OCCUPANCY,
    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
)
_EDGE_FIELD_NAMES = {
    tc.LAST_STEP_MEAN_SPEED: "mean_speed",
    tc.LAST_STEP_OCCUPANCY: "occupancy",
    tc.LAST_STEP_VEHICLE_HALTING_NUMBER: "halting_count",
}

VEHICLE_SUBSCRIBE_VARS = (
    tc.VAR_SPEED,
    tc.VAR_ROAD_ID,
    tc.VAR_WAITING_TIME,
    tc.VAR_ROUTE_INDEX,
)
_VEHICLE_FIELD_NAMES = {
    tc.VAR_SPEED: "speed",
    tc.VAR_ROAD_ID: "edge_id",
    tc.VAR_WAITING_TIME: "waiting_time",
    tc.VAR_ROUTE_INDEX: "route_index",
}
# tc.VAR_ROUTE (the full edge list) and tc.VAR_NEXT_EDGE are NOT subscribable
# in this SUMO version (confirmed empirically -- subscribing either raises
# "unsupported variable"). route_index IS subscribable, so a caller that
# needs "this vehicle's planned next edge" (e.g. reactive.py) must fetch the
# full route ONCE via a regular traci.vehicle.getRoute() call when the
# vehicle departs (see the `traci` property below), cache it, and thereafter
# combine that cached list with the subscribed route_index each step --
# still zero per-step getRoute() polling.

_SIM_SUBSCRIBE_VARS = (
    tc.VAR_DEPARTED_VEHICLES_IDS,
    tc.VAR_ARRIVED_VEHICLES_IDS,
)


class SubscriptionStateExtractor:
    """
    Connects to SUMO and maintains TraCI/libsumo subscriptions so that
    get_state() is a pure reshape of already-fetched data, never a fresh
    round-trip per object.
    """

    def __init__(self, edge_ids: List[str], use_libsumo: bool = False):
        """
        Args:
            edge_ids: Every edge to track. Subscribed once, at connect().
            use_libsumo: If True, drive the simulation in-process via libsumo
                instead of the traci socket client — same call signatures,
                no socket round-trips at all, but no SUMO-GUI support. Worth
                it once QPSO is firing hundreds of fitness evaluations per
                re-plan and the GUI isn't needed.
        """
        self.edge_ids = list(edge_ids)
        self.use_libsumo = use_libsumo

        if use_libsumo:
            try:
                import libsumo as traci_or_libsumo
            except ImportError as exc:
                raise RuntimeError(
                    "use_libsumo=True but the 'libsumo' package is not installed "
                    "in this environment. Install libsumo, or construct with "
                    "use_libsumo=False to use the standard traci socket client."
                ) from exc
        else:
            import traci as traci_or_libsumo

        self._traci = traci_or_libsumo
        self._subscribed_vehicles: Set[str] = set()
        self._connected = False

    def connect(self, sumo_cfg: str, use_gui: bool = False, step_length: float = 1.0) -> None:
        """Start SUMO and register every subscription exactly once."""
        if self._connected:
            raise RuntimeError("Already connected.")

        if use_gui and self.use_libsumo:
            raise RuntimeError("libsumo does not support sumo-gui; set use_gui=False.")

        sumo_binary = "sumo-gui" if use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", sumo_cfg,
            "--step-length", str(step_length),
            "--start",
            "--quit-on-end", "false",
            "--ignore-route-errors", "true",
        ]
        self._traci.start(sumo_cmd)
        self._connected = True

        for edge_id in self.edge_ids:
            self._traci.edge.subscribe(edge_id, EDGE_SUBSCRIBE_VARS)

        # Departures/arrivals are how we learn which vehicles to subscribe to
        # (or drop), without ever scanning getIDList() ourselves.
        self._traci.simulation.subscribe(_SIM_SUBSCRIBE_VARS)

    def step(self) -> float:
        """Advance the simulation one step and sync vehicle subscriptions."""
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        self._traci.simulationStep()
        self._sync_vehicle_subscriptions()
        return self._traci.simulation.getTime()

    def _sync_vehicle_subscriptions(self) -> None:
        """Subscribe newly-departed vehicles; drop arrived ones from tracking."""
        sim_results = self._traci.simulation.getSubscriptionResults() or {}
        departed = sim_results.get(tc.VAR_DEPARTED_VEHICLES_IDS, ())
        arrived = sim_results.get(tc.VAR_ARRIVED_VEHICLES_IDS, ())

        for veh_id in departed:
            self._traci.vehicle.subscribe(veh_id, VEHICLE_SUBSCRIBE_VARS)
            self._subscribed_vehicles.add(veh_id)

        # A vehicle that has arrived no longer exists in the simulation, so
        # there is nothing to unsubscribe on the TraCI side (and doing so
        # would error) — just stop tracking it locally.
        for veh_id in arrived:
            self._subscribed_vehicles.discard(veh_id)

    def get_state(self) -> Dict[str, Any]:
        """
        Reshape the last subscription round's results into
        {"edges": {edge_id: {...}}, "vehicles": {vehicle_id: {...}}}.

        Issues no per-object TraCI calls: everything here comes from the two
        batched getAllSubscriptionResults() calls below.
        """
        edge_raw = self._traci.edge.getAllSubscriptionResults()
        vehicle_raw = self._traci.vehicle.getAllSubscriptionResults()

        edges = {
            edge_id: {
                name: edge_raw.get(edge_id, {}).get(var_id, 0.0)
                for var_id, name in _EDGE_FIELD_NAMES.items()
            }
            for edge_id in self.edge_ids
        }

        vehicles = {
            veh_id: {
                name: raw.get(var_id)
                for var_id, name in _VEHICLE_FIELD_NAMES.items()
            }
            for veh_id, raw in vehicle_raw.items()
        }

        return {"edges": edges, "vehicles": vehicles}

    def close(self) -> None:
        if self._connected:
            self._traci.close()
            self._connected = False

    @property
    def traci(self):
        """
        The underlying traci/libsumo module, for the rare direct call that
        falls outside the batched-subscription flow above -- e.g. a one-time
        traci.vehicle.getRoute() when a vehicle departs, or issuing a
        traci.vehicle.setRoute() to apply a reactive reroute. Both are
        one-off, per-vehicle-lifecycle-event calls, not per-step polling.
        """
        return self._traci
