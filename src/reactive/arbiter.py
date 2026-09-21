"""
Replan Arbiter: reroute-rate based early-replan trigger.

Tracks how many reactive reroutes (reactive.py) have fired in the last
`window_seconds` of simulated time. A burst of reactive reroutes is a
symptom that the current QPSO plan (qpso.replan) no longer fits actual
network conditions -- rather than waiting for the next scheduled replan
tick, the arbiter lets run_hybrid.py pull one forward.
"""

from collections import deque
from typing import Deque


class ReplanArbiter:
    """
    Call record_reroute(sim_time) every time reactive.py fires a reroute,
    and should_trigger_early_replan(sim_time) once per step to check
    whether the rolling reroute count has crossed the threshold.
    """

    def __init__(self, window_seconds: float = 60.0, reroute_threshold: int = 5):
        """
        Args:
            window_seconds: rolling window, in simulated seconds, over which
                reroutes are counted.
            reroute_threshold: number of reroutes within that window that
                triggers an early replan signal.
        """
        self.window_seconds = window_seconds
        self.reroute_threshold = reroute_threshold
        self._timestamps: Deque[float] = deque()

    def record_reroute(self, sim_time: float) -> None:
        """Call once per reactive reroute event, with the sim time it occurred."""
        self._timestamps.append(sim_time)

    def _prune(self, sim_time: float) -> None:
        cutoff = sim_time - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def reroute_count(self, sim_time: float) -> int:
        """Number of reroutes recorded within the last window_seconds of sim_time."""
        self._prune(sim_time)
        return len(self._timestamps)

    def should_trigger_early_replan(self, sim_time: float) -> bool:
        return self.reroute_count(sim_time) >= self.reroute_threshold

    def notify_replanned(self, sim_time: float) -> None:
        """
        Call whenever a replan actually runs -- whether triggered by this
        arbiter or by the regular schedule -- to clear accumulated
        pressure. Reroutes recorded before a fresh plan say nothing about
        whether the NEW plan is already failing; without this, a single
        early-triggered replan would immediately re-trigger itself on the
        next step, since the same old reroutes are still inside the window.
        """
        self._timestamps.clear()
