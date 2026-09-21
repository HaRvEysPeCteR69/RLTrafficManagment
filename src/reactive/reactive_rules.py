"""
Reactive Rule Engine.

Monitors vehicle progress and local traffic conditions.
When critical conditions occur (e.g. sudden blockage, extreme delay spike,
or road closure encounter), triggers fast localized rerouting or path detour
without invoking full global QPSO replanning.
"""

from typing import Dict, List, Optional, Any


class ReactiveRuleEngine:
    """
    Evaluates rule-based triggers during delivery route execution.
    """

    def __init__(
        self,
        speed_drop_threshold: float = 0.3,   # Reroute if speed drops below 30% of free-flow
        delay_threshold_seconds: float = 120.0,
        volatility_trigger: float = 0.8,
    ):
        self.speed_drop_threshold = speed_drop_threshold
        self.delay_threshold = delay_threshold_seconds
        self.volatility_trigger = volatility_trigger

    def should_reroute(
        self,
        current_edge: str,
        current_speed: float,
        free_flow_speed: float,
        volatility: float,
        delay_accumulated: float,
    ) -> bool:
        """Check whether local conditions warrant a reactive reroute."""
        if free_flow_speed > 0 and (current_speed / free_flow_speed) < self.speed_drop_threshold:
            return True
        if delay_accumulated > self.delay_threshold:
            return True
        if volatility > self.volatility_trigger:
            return True
        return False
