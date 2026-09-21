"""Unit tests for src/reactive/arbiter.py, with hand-checkable timelines."""

from src.reactive.arbiter import ReplanArbiter


def test_below_threshold_does_not_trigger():
    arbiter = ReplanArbiter(window_seconds=60.0, reroute_threshold=5)
    for t in [1.0, 2.0, 3.0, 4.0]:  # 4 reroutes, threshold is 5
        arbiter.record_reroute(t)
    assert arbiter.reroute_count(5.0) == 4
    assert not arbiter.should_trigger_early_replan(5.0)


def test_crossing_threshold_triggers():
    arbiter = ReplanArbiter(window_seconds=60.0, reroute_threshold=5)
    for t in [1.0, 2.0, 3.0, 4.0, 5.0]:  # 5th reroute crosses the threshold
        arbiter.record_reroute(t)
    assert arbiter.reroute_count(5.0) == 5
    assert arbiter.should_trigger_early_replan(5.0)


def test_old_reroutes_fall_out_of_the_rolling_window():
    arbiter = ReplanArbiter(window_seconds=10.0, reroute_threshold=3)
    arbiter.record_reroute(0.0)
    arbiter.record_reroute(1.0)
    arbiter.record_reroute(2.0)
    assert arbiter.reroute_count(2.0) == 3
    assert arbiter.should_trigger_early_replan(2.0)

    # By t=12, the window is (2, 12]; reroutes at 0.0 and 1.0 are now outside
    # (cutoff = 12 - 10 = 2, and both are < 2), only the one at t=2.0 remains.
    assert arbiter.reroute_count(12.0) == 1
    assert not arbiter.should_trigger_early_replan(12.0)


def test_notify_replanned_clears_pressure():
    arbiter = ReplanArbiter(window_seconds=60.0, reroute_threshold=3)
    for t in [1.0, 2.0, 3.0]:
        arbiter.record_reroute(t)
    assert arbiter.should_trigger_early_replan(3.0)

    arbiter.notify_replanned(3.0)
    assert arbiter.reroute_count(3.0) == 0
    assert not arbiter.should_trigger_early_replan(3.0)

    # And it doesn't just look empty until pruned -- a fresh reroute right
    # after shouldn't immediately re-trigger on its own.
    arbiter.record_reroute(3.1)
    assert arbiter.reroute_count(3.1) == 1
    assert not arbiter.should_trigger_early_replan(3.1)


if __name__ == "__main__":
    test_below_threshold_does_not_trigger()
    test_crossing_threshold_triggers()
    test_old_reroutes_fall_out_of_the_rolling_window()
    test_notify_replanned_clears_pressure()
    print("OK: all arbiter.py unit tests passed.")
