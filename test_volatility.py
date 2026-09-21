"""
Unit tests for volatility index calculation (src/volatility/volatility_index.py).
"""

import numpy as np
from src.volatility import VolatilityIndexCalculator, NetworkVolatilityIndex


def test_volatility_calculator_basic():
    calc = VolatilityIndexCalculator(window_size=5)
    assert calc.update("edge1", 10.0, 5.0) == 0.0  # < 2 items

    v = calc.update("edge1", 12.0, 6.0)
    assert v > 0.0
    # Speeds: [10.0, 12.0], mean = 11.0, std = 1.0 -> cv ~ 1.0 / 11.0 ~ 0.0909
    assert np.isclose(v, 1.0 / 11.0, rtol=1e-3)


def test_network_volatility_index_empty_or_single():
    nvi = NetworkVolatilityIndex(window_size=15, reference_variance=0.002)
    assert nvi.update({}) == 0.0
    assert nvi.update({"e1": 10.0, "e2": 10.0}) == 0.0  # only 1 history point


def test_network_volatility_index_zero_variance():
    nvi = NetworkVolatilityIndex(window_size=15, reference_variance=0.002)
    for _ in range(10):
        val = nvi.update({"e1": 10.0, "e2": 15.0})
    # Mean is always 12.5, variance is 0.0 -> index should be 0.0
    assert val == 0.0


def test_network_volatility_index_calibrated_scaling():
    nvi = NetworkVolatilityIndex(window_size=15, reference_variance=0.002)

    # Simulate low network-mean variance ~ 0.0007
    # e.g., alternating between 7.50 and 7.55 m/s
    for i in range(15):
        speed = 7.50 if i % 2 == 0 else 7.55
        val_low = nvi.update({"e1": speed})

    assert 0.15 <= val_low <= 0.40, f"Expected low volatility ~0.2-0.3, got {val_low}"

    # Simulate higher variance ~ 0.004
    # e.g., alternating between 7.44 and 7.56 m/s (amplitude 0.06 -> var ~ 0.0036)
    for i in range(15):
        speed = 7.44 if i % 2 == 0 else 7.56
        val_high = nvi.update({"e1": speed})

    assert 0.50 <= val_high <= 0.85, f"Expected high volatility ~0.6-0.8, got {val_high}"
    assert val_high > val_low


def test_network_volatility_index_bounds():
    nvi = NetworkVolatilityIndex(window_size=10, reference_variance=0.002)
    rng = np.random.default_rng(42)
    for _ in range(50):
        speeds = {f"e{i}": rng.uniform(5.0, 20.0) for i in range(20)}
        score = nvi.update(speeds)
        assert 0.0 <= score < 1.0


if __name__ == "__main__":
    test_volatility_calculator_basic()
    test_network_volatility_index_empty_or_single()
    test_network_volatility_index_zero_variance()
    test_network_volatility_index_calibrated_scaling()
    test_network_volatility_index_bounds()
    print("OK: all volatility_index.py unit tests passed.")
