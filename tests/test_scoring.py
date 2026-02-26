"""Tests for the shared scoring primitives."""

import math

import pytest

from sparkey_reflect.core.scoring import (
    bell,
    count_score,
    diminishing,
    linear_clamp,
    sigmoid,
    weighted_sum,
)


class TestSigmoid:
    def test_midpoint_returns_half(self):
        assert sigmoid(5, 5, 1.0) == pytest.approx(0.5)

    def test_far_above_midpoint_approaches_one(self):
        assert sigmoid(100, 5, 1.0) > 0.99

    def test_far_below_midpoint_approaches_zero(self):
        assert sigmoid(-100, 5, 1.0) < 0.01

    def test_higher_steepness_sharper_transition(self):
        shallow = sigmoid(6, 5, 1.0)
        steep = sigmoid(6, 5, 10.0)
        assert steep > shallow

    def test_symmetric_around_midpoint(self):
        above = sigmoid(7, 5, 1.0)
        below = sigmoid(3, 5, 1.0)
        assert above + below == pytest.approx(1.0)

    def test_monotonically_increasing(self):
        values = [sigmoid(x, 5, 2) for x in range(0, 11)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_overflow_protection(self):
        # Should not raise even with extreme values
        assert sigmoid(-1000, 0, 100) == pytest.approx(0.0, abs=1e-10)
        assert sigmoid(1000, 0, 100) == pytest.approx(1.0, abs=1e-10)

    def test_output_range_zero_to_one(self):
        for x in [-10, -1, 0, 0.5, 1, 5, 10, 100]:
            result = sigmoid(x, 3, 2)
            assert 0 <= result <= 1


class TestBell:
    def test_peak_at_center(self):
        assert bell(5, 5, 2) == pytest.approx(1.0)

    def test_symmetric_decay(self):
        left = bell(3, 5, 2)
        right = bell(7, 5, 2)
        assert left == pytest.approx(right)

    def test_wider_width_slower_decay(self):
        narrow = bell(7, 5, 1)
        wide = bell(7, 5, 5)
        assert wide > narrow

    def test_far_from_center_approaches_zero(self):
        assert bell(100, 5, 2) < 0.01

    def test_zero_width_edge_case(self):
        assert bell(5, 5, 0) == 1.0
        assert bell(6, 5, 0) == 0.0

    def test_output_range_zero_to_one(self):
        for x in range(-10, 20):
            result = bell(x, 5, 3)
            assert 0 <= result <= 1.0


class TestLinearClamp:
    def test_below_low_returns_zero(self):
        assert linear_clamp(0, 10, 20) == 0.0

    def test_above_high_returns_one(self):
        assert linear_clamp(30, 10, 20) == 1.0

    def test_midpoint_returns_half(self):
        assert linear_clamp(15, 10, 20) == pytest.approx(0.5)

    def test_at_low_returns_zero(self):
        assert linear_clamp(10, 10, 20) == 0.0

    def test_at_high_returns_one(self):
        assert linear_clamp(20, 10, 20) == 1.0

    def test_equal_low_high(self):
        assert linear_clamp(5, 5, 5) == 1.0
        assert linear_clamp(4, 5, 5) == 0.0


class TestDiminishing:
    def test_at_scale_returns_one(self):
        assert diminishing(9, 9) == pytest.approx(1.0)

    def test_above_scale_capped_at_one(self):
        assert diminishing(100, 9) == 1.0

    def test_zero_returns_zero(self):
        assert diminishing(0, 9) == 0.0

    def test_quarter_scale_returns_half(self):
        # sqrt(0.25) = 0.5
        assert diminishing(2.25, 9) == pytest.approx(0.5)

    def test_negative_input_treated_as_zero(self):
        assert diminishing(-5, 9) == 0.0

    def test_monotonically_increasing(self):
        values = [diminishing(x, 10) for x in range(0, 15)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


class TestCountScore:
    def test_below_all_thresholds(self):
        assert count_score(0, [(1, 0.3), (3, 0.7), (5, 1.0)]) == 0.0

    def test_at_first_threshold(self):
        assert count_score(1, [(1, 0.3), (3, 0.7), (5, 1.0)]) == 0.3

    def test_between_thresholds(self):
        assert count_score(2, [(1, 0.3), (3, 0.7), (5, 1.0)]) == 0.3

    def test_at_last_threshold(self):
        assert count_score(5, [(1, 0.3), (3, 0.7), (5, 1.0)]) == 1.0

    def test_above_all_thresholds(self):
        assert count_score(10, [(1, 0.3), (3, 0.7), (5, 1.0)]) == 1.0

    def test_empty_thresholds(self):
        assert count_score(5, []) == 0.0

    def test_unsorted_thresholds_work(self):
        assert count_score(4, [(5, 1.0), (1, 0.3), (3, 0.7)]) == 0.7


class TestWeightedSum:
    def test_single_perfect_score(self):
        assert weighted_sum([(1.0, 1.0)]) == pytest.approx(100.0)

    def test_single_zero_score(self):
        assert weighted_sum([(0.0, 1.0)]) == pytest.approx(0.0)

    def test_equal_weights(self):
        result = weighted_sum([(0.5, 1.0), (0.5, 1.0)])
        assert result == pytest.approx(50.0)

    def test_different_weights(self):
        # (1.0 * 3 + 0.0 * 1) / 4 * 100 = 75
        result = weighted_sum([(1.0, 3.0), (0.0, 1.0)])
        assert result == pytest.approx(75.0)

    def test_empty_returns_zero(self):
        assert weighted_sum([]) == 0.0

    def test_zero_weights_returns_zero(self):
        assert weighted_sum([(0.5, 0.0), (0.8, 0.0)]) == 0.0

    def test_result_in_0_100_range(self):
        result = weighted_sum([(0.7, 0.2), (0.3, 0.8)])
        assert 0 <= result <= 100

    def test_typical_analyzer_pattern(self):
        # Simulates a typical analyzer with 5 dimensions
        result = weighted_sum([
            (0.8, 0.25),   # strong
            (0.6, 0.25),   # decent
            (0.4, 0.20),   # below average
            (0.9, 0.15),   # great
            (0.5, 0.15),   # average
        ])
        # Expected: (0.8*0.25 + 0.6*0.25 + 0.4*0.20 + 0.9*0.15 + 0.5*0.15) / 1.0 * 100
        expected = (0.2 + 0.15 + 0.08 + 0.135 + 0.075) * 100
        assert result == pytest.approx(expected)
