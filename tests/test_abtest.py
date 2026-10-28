from __future__ import annotations

import pytest

from abtest.stats import (
    Variant,
    required_sample_size,
    sequential_warning,
    two_proportion_z_test,
    z_critical,
)


def test_clear_winner_is_significant():
    result = two_proportion_z_test(Variant("control", 10_000, 500), Variant("treatment", 10_000, 650))
    assert result.significant and result.absolute_lift > 0
    assert result.recommendation == "ship"
    assert result.p_value < 0.001


def test_clear_loser_recommends_stopping():
    result = two_proportion_z_test(Variant("control", 10_000, 650), Variant("treatment", 10_000, 500))
    assert result.significant and result.absolute_lift < 0
    assert "worse" in result.recommendation


def test_small_difference_is_inconclusive():
    result = two_proportion_z_test(Variant("control", 1000, 100), Variant("treatment", 1000, 105))
    assert not result.significant
    assert result.ci_low < 0 < result.ci_high
    assert "inconclusive" in result.recommendation


def test_identical_variants_have_zero_effect():
    result = two_proportion_z_test(Variant("a", 500, 50), Variant("b", 500, 50))
    assert result.absolute_lift == 0
    assert result.p_value == pytest.approx(1.0)


def test_relative_lift():
    result = two_proportion_z_test(Variant("a", 1000, 100), Variant("b", 1000, 120))
    assert result.relative_lift == pytest.approx(0.2)


def test_confidence_interval_contains_the_lift():
    result = two_proportion_z_test(Variant("a", 5000, 400), Variant("b", 5000, 460))
    assert result.ci_low < result.absolute_lift < result.ci_high


def test_invalid_inputs():
    with pytest.raises(ValueError, match="visitor"):
        two_proportion_z_test(Variant("a", 0, 0), Variant("b", 10, 1))
    with pytest.raises(ValueError, match="exceed"):
        two_proportion_z_test(Variant("a", 10, 20), Variant("b", 10, 1))


def test_required_sample_size_shrinks_with_larger_effects():
    small = required_sample_size(0.05, 0.10)
    large = required_sample_size(0.05, 0.50)
    assert small > large > 100


def test_required_sample_size_is_plausible():
    # 5% baseline, 20% relative lift: the textbook answer is roughly 4-5k per arm
    assert 7000 < required_sample_size(0.05, 0.20) < 9000


def test_sample_size_validation():
    with pytest.raises(ValueError):
        required_sample_size(1.5, 0.1)
    with pytest.raises(ValueError):
        required_sample_size(0.1, 0)


def test_z_critical_table():
    assert z_critical(0.95) == pytest.approx(1.96, abs=0.01)
    with pytest.raises(ValueError):
        z_critical(0.93)


def test_peeking_warning_mentions_inflation():
    assert "false positive" in sequential_warning(5)
