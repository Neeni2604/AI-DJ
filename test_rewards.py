from __future__ import annotations

import pytest
from dj_env import harmonic_compatibility, bpm_smoothness, transition_reward


# ---------------------------------------------------------------------------
# bpm_smoothness
# ---------------------------------------------------------------------------

class TestBpmSmoothness:

    def test_identical_bpm_returns_one(self):
        assert bpm_smoothness(120.0, 120.0) == pytest.approx(1.0)

    def test_small_delta_returns_one(self):
        # delta of 3 BPM is within the <=5 threshold
        assert bpm_smoothness(120.0, 123.0) == pytest.approx(1.0)

    def test_exact_threshold_returns_one(self):
        # delta of exactly 5 BPM should still return 1.0
        assert bpm_smoothness(120.0, 125.0) == pytest.approx(1.0)

    def test_large_delta_returns_zero(self):
        # delta of 30+ BPM should return 0.0
        assert bpm_smoothness(120.0, 150.0) == pytest.approx(0.0)

    def test_very_large_delta_clamped_to_zero(self):
        # delta well beyond 30 BPM should still clamp to 0.0, not go negative
        assert bpm_smoothness(80.0, 200.0) == pytest.approx(0.0)

    def test_mid_range_delta(self):
        # delta of 17.5 BPM is halfway between 5 and 30 => score ~0.5
        score = bpm_smoothness(120.0, 137.5)
        assert 0.45 < score < 0.55

    def test_symmetry(self):
        # order should not matter
        assert bpm_smoothness(100.0, 130.0) == pytest.approx(bpm_smoothness(130.0, 100.0))


# ---------------------------------------------------------------------------
# harmonic_compatibility
# ---------------------------------------------------------------------------

class TestHarmonicCompatibility:

    def test_same_key_and_mode_is_compatible(self):
        # C major -> C major
        assert harmonic_compatibility(0, 1, 0, 1) == 1.0

    def test_adjacent_camelot_major_keys_compatible(self):
        # C major (8B) and G major (9B) are adjacent on the Camelot wheel
        assert harmonic_compatibility(0, 1, 7, 1) == 1.0

    def test_relative_minor_is_compatible(self):
        # C major (8B) and A minor (8A) share the same Camelot number
        assert harmonic_compatibility(0, 1, 9, 0) == 1.0

    def test_incompatible_keys_return_zero(self):
        # C major and F# major are on opposite sides of the wheel
        assert harmonic_compatibility(0, 1, 6, 1) == 0.0

    def test_tritone_is_incompatible(self):
        # C major and Gb major — maximum harmonic distance
        assert harmonic_compatibility(0, 1, 6, 1) == 0.0

    def test_return_type_is_float(self):
        result = harmonic_compatibility(0, 1, 0, 1)
        assert isinstance(result, float)

    def test_compatible_minor_keys(self):
        # A minor (8A) and E minor (9A) are adjacent
        assert harmonic_compatibility(9, 0, 4, 0) == 1.0


# ---------------------------------------------------------------------------
# transition_reward
# ---------------------------------------------------------------------------

class TestTransitionReward:

    # transition indices: 0=cut, 1=fade, 2=beatmatch

    def test_beatmatch_rewarded_for_small_bpm_delta(self):
        # beatmatch with delta=2 BPM should score close to 1.0
        score = transition_reward(2, bpm_delta=2.0, energy_delta=0.0)
        assert score > 0.8

    def test_beatmatch_penalised_for_large_bpm_delta(self):
        # beatmatch with delta=25 BPM should score close to 0.0
        score = transition_reward(2, bpm_delta=25.0, energy_delta=0.0)
        assert score < 0.2

    def test_cut_rewarded_for_large_energy_shift(self):
        # hard cut with a big energy jump should score high
        score = transition_reward(0, bpm_delta=0.0, energy_delta=0.6)
        assert score > 0.8

    def test_cut_penalised_for_small_energy_shift(self):
        # hard cut with almost no energy change should score low
        score = transition_reward(0, bpm_delta=0.0, energy_delta=0.05)
        assert score < 0.2

    def test_fade_rewarded_for_moderate_energy_shift(self):
        # fade with energy delta around 0.2 should score near 1.0
        score = transition_reward(1, bpm_delta=0.0, energy_delta=0.2)
        assert score > 0.8

    def test_fade_penalised_for_large_energy_shift(self):
        # fade with a huge energy shift should score low
        score = transition_reward(1, bpm_delta=0.0, energy_delta=0.9)
        assert score < 0.2

    def test_all_scores_bounded(self):
        # all transition scores must stay within [0, 1]
        for transition in [0, 1, 2]:
            for bpm_delta in [0.0, 10.0, 30.0]:
                for energy_delta in [-0.5, 0.0, 0.5, 1.0]:
                    score = transition_reward(transition, bpm_delta, energy_delta)
                    assert 0.0 <= score <= 1.0, (
                        f"Out of bounds: transition={transition}, "
                        f"bpm_delta={bpm_delta}, energy_delta={energy_delta}, score={score}"
                    )