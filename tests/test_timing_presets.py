from __future__ import annotations

import pytest

from highlightminer.config import Settings
from highlightminer.settings_presets import (
    TIMING_PRESETS,
    apply_timing_preset,
    detect_timing_preset,
)


def test_product_defaults_are_the_review_preset():
    assert detect_timing_preset(Settings()) == "Review"


def test_short_form_preset_round_trips():
    shaped = apply_timing_preset(Settings(), "Short-form")
    assert detect_timing_preset(shaped) == "Short-form"


def test_short_form_enables_shaping_and_shortens_clips():
    default = Settings()
    shaped = apply_timing_preset(default, "Short-form")
    assert shaped.short_form_mode is True
    assert shaped.max_candidate_sec < default.max_candidate_sec
    assert shaped.pre_roll_sec < default.pre_roll_sec


@pytest.mark.parametrize("name", sorted(TIMING_PRESETS))
def test_presets_only_touch_timing(name):
    """Weights, Whisper options and phrases are chosen independently."""
    default = Settings()
    shaped = apply_timing_preset(default, name)
    assert shaped.weights == default.weights
    assert shaped.whisper_model == default.whisper_model
    assert shaped.reaction_phrases == default.reaction_phrases
    assert shaped.audio_hop_sec == default.audio_hop_sec


@pytest.mark.parametrize("name", sorted(TIMING_PRESETS))
def test_every_preset_produces_valid_settings(name):
    # Settings.validate runs in __post_init__, so this would raise if a preset
    # ever set an out-of-range or self-contradictory combination.
    shaped = apply_timing_preset(Settings(), name)
    assert shaped.min_candidate_sec <= shaped.max_candidate_sec


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown timing preset"):
        apply_timing_preset(Settings(), "Cinematic")


def test_manual_edits_read_as_custom():
    shaped = apply_timing_preset(Settings(), "Short-form")
    tweaked = apply_timing_preset(Settings(), "Short-form").__class__(
        **{**shaped.__dict__, "max_candidate_sec": 33.0}
    )
    assert detect_timing_preset(tweaked) == "Custom"


def test_applying_a_preset_does_not_mutate_the_original():
    default = Settings()
    apply_timing_preset(default, "Short-form")
    assert default.short_form_mode is False
