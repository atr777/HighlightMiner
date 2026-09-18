"""The Settings page rebuilds Settings from editor state alone.

Any field it forgets is silently reset to its default on the next save, which
is exactly the kind of bug that only shows up weeks later as "my captions
turned themselves off". These tests pin the round trip.
"""

from __future__ import annotations

import dataclasses

import pytest

from highlightminer.config import Settings
from highlightminer.settings_store import load_app_settings, save_app_settings

SHORT_FORM_FIELDS = (
    "short_form_mode",
    "hook_lead_sec",
    "min_candidate_sec",
    "speech_snap_sec",
    "render_layout",
    "burn_captions",
    "caption_font_size",
    "caption_uppercase",
    "webcam_fraction",
    "webcam_rect",
    "audio_only_penalty",
    "duplicate_containment",
    "cpu_threads",
    "chat_quiet_msgs_per_min",
    "chat_active_msgs_per_min",
    "chat_min_burst_messages",
)


def _editor_module():
    import highlightminer.ui_settings as ui_settings

    return ui_settings


def test_every_short_form_field_exists_on_settings():
    names = {f.name for f in dataclasses.fields(Settings)}
    missing = [name for name in SHORT_FORM_FIELDS if name not in names]
    assert not missing, f"Settings is missing {missing}"


def test_editor_seeds_and_rebuilds_every_short_form_field(monkeypatch):
    """Seed the editor from a non-default profile, rebuild, and compare."""
    ui_settings = _editor_module()

    state: dict = {}
    monkeypatch.setattr(ui_settings.st, "session_state", state)

    original = Settings(
        short_form_mode=True,
        hook_lead_sec=2.5,
        min_candidate_sec=12.0,
        speech_snap_sec=1.25,
        render_layout="webcam",
        burn_captions=True,
        caption_font_size=120,
        caption_uppercase=True,
        webcam_fraction=0.42,
        webcam_rect={"x": 0.7, "y": 0.0, "w": 0.3, "h": 0.25},
        audio_only_penalty=0.6,
        duplicate_containment=0.55,
        cpu_threads=11,
        chat_quiet_msgs_per_min=25.0,
        chat_active_msgs_per_min=90.0,
        chat_min_burst_messages=4.0,
    )

    ui_settings._seed_editor(original, force=True)
    rebuilt = ui_settings._build_settings()

    for name in SHORT_FORM_FIELDS:
        assert getattr(rebuilt, name) == getattr(original, name), f"{name} was not preserved"


def test_webcam_rect_is_dropped_when_disabled(monkeypatch):
    ui_settings = _editor_module()
    state: dict = {}
    monkeypatch.setattr(ui_settings.st, "session_state", state)

    ui_settings._seed_editor(Settings(render_layout="crop"), force=True)
    assert ui_settings._editor_webcam_rect() is None
    assert ui_settings._build_settings().webcam_rect is None


def test_timing_preset_applies_to_editor_fields(monkeypatch):
    ui_settings = _editor_module()
    state: dict = {}
    monkeypatch.setattr(ui_settings.st, "session_state", state)

    ui_settings._seed_editor(Settings(), force=True)
    state[ui_settings._EDITOR_KEYS["timing_preset"]] = "Short-form"
    ui_settings._apply_timing_preset_to_editor()

    rebuilt = ui_settings._build_settings()
    assert rebuilt.short_form_mode is True
    assert rebuilt.max_candidate_sec == 45.0
    assert rebuilt.audio_only_penalty == 0.6


def test_unknown_timing_preset_is_a_no_op(monkeypatch):
    ui_settings = _editor_module()
    state: dict = {}
    monkeypatch.setattr(ui_settings.st, "session_state", state)
    ui_settings._seed_editor(Settings(), force=True)
    state[ui_settings._EDITOR_KEYS["timing_preset"]] = "Nonexistent"
    ui_settings._apply_timing_preset_to_editor()
    assert ui_settings._build_settings().short_form_mode is False


def test_short_form_settings_survive_a_database_round_trip(tmp_path):
    db = tmp_path / "highlightminer.db"
    original = Settings(
        short_form_mode=True,
        render_layout="crop",
        burn_captions=True,
        caption_font_size=110,
        audio_only_penalty=0.6,
        cpu_threads=11,
    )
    save_app_settings(original, db)
    loaded = load_app_settings(db)
    for name in SHORT_FORM_FIELDS:
        assert getattr(loaded, name) == getattr(original, name), f"{name} lost in storage"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"render_layout": "hologram"}, "render_layout"),
        ({"render_layout": "webcam"}, "webcam_rect"),
        ({"caption_font_size": 4}, "caption_font_size"),
        ({"webcam_fraction": 0.99}, "webcam_fraction"),
        ({"cpu_threads": -1}, "cpu_threads"),
    ],
)
def test_invalid_render_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Settings(**kwargs)
