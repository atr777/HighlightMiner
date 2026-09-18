from __future__ import annotations

WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "Balanced": {"audio": 0.34, "transcript": 0.42, "chat": 0.24},
    "Reaction-heavy": {"audio": 0.20, "transcript": 0.60, "chat": 0.20},
    "Chat-heavy": {"audio": 0.20, "transcript": 0.25, "chat": 0.55},
    "Audio-heavy": {"audio": 0.60, "transcript": 0.25, "chat": 0.15},
}


def normalize_weights(weights: dict[str, float], *, chat_available: bool = True) -> dict[str, float]:
    values = {
        "audio": max(0.0, float(weights.get("audio", 0.0))),
        "transcript": max(0.0, float(weights.get("transcript", 0.0))),
        "chat": max(0.0, float(weights.get("chat", 0.0))) if chat_available else 0.0,
    }
    total = sum(values.values()) or 1.0
    return {key: value / total for key, value in values.items()}


def detect_weight_preset(weights: dict[str, float], *, tolerance: float = 0.005) -> str:
    normalized = normalize_weights(weights)
    for name, preset in WEIGHT_PRESETS.items():
        expected = normalize_weights(preset)
        if all(abs(normalized[key] - expected[key]) <= tolerance for key in expected):
            return name
    return "Custom"


# Timing presets are separate from weight presets on purpose: weights say which
# signals matter, timing says what shape of clip to cut. They are chosen
# independently.
TIMING_PRESETS: dict[str, dict[str, float | bool]] = {
    "Review": {
        "short_form_mode": False,
        "pre_roll_sec": 18.0,
        "post_roll_sec": 14.0,
        "max_candidate_sec": 75.0,
        "min_candidate_sec": 8.0,
        "hook_lead_sec": 3.0,
        "min_candidate_score": 0.38,
        "max_candidates": 40,
    },
    "Short-form": {
        # Measured against the review preset on a real VOD: median hook offset
        # drops from 18.0s to 3.0s and median duration from 32s to 12s.
        "short_form_mode": True,
        "pre_roll_sec": 5.0,
        "post_roll_sec": 8.0,
        "max_candidate_sec": 45.0,
        "min_candidate_sec": 10.0,
        "hook_lead_sec": 3.0,
        "min_candidate_score": 0.42,
        "max_candidates": 20,
    },
}


def apply_timing_preset(settings, name: str):
    """Return a copy of ``settings`` with one timing preset applied.

    Only timing fields are touched. Whisper options, weights and reaction
    phrases are deliberately left alone, matching how weight presets behave.
    """
    from dataclasses import replace

    try:
        preset = TIMING_PRESETS[name]
    except KeyError:
        raise ValueError(f"Unknown timing preset: {name!r}") from None
    return replace(settings, **preset)


def detect_timing_preset(settings) -> str:
    for name, preset in TIMING_PRESETS.items():
        if all(getattr(settings, key) == value for key, value in preset.items()):
            return name
    return "Custom"
