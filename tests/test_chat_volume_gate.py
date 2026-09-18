from __future__ import annotations

import pytest

from highlightminer.chat import analyze_chat, messages_per_minute, volume_confidence
from highlightminer.config import Settings


def _messages(times: list[float]) -> list[dict]:
    return [{"time": t, "text": "x"} for t in times]


def _busy_chat(duration: float, per_second: int) -> list[dict]:
    return _messages([float(s) for s in range(int(duration)) for _ in range(per_second)])


def test_messages_per_minute():
    assert messages_per_minute(_messages([1.0] * 120), 60.0) == pytest.approx(120.0)
    assert messages_per_minute([], 60.0) == 0.0
    assert messages_per_minute(_messages([1.0]), 0.0) == 0.0


@pytest.mark.parametrize(
    "rate,expected",
    [
        (0.0, 0.0),
        (15.0, 0.0),      # at the quiet threshold, still untrusted
        (37.5, 0.5),      # halfway up the ramp
        (60.0, 1.0),
        (500.0, 1.0),
    ],
)
def test_volume_confidence_ramp(rate, expected):
    assert volume_confidence(rate, 15.0, 60.0) == pytest.approx(expected)


def test_volume_confidence_handles_degenerate_range():
    # quiet == active must not divide by zero
    assert volume_confidence(20.0, 15.0, 15.0) == 1.0


def test_quiet_chat_is_gated_off_entirely():
    """The measured failure case: ~14 msgs/min where one message scored 1.0."""
    # 173 messages over 723s is 14.4/min, below the 15.0 quiet threshold.
    records = _messages([float(i * 4) for i in range(173)])
    assert messages_per_minute(records, 723.0) < 15.0
    assert analyze_chat(records, 723.0) == []


def test_gated_chat_reads_as_unavailable_to_scoring():
    """An empty list is how the rest of the pipeline spells 'no chat'."""
    records = _messages([float(i * 4) for i in range(173)])
    features = analyze_chat(records, 723.0)
    # scoring.py keys every chat decision off bool(chat_features)
    assert not features
    weights = Settings().normalized_weights(chat_available=bool(features))
    assert weights["chat"] == 0.0
    assert weights["audio"] + weights["transcript"] == pytest.approx(1.0)


def test_uniform_busy_chat_has_no_bursts():
    """A flat chat is busy but eventless. Nothing should score."""
    records = _busy_chat(600.0, 2)  # steady 120 msgs/min, no spikes at all
    features = analyze_chat(records, 600.0)
    assert features, "busy chat must not be gated off"
    assert max(f["score"] for f in features) == 0.0


def test_busy_chat_with_a_real_burst_scores():
    records = _busy_chat(600.0, 2)
    records += _messages([300.0] * 30)
    features = analyze_chat(records, 600.0)
    spike = next(f for f in features if abs(f["time"] - 300.5) < 1.0)
    assert spike["score"] > 0.8
    # and the quiet stretches around it stay at zero
    calm = next(f for f in features if abs(f["time"] - 100.5) < 1.0)
    assert calm["score"] == 0.0


def test_absolute_burst_floor_damps_tiny_spikes():
    """A 2-message second against a near-zero baseline must not score 1.0."""
    # Busy enough to clear the volume gate, but the spike itself is tiny.
    records = _busy_chat(600.0, 1)          # steady 60/min baseline
    records += _messages([300.0, 300.0])    # a 2-message bump above it
    features = analyze_chat(records, 600.0, min_burst_messages=10.0)
    spike = next(f for f in features if abs(f["time"] - 300.5) < 1.0)
    assert spike["score"] < 0.5


def test_large_burst_clears_the_floor():
    records = _busy_chat(600.0, 1)
    records += _messages([300.0] * 40)
    features = analyze_chat(records, 600.0, min_burst_messages=10.0)
    spike = next(f for f in features if abs(f["time"] - 300.5) < 1.0)
    assert spike["score"] > 0.8


def test_no_records_returns_empty():
    assert analyze_chat([], 600.0) == []


def test_confidence_scales_scores_between_thresholds():
    """Mid-ramp chats contribute, but at reduced strength."""
    records = _busy_chat(600.0, 1)
    records += _messages([300.0] * 40)
    full = analyze_chat(records, 600.0, quiet_msgs_per_min=0.0, active_msgs_per_min=1.0)
    partial = analyze_chat(records, 600.0, quiet_msgs_per_min=0.0, active_msgs_per_min=120.0)
    peak_full = max(f["score"] for f in full)
    peak_partial = max(f["score"] for f in partial)
    assert 0.0 < peak_partial < peak_full


def test_settings_reject_inverted_thresholds():
    with pytest.raises(ValueError, match="chat_active_msgs_per_min"):
        Settings(chat_quiet_msgs_per_min=100.0, chat_active_msgs_per_min=50.0)


def test_settings_defaults_present():
    s = Settings()
    assert s.chat_min_burst_messages == 3.0
    assert s.chat_quiet_msgs_per_min == 15.0
    assert s.chat_active_msgs_per_min == 60.0
