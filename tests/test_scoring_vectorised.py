"""The vectorised timeline must match the per-point implementation exactly.

Ranking was quadratic in VOD length because the nearest-feature lookup rebuilt
its time array on every call and the transcript lookup scanned every segment.
Measured: 0.27s for a 12 minute VOD, 7.2s for an hour, extrapolating to roughly
22 minutes for the 13.5 hour VOD this was found on. The rewrite is only safe if
the numbers are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from highlightminer.config import Settings
from highlightminer.scoring import (
    TimelineSignal,
    _nearest_feature,
    _transcript_at,
    build_timeline,
)
from highlightminer.util import clamp


def _reference_timeline(duration, audio_features, transcript, chat_features, settings,
                        *, transcript_available=True):
    """The original per-point implementation, kept as the oracle."""
    step = max(0.5, min(1.0, settings.audio_hop_sec))
    weights = settings.normalized_weights(
        bool(chat_features), transcript_available=transcript_available
    )
    timeline = []
    t = 0.0
    while t <= duration:
        a = _nearest_feature(audio_features, t)
        tx = _transcript_at(transcript, t) if transcript_available else 0.0
        ch = _nearest_feature(chat_features, t) if chat_features else 0.0
        combined = (
            weights.get("audio", 0) * a
            + weights.get("transcript", 0) * tx
            + weights.get("chat", 0) * ch
        )
        active_values = [a]
        if transcript_available:
            active_values.append(tx)
        if chat_features:
            active_values.append(ch)
        if sum(v >= 0.68 for v in active_values) >= 2:
            combined += 0.10
        timeline.append(TimelineSignal(t, a, tx, ch, clamp(combined)))
        t += step
    return timeline


def _audio(duration, hop=0.5, seed=0):
    rng = np.random.default_rng(seed)
    count = int(duration / hop) + 2
    return [
        {"time": round(i * hop, 3), "dbfs": -20.0, "energy": 0.1, "onset": 0.0,
         "score": float(rng.random())}
        for i in range(count)
    ]


def _transcript(duration, seed=1):
    rng = np.random.default_rng(seed)
    rows, t = [], 0.0
    while t < duration:
        span = float(rng.uniform(1.0, 6.0))
        rows.append({
            "start": round(t, 3), "end": round(t + span, 3),
            "text": "words", "score": float(rng.random()), "reasons": [],
        })
        t += span + float(rng.uniform(0.0, 4.0))
    return rows


def _chat(duration, seed=2):
    rng = np.random.default_rng(seed)
    count = int(duration) + 1
    return [
        {"time": float(i), "count": 3, "ratio": 2.0, "score": float(rng.random())}
        for i in range(count)
    ]


def _assert_same(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.time == pytest.approx(b.time)
        assert a.audio == pytest.approx(b.audio)
        assert a.transcript == pytest.approx(b.transcript)
        assert a.chat == pytest.approx(b.chat)
        assert a.combined == pytest.approx(b.combined)


@pytest.mark.parametrize("duration", [0.0, 0.4, 1.0, 7.5, 60.0, 613.0])
def test_matches_the_per_point_implementation(duration):
    settings = Settings()
    audio, transcript, chat = _audio(duration), _transcript(duration), _chat(duration)
    _assert_same(
        build_timeline(duration, audio, transcript, chat, settings),
        _reference_timeline(duration, audio, transcript, chat, settings),
    )


def test_matches_without_chat():
    duration, settings = 300.0, Settings()
    audio, transcript = _audio(duration), _transcript(duration)
    _assert_same(
        build_timeline(duration, audio, transcript, [], settings),
        _reference_timeline(duration, audio, transcript, [], settings),
    )


def test_matches_without_transcript():
    duration, settings = 300.0, Settings()
    audio, chat = _audio(duration), _chat(duration)
    _assert_same(
        build_timeline(duration, audio, [], chat, settings, transcript_available=False),
        _reference_timeline(duration, audio, [], chat, settings, transcript_available=False),
    )


def test_matches_with_no_signals():
    settings = Settings()
    _assert_same(
        build_timeline(120.0, [], [], [], settings),
        _reference_timeline(120.0, [], [], [], settings),
    )


@pytest.mark.parametrize("hop", [0.5, 0.75, 1.0])
def test_grid_matches_for_each_hop(hop):
    duration = 97.0
    settings = Settings(audio_hop_sec=hop)
    audio, transcript = _audio(duration, hop), _transcript(duration)
    _assert_same(
        build_timeline(duration, audio, transcript, [], settings),
        _reference_timeline(duration, audio, transcript, [], settings),
    )


def test_nearest_feature_tie_breaks_to_the_later_feature():
    """min() over [idx, idx-1] kept the right-hand candidate on a tie."""
    features = [{"time": 0.0, "score": 0.1}, {"time": 1.0, "score": 0.9}]
    assert _nearest_feature(features, 0.5) == pytest.approx(0.9)


def test_nearest_feature_before_the_first_and_after_the_last():
    features = [{"time": 10.0, "score": 0.3}, {"time": 20.0, "score": 0.7}]
    assert _nearest_feature(features, 0.0) == pytest.approx(0.3)
    assert _nearest_feature(features, 999.0) == pytest.approx(0.7)


def test_transcript_at_uses_the_half_second_slack():
    segments = [{"start": 10.0, "end": 12.0, "score": 0.8}]
    assert _transcript_at(segments, 9.6) == pytest.approx(0.8)
    assert _transcript_at(segments, 12.4) == pytest.approx(0.8)
    assert _transcript_at(segments, 9.4) == 0.0


def test_transcript_at_takes_the_maximum_of_overlaps():
    segments = [
        {"start": 0.0, "end": 10.0, "score": 0.3},
        {"start": 5.0, "end": 15.0, "score": 0.9},
    ]
    assert _transcript_at(segments, 7.0) == pytest.approx(0.9)


def test_malformed_segments_are_skipped():
    segments = [{"start": "x"}, {"start": 1.0, "end": 3.0, "score": 0.5}]
    assert _transcript_at(segments, 2.0) == pytest.approx(0.5)


def test_scales_linearly_not_quadratically():
    """Six times the duration must cost roughly six times the work, not thirty six.

    Timed rather than counted, so it takes the best of several runs. A single
    wall-clock sample is meaningless on a machine that may be transcribing at
    the same time, which is exactly how this test first flaked.
    """
    import time

    settings = Settings()

    def best_of(duration, repeats=5):
        audio = _audio(duration)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            build_timeline(duration, audio, [], [], settings)
            samples.append(time.perf_counter() - started)
        return min(samples)

    short = best_of(3600.0)
    long = best_of(6 * 3600.0)

    # Linear predicts about 6x, quadratic about 36x. 12x separates them with
    # room for fixed overhead and a noisy machine.
    assert long < short * 12, (
        f"{short * 1000:.1f}ms for 1h vs {long * 1000:.1f}ms for 6h "
        "suggests a return to superlinear scaling"
    )
