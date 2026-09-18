from __future__ import annotations

import pytest

from highlightminer.shortform import (
    align_hook,
    self_containment,
    shape_candidate,
    snap_to_speech,
)


def _segments(*spans: tuple[float, float, str]) -> list[dict]:
    return [{"start": s, "end": e, "text": t} for s, e, t in spans]


class TestAlignHook:
    def test_moves_a_deep_payoff_to_the_front(self):
        """The default 18s pre-roll puts the payoff 18s in, which kills a Short."""
        window = align_hook(0.0, 44.0, peak_time=18.0, lead_sec=3.0)
        assert window.start == 15.0
        assert window.duration <= 45.0

    def test_leaves_an_already_early_peak_alone(self):
        window = align_hook(10.0, 40.0, peak_time=11.0, lead_sec=3.0)
        assert window.start == 10.0

    def test_never_starts_before_the_original_window(self):
        """Trimming only. Shaping must not pull in unselected material."""
        window = align_hook(10.0, 40.0, peak_time=10.5, lead_sec=30.0)
        assert window.start >= 10.0

    def test_enforces_the_duration_ceiling(self):
        window = align_hook(0.0, 300.0, peak_time=10.0, max_duration_sec=45.0)
        assert window.duration == pytest.approx(45.0)

    def test_respects_a_minimum_length(self):
        window = align_hook(0.0, 40.0, peak_time=30.0, lead_sec=3.0, min_duration_sec=8.0)
        assert window.duration >= 8.0

    def test_minimum_never_exceeds_the_original_end(self):
        window = align_hook(0.0, 5.0, peak_time=4.0, min_duration_sec=30.0)
        assert window.end <= 5.0

    def test_peak_outside_the_window_is_clamped(self):
        window = align_hook(10.0, 20.0, peak_time=999.0)
        assert 10.0 <= window.start <= 20.0

    def test_degenerate_window(self):
        window = align_hook(5.0, 5.0, peak_time=5.0)
        assert window.start == 5.0


class TestSnapToSpeech:
    def test_snaps_a_start_inside_an_utterance_outward(self):
        segments = _segments((10.0, 14.0, "a full sentence"))
        window = snap_to_speech(11.0, 20.0, segments, tolerance_sec=1.5)
        assert window.start == 10.0

    def test_leaves_a_start_alone_beyond_tolerance(self):
        segments = _segments((10.0, 14.0, "a full sentence"))
        window = snap_to_speech(13.5, 20.0, segments, tolerance_sec=1.0)
        assert window.start == 13.5

    def test_snaps_an_end_inside_an_utterance_outward(self):
        segments = _segments((10.0, 14.0, "x"), (18.0, 22.0, "y"))
        window = snap_to_speech(10.0, 21.0, segments, tolerance_sec=1.5)
        assert window.end == 22.0

    def test_no_segments_is_a_no_op(self):
        window = snap_to_speech(3.0, 9.0, [], tolerance_sec=1.5)
        assert (window.start, window.end) == (3.0, 9.0)

    def test_malformed_segments_are_ignored(self):
        segments = [{"start": "oops"}, {"end": 5.0}, {"start": 1.0, "end": 4.0, "text": "ok"}]
        window = snap_to_speech(2.0, 9.0, segments, tolerance_sec=1.5)
        assert window.start == 1.0

    def test_never_inverts_the_window(self):
        segments = _segments((0.0, 100.0, "one long segment"))
        window = snap_to_speech(50.0, 51.0, segments, tolerance_sec=100.0)
        assert window.end > window.start


class TestSelfContainment:
    def test_clean_start_and_finished_sentence_scores_high(self):
        segments = _segments((10.0, 14.0, "Something complete happens here."))
        assert self_containment(segments, 10.0, 14.0) > 0.9

    def test_starting_mid_utterance_is_penalised(self):
        segments = _segments((10.0, 20.0, "a long rambling thought that runs on"))
        mid = self_containment(segments, 15.0, 20.0)
        clean = self_containment(segments, 10.0, 20.0)
        assert mid < clean

    def test_dangling_opener_is_penalised(self):
        dangling = _segments((10.0, 14.0, "And then he did it."))
        standalone = _segments((10.0, 14.0, "Watch this happen."))
        assert self_containment(dangling, 10.0, 14.0) < self_containment(standalone, 10.0, 14.0)

    def test_unfinished_ending_is_penalised(self):
        finished = _segments((10.0, 14.0, "That is the whole thing."))
        unfinished = _segments((10.0, 14.0, "That is the whole thing"))
        assert self_containment(unfinished, 10.0, 14.0) < self_containment(finished, 10.0, 14.0)

    def test_no_transcript_is_neutral(self):
        assert self_containment([], 0.0, 10.0) == 0.5

    def test_always_within_range(self):
        segments = _segments((0.0, 100.0, "and he was like"))
        assert 0.0 <= self_containment(segments, 50.0, 60.0) <= 1.0


class TestShapeCandidate:
    def test_combines_hook_alignment_and_snapping(self):
        segments = _segments((14.0, 19.0, "the actual payoff line"), (20.0, 25.0, "aftermath"))
        window = shape_candidate(0.0, 44.0, peak_time=18.0, segments=segments, lead_sec=3.0)
        # aligned to 15.0, then snapped outward to the utterance start at 14.0
        assert window.start == 14.0

    def test_respects_the_duration_ceiling_after_snapping(self):
        """Outward snapping must not push the clip past the hard ceiling."""
        segments = _segments((0.0, 500.0, "one enormous segment"))
        window = shape_candidate(
            10.0, 400.0, peak_time=12.0, segments=segments,
            max_duration_sec=45.0, snap_tolerance_sec=100.0,
        )
        assert window.duration <= 45.0

    def test_never_starts_below_zero(self):
        segments = _segments((0.0, 10.0, "right at the start"))
        window = shape_candidate(0.0, 30.0, peak_time=1.0, segments=segments)
        assert window.start >= 0.0

    def test_clamps_to_source_duration(self):
        segments = _segments((0.0, 100.0, "x"))
        window = shape_candidate(
            10.0, 90.0, peak_time=12.0, segments=segments, source_duration=20.0
        )
        assert window.end <= 20.0

    def test_no_transcript_still_aligns_the_hook(self):
        window = shape_candidate(0.0, 44.0, peak_time=18.0, segments=[], lead_sec=3.0)
        assert window.start == 15.0
