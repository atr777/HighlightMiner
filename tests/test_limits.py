"""Limit and scale tests.

A 13.5 hour VOD broke transcription with a 3.3 GB allocation that nothing had
bounded. These pin the behaviour of the rest of the pipeline at that scale, and
at the degenerate end, so the next limit is found here rather than four hours
into an overnight run.
"""

from __future__ import annotations

import json
import time

import pytest

from highlightminer.captions import Word, build_ass, group_phrases, words_in_window
from highlightminer.chat import analyze_chat, load_chat
from highlightminer.config import Settings
from highlightminer.render import Layout, Rect, build_filter
from highlightminer.scoring import build_timeline, deduplicate_candidates, find_candidates
from highlightminer.shortform import shape_candidate
from highlightminer.titles import suggest_title

# The VOD that prompted all of this.
LONG_VOD_SECONDS = 48826.0


def _audio_features(duration: float, hop: float = 0.5) -> list[dict]:
    count = int(duration / hop) + 1
    return [
        {
            "time": round(i * hop, 3),
            "dbfs": -20.0,
            "energy": 0.1,
            "onset": 0.0,
            # A spike every ~10 minutes, so candidates actually form.
            "score": 0.95 if i % 1200 == 0 else 0.05,
        }
        for i in range(count)
    ]


def _transcript(duration: float, every: float = 30.0) -> list[dict]:
    rows = []
    t = 0.0
    while t < duration:
        rows.append({
            "start": round(t, 3),
            "end": round(t + 4.0, 3),
            "text": "something happens here and it is worth hearing",
            "score": 0.8 if int(t) % 600 == 0 else 0.1,
            "reasons": [],
        })
        t += every
    return rows


class TestLongVodScale:
    def test_timeline_covers_a_thirteen_hour_vod(self):
        settings = Settings()
        started = time.perf_counter()
        timeline = build_timeline(
            LONG_VOD_SECONDS, _audio_features(LONG_VOD_SECONDS), [], [], settings
        )
        elapsed = time.perf_counter() - started
        # 0.5s grid across 13.5 hours
        assert len(timeline) == pytest.approx(LONG_VOD_SECONDS / 0.5, rel=0.01)
        assert elapsed < 120, f"timeline build took {elapsed:.0f}s"

    def test_candidates_from_a_thirteen_hour_vod(self):
        settings = Settings(max_candidates=40)
        started = time.perf_counter()
        candidates = find_candidates(
            LONG_VOD_SECONDS,
            _audio_features(LONG_VOD_SECONDS),
            _transcript(LONG_VOD_SECONDS),
            [],
            settings,
        )
        elapsed = time.perf_counter() - started
        assert candidates
        assert len(candidates) <= settings.max_candidates
        assert elapsed < 300, f"ranking took {elapsed:.0f}s"

    def test_every_candidate_stays_inside_the_source(self):
        candidates = find_candidates(
            LONG_VOD_SECONDS,
            _audio_features(LONG_VOD_SECONDS),
            _transcript(LONG_VOD_SECONDS),
            [],
            Settings(),
        )
        for candidate in candidates:
            assert 0.0 <= candidate["start"] < candidate["end"] <= LONG_VOD_SECONDS

    def test_a_full_day_of_audio_does_not_explode(self):
        """Twitch allows very long VODs; 24 hours must still rank."""
        duration = 24 * 3600.0
        candidates = find_candidates(
            duration, _audio_features(duration), [], [], Settings(max_candidates=10)
        )
        assert len(candidates) <= 10

    def test_chat_at_realistic_volume_for_a_long_stream(self):
        # 46 msgs/min over 13.5 hours is what the real VOD carries.
        records = [
            {"time": i * (60.0 / 46.0), "text": "PogChamp"}
            for i in range(int(LONG_VOD_SECONDS / 60 * 46))
        ]
        started = time.perf_counter()
        features = analyze_chat(records, LONG_VOD_SECONDS)
        elapsed = time.perf_counter() - started
        assert features
        assert elapsed < 60, f"chat analysis took {elapsed:.0f}s"


class TestDegenerateInputs:
    def test_zero_duration(self):
        assert find_candidates(0.0, [], [], [], Settings()) == []

    def test_negative_duration_is_not_fatal(self):
        assert find_candidates(-5.0, [], [], [], Settings()) == []

    def test_no_signals_at_all(self):
        assert find_candidates(600.0, [], [], [], Settings()) == []

    def test_audio_only_with_no_transcript_or_chat(self):
        candidates = find_candidates(
            600.0, _audio_features(600.0), [], [], Settings(), transcript_available=False
        )
        for candidate in candidates:
            assert candidate["transcript_score"] == 0.0

    def test_single_sample_of_audio(self):
        features = [{"time": 0.0, "dbfs": -10.0, "energy": 1.0, "onset": 1.0, "score": 1.0}]
        find_candidates(1.0, features, [], [], Settings())

    def test_clip_at_the_very_start_of_a_vod(self):
        window = shape_candidate(0.0, 30.0, peak_time=0.0, segments=[], source_duration=600.0)
        assert window.start >= 0.0

    def test_clip_at_the_very_end_of_a_vod(self):
        window = shape_candidate(
            590.0, 600.0, peak_time=599.0, segments=[], source_duration=600.0
        )
        assert window.end <= 600.0


class TestSettingsBoundaries:
    @pytest.mark.parametrize(
        "field,low,high",
        [
            ("pre_roll_sec", 0.0, 600.0),
            ("post_roll_sec", 0.0, 600.0),
            ("merge_gap_sec", 0.0, 600.0),
            ("min_candidate_score", 0.0, 1.0),
            ("hook_lead_sec", 0.0, 60.0),
            ("speech_snap_sec", 0.0, 30.0),
            ("audio_only_penalty", 0.0, 1.0),
        ],
    )
    def test_extremes_of_each_range_are_accepted(self, field, low, high):
        for value in (low, high):
            Settings(**{field: value})

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("pre_roll_sec", 601.0),
            ("min_candidate_score", 1.1),
            ("max_candidates", 0),
            ("max_candidates", 501),
            ("beam_size", 0),
            ("beam_size", 21),
            ("caption_font_size", 401),
            ("duplicate_containment", 0.05),
        ],
    )
    def test_just_past_each_range_is_rejected(self, field, bad):
        with pytest.raises(ValueError):
            Settings(**{field: bad})

    def test_max_candidates_ceiling_is_honoured(self):
        settings = Settings(max_candidates=500, min_candidate_score=0.01)
        candidates = find_candidates(
            3600.0, _audio_features(3600.0), _transcript(3600.0), [], settings
        )
        assert len(candidates) <= 500

    def test_every_weight_at_zero_but_one(self):
        settings = Settings(weights={"audio": 1.0, "transcript": 0.0, "chat": 0.0})
        weights = settings.normalized_weights(chat_available=False)
        assert weights["audio"] == pytest.approx(1.0)

    def test_all_weights_zero_is_rejected(self):
        with pytest.raises(ValueError, match="greater than zero"):
            Settings(weights={"audio": 0.0, "transcript": 0.0, "chat": 0.0})


class TestDeduplicationAtScale:
    def test_many_overlapping_candidates(self):
        candidates = [
            {
                "id": f"H{i}",
                "start": i * 0.5,
                "end": i * 0.5 + 40.0,
                "score": 0.5,
                "peak_time": i * 0.5 + 20.0,
                "features": {},
            }
            for i in range(2000)
        ]
        started = time.perf_counter()
        kept = deduplicate_candidates(candidates, max_candidates=40)
        elapsed = time.perf_counter() - started
        assert len(kept) <= 40
        assert elapsed < 60, f"dedupe took {elapsed:.0f}s"

    def test_zero_max_candidates_returns_nothing(self):
        assert deduplicate_candidates([{"start": 0, "end": 1, "score": 1}], max_candidates=0) == []


class TestCaptionLimits:
    def test_thousands_of_words(self):
        words = [Word(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(5000)]
        started = time.perf_counter()
        document = build_ass(words)
        elapsed = time.perf_counter() - started
        assert document.count("Dialogue:") > 1000
        assert elapsed < 30, f"caption build took {elapsed:.0f}s"

    def test_a_single_very_long_word(self):
        document = build_ass([Word("x" * 5000, 0.0, 1.0)])
        assert "Dialogue:" in document

    def test_zero_length_words_still_render(self):
        document = build_ass([Word("a", 1.0, 1.0), Word("b", 1.0, 1.0)])
        assert "Dialogue:" in document

    def test_unicode_and_emoji_survive(self):
        document = build_ass([Word("日本語", 0.0, 0.5), Word("🔥", 0.5, 1.0)])
        assert "日本語" in document and "🔥" in document

    def test_words_far_outside_the_window_are_dropped(self):
        segments = [{
            "start": 0.0, "end": 100000.0, "text": "x",
            "words": [{"w": "far", "s": 99999.0, "e": 100000.0}],
        }]
        assert words_in_window(segments, 0.0, 10.0) == []

    def test_phrases_never_exceed_the_word_cap(self):
        words = [Word(f"w{i}", i * 0.05, i * 0.05 + 0.04) for i in range(500)]
        assert all(len(p.words) <= 4 for p in group_phrases(words))


class TestRenderLimits:
    @pytest.mark.parametrize("x", [0.0, 0.5, 0.683594])
    def test_extreme_but_valid_crop_positions(self, x):
        chain = build_filter(Layout(kind="crop", gameplay=Rect(x, 0.0, 0.316406, 1.0)))
        assert chain.startswith("crop=")

    def test_a_one_pixel_region_is_rejected_by_rect(self):
        with pytest.raises(ValueError):
            Rect(0.0, 0.0, 0.0, 1.0)

    def test_tiny_but_legal_region(self):
        chain = build_filter(Layout(kind="crop", gameplay=Rect(0.0, 0.0, 0.001, 0.001)))
        assert "scale=1080:1920" in chain

    @pytest.mark.parametrize("fraction", [0.05, 0.5, 0.95])
    def test_webcam_bands_always_sum_correctly(self, fraction):
        chain = build_filter(
            Layout(kind="webcam", webcam=Rect(0.7, 0.0, 0.3, 0.3), webcam_fraction=fraction)
        )
        assert "vstack=2" in chain


class TestChatLimits:
    def test_a_hundred_thousand_messages(self, tmp_path):
        path = tmp_path / "chat.json"
        records = [
            {"content_offset_seconds": i * 0.4, "message": f"msg {i}"}
            for i in range(100_000)
        ]
        path.write_text(json.dumps(records), encoding="utf-8")
        started = time.perf_counter()
        loaded = load_chat(path)
        elapsed = time.perf_counter() - started
        assert len(loaded) == 100_000
        assert elapsed < 120, f"chat load took {elapsed:.0f}s"

    def test_all_messages_at_the_same_instant(self):
        records = [{"time": 5.0, "text": f"m{i}"} for i in range(1000)]
        features = analyze_chat(records, 600.0, quiet_msgs_per_min=0.0)
        assert features

    def test_messages_beyond_the_vod_are_ignored(self):
        records = [{"time": 99999.0, "text": "late"}] * 100
        features = analyze_chat(records, 600.0, quiet_msgs_per_min=0.0)
        assert all(f["count"] == 0 for f in features)


class TestTitleLimits:
    def test_very_long_transcript_line(self):
        segments = [{"start": 0.0, "end": 5.0, "text": "word " * 5000, "score": 0.9}]
        assert len(suggest_title(segments, 0.0, 5.0)) <= 70

    def test_unicode_title(self):
        segments = [{"start": 0.0, "end": 5.0, "text": "これは本当にすごい瞬間です", "score": 0.9}]
        assert suggest_title(segments, 0.0, 5.0)

    def test_transcript_of_only_filler(self):
        segments = [{"start": 0.0, "end": 5.0, "text": "um uh like so", "score": 0.9}]
        # Nothing substantial survives cleaning, so the fallback is used.
        assert suggest_title(segments, 0.0, 5.0, fallback="H001") == "H001"
