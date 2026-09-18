from __future__ import annotations

import re

import pytest

from highlightminer.captions import (
    CaptionStyle,
    Word,
    build_ass,
    group_phrases,
    write_ass,
    words_in_window,
)


def _words(*spans: tuple[str, float, float]) -> list[Word]:
    return [Word(t, s, e) for t, s, e in spans]


def _dialogue(document: str) -> list[str]:
    return [line for line in document.splitlines() if line.startswith("Dialogue:")]


def _karaoke_total_cs(line: str) -> int:
    return sum(int(v) for v in re.findall(r"\\k(\d+)", line))


class TestWordsInWindow:
    def test_extracts_and_rebases_to_clip_time(self):
        segments = [{
            "start": 100.0, "end": 103.0, "text": "a b",
            "words": [{"w": "a", "s": 100.5, "e": 101.0}, {"w": "b", "s": 101.0, "e": 102.0}],
        }]
        words = words_in_window(segments, 100.0, 103.0)
        assert [(w.text, w.start, w.end) for w in words] == [("a", 0.5, 1.0), ("b", 1.0, 2.0)]

    def test_excludes_words_outside_the_window(self):
        segments = [{
            "start": 0.0, "end": 100.0, "text": "x",
            "words": [
                {"w": "before", "s": 1.0, "e": 2.0},
                {"w": "inside", "s": 51.0, "e": 52.0},
                {"w": "after", "s": 95.0, "e": 96.0},
            ],
        }]
        words = words_in_window(segments, 50.0, 60.0)
        assert [w.text for w in words] == ["inside"]

    def test_falls_back_to_segments_without_word_timings(self):
        """Transcripts made before word timestamps still caption, less precisely."""
        segments = [{"start": 10.0, "end": 12.0, "text": "whole segment"}]
        words = words_in_window(segments, 10.0, 12.0)
        assert [(w.text, w.start) for w in words] == [("whole segment", 0.0)]

    def test_never_produces_negative_times(self):
        segments = [{
            "start": 0.0, "end": 10.0, "text": "x",
            "words": [{"w": "straddles", "s": 4.0, "e": 6.0}],
        }]
        words = words_in_window(segments, 5.0, 10.0)
        assert words[0].start == 0.0

    def test_blank_words_are_dropped(self):
        segments = [{
            "start": 0.0, "end": 5.0, "text": "x",
            "words": [{"w": "  ", "s": 1.0, "e": 2.0}, {"w": "real", "s": 2.0, "e": 3.0}],
        }]
        assert [w.text for w in words_in_window(segments, 0.0, 5.0)] == ["real"]

    def test_empty_input(self):
        assert words_in_window([], 0.0, 10.0) == []


class TestGroupPhrases:
    def test_splits_on_word_count(self):
        words = _words(*[(f"w{i}", i * 0.1, i * 0.1 + 0.1) for i in range(9)])
        phrases = group_phrases(words, max_words=4)
        assert [len(p.words) for p in phrases] == [4, 4, 1]

    def test_splits_on_a_pause(self):
        words = _words(("a", 0.0, 0.2), ("b", 0.3, 0.5), ("c", 5.0, 5.2))
        phrases = group_phrases(words, gap_seconds=0.6)
        assert [[w.text for w in p.words] for p in phrases] == [["a", "b"], ["c"]]

    def test_splits_on_duration(self):
        words = _words(("a", 0.0, 0.2), ("b", 0.3, 4.0))
        phrases = group_phrases(words, max_seconds=2.5, gap_seconds=10.0)
        assert len(phrases) == 2

    def test_empty_input(self):
        assert group_phrases([]) == []


class TestBuildAss:
    def test_has_required_sections(self):
        document = build_ass(_words(("hi", 0.0, 0.5)))
        for section in ("[Script Info]", "[V4+ Styles]", "[Events]"):
            assert section in document

    def test_play_resolution_matches_the_output_frame(self):
        document = build_ass(_words(("hi", 0.0, 0.5)), width=1080, height=1920)
        assert "PlayResX: 1080" in document
        assert "PlayResY: 1920" in document

    def test_karaoke_durations_fill_the_event(self):
        """Gaps between words must be spent, or the highlight drifts early."""
        words = _words(("a", 0.0, 0.4), ("b", 0.9, 1.4))  # 0.5s of silence between
        line = _dialogue(build_ass(words))[0]
        assert _karaoke_total_cs(line) == 140  # the full 1.4s, not just 0.9s

    def test_karaoke_matches_event_duration_for_many_shapes(self):
        cases = [
            _words(("a", 0.0, 0.3), ("b", 0.3, 0.6)),
            _words(("a", 0.0, 0.1), ("b", 0.55, 0.9)),
            _words(("a", 1.0, 1.2), ("b", 1.2, 1.25), ("c", 1.4, 2.0)),
        ]
        for words in cases:
            line = _dialogue(build_ass(words))[0]
            expected = int(round((words[-1].end - words[0].start) * 100))
            assert _karaoke_total_cs(line) == expected

    def test_highlight_uses_primary_and_base_uses_secondary(self):
        style = CaptionStyle(primary="&H00FFFFFF", highlight="&H0000D7FF")
        document = build_ass(_words(("hi", 0.0, 0.5)), style)
        style_line = next(l for l in document.splitlines() if l.startswith("Style: Caption"))
        fields = style_line.split(",")
        assert fields[3] == "&H0000D7FF"  # PrimaryColour is what a sung word becomes
        assert fields[4] == "&H00FFFFFF"  # SecondaryColour is the unsung base

    def test_colours_swap_when_highlighting_is_off(self):
        style = CaptionStyle(primary="&H00FFFFFF", highlight="&H0000D7FF", highlight_words=False)
        document = build_ass(_words(("hi", 0.0, 0.5)), style)
        style_line = next(l for l in document.splitlines() if l.startswith("Style: Caption"))
        assert style_line.split(",")[3] == "&H00FFFFFF"

    def test_no_karaoke_tags_when_highlighting_is_off(self):
        document = build_ass(_words(("a", 0.0, 0.3), ("b", 0.3, 0.6)), CaptionStyle(highlight_words=False))
        assert "\\k" not in document
        assert "a b" in document

    def test_uppercase_option(self):
        document = build_ass(_words(("shout", 0.0, 0.4)), CaptionStyle(uppercase=True))
        assert "SHOUT" in document

    def test_braces_in_text_are_escaped(self):
        """An unescaped brace would open an ASS override block."""
        document = build_ass(_words(("{evil}", 0.0, 0.4)))
        assert r"\{evil\}" in document

    def test_very_short_word_still_gets_a_visible_event(self):
        line = _dialogue(build_ass(_words(("x", 1.0, 1.0))))[0]
        # events are widened to a minimum so a zero-length word is not invisible
        assert "0:00:01.00,0:00:01.20" in line

    def test_empty_words_produce_a_valid_document_with_no_events(self):
        document = build_ass([])
        assert "[Events]" in document
        assert _dialogue(document) == []

    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0.0, "0:00:00.00"),
            (1.5, "0:00:01.50"),
            (61.25, "0:01:01.25"),
            (3661.0, "1:01:01.00"),
            (0.999, "0:00:01.00"),   # rounding must carry, not print .100
            (59.999, "0:01:00.00"),  # and carry through a minute boundary
        ],
    )
    def test_timestamp_formatting(self, seconds, expected):
        from highlightminer.captions import _ass_timestamp
        assert _ass_timestamp(seconds) == expected


class TestWriteAss:
    def test_writes_with_a_bom_for_libass(self, tmp_path):
        path = write_ass(tmp_path / "c.ass", _words(("hi", 0.0, 0.5)))
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_creates_parent_directories(self, tmp_path):
        path = write_ass(tmp_path / "nested" / "deeper" / "c.ass", _words(("hi", 0.0, 0.5)))
        assert path.exists()
