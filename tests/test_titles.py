from __future__ import annotations

import pytest

from highlightminer.titles import (
    MAX_TITLE_CHARS,
    suggest_description,
    suggest_hashtags,
    suggest_title,
)


def _seg(start, end, text, score=0.0):
    return {"start": start, "end": end, "text": text, "score": score}


class TestSuggestTitle:
    def test_prefers_the_highest_scoring_line(self):
        segments = [
            _seg(0.0, 2.0, "nothing much happening here", 0.1),
            _seg(2.0, 4.0, "that is the craziest thing I have seen", 0.9),
        ]
        assert suggest_title(segments, 0.0, 5.0) == "that is the craziest thing I have seen"

    def test_strips_leading_filler(self):
        segments = [_seg(0.0, 2.0, "so like I actually cannot believe that", 0.9)]
        title = suggest_title(segments, 0.0, 5.0)
        assert not title.lower().startswith(("so ", "like ", "actually "))
        assert "cannot believe" in title

    def test_removes_stutters_and_fillers(self):
        segments = [_seg(0.0, 2.0, "that that that was um absolutely wild", 0.9)]
        title = suggest_title(segments, 0.0, 5.0)
        assert "um" not in title.split()
        assert title.lower().count("that") == 1

    def test_skips_grunts_in_favour_of_a_real_line(self):
        segments = [
            _seg(0.0, 1.0, "what", 0.95),
            _seg(1.0, 3.0, "he actually hit that shot", 0.5),
        ]
        assert suggest_title(segments, 0.0, 5.0) == "he actually hit that shot"

    def test_truncates_on_a_word_boundary(self):
        long_line = "this is an extremely long sentence that goes on well past any sensible title length limit"
        title = suggest_title([_seg(0.0, 5.0, long_line, 0.9)], 0.0, 5.0)
        assert len(title) <= MAX_TITLE_CHARS
        assert not title.endswith(" ")
        assert long_line.startswith(title)

    def test_only_uses_segments_inside_the_window(self):
        segments = [
            _seg(100.0, 102.0, "way outside the clip window entirely", 0.99),
            _seg(10.0, 12.0, "inside the clip window here", 0.2),
        ]
        assert suggest_title(segments, 9.0, 13.0) == "inside the clip window here"

    def test_falls_back_when_there_is_no_speech(self):
        assert suggest_title([], 0.0, 5.0, fallback="H001") == "H001"

    def test_falls_back_when_every_line_is_a_grunt(self):
        segments = [_seg(0.0, 1.0, "uh", 0.9)]
        assert suggest_title(segments, 0.0, 5.0, fallback="H002") == "H002"

    def test_malformed_segments_are_ignored(self):
        segments = [{"text": "no timing"}, _seg(1.0, 2.0, "a usable line here", 0.5)]
        assert suggest_title(segments, 0.0, 5.0) == "a usable line here"


class TestSuggestHashtags:
    def test_builds_from_the_content_label(self):
        tags = suggest_hashtags("Overwatch 2")
        assert "#overwatch" in tags
        assert "#overwatch2" in tags

    def test_always_includes_generic_tags(self):
        assert "#shorts" in suggest_hashtags(None)

    def test_deduplicates(self):
        tags = suggest_hashtags("Shorts")
        assert tags.count("#shorts") == 1

    def test_respects_the_limit(self):
        assert len(suggest_hashtags("A B C D E F G H", limit=4)) == 4

    def test_strips_punctuation(self):
        assert "#justchatting" in suggest_hashtags("Just-Chatting!")

    def test_extra_tags_are_included(self):
        assert "#clutch" in suggest_hashtags("Valorant", extra=["clutch"])


class TestSuggestDescription:
    def test_combines_title_label_and_tags(self):
        text = suggest_description("He hit the shot", "Valorant")
        assert "He hit the shot" in text
        assert "Valorant" in text
        assert "#valorant" in text

    def test_handles_a_missing_label(self):
        text = suggest_description("Something happened", None)
        assert text.startswith("Something happened")

    def test_does_not_repeat_an_identical_title_and_label(self):
        text = suggest_description("Valorant", "Valorant")
        assert text.split("\n")[0] == "Valorant"
