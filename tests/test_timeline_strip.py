"""The timeline strip and the snap helpers that replace guessing at seconds.

Retiming used to mean typing a number, re-encoding a preview, and finding out
whether the guess was right. These cover the pieces that let a boundary be
chosen from what is actually there.
"""

from __future__ import annotations

import pytest
from PIL import Image

from highlightminer.timeline_strip import (
    HEIGHT,
    WIDTH,
    flatten_words,
    render_strip,
    snap_end,
    snap_start,
    window_for,
)


def _audio(start, end, step=0.5, value=0.5):
    rows = []
    t = start
    while t <= end:
        rows.append({"time": round(t, 3), "dbfs": -20.0, "energy": value, "onset": 0.0, "score": value})
        t += step
    return rows


def _words(*spans):
    return [(s, e, t) for s, e, t in spans]


class TestWindow:
    def test_pads_either_side_of_the_clip(self):
        w = window_for(100.0, 130.0, 1000.0, pad=8.0)
        assert w.view_start == 92.0
        assert w.view_end == 138.0

    def test_clamps_to_the_start_of_the_source(self):
        assert window_for(2.0, 20.0, 1000.0, pad=8.0).view_start == 0.0

    def test_clamps_to_the_end_of_the_source(self):
        assert window_for(980.0, 999.0, 1000.0, pad=8.0).view_end == 1000.0

    def test_degenerate_window_still_has_span(self):
        w = window_for(5.0, 5.0, 0.0, pad=0.0)
        assert w.span > 0

    def test_x_maps_boundaries_inside_the_image(self):
        w = window_for(100.0, 130.0, 1000.0)
        assert 0 <= w.x(w.clip_start) < WIDTH
        assert 0 <= w.x(w.clip_end) < WIDTH
        assert w.x(w.clip_start) < w.x(w.clip_end)

    def test_x_clamps_outside_values(self):
        w = window_for(100.0, 130.0, 1000.0)
        assert w.x(-999.0) == 0
        assert w.x(999999.0) == WIDTH - 1


class TestFlattenWords:
    def test_collects_and_sorts(self):
        segments = [
            {"words": [{"w": "b", "s": 2.0, "e": 2.5}]},
            {"words": [{"w": "a", "s": 1.0, "e": 1.5}]},
        ]
        assert [w[2] for w in flatten_words(segments)] == ["a", "b"]

    def test_skips_malformed_and_blank(self):
        segments = [{"words": [
            {"w": "ok", "s": 1.0, "e": 1.5},
            {"w": "", "s": 2.0, "e": 2.5},
            {"w": "bad", "s": "x", "e": 3.0},
            {"w": "missing"},
        ]}]
        assert [w[2] for w in flatten_words(segments)] == ["ok"]

    def test_no_segments(self):
        assert flatten_words([]) == []
        assert flatten_words(None) == []


class TestSnapping:
    WORDS = _words((10.0, 10.4, "one"), (11.0, 11.5, "two"), (12.2, 12.9, "three"))

    def test_snap_start_backwards(self):
        assert snap_start(self.WORDS, 11.4, -1) == 11.0

    def test_snap_start_forwards(self):
        assert snap_start(self.WORDS, 10.5, 1) == 11.0

    def test_snap_end_backwards(self):
        assert snap_end(self.WORDS, 12.5, -1) == 11.5

    def test_snap_end_forwards(self):
        assert snap_end(self.WORDS, 10.5, 1) == 11.5

    def test_returns_none_past_the_last_word(self):
        assert snap_start(self.WORDS, 99.0, 1) is None
        assert snap_end(self.WORDS, 99.0, 1) is None

    def test_returns_none_before_the_first_word(self):
        assert snap_start(self.WORDS, 0.0, -1) is None

    def test_no_words_is_safe(self):
        assert snap_start([], 5.0, 1) is None
        assert snap_end([], 5.0, -1) is None

    def test_exact_position_moves_on_rather_than_sticking(self):
        """Pressing the button on a boundary must still move."""
        assert snap_start(self.WORDS, 11.0, 1) == 12.2
        assert snap_start(self.WORDS, 11.0, -1) == 10.0


class TestRenderStrip:
    def test_writes_an_image_of_the_expected_size(self, tmp_path):
        w = window_for(100.0, 130.0, 1000.0)
        out = render_strip(tmp_path / "s.png", w, _audio(90, 140), _words((105.0, 105.5, "hi")))
        assert Image.open(out).size == (WIDTH, HEIGHT)

    def test_kept_region_is_brighter_than_the_trimmed_surround(self, tmp_path):
        w = window_for(100.0, 130.0, 1000.0)
        out = render_strip(tmp_path / "s.png", w, _audio(90, 140, value=0.9), [])
        image = Image.open(out).convert("RGB")
        inside = image.getpixel((w.x(115.0), 20))
        outside = image.getpixel((2, 20))
        assert sum(inside) > sum(outside)

    def test_works_with_no_audio_features(self, tmp_path):
        w = window_for(100.0, 130.0, 1000.0)
        out = render_strip(tmp_path / "s.png", w, [], _words((105.0, 105.5, "hi")))
        assert out.exists()

    def test_works_with_no_words(self, tmp_path):
        w = window_for(100.0, 130.0, 1000.0)
        assert render_strip(tmp_path / "s.png", w, _audio(90, 140), []).exists()

    def test_words_outside_the_view_are_ignored(self, tmp_path):
        w = window_for(100.0, 130.0, 1000.0)
        out = render_strip(
            tmp_path / "s.png", w, _audio(90, 140), _words((5.0, 5.5, "far"), (110.0, 110.5, "near"))
        )
        assert out.exists()

    def test_creates_parent_directories(self, tmp_path):
        w = window_for(100.0, 130.0, 1000.0)
        assert render_strip(tmp_path / "a" / "b" / "s.png", w, [], []).exists()

    @pytest.mark.parametrize("clip", [(0.0, 5.0), (995.0, 1000.0), (100.0, 100.5)])
    def test_edge_positions_do_not_crash(self, tmp_path, clip):
        w = window_for(clip[0], clip[1], 1000.0)
        assert render_strip(tmp_path / "s.png", w, _audio(0, 1000, step=5.0), []).exists()
