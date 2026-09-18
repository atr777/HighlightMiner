from __future__ import annotations

import re

import pytest

from highlightminer.render import (
    SHORT_FORM_HEIGHT,
    SHORT_FORM_WIDTH,
    Layout,
    LayoutError,
    Rect,
    build_filter,
    centered_vertical_slice,
    escape_filter_path,
)


class TestRect:
    def test_defaults_to_full_frame(self):
        assert Rect().is_full_frame()

    def test_partial_rect_is_not_full_frame(self):
        assert not Rect(0.1, 0.0, 0.5, 1.0).is_full_frame()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"x": -0.1},
            {"y": 1.5},
            {"w": 0.0},
            {"h": -1.0},
            {"x": 0.8, "w": 0.5},   # runs off the right edge
            {"y": 0.9, "h": 0.5},   # runs off the bottom edge
        ],
    )
    def test_rejects_out_of_bounds(self, kwargs):
        with pytest.raises(LayoutError):
            Rect(**kwargs)

    def test_crop_filter_is_resolution_independent(self):
        # Fractions of iw/ih, so a template drawn on 1080p works on 720p.
        assert Rect(0.25, 0.1, 0.5, 0.8).crop_filter() == "crop=iw*0.5:ih*0.8:iw*0.25:ih*0.1"

    def test_round_trips_through_dict(self):
        rect = Rect(0.25, 0.1, 0.5, 0.8)
        assert Rect.from_dict(rect.as_dict()) == rect


class TestCenteredVerticalSlice:
    def test_keeps_about_28_percent_of_a_16_9_width(self):
        slice_ = centered_vertical_slice()
        assert slice_.w == pytest.approx(0.31640625)
        assert slice_.h == 1.0

    def test_is_horizontally_centered(self):
        slice_ = centered_vertical_slice()
        assert slice_.x == pytest.approx((1.0 - slice_.w) / 2)


class TestLayout:
    def test_webcam_layout_requires_a_webcam_rect(self):
        with pytest.raises(LayoutError, match="webcam rectangle"):
            Layout(kind="webcam")

    def test_rejects_unknown_kind(self):
        with pytest.raises(LayoutError, match="Unknown layout kind"):
            Layout(kind="hologram")

    @pytest.mark.parametrize("fraction", [0.0, 0.99, 1.5])
    def test_rejects_absurd_webcam_fractions(self, fraction):
        with pytest.raises(LayoutError, match="webcam_fraction"):
            Layout(kind="webcam", webcam=Rect(0.7, 0.0, 0.3, 0.3), webcam_fraction=fraction)

    def test_round_trips_through_dict(self):
        layout = Layout(kind="webcam", webcam=Rect(0.7, 0.0, 0.3, 0.3), webcam_fraction=0.4)
        assert Layout.from_dict(layout.as_dict()) == layout

    def test_from_dict_defaults_to_letterbox(self):
        assert Layout.from_dict({}).kind == "letterbox"


class TestBuildFilter:
    def test_defaults_to_letterbox(self):
        assert "overlay" in build_filter()

    def test_letterbox_blurs_a_background_copy(self):
        chain = build_filter(Layout(kind="letterbox"))
        assert "split=2[bg][fg]" in chain
        assert "boxblur" in chain
        # foreground is contained, background covers
        assert "force_original_aspect_ratio=decrease" in chain
        assert "force_original_aspect_ratio=increase" in chain

    def test_letterbox_without_a_source_rect_skips_the_crop(self):
        assert "crop=iw" not in build_filter(Layout(kind="letterbox"))

    def test_letterbox_with_a_source_rect_crops_first(self):
        chain = build_filter(Layout(kind="letterbox", gameplay=Rect(0.0, 0.0, 0.5, 1.0)))
        assert chain.startswith("crop=iw*0.5:ih*1:iw*0:ih*0,split=2")

    def test_crop_defaults_to_the_centered_slice(self):
        chain = build_filter(Layout(kind="crop"))
        assert chain.startswith("crop=iw*0.316406")

    def test_crop_honours_an_explicit_region(self):
        chain = build_filter(Layout(kind="crop", gameplay=Rect(0.0, 0.0, 0.4, 1.0)))
        assert chain.startswith("crop=iw*0.4:")

    def test_webcam_splits_height_into_explicit_bands(self):
        chain = build_filter(
            Layout(kind="webcam", webcam=Rect(0.7, 0.0, 0.3, 0.3), webcam_fraction=0.3)
        )
        # 30% of 1920 = 576, remainder 1344, and both are even
        assert "crop=1080:576" in chain
        assert "crop=1080:1344" in chain
        assert "vstack=2" in chain

    @pytest.mark.parametrize("fraction", [0.1, 0.25, 0.3, 0.5, 0.75, 0.9])
    def test_webcam_bands_always_sum_to_the_output_height(self, fraction):
        chain = build_filter(
            Layout(kind="webcam", webcam=Rect(0.7, 0.0, 0.3, 0.3), webcam_fraction=fraction)
        )
        heights = [int(h) for h in re.findall(rf"crop={SHORT_FORM_WIDTH}:(\d+)", chain)]
        # the two bands, then the final safety cover-crop
        assert len(heights) == 3
        assert heights[0] + heights[1] == SHORT_FORM_HEIGHT
        assert all(h % 2 == 0 for h in heights)

    def test_every_layout_ends_with_square_pixels(self):
        for layout in (
            Layout(kind="letterbox"),
            Layout(kind="crop"),
            Layout(kind="webcam", webcam=Rect(0.7, 0.0, 0.3, 0.3)),
        ):
            assert build_filter(layout).endswith("setsar=1")

    def test_subtitles_are_burned_last(self):
        chain = build_filter(Layout(kind="crop"), subtitles="C\\:/work/clip.ass")
        assert chain.endswith("ass=C\\:/work/clip.ass")
        # after setsar, so captions are not scaled or cropped with the source
        assert chain.index("setsar=1") < chain.index("ass=")

    def test_no_subtitles_means_no_ass_filter(self):
        assert "ass=" not in build_filter(Layout(kind="crop"))

    def test_custom_output_size(self):
        chain = build_filter(Layout(kind="crop"), width=720, height=1280)
        assert "crop=720:1280" in chain

    def test_default_output_is_1080x1920(self):
        assert f"crop={SHORT_FORM_WIDTH}:{SHORT_FORM_HEIGHT}" in build_filter(Layout(kind="crop"))


class TestEscapeFilterPath:
    def test_escapes_a_windows_path(self):
        assert escape_filter_path(r"C:\work\clip.ass") == r"C\:/work/clip.ass"

    def test_leaves_posix_paths_alone_except_colons(self):
        assert escape_filter_path("/home/a/clip.ass") == "/home/a/clip.ass"
