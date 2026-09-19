from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from highlightminer import crop_tool
from highlightminer.crop_tool import (
    DEFAULT_SAMPLE_COUNT,
    CropToolError,
    annotate_frame,
    crop_preview,
    rect_for_aspect,
    sample_frames,
    sample_times,
)
from highlightminer.render import Rect


def _frame(path: Path, size=(960, 540), colour=(20, 120, 200)) -> Path:
    Image.new("RGB", size, colour).save(path)
    return path


class TestSampleTimes:
    def test_spreads_across_the_vod(self):
        times = sample_times(1000.0, 5)
        assert len(times) == 5
        assert times == sorted(times)

    def test_skips_the_very_start_and_end(self):
        """Openers and outros are not representative gameplay."""
        times = sample_times(1000.0, 5)
        assert times[0] > 0.0
        assert times[-1] < 1000.0

    def test_thirteen_hour_vod_spans_the_whole_thing(self):
        times = sample_times(48826.0, 6)
        assert times[0] / 3600 == pytest.approx(0.54, abs=0.05)
        assert times[-1] / 3600 == pytest.approx(13.02, abs=0.05)

    def test_single_sample_lands_in_the_middle(self):
        assert sample_times(1000.0, 1) == [pytest.approx(500.0)]

    def test_zero_duration_is_safe(self):
        assert sample_times(0.0, 3) == [0.0, 0.0, 0.0]

    def test_count_is_clamped_to_at_least_one(self):
        assert len(sample_times(100.0, 0)) == 1

    def test_default_count(self):
        assert len(sample_times(100.0)) == DEFAULT_SAMPLE_COUNT


class TestRectForAspect:
    def test_full_height_gives_the_916_width_of_a_169_frame(self):
        rect = rect_for_aspect(x=0.0, height=1.0)
        assert rect.w == pytest.approx(0.316406, abs=1e-5)

    def test_half_height_halves_the_width(self):
        assert rect_for_aspect(x=0.0, height=0.5).w == pytest.approx(0.158203, abs=1e-5)

    def test_x_is_clamped_so_the_rect_stays_inside(self):
        rect = rect_for_aspect(x=0.95, height=1.0)
        assert rect.x + rect.w <= 1.0 + 1e-9

    def test_y_is_clamped_too(self):
        rect = rect_for_aspect(x=0.0, y=0.9, height=0.5)
        assert rect.y + rect.h <= 1.0 + 1e-9

    def test_negative_inputs_are_clamped(self):
        rect = rect_for_aspect(x=-1.0, y=-1.0, height=1.0)
        assert rect.x == 0.0 and rect.y == 0.0

    def test_result_is_always_a_valid_rect(self):
        # Rect validates in __post_init__, so this would raise on a bad value.
        for x in (0.0, 0.3, 0.7, 1.0):
            for h in (0.3, 0.5, 1.0):
                rect = rect_for_aspect(x=x, height=h)
                assert isinstance(rect, Rect)

    @pytest.mark.parametrize("height", [0.4, 0.75, 1.0])
    def test_pixel_aspect_of_the_region_is_9_by_16(self, height):
        """In a 16:9 frame, pixel aspect is (w/h) * (16/9), so w/h must be 81/256."""
        rect = rect_for_aspect(x=0.0, height=height, source_aspect=16 / 9)
        pixel_aspect = (rect.w / rect.h) * (16 / 9)
        assert pixel_aspect == pytest.approx(9 / 16, abs=1e-4)


class TestAnnotateFrame:
    def test_writes_an_image_of_the_same_size(self, tmp_path):
        src = _frame(tmp_path / "f.png")
        out = annotate_frame(src, Rect(0.3, 0.0, 0.32, 1.0), tmp_path / "marked.png")
        assert Image.open(out).size == Image.open(src).size

    def test_kept_region_stays_brighter_than_the_surround(self, tmp_path):
        src = _frame(tmp_path / "f.png")
        out = annotate_frame(src, Rect(0.4, 0.0, 0.2, 1.0), tmp_path / "marked.png")
        image = Image.open(out).convert("RGB")
        width, height = image.size
        inside = image.getpixel((int(width * 0.5), height // 2))
        outside = image.getpixel((int(width * 0.05), height // 2))
        assert sum(inside) > sum(outside), "the kept region must not be dimmed"

    def test_full_frame_rect_dims_nothing(self, tmp_path):
        src = _frame(tmp_path / "f.png")
        out = annotate_frame(src, Rect(), tmp_path / "marked.png")
        image = Image.open(out).convert("RGB")
        centre = image.getpixel((image.size[0] // 2, image.size[1] // 2))
        assert sum(centre) > 200

    def test_creates_parent_directories(self, tmp_path):
        src = _frame(tmp_path / "f.png")
        out = annotate_frame(src, Rect(0.3, 0.0, 0.3, 1.0), tmp_path / "a" / "b" / "m.png")
        assert out.exists()


class TestCropPreview:
    def test_output_is_the_short_form_aspect(self, tmp_path):
        src = _frame(tmp_path / "f.png")
        out = crop_preview(src, Rect(0.3, 0.0, 0.316406, 1.0), tmp_path / "c.png", height=640)
        width, height = Image.open(out).size
        assert height == 640
        assert width / height == pytest.approx(1080 / 1920, abs=0.01)

    def test_narrow_rect_still_produces_an_image(self, tmp_path):
        src = _frame(tmp_path / "f.png")
        out = crop_preview(src, Rect(0.0, 0.0, 0.01, 0.01), tmp_path / "c.png", height=320)
        assert Image.open(out).size[1] == 320

    def test_respects_the_vertical_offset(self, tmp_path):
        # Top half red, bottom half blue; cropping the bottom must read blue.
        image = Image.new("RGB", (960, 540), (255, 0, 0))
        for y in range(270, 540):
            for x in range(0, 960, 8):
                image.putpixel((x, y), (0, 0, 255))
        src = tmp_path / "split.png"
        image.save(src)
        out = crop_preview(src, Rect(0.4, 0.6, 0.2, 0.4), tmp_path / "c.png", height=200)
        image = Image.open(out).convert("RGB")
        width, height = image.size
        blueish = sum(
            1
            for y in range(0, height, 8)
            for x in range(0, width, 8)
            if image.getpixel((x, y))[2] > image.getpixel((x, y))[0]
        )
        assert blueish > 0


class TestSampleFrames:
    def test_skips_frames_that_cannot_be_read(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def flaky(video, when, out, width=960):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise CropToolError("bad seek")
            return _frame(Path(out))

        monkeypatch.setattr(crop_tool, "extract_frame", flaky)
        frames = sample_frames("v.mp4", 600.0, tmp_path, count=4)
        assert len(frames) == 2

    def test_raises_when_nothing_can_be_read(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            crop_tool, "extract_frame",
            lambda *a, **k: (_ for _ in ()).throw(CropToolError("nope")),
        )
        with pytest.raises(CropToolError, match="No frames"):
            sample_frames("v.mp4", 600.0, tmp_path, count=3)

    def test_returns_times_alongside_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            crop_tool, "extract_frame", lambda v, when, out, width=960: _frame(Path(out))
        )
        frames = sample_frames("v.mp4", 600.0, tmp_path, count=3)
        assert [round(f.time) for f in frames] == [24, 300, 576]


class TestExtractFrame:
    def test_ffmpeg_failure_becomes_a_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crop_tool, "require_ffmpeg", lambda: None)
        monkeypatch.setattr(crop_tool, "require_executable", lambda _n: "ffmpeg")

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, "ffmpeg")

        monkeypatch.setattr(crop_tool.subprocess, "run", boom)
        with pytest.raises(CropToolError, match="Could not read a frame"):
            crop_tool.extract_frame("v.mp4", 10.0, tmp_path / "f.png")

    def test_missing_output_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(crop_tool, "require_ffmpeg", lambda: None)
        monkeypatch.setattr(crop_tool, "require_executable", lambda _n: "ffmpeg")
        monkeypatch.setattr(crop_tool.subprocess, "run", lambda *a, **k: None)
        with pytest.raises(CropToolError, match="wrote no frame"):
            crop_tool.extract_frame("v.mp4", 10.0, tmp_path / "missing.png")
