"""Crop regions are remembered per source, not globally.

Where the 9:16 window belongs depends on the channel's overlay layout. The
region chosen on one streamer's VOD sliced another streamer's alert box in
half, which is what prompted this.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from highlightminer.config import Settings
from highlightminer.render import layout_from_settings
from highlightminer.storage import (
    analysis_crop_rect,
    save_analysis,
    save_source_crop_rect,
    source_crop_rect,
)

CENTRE = {"x": 0.341797, "y": 0.0, "w": 0.316406, "h": 1.0}
NUDGED = {"x": 0.27, "y": 0.0, "w": 0.316406, "h": 1.0}


@pytest.fixture
def analysis(tmp_path):
    """A stored analysis with a known source fingerprint."""
    video = tmp_path / "stream.mp4"
    video.write_bytes(b"\0" * 4096)
    db = tmp_path / "highlightminer.db"
    payload = {
        "version": 1,
        "video_path": str(video),
        "content_label": "T",
        "duration": 600.0,
        "media": {"duration": 600.0, "streams": []},
        "transcription": {"language": "en"},
        "chat": {"path": None, "messages": 0},
        "settings": asdict(Settings()),
        "candidates": [],
    }
    analysis_id = save_analysis(db, payload, [], [], [], work_dir=str(tmp_path))
    from highlightminer.identity import describe_source

    return db, analysis_id, describe_source(video)["fingerprint"]


class TestStorage:
    def test_nothing_remembered_by_default(self, analysis):
        db, analysis_id, fingerprint = analysis
        assert source_crop_rect(db, fingerprint) is None
        assert analysis_crop_rect(db, analysis_id) is None

    def test_round_trips_by_fingerprint(self, analysis):
        db, _, fingerprint = analysis
        assert save_source_crop_rect(db, fingerprint, NUDGED) is True
        assert source_crop_rect(db, fingerprint) == NUDGED

    def test_reachable_from_the_analysis(self, analysis):
        db, analysis_id, fingerprint = analysis
        save_source_crop_rect(db, fingerprint, NUDGED)
        assert analysis_crop_rect(db, analysis_id) == NUDGED

    def test_can_be_cleared(self, analysis):
        db, _, fingerprint = analysis
        save_source_crop_rect(db, fingerprint, NUDGED)
        save_source_crop_rect(db, fingerprint, None)
        assert source_crop_rect(db, fingerprint) is None

    def test_unknown_source_reports_failure(self, analysis):
        db, _, _ = analysis
        assert save_source_crop_rect(db, "not-a-real-fingerprint", NUDGED) is False

    def test_unknown_analysis_returns_none(self, analysis):
        db, _, _ = analysis
        assert analysis_crop_rect(db, "no-such-analysis") is None

    def test_two_sources_keep_different_regions(self, tmp_path):
        from highlightminer.identity import describe_source

        db = tmp_path / "highlightminer.db"
        fingerprints = []
        for index, filler in enumerate((b"a", b"b")):
            video = tmp_path / f"stream{index}.mp4"
            video.write_bytes(filler * 4096)
            payload = {
                "version": 1, "video_path": str(video), "content_label": "T",
                "duration": 600.0, "media": {"duration": 600.0, "streams": []},
                "transcription": {"language": "en"},
                "chat": {"path": None, "messages": 0},
                "settings": asdict(Settings()), "candidates": [],
            }
            save_analysis(db, payload, [], [], [], work_dir=str(tmp_path))
            fingerprints.append(describe_source(video)["fingerprint"])

        save_source_crop_rect(db, fingerprints[0], CENTRE)
        save_source_crop_rect(db, fingerprints[1], NUDGED)
        assert source_crop_rect(db, fingerprints[0]) == CENTRE
        assert source_crop_rect(db, fingerprints[1]) == NUDGED


class TestLayoutOverride:
    def test_source_region_wins_over_the_profile_default(self):
        settings = Settings(render_layout="crop", gameplay_rect=NUDGED)
        layout = layout_from_settings(settings, crop_rect=CENTRE)
        assert layout.gameplay.x == pytest.approx(CENTRE["x"])

    def test_profile_default_is_used_when_the_source_has_none(self):
        settings = Settings(render_layout="crop", gameplay_rect=NUDGED)
        layout = layout_from_settings(settings, crop_rect=None)
        assert layout.gameplay.x == pytest.approx(NUDGED["x"])

    def test_centred_slice_when_neither_is_set(self):
        layout = layout_from_settings(Settings(render_layout="crop"))
        assert layout.gameplay is None  # build_filter falls back to the centred slice

    def test_override_is_ignored_for_source_aspect(self):
        assert layout_from_settings(Settings(render_layout="source"), crop_rect=CENTRE) is None

    def test_override_applies_to_the_built_filter(self):
        from highlightminer.render import build_filter

        settings = Settings(render_layout="crop", gameplay_rect=NUDGED)
        chain = build_filter(layout_from_settings(settings, crop_rect=CENTRE))
        assert "iw*0.341797" in chain
