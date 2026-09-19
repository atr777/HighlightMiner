"""Crop framing resolves per clip first, then per source.

Framing is an editorial decision about one moment. Adjusting the window while
reviewing one candidate must not silently reframe the rest of the queue, which
is what a source-only setting did.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from highlightminer.config import Settings
from highlightminer.storage import (
    analysis_crop_rect,
    candidate_crop_rect,
    load_review,
    resolve_crop_rect,
    save_analysis,
    save_candidate_crop_rect,
    save_source_crop_rect,
)

CLIP = {"x": 0.10, "y": 0.0, "w": 0.316406, "h": 1.0}
SOURCE = {"x": 0.27, "y": 0.0, "w": 0.316406, "h": 1.0}


@pytest.fixture
def analysis(tmp_path):
    video = tmp_path / "stream.mp4"
    video.write_bytes(b"\0" * 4096)
    db = tmp_path / "highlightminer.db"
    payload = {
        "version": 1, "video_path": str(video), "content_label": "T",
        "duration": 600.0, "media": {"duration": 600.0, "streams": []},
        "transcription": {"language": "en"},
        "chat": {"path": None, "messages": 0},
        "settings": asdict(Settings()),
        "candidates": [
            {
                "id": cid, "rank": rank, "score": 0.9, "peak_time": 10.0 * rank,
                "start": 10.0 * rank, "end": 10.0 * rank + 20,
                "start_label": "00:10", "end_label": "00:30",
                "audio_score": 0.9, "transcript_score": 0.4, "chat_score": 0.0,
                "reason": "audio spike", "transcript": "x", "content_label": "T",
                "features": {},
            }
            for rank, cid in enumerate(("H001", "H002"), start=1)
        ],
    }
    analysis_id = save_analysis(db, payload, [], [], [], work_dir=str(tmp_path))
    from highlightminer.identity import describe_source

    return db, analysis_id, describe_source(video)["fingerprint"]


class TestPerClipStorage:
    def test_nothing_set_by_default(self, analysis):
        db, analysis_id, _ = analysis
        assert candidate_crop_rect(db, analysis_id, "H001") is None

    def test_round_trips(self, analysis):
        db, analysis_id, _ = analysis
        assert save_candidate_crop_rect(db, analysis_id, "H001", CLIP) is True
        assert candidate_crop_rect(db, analysis_id, "H001") == CLIP

    def test_one_clip_does_not_affect_another(self, analysis):
        """The whole point of the change."""
        db, analysis_id, _ = analysis
        save_candidate_crop_rect(db, analysis_id, "H001", CLIP)
        assert candidate_crop_rect(db, analysis_id, "H002") is None

    def test_can_be_cleared(self, analysis):
        db, analysis_id, _ = analysis
        save_candidate_crop_rect(db, analysis_id, "H001", CLIP)
        save_candidate_crop_rect(db, analysis_id, "H001", None)
        assert candidate_crop_rect(db, analysis_id, "H001") is None

    def test_unknown_candidate_reports_failure(self, analysis):
        db, analysis_id, _ = analysis
        assert save_candidate_crop_rect(db, analysis_id, "NOPE", CLIP) is False

    def test_appears_in_loaded_review_state(self, analysis):
        db, analysis_id, _ = analysis
        save_candidate_crop_rect(db, analysis_id, "H001", CLIP)
        review = load_review(db, analysis_id)
        assert review["items"]["H001"]["crop_rect"] == CLIP
        assert review["items"]["H002"]["crop_rect"] is None


class TestResolution:
    def test_clip_framing_wins(self, analysis):
        db, analysis_id, fingerprint = analysis
        save_source_crop_rect(db, fingerprint, SOURCE)
        save_candidate_crop_rect(db, analysis_id, "H001", CLIP)
        assert resolve_crop_rect(db, analysis_id, "H001") == CLIP

    def test_falls_back_to_the_source(self, analysis):
        db, analysis_id, fingerprint = analysis
        save_source_crop_rect(db, fingerprint, SOURCE)
        assert resolve_crop_rect(db, analysis_id, "H002") == SOURCE

    def test_other_clips_keep_the_source_framing(self, analysis):
        db, analysis_id, fingerprint = analysis
        save_source_crop_rect(db, fingerprint, SOURCE)
        save_candidate_crop_rect(db, analysis_id, "H001", CLIP)
        assert resolve_crop_rect(db, analysis_id, "H002") == SOURCE

    def test_none_when_neither_is_set(self, analysis):
        db, analysis_id, _ = analysis
        assert resolve_crop_rect(db, analysis_id, "H001") is None

    def test_without_a_candidate_it_is_the_source_window(self, analysis):
        db, analysis_id, fingerprint = analysis
        save_source_crop_rect(db, fingerprint, SOURCE)
        save_candidate_crop_rect(db, analysis_id, "H001", CLIP)
        assert resolve_crop_rect(db, analysis_id) == SOURCE
        assert analysis_crop_rect(db, analysis_id) == SOURCE

    def test_clearing_a_clip_restores_the_source_framing(self, analysis):
        db, analysis_id, fingerprint = analysis
        save_source_crop_rect(db, fingerprint, SOURCE)
        save_candidate_crop_rect(db, analysis_id, "H001", CLIP)
        save_candidate_crop_rect(db, analysis_id, "H001", None)
        assert resolve_crop_rect(db, analysis_id, "H001") == SOURCE


class TestTableStyling:
    def _rows(self):
        return [
            {"#": 1, "ID": "H001", "Score": 9.5, "Start": "00:10", "End": "00:40",
             "Why": "audio spike", "Status": "keep"},
            {"#": 2, "ID": "H002", "Score": 8.7, "Start": "01:10", "End": "01:40",
             "Why": "chat burst", "Status": "reject"},
            {"#": 3, "ID": "H003", "Score": 8.0, "Start": "02:10", "End": "02:40",
             "Why": "speech", "Status": "unreviewed"},
        ]

    def test_kept_and_rejected_rows_are_tinted(self):
        from highlightminer.ui_mine import _STATUS_STYLES, _style_candidate_rows

        html = _style_candidate_rows(self._rows()).to_html()
        assert _STATUS_STYLES["keep"].split(":")[1].strip() in html
        assert _STATUS_STYLES["reject"].split(":")[1].strip() in html

    def test_unreviewed_rows_are_left_plain(self):
        from highlightminer.ui_mine import _style_candidate_rows

        html = _style_candidate_rows([self._rows()[2]]).to_html()
        assert "background-color" not in html

    def test_empty_table_is_safe(self):
        from highlightminer.ui_mine import _style_candidate_rows

        assert _style_candidate_rows([]).empty
