"""The database location is overridable.

Working files often belong on a different drive from the application. Without
an override the desktop UI could only open the database beside the executable,
while the CLI could be pointed anywhere with --db. An analysis run on another
drive was therefore invisible in the app, which is exactly what happened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from highlightminer import storage
from highlightminer.storage import DATABASE_PATH_ENV, default_db_path


def test_defaults_beside_the_application(monkeypatch):
    monkeypatch.delenv(DATABASE_PATH_ENV, raising=False)
    assert default_db_path().name == "highlightminer.db"


def test_environment_override_is_used(monkeypatch, tmp_path):
    target = tmp_path / "elsewhere" / "hm.db"
    monkeypatch.setenv(DATABASE_PATH_ENV, str(target))
    assert default_db_path() == target


def test_blank_override_falls_back(monkeypatch):
    monkeypatch.setenv(DATABASE_PATH_ENV, "   ")
    assert default_db_path().name == "highlightminer.db"


def test_user_home_is_expanded(monkeypatch):
    monkeypatch.setenv(DATABASE_PATH_ENV, "~/hm.db")
    assert "~" not in str(default_db_path())


def test_override_does_not_need_to_exist_yet(monkeypatch, tmp_path):
    """A fresh database on another drive is created on first use."""
    target = tmp_path / "new" / "hm.db"
    monkeypatch.setenv(DATABASE_PATH_ENV, str(target))
    resolved = default_db_path()
    assert not resolved.exists()
    with storage.connect(resolved) as conn:
        storage.initialize(conn)
    assert resolved.exists()


def test_data_written_through_the_override_reads_back(monkeypatch, tmp_path):
    target = tmp_path / "hm.db"
    monkeypatch.setenv(DATABASE_PATH_ENV, str(target))
    from dataclasses import asdict

    from highlightminer.config import Settings

    video = tmp_path / "v.mp4"
    video.write_bytes(b"\0" * 2048)
    payload = {
        "version": 1, "video_path": str(video), "content_label": "T",
        "duration": 60.0, "media": {"duration": 60.0, "streams": []},
        "transcription": {"language": "en"},
        "chat": {"path": None, "messages": 0},
        "settings": asdict(Settings()), "candidates": [],
    }
    analysis_id = storage.save_analysis(
        default_db_path(), payload, [], [], [], work_dir=str(tmp_path)
    )
    assert storage.load_analysis(default_db_path(), analysis_id)["duration"] == 60.0
