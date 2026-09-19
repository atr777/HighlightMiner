"""Where the database and work folder live, when the drive is not the default.

Working files are large and usually belong on a different drive from the
application. The database already had an override; the work folder did not,
which meant retyping it in the sidebar on every launch. Forgetting once sent a
13 GB download to the system drive and started a second, empty database with
stock settings, so a carefully configured run silently used none of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from highlightminer.storage import DATABASE_PATH_ENV, default_db_path
from highlightminer.ui_common import WORK_DIR_ENV, default_work_dir


class TestWorkDirOverride:
    def test_falls_back_to_the_application_folder(self, monkeypatch):
        monkeypatch.delenv(WORK_DIR_ENV, raising=False)
        assert default_work_dir().endswith("highlightminer_work")

    def test_the_environment_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv(WORK_DIR_ENV, str(tmp_path))
        assert default_work_dir() == str(tmp_path)

    def test_blank_is_treated_as_unset(self, monkeypatch):
        """An empty variable must not point the work folder at the drive root."""
        monkeypatch.setenv(WORK_DIR_ENV, "   ")
        assert default_work_dir().endswith("highlightminer_work")

    def test_a_home_relative_path_is_expanded(self, monkeypatch):
        monkeypatch.setenv(WORK_DIR_ENV, "~/vods")
        assert "~" not in default_work_dir()

    def test_it_does_not_have_to_exist_yet(self, monkeypatch, tmp_path):
        target = tmp_path / "not" / "created"
        monkeypatch.setenv(WORK_DIR_ENV, str(target))
        assert default_work_dir() == str(target)


class TestDatabaseOverride:
    def test_the_environment_wins(self, monkeypatch, tmp_path):
        db = tmp_path / "elsewhere.db"
        monkeypatch.setenv(DATABASE_PATH_ENV, str(db))
        assert default_db_path() == db

    def test_blank_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(DATABASE_PATH_ENV, "")
        assert default_db_path().name.endswith(".db")


class TestTheTwoAreIndependent:
    def test_the_database_can_sit_outside_the_work_folder(self, monkeypatch, tmp_path):
        """They are set separately, and one must not quietly relocate the other."""
        monkeypatch.setenv(DATABASE_PATH_ENV, str(tmp_path / "a" / "hm.db"))
        monkeypatch.setenv(WORK_DIR_ENV, str(tmp_path / "b"))
        assert Path(default_work_dir()) not in default_db_path().parents
