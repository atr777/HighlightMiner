"""Disk-space preflight for downloads.

Written after a 13.5 hour source-quality Twitch VOD reached 28 GB on a drive
with 39 GB free, which would have left Windows near zero free space partway
through an unattended run.
"""

from __future__ import annotations

import shutil

import pytest

from highlightminer.ingest import base
from highlightminer.ingest.base import (
    InsufficientDiskSpace,
    check_disk_space,
    estimate_download_bytes,
)

GB = 1024**3


class TestEstimateDownloadBytes:
    def test_prefers_an_explicit_filesize(self):
        assert estimate_download_bytes({"filesize": 1234, "tbr": 9999, "duration": 60}) == 1234

    def test_falls_back_to_the_approximate_filesize(self):
        assert estimate_download_bytes({"filesize_approx": 4321}) == 4321

    def test_computes_from_bitrate_and_duration(self):
        # 8000 kbps for 10s is 10 MB
        assert estimate_download_bytes({"tbr": 8000, "duration": 10}) == 10_000_000

    def test_matches_the_real_measured_vod(self):
        """13.56h at the 6.4 Mbps measured on the actual playlist."""
        estimate = estimate_download_bytes({"tbr": 6370, "duration": 48826})
        assert 35 * GB < estimate < 40 * GB

    def test_sums_split_audio_and_video_formats(self):
        info = {"duration": 100, "requested_formats": [{"tbr": 6000}, {"tbr": 2000}]}
        assert estimate_download_bytes(info) == 100_000_000

    def test_uses_a_conservative_default_without_a_bitrate(self):
        # Not optimistic: guessing low would defeat the whole guard.
        assert estimate_download_bytes({"duration": 100}) == 100_000_000

    def test_returns_none_without_a_duration(self):
        assert estimate_download_bytes({}) is None

    def test_explicit_duration_overrides_metadata(self):
        assert estimate_download_bytes({"tbr": 8000, "duration": 1}, duration=10) == 10_000_000


class TestCheckDiskSpace:
    def _free(self, monkeypatch, free_bytes):
        monkeypatch.setattr(
            base.shutil, "disk_usage",
            lambda _p: shutil._ntuple_diskusage(100 * GB, 100 * GB - free_bytes, free_bytes),
        )

    def test_passes_when_there_is_room(self, monkeypatch, tmp_path):
        self._free(monkeypatch, 100 * GB)
        check_disk_space(tmp_path, 10 * GB)

    def test_raises_when_it_will_not_fit(self, monkeypatch, tmp_path):
        self._free(monkeypatch, 39 * GB)
        with pytest.raises(InsufficientDiskSpace, match="Not enough space"):
            check_disk_space(tmp_path, 39 * GB)

    def test_the_real_case_is_rejected(self, monkeypatch, tmp_path):
        """39 GB download onto 39 GB free must not start."""
        self._free(monkeypatch, 39 * GB)
        with pytest.raises(InsufficientDiskSpace):
            check_disk_space(tmp_path, 39 * GB)

    def test_the_same_download_fits_on_the_big_drive(self, monkeypatch, tmp_path):
        self._free(monkeypatch, 806 * GB)
        check_disk_space(tmp_path, 39 * GB)

    def test_message_reports_both_numbers(self, monkeypatch, tmp_path):
        self._free(monkeypatch, 39 * GB)
        with pytest.raises(InsufficientDiskSpace) as exc:
            check_disk_space(tmp_path, 39 * GB)
        assert "39.0 GB free" in str(exc.value)
        assert "skip_space_check" in str(exc.value)

    def test_unknown_size_is_not_treated_as_fitting(self, monkeypatch, tmp_path):
        """No estimate means cannot check, so do not block the download."""
        self._free(monkeypatch, 1 * GB)
        check_disk_space(tmp_path, None)

    def test_headroom_is_required_beyond_the_download(self, monkeypatch, tmp_path):
        # Exactly the download size is not enough: the analysis WAV and previews
        # still need somewhere to live.
        self._free(monkeypatch, 10 * GB)
        with pytest.raises(InsufficientDiskSpace):
            check_disk_space(tmp_path, 10 * GB)

    def test_walks_up_to_an_existing_parent(self, monkeypatch, tmp_path):
        """The destination usually does not exist yet."""
        self._free(monkeypatch, 806 * GB)
        check_disk_space(tmp_path / "not" / "created" / "yet", 39 * GB)

    def test_unreadable_drive_does_not_block(self, monkeypatch, tmp_path):
        def boom(_p):
            raise OSError("no such drive")

        monkeypatch.setattr(base.shutil, "disk_usage", boom)
        check_disk_space(tmp_path, 39 * GB)


class TestDownloadPreflight:
    def test_download_aborts_before_writing_anything(self, monkeypatch, tmp_path):
        """The whole point: fail before a part-file exists."""
        class FakeYDL:
            def __init__(self, options):
                self.options = options

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download=False):
                assert download is False, "preflight must not download"
                return {"tbr": 6370, "duration": 48826}

        monkeypatch.setattr(base, "_import_yt_dlp", lambda: type("M", (), {"YoutubeDL": FakeYDL}))
        monkeypatch.setattr(
            base.shutil, "disk_usage",
            lambda _p: shutil._ntuple_diskusage(100 * GB, 61 * GB, 39 * GB),
        )
        with pytest.raises(InsufficientDiskSpace):
            base.download_video("https://twitch.tv/videos/1", tmp_path)
        assert list(tmp_path.glob("*.part")) == []

    def test_skip_space_check_bypasses_the_preflight(self, monkeypatch, tmp_path):
        calls = []

        class FakeYDL:
            def __init__(self, options):
                calls.append(options.get("skip_download", False))

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download=False):
                path = tmp_path / "v.mp4"
                path.write_bytes(b"x")
                return {"filepath": str(path), "id": "v", "ext": "mp4", "extractor_key": "T"}

        monkeypatch.setattr(base, "_import_yt_dlp", lambda: type("M", (), {"YoutubeDL": FakeYDL}))
        base.download_video("https://twitch.tv/videos/1", tmp_path, skip_space_check=True)
        # only the real download pass, no metadata-only pass
        assert calls == [False]
