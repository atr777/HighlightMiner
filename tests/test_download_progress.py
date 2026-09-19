"""Download progress, measured from the part file on disk.

Nothing else can report it. HLS is routed through ffmpeg as an external
downloader, which bypasses yt-dlp's progress hooks entirely, and ffmpeg is run
with its output suppressed. A twelve gigabyte Twitch VOD showed nothing at all
between "estimated download 12.5 GB" and "finished".

The file on disk knows, so that is what gets watched.
"""

from __future__ import annotations

import threading
import time

import pytest

from highlightminer.ingest.base import (
    _format_bytes,
    _format_eta,
    _format_rate,
    _newest_part_file,
    _watch_part_file,
)

GB = 1024**3
MB = 1024**2


class TestFormatting:
    @pytest.mark.parametrize(
        "value,expected",
        [(12_500 * MB, "12.2 GB"), (500 * MB, "500 MB"), (0, "0 MB")],
    )
    def test_sizes(self, value, expected):
        assert _format_bytes(value) == expected

    def test_rate(self):
        assert _format_rate(18 * MB) == "18.0 MB/s"

    @pytest.mark.parametrize(
        "seconds,expected",
        [(45, "45s left"), (330, "6 min left"), (3600 * 2, "2.0 h left")],
    )
    def test_eta(self, seconds, expected):
        assert _format_eta(seconds) == expected

    def test_eta_crosses_to_minutes_rather_than_showing_120s(self):
        assert "min" in _format_eta(100)


class TestFindingThePartFile:
    def test_finds_it(self, tmp_path):
        part = tmp_path / "TwitchVod-v1.mp4.part"
        part.write_bytes(b"x")
        assert _newest_part_file(tmp_path) == part

    def test_nothing_to_find(self, tmp_path):
        assert _newest_part_file(tmp_path) is None

    def test_ignores_the_finished_file(self, tmp_path):
        (tmp_path / "done.mp4").write_bytes(b"x")
        assert _newest_part_file(tmp_path) is None

    def test_picks_the_most_recent_of_several(self, tmp_path):
        old = tmp_path / "old.mp4.part"
        old.write_bytes(b"x")
        time.sleep(0.01)
        new = tmp_path / "new.mp4.part"
        new.write_bytes(b"x")
        assert _newest_part_file(tmp_path) == new

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert _newest_part_file(tmp_path / "nope") is None


def _run_watcher(out_dir, estimated, writes, interval=0.02):
    """Drive the watcher against a part file that grows, and collect reports."""
    reports: list[tuple[str, float, str]] = []
    stop = threading.Event()
    thread = threading.Thread(
        target=_watch_part_file,
        args=(out_dir, estimated, lambda *a: reports.append(a), stop),
        daemon=True,
    )
    thread.start()
    part = out_dir / "TwitchVod-v1.mp4.part"
    for size in writes:
        part.write_bytes(b"\0" * size)
        time.sleep(interval)
    stop.set()
    thread.join(timeout=5)
    return reports


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch):
    """The real interval is tuned for a download, not for a test."""
    monkeypatch.setattr("highlightminer.ingest.base._WATCH_INTERVAL_SEC", 0.01)


class TestWatching:
    def test_reports_bytes_written(self, tmp_path):
        reports = _run_watcher(tmp_path, None, [1000, 2000])
        assert reports, "the watcher reported nothing at all"
        assert any("Downloading" in message for _, _, message in reports)

    def test_reports_a_percentage_against_the_estimate(self, tmp_path):
        reports = _run_watcher(tmp_path, 1000, [500])
        assert any("50%" in message for _, _, message in reports)

    def test_the_fraction_drives_the_progress_bar(self, tmp_path):
        reports = _run_watcher(tmp_path, 1000, [500])
        assert any(fraction == pytest.approx(0.5) for _, fraction, _ in reports)

    def test_the_stage_is_always_download(self, tmp_path):
        reports = _run_watcher(tmp_path, 1000, [500])
        assert {stage for stage, _, _ in reports} == {"download"}

    def test_never_claims_completion_before_the_download_says_so(self, tmp_path):
        """The estimate can undershoot; a bar that sits at 100% is a lie."""
        reports = _run_watcher(tmp_path, 1000, [5000])
        assert all(fraction < 1.0 for _, fraction, _ in reports)

    def test_without_an_estimate_it_still_reports_bytes(self, tmp_path):
        reports = _run_watcher(tmp_path, None, [4096])
        assert reports
        assert not any("%" in message for _, _, message in reports)

    def test_waits_quietly_until_the_part_file_appears(self, tmp_path):
        reports = []
        stop = threading.Event()
        thread = threading.Thread(
            target=_watch_part_file,
            args=(tmp_path, 1000, lambda *a: reports.append(a), stop),
            daemon=True,
        )
        thread.start()
        time.sleep(0.05)
        stop.set()
        thread.join(timeout=5)
        assert reports == []

    def test_stops_when_asked(self, tmp_path):
        (tmp_path / "a.mp4.part").write_bytes(b"x")
        stop = threading.Event()
        thread = threading.Thread(
            target=_watch_part_file, args=(tmp_path, 100, lambda *a: None, stop), daemon=True
        )
        thread.start()
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_a_failing_callback_does_not_kill_the_download(self, tmp_path):
        """The UI is not allowed to take the download down with it."""
        (tmp_path / "a.mp4.part").write_bytes(b"x" * 100)
        stop = threading.Event()

        def explode(*args):
            raise RuntimeError("no script run context")

        thread = threading.Thread(
            target=_watch_part_file, args=(tmp_path, 1000, explode, stop), daemon=True
        )
        thread.start()
        time.sleep(0.05)
        assert thread.is_alive()
        stop.set()
        thread.join(timeout=5)

    def test_reports_a_rate_and_an_eta_once_there_is_enough_history(self, tmp_path):
        reports = _run_watcher(tmp_path, 10_000, [1000 * i for i in range(1, 8)], interval=0.4)
        messages = " ".join(m for _, _, m in reports)
        assert "MB/s" in messages
        assert "left" in messages


class TestWiring:
    def test_download_video_accepts_a_thread_hook(self):
        """Streamlit drops writes from threads it has not been told about."""
        import inspect

        from highlightminer.ingest.base import download_video

        assert "thread_hook" in inspect.signature(download_video).parameters

    def test_the_router_passes_it_through(self):
        import inspect

        from highlightminer.ingest.router import ingest

        assert "thread_hook" in inspect.signature(ingest).parameters
