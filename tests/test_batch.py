from __future__ import annotations

from pathlib import Path

import pytest

from highlightminer import batch as batch_module
from highlightminer.batch import BatchJob, parse_sources, run_batch
from highlightminer.config import Settings
from highlightminer.ingest.base import IngestResult, VodInfo


class TestParseSources:
    def test_passes_through_urls_and_paths(self):
        values = ["https://twitch.tv/videos/1", "C:/vods/a.mp4"]
        assert parse_sources(values) == values

    def test_expands_a_queue_file(self, tmp_path):
        queue = tmp_path / "queue.txt"
        queue.write_text(
            "# tonight's batch\n"
            "https://twitch.tv/videos/1\n"
            "\n"
            "https://kick.com/video/abc12345\n",
            encoding="utf-8",
        )
        assert parse_sources([str(queue)]) == [
            "https://twitch.tv/videos/1",
            "https://kick.com/video/abc12345",
        ]

    def test_ignores_blank_entries(self):
        assert parse_sources(["", "   ", "x"]) == ["x"]

    def test_missing_txt_is_treated_as_a_literal_source(self, tmp_path):
        missing = str(tmp_path / "nope.txt")
        assert parse_sources([missing]) == [missing]


class TestBatchJob:
    def test_url_detection(self):
        assert BatchJob("https://twitch.tv/videos/1").is_url
        assert not BatchJob("C:/vods/a.mp4").is_url

    def test_label_shortens_local_paths(self):
        assert BatchJob("C:/vods/stream.mp4").label == "stream.mp4"
        assert BatchJob("https://twitch.tv/videos/1").label == "https://twitch.tv/videos/1"


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Stub ingest and analysis so batch orchestration can be tested alone."""
    calls: dict[str, list] = {"ingest": [], "analyze": []}
    failures: dict[str, str] = {}

    def fake_ingest(url, video_dir, **kwargs):
        calls["ingest"].append(url)
        if url in failures:
            raise RuntimeError(failures[url])
        video = tmp_path / f"{abs(hash(url))}.mp4"
        video.write_bytes(b"x")
        return IngestResult(
            video_path=video,
            chat_path=None,
            info=VodInfo(platform="twitch", video_id="1", uploader="Streamer"),
            chat_error="no chat replay",
        )

    def fake_analyze(video_path, work_dir, settings, **kwargs):
        calls["analyze"].append((Path(video_path).name, kwargs.get("content_label")))
        if str(video_path) in failures:
            raise RuntimeError(failures[str(video_path)])
        return "analysis-id"

    monkeypatch.setattr(batch_module, "ingest", fake_ingest)
    monkeypatch.setattr(batch_module, "analyze_vod", fake_analyze)
    return calls, failures


class TestRunBatch:
    def test_processes_every_source(self, harness, tmp_path):
        calls, _ = harness
        result = run_batch(
            ["https://twitch.tv/videos/1", "https://twitch.tv/videos/2"],
            tmp_path / "work",
            Settings(),
        )
        assert len(result.succeeded) == 2
        assert len(calls["analyze"]) == 2

    def test_one_failure_does_not_stop_the_run(self, harness, tmp_path):
        """The worst outcome for an overnight job is losing it to one dead URL."""
        calls, failures = harness
        failures["https://twitch.tv/videos/2"] = "gone"
        result = run_batch(
            [
                "https://twitch.tv/videos/1",
                "https://twitch.tv/videos/2",
                "https://twitch.tv/videos/3",
            ],
            tmp_path / "work",
            Settings(),
        )
        assert len(result.succeeded) == 2
        assert len(result.failed) == 1
        assert "gone" in result.failed[0].error

    def test_failure_is_reflected_in_the_summary(self, harness, tmp_path):
        _, failures = harness
        failures["https://twitch.tv/videos/1"] = "boom"
        result = run_batch(["https://twitch.tv/videos/1"], tmp_path / "work", Settings())
        assert result.summary() == "0/1 analyses completed, 1 failed"

    def test_uploader_becomes_the_content_label(self, harness, tmp_path):
        calls, _ = harness
        run_batch(["https://twitch.tv/videos/1"], tmp_path / "work", Settings())
        assert calls["analyze"][0][1] == "Streamer"

    def test_explicit_label_wins_over_the_uploader(self, harness, tmp_path):
        calls, _ = harness
        run_batch(
            ["https://twitch.tv/videos/1"], tmp_path / "work", Settings(), content_label="Pokemon"
        )
        assert calls["analyze"][0][1] == "Pokemon"

    def test_local_files_skip_ingest(self, harness, tmp_path):
        calls, _ = harness
        local = tmp_path / "local.mp4"
        local.write_bytes(b"x")
        result = run_batch([str(local)], tmp_path / "work", Settings())
        assert calls["ingest"] == []
        assert len(result.succeeded) == 1

    def test_missing_local_file_fails_that_job_only(self, harness, tmp_path):
        result = run_batch(
            [str(tmp_path / "ghost.mp4"), str(tmp_path / "ghost2.mp4")],
            tmp_path / "work",
            Settings(),
        )
        assert len(result.failed) == 2
        assert "not a local file" in result.failed[0].error

    def test_chat_note_is_carried_through(self, harness, tmp_path):
        result = run_batch(["https://twitch.tv/videos/1"], tmp_path / "work", Settings())
        assert result.succeeded[0].chat_note == "no chat replay"

    def test_progress_reports_each_job(self, harness, tmp_path):
        seen: list[tuple[str, str]] = []
        run_batch(
            ["https://twitch.tv/videos/1"],
            tmp_path / "work",
            Settings(),
            progress=lambda label, message: seen.append((label, message)),
        )
        assert seen[-1][1] == "done"

    def test_timing_is_recorded(self, harness, tmp_path):
        result = run_batch(["https://twitch.tv/videos/1"], tmp_path / "work", Settings())
        assert result.succeeded[0].seconds >= 0.0

    def test_empty_source_list(self, tmp_path):
        result = run_batch([], tmp_path / "work", Settings())
        assert result.jobs == []
        assert result.summary() == "0/0 analyses completed, 0 failed"
