"""Unattended batch ingest and analysis.

Transcription runs at roughly 3x realtime on CPU, so a long VOD is a background
job rather than something to sit through. Batch mode is what makes that
workable: queue several sources at night, review a filled queue later.

Each job is independent. One failure never stops the run, because losing four
queued hours to a single dead URL is the worst possible outcome for an
overnight job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

import psutil

from .config import Settings
from .ingest import IngestError, ingest, is_supported_url
from .pipeline import analyze_vod

BatchProgress = Callable[[str, str], None]

_STATUS_PENDING = "pending"
_STATUS_RUNNING = "running"
_STATUS_DONE = "done"
_STATUS_FAILED = "failed"


@dataclass
class BatchJob:
    """One source to process. Either a URL or an already-local file."""

    source: str
    content_label: str | None = None
    status: str = _STATUS_PENDING
    video_path: Path | None = None
    chat_path: Path | None = None
    analysis_id: str | None = None
    error: str | None = None
    chat_note: str | None = None
    seconds: float = 0.0

    @property
    def is_url(self) -> bool:
        return is_supported_url(self.source)

    @property
    def label(self) -> str:
        return Path(self.source).name if not self.is_url else self.source


@dataclass
class BatchResult:
    jobs: list[BatchJob] = field(default_factory=list)

    @property
    def succeeded(self) -> list[BatchJob]:
        return [j for j in self.jobs if j.status == _STATUS_DONE]

    @property
    def failed(self) -> list[BatchJob]:
        return [j for j in self.jobs if j.status == _STATUS_FAILED]

    def summary(self) -> str:
        total = len(self.jobs)
        ok = len(self.succeeded)
        return f"{ok}/{total} analyses completed, {len(self.failed)} failed"


def parse_sources(values: Iterable[str]) -> list[str]:
    """Accept URLs, local paths, or a text file listing one source per line.

    Blank lines and ``#`` comments are ignored so a queue file can be annotated.
    """
    sources: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        candidate = Path(text)
        if candidate.suffix.lower() == ".txt" and candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    sources.append(line)
            continue
        sources.append(text)
    return sources


def run_batch(
    sources: Iterable[str],
    work_root: str | Path,
    settings: Settings,
    *,
    db_path: str | Path | None = None,
    video_dir: str | Path | None = None,
    content_label: str | None = None,
    allow_model_download: bool = True,
    cpu_threads: int | None = None,
    progress: BatchProgress | None = None,
) -> BatchResult:
    """Ingest and analyze every source, continuing past individual failures."""
    # Interactive analysis reserves a core so the desktop stays usable. An
    # unattended batch has no desktop to protect, and measured 24% faster
    # transcription with all logical cores minus one (103.5s against 128.3s
    # on a 4 minute sample).
    if cpu_threads is None:
        logical = psutil.cpu_count(logical=True) or 0
        cpu_threads = max(1, logical - 1) if logical > 1 else 0
    if cpu_threads:
        settings = replace(settings, cpu_threads=int(cpu_threads))
    work_root = Path(work_root).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    videos = Path(video_dir).expanduser().resolve() if video_dir else work_root / "vods"

    jobs = [BatchJob(source=s, content_label=content_label) for s in parse_sources(sources)]
    result = BatchResult(jobs=jobs)

    def report(job: BatchJob, message: str) -> None:
        if progress:
            progress(job.label, message)

    for index, job in enumerate(jobs, start=1):
        job.status = _STATUS_RUNNING
        started = time.perf_counter()
        report(job, f"[{index}/{len(jobs)}] starting")
        try:
            if job.is_url:
                report(job, "downloading")
                ingested = ingest(job.source, videos)
                job.video_path = ingested.video_path
                job.chat_path = ingested.chat_path
                job.chat_note = ingested.chat_error
                if not job.content_label:
                    job.content_label = ingested.info.uploader or None
            else:
                local = Path(job.source).expanduser()
                if not local.is_file():
                    raise IngestError(f"Not a URL and not a local file: {job.source}")
                job.video_path = local.resolve()

            report(job, "analyzing")
            work_dir = work_root / job.video_path.stem
            analysis_id = analyze_vod(
                job.video_path,
                work_dir,
                settings,
                chat_path=job.chat_path,
                db_path=db_path,
                content_label=job.content_label,
                allow_model_download=allow_model_download,
            )
            job.analysis_id = str(analysis_id)
            job.status = _STATUS_DONE
            report(job, "done")
        except Exception as exc:  # noqa: BLE001 - one bad source must not end the run
            job.status = _STATUS_FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            report(job, f"failed: {job.error}")
        finally:
            job.seconds = round(time.perf_counter() - started, 1)

    return result
