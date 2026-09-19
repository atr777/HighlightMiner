"""Shared ingest machinery: video download and the adapter contract.

Adapters turn a platform URL into local files the existing pipeline already
understands. Video always goes through yt-dlp, which covers all three platforms
with one code path; chat is where the per-platform work lives.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from ..media import find_executable

# Reported as (stage, fraction 0..1, message).
IngestProgress = Callable[[str, float, str], None]


class IngestError(RuntimeError):
    """A VOD or its chat could not be retrieved."""


class InsufficientDiskSpace(IngestError):
    """The estimated download will not fit on the target drive.

    Worth failing fast and loudly: a 13.5 hour source-quality Twitch VOD is
    about 39 GB, and discovering that at 3am partway through an unattended
    batch leaves a part-file occupying the disk and Windows near zero free
    space, which is worse than not starting.
    """


class ChatUnavailable(IngestError):
    """The platform has no chat replay for this VOD, or we cannot reach it.

    Distinct from IngestError because it is recoverable: analysis continues on
    audio and transcript alone, with the chat weight renormalized away.
    """


@dataclass
class VodInfo:
    platform: str
    video_id: str
    title: str = ""
    uploader: str = ""
    duration: float = 0.0
    was_live: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestResult:
    video_path: Path
    chat_path: Path | None
    info: VodInfo
    chat_error: str | None = None


class VodAdapter(Protocol):
    platform: str

    def matches(self, url: str) -> bool: ...
    def video_id(self, url: str) -> str: ...
    def fetch_chat(self, url: str, out_path: str | Path) -> Path: ...


def _import_yt_dlp():
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise IngestError(
            "yt-dlp is required for URL ingest. Install it with: pip install yt-dlp"
        ) from exc
    return yt_dlp


def _ffmpeg_dir() -> str | None:
    ffmpeg = find_executable("ffmpeg")
    return str(Path(ffmpeg).parent) if ffmpeg else None



# Twitch source ("chunked") 1080p60 measures around 6.4 Mbps. Used only when the
# metadata carries no bitrate at all, and deliberately not optimistic.
_FALLBACK_BITRATE_KBPS = 8000.0

# Keep this much free beyond the download itself, for the analysis WAV, previews
# and the operating system.
_DISK_HEADROOM_BYTES = 5 * 1024**3
_SIZE_SAFETY_FACTOR = 1.15


def _format_bytes(value: float) -> str:
    gb = value / 1024**3
    return f"{gb:.1f} GB" if gb >= 1 else f"{value / 1024**2:.0f} MB"


def _format_rate(bytes_per_second: float) -> str:
    return f"{bytes_per_second / 1024**2:.1f} MB/s"


def _format_eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s left"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min left"
    return f"{minutes / 60:.1f} h left"


# How often the part file is measured. Often enough to feel live, rarely
# enough that it costs nothing next to an 18 MB/s download.
_WATCH_INTERVAL_SEC = 1.5

# Rate is averaged over this long, so a momentary stall does not read as a
# finished download and a burst does not promise an ETA it cannot keep.
_RATE_WINDOW_SEC = 10.0


def _newest_part_file(out_dir: Path) -> Path | None:
    """The part file ffmpeg is currently writing, if there is one."""
    try:
        parts = [p for p in out_dir.glob("*.part") if p.is_file()]
    except OSError:
        return None
    if not parts:
        return None
    return max(parts, key=lambda p: p.stat().st_mtime)


def _watch_part_file(
    out_dir: Path,
    estimated_bytes: int | None,
    progress: IngestProgress,
    stop: threading.Event,
) -> None:
    """Report download progress by watching the part file grow.

    yt-dlp's progress hooks do not fire here. HLS is routed through ffmpeg as
    an external downloader, which bypasses them, and ffmpeg itself is run with
    output suppressed. That left a twelve gigabyte download showing nothing
    between "estimated size" and "finished".

    Measuring the file on disk needs neither, and works the same for every
    platform because they all take the same ffmpeg path.
    """
    samples: deque[tuple[float, int]] = deque()
    while not stop.is_set():
        part = _newest_part_file(out_dir)
        if part is not None:
            try:
                done = part.stat().st_size
            except OSError:
                done = 0
            now = time.monotonic()
            samples.append((now, done))
            while len(samples) > 2 and now - samples[0][0] > _RATE_WINDOW_SEC:
                samples.popleft()

            fraction = min(0.999, done / estimated_bytes) if estimated_bytes else 0.0
            message = f"Downloading {_format_bytes(done)}"
            if estimated_bytes:
                message += f" of {_format_bytes(estimated_bytes)} ({fraction * 100:.0f}%)"

            elapsed = now - samples[0][0]
            gained = done - samples[0][1]
            if elapsed >= 2.0 and gained > 0:
                rate = gained / elapsed
                message += f" · {_format_rate(rate)}"
                if estimated_bytes and done < estimated_bytes:
                    message += f" · {_format_eta((estimated_bytes - done) / rate)}"

            try:
                progress("download", fraction, message)
            except Exception:
                # A failing UI callback must not take the download down.
                pass
        stop.wait(_WATCH_INTERVAL_SEC)


def estimate_download_bytes(info: dict, duration: float | None = None) -> int | None:
    """Best estimate of a download's size from yt-dlp metadata.

    Prefers an explicit filesize, then bitrate times duration. Returns None when
    there is nothing to go on, which callers treat as "cannot check" rather than
    "fits".
    """
    for key in ("filesize", "filesize_approx"):
        value = info.get(key)
        if value:
            return int(value)

    seconds = float(duration or info.get("duration") or 0.0)
    if seconds <= 0:
        return None

    bitrate = info.get("tbr") or info.get("vbr")
    if not bitrate:
        formats = info.get("requested_formats") or []
        bitrate = sum(float(f.get("tbr") or 0.0) for f in formats) or None
    if not bitrate:
        bitrate = _FALLBACK_BITRATE_KBPS
    return int(float(bitrate) * 1000.0 / 8.0 * seconds)


def check_disk_space(
    dest_dir: str | Path,
    estimated_bytes: int | None,
    *,
    headroom_bytes: int = _DISK_HEADROOM_BYTES,
) -> None:
    """Raise if an estimated download plus headroom will not fit."""
    if not estimated_bytes:
        return
    target = Path(dest_dir).expanduser().resolve()
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return

    needed = int(estimated_bytes * _SIZE_SAFETY_FACTOR) + headroom_bytes
    if free < needed:
        raise InsufficientDiskSpace(
            f"Not enough space on {probe.drive or probe}: "
            f"{_format_bytes(free)} free, need about {_format_bytes(needed)} "
            f"(estimated download {_format_bytes(estimated_bytes)} plus "
            f"{_format_bytes(headroom_bytes)} working headroom). "
            "Choose another drive, or pass skip_space_check to override."
        )


def probe_url(url: str) -> VodInfo:
    """Read VOD metadata without downloading anything."""
    yt_dlp = _import_yt_dlp()
    options = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            data = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise IngestError(f"Could not read VOD metadata: {exc}") from exc
    if not isinstance(data, dict):
        raise IngestError("Unexpected metadata response.")

    return VodInfo(
        platform=str(data.get("extractor_key", "") or "").lower(),
        video_id=str(data.get("id", "") or ""),
        title=str(data.get("title", "") or ""),
        uploader=str(data.get("uploader", "") or ""),
        duration=float(data.get("duration") or 0.0),
        was_live=bool(data.get("was_live")),
        raw=data,
    )


def download_video(
    url: str,
    dest_dir: str | Path,
    *,
    max_height: int = 1080,
    progress: IngestProgress | None = None,
    skip_space_check: bool = False,
    thread_hook: Callable[[threading.Thread], Any] | None = None,
) -> Path:
    """Download a VOD to ``dest_dir`` and return its path.

    Routes HLS through ffmpeg rather than yt-dlp's native downloader. Twitch
    VODs otherwise fail with "Initialization fragment found after media
    fragments", and sending every platform down the same path keeps one code
    path rather than a Twitch special case.

    ``thread_hook`` is handed the progress watcher thread before it starts, so
    a UI framework that needs threads registered can do so. Streamlit drops
    writes from threads it does not know about.
    """
    yt_dlp = _import_yt_dlp()
    out_dir = Path(dest_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[str] = []
    estimated_bytes: int | None = None

    def hook(status: dict) -> None:
        if status.get("status") == "finished" and status.get("filename"):
            downloaded.append(status["filename"])
        if progress and status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            done = status.get("downloaded_bytes") or 0
            fraction = (done / total) if total else 0.0
            progress("download", min(1.0, fraction), "Downloading VOD")

    options = {
        "quiet": True,
        "no_warnings": True,
        "format": f"best[height<={int(max_height)}]/best",
        # extractor_key, not extractor: the latter is "twitch:vod", and the
        # colon gets sanitized into a fullwidth U+FF1A in the filename.
        "outtmpl": str(out_dir / "%(extractor_key)s-%(id)s.%(ext)s"),
        # Both are needed: hls_prefer_native turns the native HLS downloader
        # off, external_downloader names ffmpeg as the replacement.
        "hls_prefer_native": False,
        "external_downloader": {"default": "ffmpeg"},
        "progress_hooks": [hook],
        "noprogress": True,
    }
    ffmpeg_dir = _ffmpeg_dir()
    if ffmpeg_dir:
        options["ffmpeg_location"] = ffmpeg_dir

    if not skip_space_check:
        # Metadata-only pass first, so an impossible download fails before it
        # has written 28 GB of part-file to a drive that cannot hold it.
        try:
            with yt_dlp.YoutubeDL({**options, "skip_download": True}) as ydl:
                preflight = ydl.extract_info(url, download=False)
        except Exception as exc:
            raise IngestError(f"Could not read VOD metadata: {exc}") from exc
        if isinstance(preflight, dict):
            estimated = estimate_download_bytes(preflight)
            estimated_bytes = estimated
            if progress and estimated:
                progress(
                    "preflight", 0.0,
                    f"Estimated download {_format_bytes(estimated)}",
                )
            check_disk_space(out_dir, estimated)

    stop_watching = threading.Event()
    watcher: threading.Thread | None = None
    if progress is not None:
        watcher = threading.Thread(
            target=_watch_part_file,
            args=(out_dir, estimated_bytes, progress, stop_watching),
            name="highlightminer-download-progress",
            daemon=True,
        )
        if thread_hook is not None:
            thread_hook(watcher)
        watcher.start()

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            data = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise IngestError(f"Video download failed: {exc}") from exc
    finally:
        stop_watching.set()
        if watcher is not None:
            watcher.join(timeout=_WATCH_INTERVAL_SEC * 2)

    # yt-dlp reports the final path through the hook; fall back to its own
    # filename prediction when ffmpeg downloading bypassed the hook.
    if downloaded:
        path = Path(downloaded[-1])
        if path.exists():
            return path
    if isinstance(data, dict):
        for key in ("filepath", "_filename"):
            value = data.get(key)
            if value and Path(value).exists():
                return Path(value)
        predicted = out_dir / f"{data.get('extractor_key')}-{data.get('id')}.{data.get('ext', 'mp4')}"
        if predicted.exists():
            return predicted

    raise IngestError("Video download reported success but no file was found.")
