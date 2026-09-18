"""Shared ingest machinery: video download and the adapter contract.

Adapters turn a platform URL into local files the existing pipeline already
understands. Video always goes through yt-dlp, which covers all three platforms
with one code path; chat is where the per-platform work lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from ..media import find_executable

# Reported as (stage, fraction 0..1, message).
IngestProgress = Callable[[str, float, str], None]


class IngestError(RuntimeError):
    """A VOD or its chat could not be retrieved."""


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
) -> Path:
    """Download a VOD to ``dest_dir`` and return its path.

    Routes HLS through ffmpeg rather than yt-dlp's native downloader. Twitch
    VODs otherwise fail with "Initialization fragment found after media
    fragments", and sending every platform down the same path keeps one code
    path rather than a Twitch special case.
    """
    yt_dlp = _import_yt_dlp()
    out_dir = Path(dest_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[str] = []

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

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            data = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise IngestError(f"Video download failed: {exc}") from exc

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
