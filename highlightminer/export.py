from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .categorization import content_folder_name
from .diagnostics import ffmpeg_failure, log_detailed, log_event, log_exception
from .media import has_encoder, probe_media, require_executable, require_ffmpeg
from .timestamps import ClipBounds, normalize_clip_bounds
from .util import ensure_dir

_PREVIEW_CACHE_KEEP = 4
_PREVIEW_CLEANUP_RETRIES = 3
_PREVIEW_CLEANUP_DELAY_SEC = 0.05
_PREVIEW_GENERATION_LOCK = threading.RLock()


@dataclass(frozen=True)
class PreviewClipResult:
    path: Path
    cleanup_failures: int = 0


class PreviewFileLockError(PermissionError):
    """A temporary preview could not be removed or replaced due to file access."""


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return text[:80] or "highlight"


def _non_overwriting_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not find a free export filename.")


def _retry_unlink(path: Path) -> bool:
    """Best-effort deletion for preview files that may still be held by Windows."""
    for attempt in range(_PREVIEW_CLEANUP_RETRIES):
        try:
            path.unlink(missing_ok=True)
            return True
        except FileNotFoundError:
            return True
        except (PermissionError, OSError):
            if attempt + 1 < _PREVIEW_CLEANUP_RETRIES:
                time.sleep(_PREVIEW_CLEANUP_DELAY_SEC)
    return False


def _prune_preview_files(out_dir: Path, stem: str, *, keep_path: Path) -> int:
    """Prune old previews and return the number that remained after retries."""
    previews: list[Path] = []
    for path in out_dir.glob(f"{stem}_*.mp4"):
        try:
            if path.is_file():
                previews.append(path)
        except OSError:
            continue

    def modified(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    previews.sort(key=modified, reverse=True)
    protected = {keep_path}
    cleanup_failures = 0
    for path in previews:
        if path == keep_path:
            continue
        if len(protected) < _PREVIEW_CACHE_KEEP:
            protected.add(path)
            continue
        if not _retry_unlink(path):
            cleanup_failures += 1
    if cleanup_failures:
        log_event(
            "preview.cleanup_failed",
            level=logging.WARNING,
            failed_files=cleanup_failures,
        )
    return cleanup_failures


def _source_clip_bounds(src: Path, start: float, end: float, *, operation: str, clip_id: str) -> ClipBounds:
    source_duration = float(probe_media(src)["duration"])
    bounds = normalize_clip_bounds(start, end, source_duration)
    if bounds.meaningfully_invalid:
        log_event(
            "clip.bounds_normalized",
            level=logging.WARNING,
            operation=operation,
            clip_id=safe_name(clip_id),
        )
    return bounds


def _run_encode(command: list[str], *, encoder: str) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            shell=False,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as exc:
        ffmpeg_failure("ffmpeg", exc)
        raise
    log_detailed("encoder.complete", encoder=encoder, exit_code=0)


# Hardware encoders are tried in order, then libx264 as the guaranteed fallback.
# ffmpeg advertising an encoder does not mean usable hardware exists behind it:
# a build with NVENC compiled in still lists h264_nvenc on a machine with an AMD
# card, and only fails once it actually tries to open a session. So every entry
# stays inside the same try/except and falls through on failure.
_HARDWARE_ENCODERS: tuple[tuple[str, list[str], list[str]], ...] = (
    (
        "h264_nvenc",
        ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "3M", "-maxrate", "4M", "-bufsize", "8M"],
        ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"],
    ),
    (
        "h264_amf",
        ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp_i", "26", "-qp_p", "26"],
        ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20"],
    ),
    (
        "h264_qsv",
        ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "26"],
        ["-c:v", "h264_qsv", "-preset", "medium", "-global_quality", "20"],
    ),
)

_SOFTWARE_ENCODER = (
    "libx264",
    ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"],
    ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
)

_PREVIEW_SCALE_FILTER = "scale='min(1280,iw)':-2,fps=30"


def _run_h264_encode(
    ffmpeg: str,
    src: Path,
    out: Path,
    start: float,
    duration: float,
    *,
    preview: bool = False,
    video_filters: str | None = None,
    extra_inputs: list[str] | None = None,
) -> None:
    """Encode one clip, preferring hardware acceleration where it actually works.

    ``video_filters`` replaces the default preview downscale when given, which is
    how vertical reframing and burned captions are applied.
    """
    common = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(start):.3f}",
        "-i",
        str(src),
        *(extra_inputs or []),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
    ]

    filters = video_filters if video_filters is not None else (_PREVIEW_SCALE_FILTER if preview else None)
    if filters:
        common += ["-vf", filters]

    audio_bitrate = "128k" if preview else "192k"

    def finish(video_args: list[str]) -> list[str]:
        return [
            *common,
            *video_args,
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-movflags",
            "+faststart",
            str(out),
        ]

    candidates = [
        (name, preview_args if preview else final_args)
        for name, preview_args, final_args in _HARDWARE_ENCODERS
        if has_encoder(name)
    ]
    software_name, software_preview, software_final = _SOFTWARE_ENCODER

    for index, (name, video_args) in enumerate(candidates):
        next_encoder = candidates[index + 1][0] if index + 1 < len(candidates) else software_name
        try:
            log_detailed("encoder.selection", encoder=name, preview=preview)
            _run_encode(finish(video_args), encoder=name)
            return
        except subprocess.CalledProcessError:
            log_event(
                "encoder.fallback",
                level=logging.WARNING,
                from_encoder=name,
                to_encoder=next_encoder,
                preview=preview,
            )
            out.unlink(missing_ok=True)

    video_args = software_preview if preview else software_final
    log_detailed("encoder.selection", encoder=software_name, preview=preview)
    try:
        _run_encode(finish(video_args), encoder=software_name)
    except subprocess.CalledProcessError as exc:
        log_exception("encoder.error", exc, encoder=software_name, preview=preview)
        raise


def create_preview_clip(
    video_path: str | Path,
    output_dir: str | Path,
    clip_id: str,
    start: float,
    end: float,
) -> PreviewClipResult:
    require_ffmpeg()
    ffmpeg = require_executable("ffmpeg")
    src = Path(video_path).expanduser().resolve()
    out_dir = ensure_dir(output_dir)

    bounds = _source_clip_bounds(src, float(start), float(end), operation="preview", clip_id=clip_id)
    start = bounds.start
    end = bounds.end
    duration = end - start

    stem = safe_name(clip_id)
    signature = f"{start:.3f}_{end:.3f}".replace(".", "_")
    out = out_dir / f"{stem}_{signature}.mp4"
    partial = out.with_name(f".{out.stem}.partial{out.suffix}")

    # Streamlit/browser video playback can keep an earlier preview open on
    # Windows. Never delete the currently displayed predecessor before the
    # replacement has been encoded successfully.
    with _PREVIEW_GENERATION_LOCK:
        if out.exists() and out.stat().st_size > 0:
            cleanup_failures = _prune_preview_files(out_dir, stem, keep_path=out)
            return PreviewClipResult(out, cleanup_failures)

        if not _retry_unlink(partial):
            raise PreviewFileLockError("Could not remove an incomplete preview from an earlier attempt.")
        try:
            _run_h264_encode(ffmpeg, src, partial, start, duration, preview=True)
            try:
                partial.replace(out)
            except PermissionError as exc:
                raise PreviewFileLockError("Could not replace the temporary preview file.") from exc
        except Exception:
            _retry_unlink(partial)
            raise
        cleanup_failures = _prune_preview_files(out_dir, stem, keep_path=out)
        return PreviewClipResult(out, cleanup_failures)


def export_clip(
    video_path: str | Path,
    output_dir: str | Path,
    clip_id: str,
    start: float,
    end: float,
    title: str | None = None,
    category: str | None = None,
) -> Path:
    """Export a clip without silently overwriting an older export."""
    require_ffmpeg()
    ffmpeg = require_executable("ffmpeg")
    src = Path(video_path).expanduser().resolve()
    base_dir = ensure_dir(output_dir)
    out_dir = ensure_dir(base_dir / content_folder_name(category))
    bounds = _source_clip_bounds(src, float(start), float(end), operation="export", clip_id=clip_id)
    duration = bounds.end - bounds.start
    # A deliberate title is the filename, not decorative text after an
    # internal candidate ID. Untitled clips retain the stable candidate ID.
    stem = safe_name(title if title else clip_id)
    out = _non_overwriting_path(out_dir / f"{stem}.mp4")

    _run_h264_encode(ffmpeg, src, out, bounds.start, duration, preview=False)
    log_event("export.complete", count=1)
    return out
