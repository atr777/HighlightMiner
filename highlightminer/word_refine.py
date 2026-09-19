"""Word timings for the candidate windows only.

The sweep backend (whisper.cpp on Vulkan) is roughly five times faster than
CPU faster-whisper but cannot produce usable word timings alongside flash
attention. It does not need to. Word timings are read in exactly two places,
karaoke captions at export and the timeline strip at review, and both only
ever look inside a candidate window.

So the sweep transcribes thirteen hours without words, and this re-transcribes
the twelve or so minutes that candidates actually occupy, with faster-whisper,
which produces the word timings the captions were built against.

The model is loaded once for the whole batch. Loading the turbo model costs
around twenty-five seconds, which dwarfs a thirty-second window on its own, so
loading per window would cost more than the refinement itself.
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .diagnostics import log_detailed, log_event
from .model_access import ModelAccessPreferences, PreparedModelReference, prepare_model_reference

# Extra audio either side of a window, so a word straddling the boundary is
# transcribed in full. Segments starting outside the window are discarded.
PAD_SECONDS = 3.0

# Windows closer together than this are read as one. Two candidates twenty
# seconds apart cost less as a single pass than as two.
MERGE_GAP_SECONDS = 10.0


@dataclass(frozen=True)
class Window:
    start: float
    end: float

    @property
    def span(self) -> float:
        return max(0.0, self.end - self.start)


def windows_for_candidates(candidates: list[dict], pad: float = 0.0) -> list[Window]:
    """Merged, ordered windows covering every candidate."""
    spans: list[tuple[float, float]] = []
    for candidate in candidates or []:
        try:
            start = float(candidate["start"])
            end = float(candidate["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        spans.append((max(0.0, start - pad), end + pad))
    if not spans:
        return []

    spans.sort()
    merged: list[list[float]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= MERGE_GAP_SECONDS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [Window(round(s, 3), round(e, 3)) for s, e in merged]


def _read_window(wav_path: str | Path, start: float, end: float) -> tuple[np.ndarray, float]:
    """Samples covering [start, end] plus padding, and the true offset read from."""
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV from FFmpeg")
        rate = wf.getframerate()
        channels = wf.getnchannels()
        total = wf.getnframes()

        first = max(0, int((start - PAD_SECONDS) * rate))
        last = min(total, int((end + PAD_SECONDS) * rate))
        if last <= first:
            return np.zeros(0, dtype=np.float32), first / rate
        wf.setpos(first)
        raw = wf.readframes(last - first)

    block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        block = block.reshape(-1, channels).mean(axis=1)
    return block, first / rate


def _load_model(settings: Settings, model_access: ModelAccessPreferences | None, prepared: PreparedModelReference | None):
    from faster_whisper import WhisperModel

    from .runtime import configure_windows_cuda_dll_search
    from .transcribe import _cpu_thread_count, resolve_device

    configure_windows_cuda_dll_search()
    reference = prepared or prepare_model_reference(settings, model_access or ModelAccessPreferences())
    device, compute_type = resolve_device(settings)
    kwargs = {"local_files_only": reference.local_files_only}
    if device == "cpu":
        kwargs["cpu_threads"] = int(settings.cpu_threads) or _cpu_thread_count()
    return WhisperModel(reference.reference, device=device, compute_type=compute_type, **kwargs), device


def refine_windows(
    wav_path: str | Path,
    windows: list[Window],
    settings: Settings,
    *,
    model_access: ModelAccessPreferences | None = None,
    prepared_model: PreparedModelReference | None = None,
    progress=None,
) -> list[dict]:
    """Transcribe each window with word timings, in absolute source time."""
    windows = [w for w in windows if w.span > 0]
    if not windows:
        return []

    from .transcribe import _segment_words

    started_at = time.perf_counter()
    total_span = sum(w.span for w in windows)
    log_event("transcription.refine_start", windows=len(windows), audio_seconds=round(total_span, 1))

    model, device = _load_model(settings, model_access, prepared_model)
    rows: list[dict] = []
    done_span = 0.0

    for index, window in enumerate(windows, start=1):
        if progress is not None:
            progress(
                f"Refining word timings — window {index} of {len(windows)}",
                done_span / total_span if total_span else 0.0,
            )
        samples, offset = _read_window(wav_path, window.start, window.end)
        if samples.size == 0:
            done_span += window.span
            continue

        segments, _ = model.transcribe(
            samples,
            language=settings.language,
            beam_size=settings.beam_size,
            vad_filter=settings.vad_filter,
            word_timestamps=True,
        )
        for segment in segments:
            seg_start = float(getattr(segment, "start", 0.0) or 0.0) + offset
            seg_end = float(getattr(segment, "end", 0.0) or 0.0) + offset
            # Padding exists for context, not for output. Anything beginning
            # outside the window belongs to a neighbouring stretch of the VOD.
            if seg_start < window.start or seg_start > window.end:
                continue
            text = str(getattr(segment, "text", "") or "").strip()
            if not text:
                continue
            rows.append({
                "start": round(seg_start, 3),
                "end": round(max(seg_start, seg_end), 3),
                "text": text,
                "words": _segment_words(segment, seg_start, max(seg_start, seg_end), offset),
            })
        done_span += window.span

    rows.sort(key=lambda r: (r["start"], r["end"]))
    elapsed = time.perf_counter() - started_at
    log_event(
        "transcription.refine_complete",
        windows=len(windows),
        segments=len(rows),
        device=device,
        elapsed_seconds=round(elapsed, 1),
        realtime_factor=round(total_span / elapsed, 2) if elapsed > 0 else None,
    )
    return rows


def merge_refined(transcript: list[dict], refined: list[dict], windows: list[Window]) -> list[dict]:
    """Replace the sweep's wordless rows inside each window with the refined ones.

    Rows outside every window are kept untouched: they still carry the text
    that scoring and the transcript view use, they simply have no word timings,
    and nothing outside a candidate window ever asks for them.
    """
    if not windows:
        return list(transcript or [])

    def inside(when: float) -> bool:
        return any(w.start <= when <= w.end for w in windows)

    kept = [row for row in (transcript or []) if not inside(float(row.get("start", 0.0)))]
    merged = kept + list(refined or [])
    merged.sort(key=lambda r: (float(r.get("start", 0.0)), float(r.get("end", 0.0))))
    log_detailed(
        "transcription.refine_merge",
        replaced=len(transcript or []) - len(kept),
        inserted=len(refined or []),
    )
    return merged
