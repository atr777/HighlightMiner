from __future__ import annotations

import logging
import math
import re
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .config import Settings
from .diagnostics import log_detailed, log_event, safe_model_name
from .model_access import ModelAccessPreferences, PreparedModelReference, prepare_model_reference
from .runtime import configure_windows_cuda_dll_search
from .transcript_checkpoint import open_checkpoint
from .transcription_status import TRANSCRIPTION_AVAILABLE
from .util import clamp

# Public faster-whisper/CTranslate2 API usage is based on upstream documentation.
# No faster-whisper source code is vendored here; see ATTRIBUTIONS.md.

_LAUGH_RE = re.compile(r"\b(?:ha(?:ha)+|he(?:he)+|lol|lmao|rofl)\b", re.IGNORECASE)
_PROFANITY = {
    "fuck", "fucking", "shit", "damn", "bitch", "asshole",
    "vittu", "saatana", "perkele", "jumalauta", "helvetti",
}
_REACTION_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_REACTION_FIRST_OCCURRENCE_SCORE = 0.49
_REACTION_REPEAT_LOG_BONUS = 0.10
_REACTION_SCORE_CAP = 0.70

TranscriptionProgress = Callable[[str, float], None]
_PROGRESS_REPORT_INTERVAL_SECONDS = 1.0
_PROGRESS_REPORT_FRACTION_DELTA = 0.005
_CPU_THREAD_FALLBACK_CAP = 4


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and math.isfinite(parsed) else None


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _compute_label(compute_type: str) -> str:
    labels = {
        "float16": "FP16",
        "float32": "FP32",
        "int8": "INT8",
        "int8_float16": "INT8/FP16",
        "int8_float32": "INT8/FP32",
        "int16": "INT16",
    }
    return labels.get(compute_type, compute_type.upper())


def _cpu_thread_count() -> int:
    """Reserve one physical core for the desktop; use a capped logical fallback."""
    physical = psutil.cpu_count(logical=False)
    if physical is not None and physical > 0:
        return max(1, int(physical) - 1)

    logical = psutil.cpu_count(logical=True)
    if logical is None or logical <= 1:
        return 1
    return max(1, min(_CPU_THREAD_FALLBACK_CAP, int(logical) - 1))


def _runtime_label(
    device: str,
    compute_type: str,
    model_name: str,
    cpu_threads: int | None = None,
) -> str:
    if device == "cuda":
        return f"GPU (CUDA · {_compute_label(compute_type)} · {model_name})"
    if cpu_threads is not None:
        unit = "thread" if cpu_threads == 1 else "threads"
        return f"CPU ({_compute_label(compute_type)} · {model_name} · {cpu_threads} {unit})"
    return f"CPU ({_compute_label(compute_type)} · {model_name})"


def resolve_device(settings: Settings) -> tuple[str, str]:
    if settings.device != "auto":
        compute = settings.compute_type
        if compute == "auto":
            compute = "float16" if settings.device == "cuda" else "int8"
        return settings.device, compute

    try:
        configure_windows_cuda_dll_search()
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16" if settings.compute_type == "auto" else settings.compute_type
    except Exception as exc:
        log_detailed("model.device_probe", decision="cpu", probe_error_type=type(exc).__name__)
    return "cpu", "int8" if settings.compute_type == "auto" else settings.compute_type


def _reaction_tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in _REACTION_TOKEN_RE.finditer(text))


def count_reaction_phrase_occurrences(text: str, reaction_phrases: list[str]) -> int:
    """Count configured phrase sequences using punctuation-insensitive tokens.

    Each unique configured phrase is counted independently and matches may
    overlap. This makes a deliberate multi-word phrase exact while allowing a
    sequence such as ``ha ha`` to occur twice in ``ha ha ha``. Duplicate
    configuration entries do not multiply the score.
    """
    text_tokens = _reaction_tokens(text)
    configured = {_reaction_tokens(phrase) for phrase in reaction_phrases if phrase.strip()}
    configured.discard(())

    occurrences = 0
    for phrase_tokens in configured:
        width = len(phrase_tokens)
        occurrences += sum(
            text_tokens[index:index + width] == phrase_tokens
            for index in range(len(text_tokens) - width + 1)
        )
    return occurrences


def _reaction_occurrence_score(occurrences: int) -> float:
    if occurrences <= 0:
        return 0.0
    return min(
        _REACTION_SCORE_CAP,
        _REACTION_FIRST_OCCURRENCE_SCORE
        + _REACTION_REPEAT_LOG_BONUS * math.log2(occurrences),
    )


def score_text(text: str, reaction_phrases: list[str]) -> tuple[float, list[str]]:
    raw = text.strip()
    lower = raw.lower()
    reasons: list[str] = []
    score = 0.0

    reaction_occurrences = count_reaction_phrase_occurrences(raw, reaction_phrases)
    if reaction_occurrences:
        score += _reaction_occurrence_score(reaction_occurrences)
        reasons.append("reaction phrase")

    if _LAUGH_RE.search(lower):
        score += 0.46
        reasons.append("laughter")

    words = re.findall(r"[\w']+", raw, flags=re.UNICODE)
    profanity_hits = sum(1 for w in words if w.lower() in _PROFANITY)
    if profanity_hits:
        score += min(0.28, 0.12 * profanity_hits)
        reasons.append("strong reaction")

    exclamations = raw.count("!")
    questions = raw.count("?")
    if exclamations:
        score += min(0.18, exclamations * 0.06)
        reasons.append("exclamation")
    if questions >= 2:
        score += 0.10
        reasons.append("surprise/questioning")

    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    caps = [w for w in alpha_words if len(w) >= 3 and w.isupper()]
    if caps:
        score += min(0.14, 0.05 * len(caps))
        reasons.append("raised/emphatic wording")

    if 1 <= len(words) <= 7 and (exclamations or profanity_hits or caps):
        score += 0.08

    return clamp(score), reasons


def _segment_words(
    segment: object,
    seg_start: float,
    seg_end: float,
    offset: float = 0.0,
) -> list[dict]:
    """Extract per-word timings, when the model was asked for them.

    Captions need word timing, and tighter clip boundaries fall out of it for
    free. Whisper occasionally emits a word whose timings are missing or run
    backwards, so each one is clamped into its own segment rather than trusted.
    """
    words = getattr(segment, "words", None) or []
    out: list[dict] = []
    for word in words:
        text = getattr(word, "word", None)
        text = "" if text is None else str(text).strip()
        if not text:
            continue
        start = _safe_float(getattr(word, "start", None))
        end = _safe_float(getattr(word, "end", None))
        if start is None or end is None:
            continue
        # Word times are relative to the chunk the model saw.
        start += offset
        end += offset
        start = min(max(start, seg_start), seg_end)
        end = min(max(end, start), seg_end)
        out.append({"w": text, "s": round(start, 3), "e": round(end, 3)})
    return out



# Each chunk is fed this many extra seconds past the region it owns, so speech
# straddling a boundary is transcribed in full rather than truncated. Measured
# without it: transcript gaps landing exactly on every boundary.
CHUNK_OVERLAP_SEC = 15.0


def _iter_audio_chunks(wav_path: str | Path, chunk_sec: float, overlap_sec: float = CHUNK_OVERLAP_SEC):
    """Yield (offset_seconds, samples, owned_end_seconds) without loading the file.

    Chunk *i* owns ``[i*chunk, (i+1)*chunk)`` but is handed ``overlap_sec`` of
    extra audio past that. A segment beginning inside the owned region is
    therefore complete even when it runs past the boundary, and the caller keeps
    only segments that begin in the owned region, so nothing is duplicated.

    faster-whisper accepts a numpy array directly, so chunks never touch disk.
    """
    with wave.open(str(wav_path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV from FFmpeg")
        rate = wf.getframerate()
        channels = wf.getnchannels()
        total_frames = wf.getnframes()
        step = max(1, int(chunk_sec * rate))
        overlap = max(0, int(overlap_sec * rate))

        position = 0
        while position < total_frames:
            wf.setpos(position)
            wanted = min(step + overlap, total_frames - position)
            raw = wf.readframes(wanted)
            if not raw:
                return
            block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                block = block.reshape(-1, channels).mean(axis=1)
            owned_end = min(position + step, total_frames) / rate
            yield position / rate, block, owned_end
            position += step


def transcribe_audio(
    audio_path: str | Path,
    settings: Settings,
    model_access: ModelAccessPreferences | None = None,
    prepared_model: PreparedModelReference | None = None,
    *,
    audio_duration: float | None = None,
    progress: TranscriptionProgress | None = None,
    work_dir: str | Path | None = None,
) -> tuple[list[dict], dict]:
    configure_windows_cuda_dll_search()
    from faster_whisper import WhisperModel

    started_at = time.perf_counter()
    prepared = prepared_model or prepare_model_reference(
        settings,
        model_access or ModelAccessPreferences(),
    )
    device, compute_type = resolve_device(settings)
    fallback_reason = None
    # 0 means auto, which reserves a core for the desktop. An unattended
    # batch has no desktop to protect and can take the whole machine.
    if device == "cpu":
        cpu_threads = int(settings.cpu_threads) or _cpu_thread_count()
    else:
        cpu_threads = None
    model_kwargs: dict[str, Any] = {
        "local_files_only": prepared.local_files_only,
    }
    if cpu_threads is not None:
        model_kwargs["cpu_threads"] = cpu_threads

    model_name = safe_model_name(prepared.display_name, prepared.source)
    log_event(
        "model.load_start",
        model_name=model_name,
        model_source=prepared.source,
        device=device,
        compute_type=compute_type,
        fallback=False,
    )
    log_detailed(
        "model.load_options",
        model_name=model_name,
        model_source=prepared.source,
        local_files_only=prepared.local_files_only,
        cpu_threads=cpu_threads,
    )

    def report(message: str, fraction: float) -> None:
        if progress is not None:
            progress(message, clamp(fraction))

    report(
        f"Loading faster-whisper model — "
        f"{_runtime_label(device, compute_type, prepared.display_name, cpu_threads)} · "
        f"elapsed {_format_elapsed(time.perf_counter() - started_at)}",
        0.0,
    )
    try:
        model = WhisperModel(prepared.reference, device=device, compute_type=compute_type, **model_kwargs)
    except Exception as exc:
        if device != "cuda":
            raise
        fallback_reason = f"CUDA initialization failed: {type(exc).__name__}: {exc}"
        log_event(
            "model.fallback",
            level=logging.WARNING,
            model_name=model_name,
            model_source=prepared.source,
            from_device="cuda",
            to_device="cpu",
            reason_type=type(exc).__name__,
        )
        device, compute_type = "cpu", "int8"
        cpu_threads = _cpu_thread_count()
        model_kwargs["cpu_threads"] = cpu_threads
        report(
            f"CUDA initialization failed; retrying on "
            f"{_runtime_label(device, compute_type, prepared.display_name, cpu_threads)} · "
            f"elapsed {_format_elapsed(time.perf_counter() - started_at)}",
            0.0,
        )
        model = WhisperModel(prepared.reference, device=device, compute_type=compute_type, **model_kwargs)

    log_event(
        "model.load_complete",
        model_name=model_name,
        model_source=prepared.source,
        device=device,
        compute_type=compute_type,
        fallback=bool(fallback_reason),
    )

    kwargs = {
        "beam_size": int(settings.beam_size),
        "vad_filter": bool(settings.vad_filter),
        "word_timestamps": bool(settings.word_timestamps),
    }
    if settings.language:
        kwargs["language"] = settings.language

    duration_hint = max(0.0, float(audio_duration or 0.0))
    chunk_sec = float(getattr(settings, "transcribe_chunk_sec", 0.0) or 0.0)
    # Chunk only when the audio is known to be longer than one chunk. Short
    # audio and an unknown duration both take the original single-pass path, so
    # nothing changes for the common case and chunk boundaries are never
    # introduced needlessly.
    if chunk_sec > 0 and duration_hint <= chunk_sec:
        chunk_sec = 0.0
    if chunk_sec > 0:
        log_detailed("transcription.chunked", chunk_seconds=chunk_sec)

    checkpoint = open_checkpoint(work_dir, settings, audio_path, chunk_sec, CHUNK_OVERLAP_SEC)
    recovered = checkpoint.load() if checkpoint else 0
    if recovered:
        log_event("transcription.resumed", chunks_recovered=recovered)

    info = None
    rows: list[dict] = []
    duration = max(0.0, float(audio_duration or 0.0))
    last_report_at = started_at
    last_fraction = -1.0
    furthest_audio_second = 0.0

    def report_transcription(*, force: bool = False) -> None:
        nonlocal last_report_at, last_fraction
        now = time.perf_counter()
        fraction = clamp(furthest_audio_second / duration) if duration > 0 else 0.0
        if (
            not force
            and (now - last_report_at) < _PROGRESS_REPORT_INTERVAL_SECONDS
            and (fraction - last_fraction) < _PROGRESS_REPORT_FRACTION_DELTA
        ):
            return
        audio_text = ""
        if duration > 0:
            audio_text = f" · {_format_elapsed(furthest_audio_second)} / {_format_elapsed(duration)} audio"
        resumed = f" · resumed {recovered} chunks" if recovered else ""
        report(
            f"Transcribing — {_runtime_label(device, compute_type, prepared.display_name, cpu_threads)}"
            f"{audio_text}{resumed} · elapsed {_format_elapsed(now - started_at)}",
            fraction,
        )
        last_report_at = now
        last_fraction = fraction

    def build_row(seg: object, offset: float) -> dict | None:
        """Normalise one segment to absolute time, or None if unusable."""
        start = _safe_float(getattr(seg, "start", None))
        end = _safe_float(getattr(seg, "end", None))
        raw_text = getattr(seg, "text", "")
        text = "" if raw_text is None else str(raw_text).strip()
        if start is None or end is None or not text:
            return None
        start = max(0.0, start + offset)
        end = max(start, end + offset)
        return {
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "words": _segment_words(seg, start, end, offset),
        }

    report_transcription(force=True)

    if chunk_sec <= 0:
        segments, info = model.transcribe(str(audio_path), **kwargs)
        for seg in segments:
            end = _safe_float(getattr(seg, "end", None))
            if end is not None:
                furthest_audio_second = max(furthest_audio_second, max(0.0, end))
                report_transcription()
            row = build_row(seg, 0.0)
            if row:
                rows.append(row)
    else:
        for index, (offset, samples, owned_end) in enumerate(
            _iter_audio_chunks(audio_path, chunk_sec)
        ):
            cached = checkpoint.rows_for(index) if checkpoint else None
            if cached is not None:
                rows.extend(cached)
                furthest_audio_second = max(furthest_audio_second, owned_end)
                report_transcription()
                continue

            segments, chunk_info = model.transcribe(samples, **kwargs)
            if info is None:
                info = chunk_info

            chunk_rows: list[dict] = []
            for seg in segments:
                start = _safe_float(getattr(seg, "start", None))
                # Segments starting in the overlap tail belong to the next
                # chunk, which sees more of their audio.
                if start is not None and (offset + start) >= owned_end:
                    continue
                row = build_row(seg, offset)
                if row:
                    chunk_rows.append(row)

            if checkpoint:
                checkpoint.append(index, chunk_rows)
            rows.extend(chunk_rows)
            furthest_audio_second = max(furthest_audio_second, owned_end)
            report_transcription()

    rows.sort(key=lambda r: (r["start"], r["end"]))
    # Reaction scoring is cheap and depends on the phrase list, so it happens
    # here rather than being baked into the checkpoint.
    for row in rows:
        score, reasons = score_text(row["text"], settings.reaction_phrases)
        row["score"] = round(score, 4)
        row["reasons"] = reasons

    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    if duration > 0:
        furthest_audio_second = max(furthest_audio_second, duration)
    report_transcription(force=True)

    language_probability = _safe_float(getattr(info, "language_probability", None))
    metadata = {
        "status": TRANSCRIPTION_AVAILABLE,
        "language": getattr(info, "language", None),
        "language_probability": language_probability if language_probability is not None else 0.0,
        "device": device,
        "compute_type": compute_type,
        "model": prepared.display_name,
        "model_source": prepared.source,
        "fallback_reason": fallback_reason,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "audio_duration_seconds": round(duration, 3) if duration > 0 else None,
        "real_time_factor": round(elapsed_seconds / duration, 6) if duration > 0 else None,
    }
    if cpu_threads is not None:
        metadata["cpu_threads"] = cpu_threads
    if checkpoint:
        metadata["resumed_chunks"] = recovered
    return rows, metadata
