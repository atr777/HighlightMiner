"""whisper.cpp backend, for GPU transcription on hardware CTranslate2 cannot use.

faster-whisper runs through CTranslate2, whose only GPU backend is CUDA. On an
AMD card that means CPU, measured at 2.95x realtime. whisper.cpp has a Vulkan
backend that works on AMD, Intel and NVIDIA alike, measured at 15.9x realtime
on the same audio and model.

This backend deliberately does **not** produce word timings. whisper.cpp can,
via DTW, but DTW and flash attention are mutually exclusive and flash attention
is where most of the speed comes from:

    dtw_token_timestamps is not supported with flash_attn - disabling

Forcing one segment per word (-ml 1 -sow) keeps flash attention but produces
degenerate timings: the first word of a segment absorbs the whole preceding
gap and the rest bunch up 30ms apart. Unusable for karaoke captions.

So the sweep runs here without word timings, and word timings are refined
afterwards by faster-whisper over the candidate windows only. Those are the
only places word timings are ever used, and it is about twelve minutes of audio
rather than thirteen hours.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .diagnostics import log_detailed, log_event
from .runtime import app_root

# Where a local build lands, relative to the application root.
_BINARY_NAMES = ("whisper-cli.exe", "whisper-cli", "main.exe", "main")
_SEARCH_SUBDIRS = (
    Path("tools/whisper.cpp/build/bin/Release"),
    Path("tools/whisper.cpp/build/bin"),
    Path("whispercpp"),
    Path("bin"),
)
_MODEL_SUBDIRS = (Path("tools/models"), Path("models"), Path("bin"))


class WhisperCppUnavailable(RuntimeError):
    """The whisper.cpp binary or model could not be found."""


@dataclass(frozen=True)
class WhisperCppTools:
    binary: Path
    model: Path
    vad_model: Path | None = None


def _search_roots() -> list[Path]:
    root = app_root()
    return [root, root.parent]


def find_binary(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None
    for root in _search_roots():
        for subdir in _SEARCH_SUBDIRS:
            for name in _BINARY_NAMES:
                candidate = root / subdir / name
                if candidate.is_file():
                    return candidate
    found = shutil.which("whisper-cli")
    return Path(found) if found else None


# Tried in order when the configured Whisper model has no ggml counterpart on
# disk. turbo first: it is what the speed measurements were taken against.
_MODEL_FALLBACKS = ("large-v3-turbo", "large-v3", "medium", "small", "base")


def find_model(explicit: str | Path | None = None, model_name: str = "large-v3-turbo") -> Path | None:
    """Locate a ggml model, preferring the exact counterpart of the configured one.

    A substitution is a real change of output, not a detail, so the caller
    reports which model was actually used rather than the one that was asked
    for. See ``transcribe``'s metadata.
    """
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None
    # ggml naming, with the plain "turbo" alias pointing at large-v3-turbo.
    stem = "large-v3-turbo" if model_name == "turbo" else model_name
    for wanted in (stem, *(f for f in _MODEL_FALLBACKS if f != stem)):
        for root in _search_roots():
            for subdir in _MODEL_SUBDIRS:
                candidate = root / subdir / f"ggml-{wanted}.bin"
                if candidate.is_file():
                    return candidate
    return None


def find_vad_model(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None
    for root in _search_roots():
        for subdir in _MODEL_SUBDIRS:
            for name in ("ggml-silero-v5.1.2.bin", "ggml-silero-v5.1.bin"):
                candidate = root / subdir / name
                if candidate.is_file():
                    return candidate
    return None


def resolve_tools(settings) -> WhisperCppTools:
    """Locate everything the backend needs, or explain what is missing."""
    binary = find_binary(getattr(settings, "whispercpp_binary", "") or None)
    if binary is None:
        raise WhisperCppUnavailable(
            "whisper-cli was not found. Build whisper.cpp with -DGGML_VULKAN=ON, "
            "or set whispercpp_binary to its path."
        )
    model = find_model(
        getattr(settings, "whispercpp_model", "") or None,
        getattr(settings, "whisper_model", "large-v3-turbo"),
    )
    if model is None:
        raise WhisperCppUnavailable(
            "No ggml model found. Download ggml-large-v3-turbo.bin into tools/models, "
            "or set whispercpp_model to its path."
        )
    return WhisperCppTools(binary, model, find_vad_model())


def is_available(settings) -> bool:
    try:
        resolve_tools(settings)
    except WhisperCppUnavailable:
        return False
    return True


def _parse_json(payload: dict) -> list[dict]:
    """Convert whisper.cpp JSON into the row shape the pipeline expects.

    Word timings are deliberately absent; see the module docstring.
    """
    rows: list[dict] = []
    for segment in payload.get("transcription", []) or []:
        offsets = segment.get("offsets") or {}
        try:
            start = float(offsets["from"]) / 1000.0
            end = float(offsets["to"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = max(0.0, start)
        rows.append({
            "start": round(start, 3),
            "end": round(max(start, end), 3),
            "text": text,
            "words": [],
        })
    rows.sort(key=lambda r: (r["start"], r["end"]))
    return rows


def transcribe(
    audio_path: str | Path,
    settings,
    *,
    threads: int | None = None,
    timeout: float | None = None,
) -> tuple[list[dict], dict]:
    """Transcribe a whole file with whisper.cpp, returning rows without words."""
    tools = resolve_tools(settings)
    audio = Path(audio_path).expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="highlightminer-wcpp-") as work:
        out_stem = Path(work) / "out"
        command = [
            str(tools.binary),
            "-m", str(tools.model),
            "-f", str(audio),
            "-oj",
            "-of", str(out_stem),
            "-bs", str(int(getattr(settings, "beam_size", 5))),
        ]
        if threads:
            command += ["-t", str(int(threads))]
        if getattr(settings, "language", None):
            command += ["-l", str(settings.language)]
        # Flash attention is the default and is what makes this worth using.
        if getattr(settings, "vad_filter", True) and tools.vad_model is not None:
            command += ["--vad", "-vm", str(tools.vad_model)]

        log_detailed(
            "transcription.whispercpp_start",
            model=tools.model.name,
            vad=tools.vad_model is not None,
        )
        try:
            result = subprocess.run(
                command,
                check=True,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            raise WhisperCppUnavailable(
                f"whisper.cpp failed: {(exc.stderr or '').strip()[:300]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise WhisperCppUnavailable("whisper.cpp timed out.") from exc

        produced = out_stem.with_suffix(".json")
        if not produced.exists():
            raise WhisperCppUnavailable("whisper.cpp produced no JSON output.")
        payload = json.loads(produced.read_text(encoding="utf-8"))

    rows = _parse_json(payload)
    stderr = result.stderr or ""
    device = "vulkan" if "ggml_vulkan" in stderr else "cpu"
    log_event("transcription.whispercpp_complete", segments=len(rows), device=device)

    used_model = tools.model.stem.replace("ggml-", "")
    requested = str(getattr(settings, "whisper_model", "") or "")
    metadata = {
        "backend": "whisper.cpp",
        "device": device,
        "model": used_model,
        "language": (payload.get("result") or {}).get("language"),
        # No word timings by design; the refinement pass supplies them.
        "word_timestamps": False,
    }
    if requested and requested not in {used_model, "turbo"}:
        # Different weights mean different words. Say so rather than letting
        # the settings page imply a model that never ran.
        metadata["requested_model"] = requested
        log_event("transcription.whispercpp_model_substituted", requested=requested, used=used_model)
    return rows, metadata
