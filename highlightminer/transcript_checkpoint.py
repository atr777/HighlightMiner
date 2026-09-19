"""Per-chunk transcription checkpointing.

Transcribing a 13.5 hour VOD takes hours and the pipeline only persists at the
very end, so a crash at 66% lost everything. That is not hypothetical on a
machine that stops unexpectedly every few days.

Each completed chunk is appended to a JSONL sidecar in the work directory, so a
resumed run costs at most one chunk rather than the whole VOD.

Rows are stored without reaction scores, deliberately. Scoring is cheap and
depends on the reaction-phrase list, so a checkpoint stays valid when those
change, exactly as the database transcript cache does.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


def signature_for(settings, audio_path: str | Path, chunk_sec: float, overlap_sec: float) -> dict:
    """Everything that would invalidate previously transcribed chunks."""
    audio = Path(audio_path)
    try:
        size = audio.stat().st_size
    except OSError:
        size = -1
    return {
        "version": CHECKPOINT_VERSION,
        "model": str(getattr(settings, "whisper_model", "")),
        "compute_type": str(getattr(settings, "compute_type", "")),
        "language": getattr(settings, "language", None),
        "beam_size": int(getattr(settings, "beam_size", 0)),
        "vad_filter": bool(getattr(settings, "vad_filter", False)),
        "word_timestamps": bool(getattr(settings, "word_timestamps", False)),
        "chunk_sec": float(chunk_sec),
        "overlap_sec": float(overlap_sec),
        "audio_bytes": size,
    }


class TranscriptCheckpoint:
    """Append-only record of which chunks are already transcribed."""

    def __init__(self, path: str | Path, signature: dict):
        self.path = Path(path)
        self.signature = signature
        self._chunks: dict[int, list[dict]] = {}
        self._started = False

    # -- reading ---------------------------------------------------------

    def load(self) -> int:
        """Read any usable checkpoint. Returns how many chunks were recovered.

        A truncated final line is expected after a hard crash and is discarded
        rather than treated as corruption.
        """
        if not self.path.exists():
            return 0
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return 0
        if not lines:
            return 0

        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError:
            log.warning("Transcript checkpoint header unreadable; starting fresh.")
            return 0

        if header.get("signature") != self.signature:
            log.info("Transcript checkpoint does not match current settings; starting fresh.")
            return 0

        recovered = 0
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Only ever the last line, half written when the power went.
                break
            index = record.get("chunk")
            rows = record.get("rows")
            if isinstance(index, int) and isinstance(rows, list):
                self._chunks[index] = rows
                recovered += 1
        self._started = True
        return recovered

    def rows_for(self, index: int) -> list[dict] | None:
        return self._chunks.get(index)

    @property
    def completed(self) -> int:
        return len(self._chunks)

    # -- writing ---------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._started:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"signature": self.signature}, ensure_ascii=False) + "\n")
        self._started = True

    def append(self, index: int, rows: list[dict]) -> None:
        """Record one finished chunk, flushed so a crash cannot lose it."""
        self._ensure_started()
        self._chunks[index] = rows
        record = {"chunk": index, "rows": rows}
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
        except OSError as exc:
            # A checkpoint that cannot be written must not fail the run.
            log.warning("Could not write transcript checkpoint: %s", exc)

    def discard(self) -> None:
        """Remove the checkpoint once its transcript is safely stored."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def open_checkpoint(
    work_dir: str | Path | None,
    settings,
    audio_path: str | Path,
    chunk_sec: float,
    overlap_sec: float,
) -> TranscriptCheckpoint | None:
    """Build a checkpoint for this run, or None when chunking is off."""
    if not work_dir or chunk_sec <= 0:
        return None
    path = Path(work_dir) / "transcript-progress.jsonl"
    return TranscriptCheckpoint(path, signature_for(settings, audio_path, chunk_sec, overlap_sec))
