from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from .util import clamp


def _robust_scale(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    p50 = float(np.percentile(values, 50))
    p95 = float(np.percentile(values, 95))
    span = max(1e-6, p95 - p50)
    return np.clip((values - p50) / span, 0.0, 1.0)


# Read the WAV in chunks of roughly this many seconds. A 13.5 hour VOD is about
# 781 million samples, which is 3.1 GB once widened to float32, so reading it in
# one go is not an option on a 16 GB machine.
_READ_BLOCK_SEC = 120.0


def _iter_windows(wf: wave.Wave_read, win: int, hop: int, channels: int):
    """Yield (start_sample, window) pairs without holding the whole file.

    Blocks are read sequentially and the unconsumed tail is carried into the
    next one, so windows straddling a block boundary are still emitted exactly
    once and at the correct offset.
    """
    block_frames = max(win, int(_READ_BLOCK_SEC * wf.getframerate()))
    carry = np.empty(0, dtype=np.float32)
    carry_start = 0
    produced = False

    while True:
        raw = wf.readframes(block_frames)
        if not raw:
            break
        block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            block = block.reshape(-1, channels).mean(axis=1)
        buffer = np.concatenate((carry, block)) if carry.size else block

        offset = 0
        while offset + win <= buffer.size:
            yield carry_start + offset, buffer[offset:offset + win]
            produced = True
            offset += hop
        carry = buffer[offset:]
        carry_start += offset

    # A file shorter than one window still produces a single short window, which
    # is what the whole-file implementation did.
    if not produced and carry.size:
        yield carry_start, carry


def analyze_audio(wav_path: str | Path, window_sec: float = 1.0, hop_sec: float = 0.5) -> list[dict]:
    times: list[float] = []
    db_values: list[float] = []

    with wave.open(str(wav_path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV from FFmpeg")
        channels = wf.getnchannels()
        rate = wf.getframerate()
        win = max(1, int(window_sec * rate))
        hop = max(1, int(hop_sec * rate))

        for start, chunk in _iter_windows(wf, win, hop, channels):
            rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
            dbfs = 20.0 * math.log10(max(rms, 1e-7))
            times.append((start + chunk.size / 2) / rate)
            db_values.append(dbfs)

    db = np.asarray(db_values, dtype=np.float32)
    energy = _robust_scale(db)
    delta = np.maximum(0.0, np.diff(db, prepend=db[0] if db.size else 0.0))
    onset = _robust_scale(delta)
    excitement = np.clip(0.76 * energy + 0.24 * onset, 0.0, 1.0)

    return [
        {
            "time": round(float(t), 3),
            "dbfs": round(float(d), 3),
            "energy": round(float(e), 4),
            "onset": round(float(o), 4),
            "score": round(clamp(float(x)), 4),
        }
        for t, d, e, o, x in zip(times, db, energy, onset, excitement)
    ]
