"""Streaming audio analysis must match the whole-file implementation exactly.

A 13.5 hour VOD is ~781 million samples, 3.1 GB once widened to float32.
Reading it in one go is not viable on a 16 GB machine, but the rewrite is only
safe if it produces identical numbers.
"""

from __future__ import annotations

import math
import wave

import numpy as np
import pytest

from highlightminer import audio as audio_module
from highlightminer.audio import analyze_audio


def _reference(wav_path, window_sec=1.0, hop_sec=0.5):
    """The original whole-file implementation, kept as the oracle."""
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    win = max(1, int(window_sec * rate))
    hop = max(1, int(hop_sec * rate))
    times, db_values = [], []
    for start in range(0, max(1, len(samples) - win + 1), hop):
        chunk = samples[start:start + win]
        if chunk.size == 0:
            break
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
        db_values.append(20.0 * math.log10(max(rms, 1e-7)))
        times.append((start + chunk.size / 2) / rate)
    return times, db_values


def _write_wav(path, samples, rate=16000, channels=1):
    data = np.clip(samples, -1.0, 1.0)
    pcm = (data * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    return path


def _signal(seconds, rate=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    # Tone plus noise plus a couple of bursts, so windows differ from each other.
    sig = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)
    for centre in (seconds * 0.3, seconds * 0.7):
        lo, hi = int((centre - 0.2) * rate), int((centre + 0.2) * rate)
        sig[lo:hi] *= 4.0
    return sig


@pytest.mark.parametrize("seconds", [0.3, 1.0, 3.7, 12.5, 30.0])
def test_matches_the_whole_file_implementation(tmp_path, seconds):
    path = _write_wav(tmp_path / "a.wav", _signal(seconds))
    rows = analyze_audio(path)
    times, dbs = _reference(path)

    assert len(rows) == len(times), f"window count differs at {seconds}s"
    for row, t, d in zip(rows, times, dbs):
        assert row["time"] == pytest.approx(round(t, 3))
        assert row["dbfs"] == pytest.approx(round(d, 3), abs=1e-3)


def test_matches_across_a_block_boundary(tmp_path, monkeypatch):
    """Windows straddling a read block must appear once, at the right offset."""
    # Force a tiny block so a 5 second file spans many of them.
    monkeypatch.setattr(audio_module, "_READ_BLOCK_SEC", 0.25)
    path = _write_wav(tmp_path / "a.wav", _signal(5.0))
    rows = analyze_audio(path)
    times, dbs = _reference(path)

    assert len(rows) == len(times)
    for row, t, d in zip(rows, times, dbs):
        assert row["time"] == pytest.approx(round(t, 3))
        assert row["dbfs"] == pytest.approx(round(d, 3), abs=1e-3)


@pytest.mark.parametrize("block_sec", [0.1, 0.5, 1.0, 7.0, 1000.0])
def test_result_is_independent_of_block_size(tmp_path, monkeypatch, block_sec):
    path = _write_wav(tmp_path / "a.wav", _signal(6.0))
    monkeypatch.setattr(audio_module, "_READ_BLOCK_SEC", 1000.0)
    expected = analyze_audio(path)
    monkeypatch.setattr(audio_module, "_READ_BLOCK_SEC", block_sec)
    assert analyze_audio(path) == expected


def test_file_shorter_than_one_window_still_yields_a_row(tmp_path):
    path = _write_wav(tmp_path / "a.wav", _signal(0.4))
    rows = analyze_audio(path, window_sec=1.0, hop_sec=0.5)
    assert len(rows) == 1


def test_stereo_is_averaged_to_mono(tmp_path):
    rate = 16000
    mono = _signal(2.0, rate)
    stereo = np.repeat(mono, 2)
    mono_path = _write_wav(tmp_path / "m.wav", mono, rate, channels=1)
    stereo_path = _write_wav(tmp_path / "s.wav", stereo, rate, channels=2)
    assert len(analyze_audio(stereo_path)) == len(analyze_audio(mono_path))


def test_rejects_non_16_bit_audio(tmp_path):
    path = tmp_path / "bad.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * 1000)
    with pytest.raises(ValueError, match="16-bit"):
        analyze_audio(path)


def test_peak_windows_land_on_the_loud_bursts(tmp_path):
    path = _write_wav(tmp_path / "a.wav", _signal(20.0))
    rows = analyze_audio(path)
    loudest = max(rows, key=lambda r: r["dbfs"])
    assert 5.0 < loudest["time"] < 15.0


def test_memory_stays_bounded_on_a_long_file(tmp_path, monkeypatch):
    """The whole point: never materialise the full signal."""
    seen: list[int] = []
    real_frombuffer = np.frombuffer

    def spy(buffer, *args, **kwargs):
        result = real_frombuffer(buffer, *args, **kwargs)
        seen.append(result.size)
        return result

    monkeypatch.setattr(audio_module.np, "frombuffer", spy)
    monkeypatch.setattr(audio_module, "_READ_BLOCK_SEC", 2.0)
    path = _write_wav(tmp_path / "a.wav", _signal(30.0))
    analyze_audio(path)

    assert seen, "no reads happened"
    # No single read may cover the whole 30 second file.
    assert max(seen) <= 2.0 * 16000
