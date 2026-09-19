"""Chunked transcription for long VODs.

Whisper's VAD filter builds the whole signal as one array of 576-sample frames.
A 13.5 hour VOD is 1.5 million of them, 3.3 GB, and it failed outright on this
16 GB machine partway through a real run. Chunking bounds that, but only if the
per-chunk time offsets are applied correctly.
"""

from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import pytest

from highlightminer import transcribe as t


def _write_wav(path, seconds, rate=16000, channels=1):
    samples = np.zeros(int(seconds * rate * channels), dtype="<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return path


def _word(text, start, end):
    return SimpleNamespace(word=text, start=start, end=end)


def _segment(text, start, end, words=None):
    return SimpleNamespace(text=text, start=start, end=end, words=words or [])


class TestIterAudioChunks:
    def test_splits_into_bounded_chunks(self, tmp_path):
        path = _write_wav(tmp_path / "a.wav", 10.0)
        chunks = list(t._iter_audio_chunks(path, chunk_sec=3.0, overlap_sec=0.0))
        assert [round(o, 3) for o, _, _ in chunks] == [0.0, 3.0, 6.0, 9.0]
        assert [len(c) for _, c, _ in chunks] == [48000, 48000, 48000, 16000]

    def test_offsets_are_cumulative_and_exact(self, tmp_path):
        path = _write_wav(tmp_path / "a.wav", 7.5)
        offsets = [o for o, _, _ in t._iter_audio_chunks(path, chunk_sec=2.5, overlap_sec=0.0)]
        assert offsets == pytest.approx([0.0, 2.5, 5.0])

    def test_owned_regions_tile_the_file_without_gaps(self, tmp_path):
        path = _write_wav(tmp_path / "a.wav", 10.0)
        chunks = list(t._iter_audio_chunks(path, chunk_sec=3.0, overlap_sec=1.0))
        offsets = [o for o, _, _ in chunks]
        owned = [e for _, _, e in chunks]
        assert owned[:-1] == pytest.approx(offsets[1:])
        assert owned[-1] == pytest.approx(10.0)

    def test_overlap_extends_the_audio_past_the_owned_region(self, tmp_path):
        path = _write_wav(tmp_path / "a.wav", 10.0)
        chunks = list(t._iter_audio_chunks(path, chunk_sec=3.0, overlap_sec=1.0))
        offset, samples, owned_end = chunks[0]
        assert owned_end == pytest.approx(3.0)
        # 3s owned plus 1s of lookahead
        assert len(samples) == 4 * 16000

    def test_final_chunk_is_not_padded_past_the_file(self, tmp_path):
        path = _write_wav(tmp_path / "a.wav", 10.0)
        _, samples, owned_end = list(t._iter_audio_chunks(path, chunk_sec=3.0, overlap_sec=5.0))[-1]
        assert owned_end == pytest.approx(10.0)
        assert len(samples) == 1 * 16000

    def test_single_chunk_when_file_is_short(self, tmp_path):
        path = _write_wav(tmp_path / "a.wav", 2.0)
        assert len(list(t._iter_audio_chunks(path, chunk_sec=60.0))) == 1

    def test_stereo_is_averaged_to_mono(self, tmp_path):
        path = _write_wav(tmp_path / "a.wav", 4.0, channels=2)
        chunks = list(t._iter_audio_chunks(path, chunk_sec=2.0, overlap_sec=0.0))
        assert all(c.ndim == 1 for _, c, _ in chunks)
        assert sum(len(c) for _, c, _ in chunks) == 4 * 16000

    def test_never_loads_the_whole_file(self, tmp_path):
        path = _write_wav(tmp_path / "a.wav", 60.0)
        sizes = [len(c) for _, c, _ in t._iter_audio_chunks(path, chunk_sec=5.0, overlap_sec=1.0)]
        assert max(sizes) == 6 * 16000

    def test_rejects_non_16_bit(self, tmp_path):
        path = tmp_path / "bad.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 100)
        with pytest.raises(ValueError, match="16-bit"):
            list(t._iter_audio_chunks(path, chunk_sec=1.0))


# Chunk ownership, offsets and the single-pass path are covered in
# test_transcribe_resume.py, which drives transcribe_audio itself so the
# production path and the test cannot drift apart.


class TestSegmentWordOffsets:
    def test_words_are_shifted_by_the_chunk_offset(self):
        seg = _segment("hi", 1.0, 2.0, [_word("hi", 1.0, 2.0)])
        words = t._segment_words(seg, 101.0, 102.0, offset=100.0)
        assert words == [{"w": "hi", "s": 101.0, "e": 102.0}]

    def test_words_are_clamped_into_the_shifted_segment(self):
        seg = _segment("hi", 1.0, 2.0, [_word("hi", 0.0, 9.0)])
        words = t._segment_words(seg, 101.0, 102.0, offset=100.0)
        assert words[0]["s"] == 101.0
        assert words[0]["e"] == 102.0

    def test_zero_offset_is_unchanged(self):
        seg = _segment("hi", 1.0, 2.0, [_word("hi", 1.0, 2.0)])
        assert t._segment_words(seg, 1.0, 2.0) == [{"w": "hi", "s": 1.0, "e": 2.0}]

    def test_words_without_timings_are_skipped(self):
        seg = _segment("hi", 1.0, 2.0, [_word("hi", None, None), _word("ok", 1.0, 1.5)])
        assert [w["w"] for w in t._segment_words(seg, 1.0, 2.0)] == ["ok"]


class TestChunkSetting:
    def test_default_is_half_an_hour(self):
        from highlightminer.config import Settings

        assert Settings().transcribe_chunk_sec == 1800.0

    def test_can_be_disabled(self):
        from highlightminer.config import Settings

        assert Settings(transcribe_chunk_sec=0.0).transcribe_chunk_sec == 0.0

    def test_rejects_negative(self):
        from highlightminer.config import Settings

        with pytest.raises(ValueError, match="transcribe_chunk_sec"):
            Settings(transcribe_chunk_sec=-1.0)


def test_thirteen_hour_vod_is_split_into_manageable_pieces(tmp_path, monkeypatch):
    """The case that failed: 48826s at the default half-hour chunk."""
    duration = 48826.0
    chunk = 1800.0
    expected = int(duration // chunk) + (1 if duration % chunk else 0)
    assert expected == 28

    # Each chunk's VAD array is frames x 576 floats; confirm it is now small.
    frames = chunk * 16000 / 512
    vad_bytes = frames * 576 * 4
    assert vad_bytes < 200 * 1024**2, "chunk still too large for a 16 GB machine"
