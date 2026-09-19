"""Chunked transcription, ownership and crash resume, driven through the real path.

A 13.5 hour transcription reached 9 hours and wrote nothing, because the
pipeline only persisted at the end. The machine it was running on stops
unexpectedly every few days, so losing a whole run to that is not hypothetical.

These drive transcribe_audio itself with a stub model rather than testing a
helper, so the ownership filter and the checkpoint cannot drift apart.
"""

from __future__ import annotations

import json
import sys
import types
import wave

import numpy as np
import pytest

from highlightminer import transcribe
from highlightminer.config import Settings
from highlightminer.model_access import PreparedModelReference
from highlightminer.transcribe import transcribe_audio
from highlightminer.transcript_checkpoint import TranscriptCheckpoint, signature_for


def _write_wav(path, seconds, rate=16000):
    samples = np.zeros(int(seconds * rate), dtype="<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return path


def _segment(text, start, end, words=None):
    return types.SimpleNamespace(text=text, start=start, end=end, words=words or [])


def _install_model(monkeypatch, segments_for_chunk, record=None):
    """Stub faster_whisper so each chunk gets scripted segments."""
    calls = record if record is not None else []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, **kwargs):
            index = len(calls)
            calls.append(audio)
            return segments_for_chunk(index), types.SimpleNamespace(
                language="en", language_probability=0.9
            )

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return calls


def _prepared():
    return PreparedModelReference(
        reference="turbo", display_name="turbo", source="standard", local_files_only=False
    )


def _settings(**kw):
    base = dict(whisper_model="turbo", transcribe_chunk_sec=3.0, word_timestamps=False)
    base.update(kw)
    return Settings(**base)


class TestChunking:
    def test_long_audio_is_split_into_chunks(self, tmp_path, monkeypatch):
        wav = _write_wav(tmp_path / "a.wav", 9.0)
        calls = _install_model(monkeypatch, lambda i: [_segment(f"chunk{i}", 0.5, 1.0)])
        rows, meta = transcribe_audio(
            wav, _settings(), prepared_model=_prepared(), audio_duration=9.0
        )
        assert len(calls) == 3
        assert [r["text"] for r in rows] == ["chunk0", "chunk1", "chunk2"]

    def test_chunk_offsets_are_applied_to_times(self, tmp_path, monkeypatch):
        wav = _write_wav(tmp_path / "a.wav", 9.0)
        _install_model(monkeypatch, lambda i: [_segment(f"c{i}", 0.5, 1.0)])
        rows, _ = transcribe_audio(
            wav, _settings(), prepared_model=_prepared(), audio_duration=9.0
        )
        assert [r["start"] for r in rows] == pytest.approx([0.5, 3.5, 6.5])

    def test_short_audio_uses_a_single_pass(self, tmp_path, monkeypatch):
        wav = _write_wav(tmp_path / "a.wav", 2.0)
        calls = _install_model(monkeypatch, lambda i: [_segment("all", 0.1, 0.5)])
        transcribe_audio(wav, _settings(), prepared_model=_prepared(), audio_duration=2.0)
        assert len(calls) == 1
        assert isinstance(calls[0], str), "single pass should hand over a path"

    def test_segments_in_the_overlap_tail_are_left_to_the_next_chunk(self, tmp_path, monkeypatch):
        """Otherwise the same speech lands in the transcript twice."""
        wav = _write_wav(tmp_path / "a.wav", 9.0)
        # 0.5s in is owned; 3.5s in is past the 3.0s boundary, so deferred.
        _install_model(
            monkeypatch,
            lambda i: [_segment(f"kept{i}", 0.5, 1.0), _segment(f"tail{i}", 3.5, 3.9)],
        )
        rows, _ = transcribe_audio(
            wav, _settings(), prepared_model=_prepared(), audio_duration=9.0
        )
        assert [r["text"] for r in rows] == ["kept0", "kept1", "kept2"]

    def test_rows_come_back_in_time_order(self, tmp_path, monkeypatch):
        wav = _write_wav(tmp_path / "a.wav", 9.0)
        _install_model(monkeypatch, lambda i: [_segment(f"c{i}", 0.5, 1.0)])
        rows, _ = transcribe_audio(
            wav, _settings(), prepared_model=_prepared(), audio_duration=9.0
        )
        assert [r["start"] for r in rows] == sorted(r["start"] for r in rows)


class TestCheckpoint:
    def test_each_chunk_is_written_as_it_finishes(self, tmp_path, monkeypatch):
        wav = _write_wav(tmp_path / "a.wav", 9.0)
        _install_model(monkeypatch, lambda i: [_segment(f"c{i}", 0.5, 1.0)])
        work = tmp_path / "work"
        transcribe_audio(
            wav, _settings(), prepared_model=_prepared(), audio_duration=9.0, work_dir=work
        )
        lines = (work / "transcript-progress.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4  # header plus three chunks

    def test_a_resumed_run_only_transcribes_what_is_missing(self, tmp_path, monkeypatch):
        """The whole point: a crash costs one chunk, not the whole VOD."""
        wav = _write_wav(tmp_path / "a.wav", 9.0)
        work = tmp_path / "work"
        settings = _settings()

        # First run dies after the first chunk.
        class Boom(RuntimeError):
            pass

        def first_run(index):
            if index >= 1:
                raise Boom("power cut")
            return [_segment("c0", 0.5, 1.0)]

        _install_model(monkeypatch, first_run)
        with pytest.raises(Boom):
            transcribe_audio(
                wav, settings, prepared_model=_prepared(), audio_duration=9.0, work_dir=work
            )

        # Second run must skip chunk 0 and do only 1 and 2.
        second_calls = _install_model(
            monkeypatch, lambda i: [_segment(f"resumed{i}", 0.5, 1.0)]
        )
        rows, meta = transcribe_audio(
            wav, settings, prepared_model=_prepared(), audio_duration=9.0, work_dir=work
        )
        assert len(second_calls) == 2, "chunk 0 should have been recovered"
        assert meta["resumed_chunks"] == 1
        assert [r["text"] for r in rows] == ["c0", "resumed0", "resumed1"]

    def test_changed_settings_invalidate_the_checkpoint(self, tmp_path, monkeypatch):
        wav = _write_wav(tmp_path / "a.wav", 9.0)
        work = tmp_path / "work"
        _install_model(monkeypatch, lambda i: [_segment(f"c{i}", 0.5, 1.0)])
        transcribe_audio(
            wav, _settings(), prepared_model=_prepared(), audio_duration=9.0, work_dir=work
        )
        calls = _install_model(monkeypatch, lambda i: [_segment(f"x{i}", 0.5, 1.0)])
        transcribe_audio(
            wav, _settings(beam_size=3), prepared_model=_prepared(),
            audio_duration=9.0, work_dir=work,
        )
        assert len(calls) == 3, "a different model setting must not reuse chunks"

    def test_reaction_phrases_do_not_invalidate_the_checkpoint(self, tmp_path, monkeypatch):
        """Scoring is cheap and phrase-dependent, so it is not checkpointed."""
        wav = _write_wav(tmp_path / "a.wav", 9.0)
        work = tmp_path / "work"
        _install_model(monkeypatch, lambda i: [_segment("no way that worked", 0.5, 1.0)])
        transcribe_audio(
            wav, _settings(reaction_phrases=[]), prepared_model=_prepared(),
            audio_duration=9.0, work_dir=work,
        )
        calls = _install_model(monkeypatch, lambda i: [_segment("unused", 0.5, 1.0)])
        rows, meta = transcribe_audio(
            wav, _settings(reaction_phrases=["no way"]), prepared_model=_prepared(),
            audio_duration=9.0, work_dir=work,
        )
        assert calls == [], "nothing should be re-transcribed"
        assert meta["resumed_chunks"] == 3
        assert rows[0]["score"] > 0, "rescored with the new phrase list"

    def test_a_truncated_final_line_is_tolerated(self, tmp_path):
        """Exactly what a hard power cut leaves behind."""
        settings = _settings()
        wav = _write_wav(tmp_path / "a.wav", 9.0)
        path = tmp_path / "progress.jsonl"
        signature = signature_for(settings, wav, 3.0, 15.0)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"signature": signature}) + "\n")
            handle.write(json.dumps({"chunk": 0, "rows": []}) + "\n")
            handle.write('{"chunk": 1, "rows": [{"start"')  # cut off mid write

        checkpoint = TranscriptCheckpoint(path, signature)
        assert checkpoint.load() == 1
        assert checkpoint.rows_for(0) == []
        assert checkpoint.rows_for(1) is None

    def test_missing_checkpoint_is_not_an_error(self, tmp_path):
        settings = _settings()
        wav = _write_wav(tmp_path / "a.wav", 1.0)
        checkpoint = TranscriptCheckpoint(
            tmp_path / "nope.jsonl", signature_for(settings, wav, 3.0, 15.0)
        )
        assert checkpoint.load() == 0

    def test_no_checkpoint_written_without_a_work_dir(self, tmp_path, monkeypatch):
        wav = _write_wav(tmp_path / "a.wav", 9.0)
        _install_model(monkeypatch, lambda i: [_segment(f"c{i}", 0.5, 1.0)])
        _, meta = transcribe_audio(
            wav, _settings(), prepared_model=_prepared(), audio_duration=9.0
        )
        assert "resumed_chunks" not in meta
        assert not list(tmp_path.glob("*.jsonl"))

    def test_signature_tracks_the_audio_file(self, tmp_path):
        settings = _settings()
        short = _write_wav(tmp_path / "short.wav", 1.0)
        long = _write_wav(tmp_path / "long.wav", 5.0)
        assert signature_for(settings, short, 3.0, 15.0) != signature_for(settings, long, 3.0, 15.0)
