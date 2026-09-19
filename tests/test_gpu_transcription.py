"""The GPU sweep backend and the word-timing pass that follows it.

faster-whisper runs through CTranslate2, which only has a CUDA GPU backend, so
on an AMD card it runs on the CPU at 2.95x realtime. whisper.cpp on Vulkan
manages 15.9x on the same audio, but only with flash attention, which is
mutually exclusive with its DTW word timings.

The resolution is to split the work: sweep the whole source on the GPU without
word timings, then re-transcribe the candidate windows with faster-whisper.
Word timings are only ever read inside a candidate window, so that is minutes
of audio instead of hours. These cover the seams in that split.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from highlightminer import whispercpp
from highlightminer.config import Settings
from highlightminer.word_refine import (
    MERGE_GAP_SECONDS,
    PAD_SECONDS,
    Window,
    _read_window,
    merge_refined,
    windows_for_candidates,
)


class TestBackendSetting:
    def test_defaults_to_the_proven_backend(self):
        """Nothing changes for an existing install until it is asked to."""
        assert Settings().transcription_backend == "faster-whisper"
        assert Settings().refine_word_timings is True

    @pytest.mark.parametrize("backend", ["auto", "faster-whisper", "whispercpp"])
    def test_accepts_known_backends(self, backend):
        assert Settings(transcription_backend=backend).transcription_backend == backend

    def test_normalises_case(self):
        assert Settings(transcription_backend="Whispercpp").transcription_backend == "whispercpp"

    def test_rejects_anything_else(self):
        with pytest.raises(ValueError, match="transcription_backend"):
            Settings(transcription_backend="vosk")


class TestToolDiscovery:
    def _tree(self, tmp_path, binary="whisper-cli.exe", model="ggml-large-v3-turbo.bin"):
        release = tmp_path / "tools" / "whisper.cpp" / "build" / "bin" / "Release"
        release.mkdir(parents=True)
        (tmp_path / "tools" / "models").mkdir(parents=True)
        if binary:
            (release / binary).write_bytes(b"x")
        if model:
            (tmp_path / "tools" / "models" / model).write_bytes(b"x")
        return tmp_path

    @pytest.fixture
    def rooted(self, tmp_path, monkeypatch):
        def install(**kwargs):
            root = self._tree(tmp_path, **kwargs)
            monkeypatch.setattr(whispercpp, "_search_roots", lambda: [root])
            return root

        return install

    def test_finds_a_local_build(self, rooted):
        root = rooted()
        assert whispercpp.find_binary().name == "whisper-cli.exe"
        assert whispercpp.find_model().parent == root / "tools" / "models"

    def test_an_explicit_path_wins(self, rooted, tmp_path):
        rooted()
        elsewhere = tmp_path / "custom.exe"
        elsewhere.write_bytes(b"x")
        assert whispercpp.find_binary(elsewhere) == elsewhere

    def test_an_explicit_path_that_does_not_exist_is_not_silently_ignored(self, rooted, tmp_path):
        rooted()
        assert whispercpp.find_binary(tmp_path / "nope.exe") is None

    def test_falls_back_when_the_exact_model_is_absent(self, rooted):
        """Asking for large-v3 with only turbo on disk uses turbo, not nothing."""
        rooted()
        assert whispercpp.find_model(model_name="large-v3").name == "ggml-large-v3-turbo.bin"

    def test_prefers_the_exact_model_when_it_is_there(self, rooted):
        root = rooted()
        (root / "tools" / "models" / "ggml-medium.bin").write_bytes(b"x")
        assert whispercpp.find_model(model_name="medium").name == "ggml-medium.bin"

    def test_turbo_is_an_alias(self, rooted):
        rooted()
        assert whispercpp.find_model(model_name="turbo").name == "ggml-large-v3-turbo.bin"

    def test_unavailable_without_a_binary(self, rooted):
        rooted(binary=None)
        assert whispercpp.is_available(Settings()) is False

    def test_unavailable_without_a_model(self, rooted):
        rooted(model=None)
        assert whispercpp.is_available(Settings()) is False

    def test_the_missing_piece_is_named(self, rooted):
        rooted(model=None)
        with pytest.raises(whispercpp.WhisperCppUnavailable, match="ggml model"):
            whispercpp.resolve_tools(Settings())


class TestJsonParsing:
    def _payload(self, *segments):
        return {
            "transcription": [
                {"offsets": {"from": int(s * 1000), "to": int(e * 1000)}, "text": t}
                for s, e, t in segments
            ]
        }

    def test_converts_millisecond_offsets_to_seconds(self):
        rows = whispercpp._parse_json(self._payload((1.5, 3.25, " hello ")))
        assert rows == [{"start": 1.5, "end": 3.25, "text": "hello", "words": []}]

    def test_carries_no_word_timings(self):
        """By design: the refinement pass supplies them."""
        rows = whispercpp._parse_json(self._payload((0.0, 1.0, "a")))
        assert rows[0]["words"] == []

    def test_orders_by_time(self):
        rows = whispercpp._parse_json(self._payload((5.0, 6.0, "b"), (1.0, 2.0, "a")))
        assert [r["text"] for r in rows] == ["a", "b"]

    def test_drops_blank_text(self):
        assert whispercpp._parse_json(self._payload((0.0, 1.0, "   "))) == []

    def test_drops_malformed_offsets(self):
        payload = {"transcription": [{"offsets": {"from": "x", "to": 1000}, "text": "a"}]}
        assert whispercpp._parse_json(payload) == []

    def test_survives_a_missing_offsets_block(self):
        assert whispercpp._parse_json({"transcription": [{"text": "a"}]}) == []

    def test_empty_payload(self):
        assert whispercpp._parse_json({}) == []

    def test_negative_start_is_clamped(self):
        rows = whispercpp._parse_json(self._payload((-0.5, 1.0, "a")))
        assert rows[0]["start"] == 0.0

    def test_end_never_precedes_start(self):
        rows = whispercpp._parse_json(self._payload((3.0, 1.0, "a")))
        assert rows[0]["end"] >= rows[0]["start"]


class TestCandidateWindows:
    def test_one_candidate_is_one_window(self):
        assert windows_for_candidates([{"start": 10.0, "end": 45.0}]) == [Window(10.0, 45.0)]

    def test_nearby_candidates_merge(self):
        """Two passes over overlapping audio cost more than one."""
        windows = windows_for_candidates(
            [{"start": 10.0, "end": 45.0}, {"start": 50.0, "end": 80.0}]
        )
        assert windows == [Window(10.0, 80.0)]

    def test_distant_candidates_stay_apart(self):
        far = 45.0 + MERGE_GAP_SECONDS + 5
        windows = windows_for_candidates(
            [{"start": 10.0, "end": 45.0}, {"start": far, "end": 120.0}]
        )
        assert len(windows) == 2

    def test_overlapping_candidates_collapse(self):
        windows = windows_for_candidates(
            [{"start": 10.0, "end": 60.0}, {"start": 30.0, "end": 45.0}]
        )
        assert windows == [Window(10.0, 60.0)]

    def test_out_of_order_input_is_handled(self):
        windows = windows_for_candidates(
            [{"start": 200.0, "end": 230.0}, {"start": 10.0, "end": 40.0}]
        )
        assert [w.start for w in windows] == [10.0, 200.0]

    def test_skips_zero_length_and_inverted(self):
        rows = [{"start": 10.0, "end": 10.0}, {"start": 50.0, "end": 20.0}]
        assert windows_for_candidates(rows) == []

    def test_skips_malformed(self):
        assert windows_for_candidates([{"start": "x", "end": 10.0}, {}]) == []

    def test_no_candidates(self):
        assert windows_for_candidates([]) == []

    def test_does_not_run_off_the_front_of_the_source(self):
        assert windows_for_candidates([{"start": 1.0, "end": 20.0}], pad=5.0)[0].start == 0.0

    def test_total_covered_audio_is_a_fraction_of_a_long_source(self):
        """The whole reason the split is worth doing."""
        candidates = [{"start": i * 2000.0, "end": i * 2000.0 + 35.0} for i in range(20)]
        covered = sum(w.span for w in windows_for_candidates(candidates))
        assert covered == pytest.approx(700.0)


def _wav(path, seconds=30.0, rate=16000):
    """A real 16-bit mono PCM WAV, matching what FFmpeg hands the pipeline."""
    samples = (np.sin(np.linspace(0, 400 * np.pi, int(seconds * rate))) * 8000).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return path


class TestWindowReading:
    def test_reads_the_window_plus_padding(self, tmp_path):
        block, offset = _read_window(_wav(tmp_path / "a.wav"), 10.0, 15.0)
        assert offset == pytest.approx(10.0 - PAD_SECONDS)
        assert block.size / 16000 == pytest.approx(5.0 + 2 * PAD_SECONDS, abs=0.01)

    def test_padding_is_clamped_at_the_start(self, tmp_path):
        _, offset = _read_window(_wav(tmp_path / "a.wav"), 1.0, 5.0)
        assert offset == 0.0

    def test_padding_is_clamped_at_the_end(self, tmp_path):
        block, offset = _read_window(_wav(tmp_path / "a.wav", seconds=20.0), 15.0, 20.0)
        assert offset + block.size / 16000 <= 20.0 + 1e-6

    def test_a_window_past_the_end_reads_nothing(self, tmp_path):
        block, _ = _read_window(_wav(tmp_path / "a.wav", seconds=10.0), 100.0, 110.0)
        assert block.size == 0

    def test_rejects_audio_that_is_not_the_expected_format(self, tmp_path):
        path = tmp_path / "b.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)
            wf.setframerate(16000)
            wf.writeframes(b"\0" * 1000)
        with pytest.raises(ValueError, match="16-bit"):
            _read_window(path, 0.0, 0.05)


class TestMergingRefinedWords:
    SWEEP = [
        {"start": 5.0, "end": 8.0, "text": "outside before", "words": []},
        {"start": 20.0, "end": 24.0, "text": "inside rough", "words": []},
        {"start": 26.0, "end": 29.0, "text": "inside rough two", "words": []},
        {"start": 90.0, "end": 93.0, "text": "outside after", "words": []},
    ]
    REFINED = [
        {
            "start": 20.1,
            "end": 24.2,
            "text": "inside precise",
            "words": [{"w": "inside", "s": 20.1, "e": 20.6}],
        }
    ]
    WINDOWS = [Window(18.0, 30.0)]

    def test_rows_inside_the_window_are_replaced(self):
        merged = merge_refined(self.SWEEP, self.REFINED, self.WINDOWS)
        assert [r["text"] for r in merged] == [
            "outside before",
            "inside precise",
            "outside after",
        ]

    def test_rows_outside_the_window_keep_their_text(self):
        """They still feed scoring; they simply have no word timings."""
        merged = merge_refined(self.SWEEP, self.REFINED, self.WINDOWS)
        assert merged[0]["words"] == []
        assert merged[-1]["text"] == "outside after"

    def test_word_timings_survive_the_merge(self):
        merged = merge_refined(self.SWEEP, self.REFINED, self.WINDOWS)
        assert merged[1]["words"][0]["w"] == "inside"

    def test_output_stays_ordered(self):
        merged = merge_refined(self.SWEEP, self.REFINED, self.WINDOWS)
        assert merged == sorted(merged, key=lambda r: r["start"])

    def test_no_windows_leaves_the_transcript_alone(self):
        assert merge_refined(self.SWEEP, [], []) == self.SWEEP

    def test_a_window_the_sweep_heard_nothing_in_still_takes_its_words(self):
        assert len(merge_refined([], self.REFINED, self.WINDOWS)) == 1

    def test_a_window_with_no_speech_simply_drops_the_rough_rows(self):
        merged = merge_refined(self.SWEEP, [], self.WINDOWS)
        assert [r["text"] for r in merged] == ["outside before", "outside after"]


class TestPipelineDispatch:
    def test_refinement_runs_when_the_sweep_produced_no_words(self):
        from highlightminer.pipeline import _needs_word_refinement

        assert _needs_word_refinement({"word_timestamps": False}, Settings()) is True

    def test_refinement_is_skipped_when_words_are_already_there(self):
        from highlightminer.pipeline import _needs_word_refinement

        assert _needs_word_refinement({"word_timestamps": True}, Settings()) is False

    def test_refinement_is_skipped_when_captions_are_not_wanted(self):
        from highlightminer.pipeline import _needs_word_refinement

        settings = Settings(word_timestamps=False)
        assert _needs_word_refinement({"word_timestamps": False}, settings) is False

    def test_refinement_can_be_turned_off(self):
        from highlightminer.pipeline import _needs_word_refinement

        settings = Settings(refine_word_timings=False)
        assert _needs_word_refinement({"word_timestamps": False}, settings) is False

    def test_a_transcript_that_says_nothing_is_assumed_to_have_words(self):
        """faster-whisper metadata predates this field; it always had words."""
        from highlightminer.pipeline import _needs_word_refinement

        assert _needs_word_refinement({}, Settings()) is False

    def test_explicit_whispercpp_reports_what_is_missing(self, tmp_path, monkeypatch):
        """Silently taking five times longer than asked for would be worse."""
        from highlightminer.pipeline import _sweep_transcript

        monkeypatch.setattr(whispercpp, "_search_roots", lambda: [tmp_path])
        with pytest.raises(whispercpp.WhisperCppUnavailable):
            _sweep_transcript(
                tmp_path / "a.wav",
                Settings(transcription_backend="whispercpp"),
                model_access=None,
                prepared_model=None,
                duration=1.0,
                progress=lambda *a: None,
                work_dir=tmp_path,
            )

    def test_auto_falls_back_quietly(self, tmp_path, monkeypatch):
        """That is what choosing auto asks for."""
        from highlightminer import pipeline

        monkeypatch.setattr(whispercpp, "_search_roots", lambda: [tmp_path])
        called = {}

        def fake(wav, settings, **kwargs):
            called["yes"] = True
            return [], {"backend": "faster-whisper"}

        monkeypatch.setattr(pipeline, "transcribe_audio", fake)
        _, meta = pipeline._sweep_transcript(
            tmp_path / "a.wav",
            Settings(transcription_backend="auto"),
            model_access=None,
            prepared_model=None,
            duration=1.0,
            progress=lambda *a: None,
            work_dir=tmp_path,
        )
        assert called and meta["backend"] == "faster-whisper"

    def test_faster_whisper_never_reaches_the_gpu_backend(self, tmp_path, monkeypatch):
        from highlightminer import pipeline

        def refuse(settings):
            raise AssertionError("the GPU backend must not be consulted")

        monkeypatch.setattr(whispercpp, "is_available", refuse)
        monkeypatch.setattr(pipeline, "transcribe_audio", lambda *a, **k: ([], {}))
        pipeline._sweep_transcript(
            tmp_path / "a.wav",
            Settings(transcription_backend="faster-whisper"),
            model_access=None,
            prepared_model=None,
            duration=1.0,
            progress=lambda *a: None,
            work_dir=tmp_path,
        )


class TestCacheSignature:
    """A transcript from one engine must not satisfy a run configured for another."""

    def _signature(self, **kwargs):
        from highlightminer.model_access import ModelAccessPreferences
        from highlightminer.pipeline import _stage_signatures

        return _stage_signatures(Settings(**kwargs), None, ModelAccessPreferences())["transcript"]

    def test_changing_the_backend_invalidates_the_cache(self):
        assert self._signature(transcription_backend="faster-whisper") != self._signature(
            transcription_backend="whispercpp"
        )

    def test_changing_the_ggml_model_invalidates_the_cache(self):
        assert self._signature(whispercpp_model="") != self._signature(
            whispercpp_model="/m/ggml-medium.bin"
        )

    def test_turning_refinement_off_invalidates_the_cache(self):
        assert self._signature(refine_word_timings=True) != self._signature(
            refine_word_timings=False
        )

    def test_the_same_settings_give_the_same_signature(self):
        assert self._signature() == self._signature()
