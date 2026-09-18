from __future__ import annotations

import pytest

from highlightminer.config import Settings
from highlightminer.scoring import deduplicate_candidates, find_candidates


def _audio(duration: float, spikes: dict[float, float], hop: float = 0.5) -> list[dict]:
    rows = []
    t = 0.0
    while t <= duration:
        score = 0.05
        for centre, value in spikes.items():
            if abs(t - centre) <= 1.0:
                score = max(score, value)
        rows.append({"time": round(t, 3), "dbfs": -20.0, "energy": score, "onset": 0.0, "score": score})
        t += hop
    return rows


def _speech(start: float, end: float, score: float, text: str = "words") -> dict:
    return {"start": start, "end": end, "text": text, "score": score, "reasons": []}


class TestAudioOnlyPenalty:
    def _settings(self, penalty: float) -> Settings:
        return Settings(
            audio_only_penalty=penalty,
            min_candidate_score=0.2,
            pre_roll_sec=2.0,
            post_roll_sec=2.0,
            merge_gap_sec=5.0,
        )

    def test_uncorroborated_audio_is_penalised(self):
        """Measured case: PC fan noise scored audio 0.97, transcript exactly 0.00."""
        audio = _audio(120.0, {60.0: 0.97})
        full = find_candidates(120.0, audio, [], [], self._settings(1.0))
        damped = find_candidates(120.0, audio, [], [], self._settings(0.5))
        assert full and damped
        assert damped[0]["score"] < full[0]["score"]

    def test_candidate_is_penalised_not_dropped(self):
        """It stays reviewable, and still yields a learning label."""
        audio = _audio(120.0, {60.0: 0.97})
        damped = find_candidates(120.0, audio, [], [], self._settings(0.1))
        assert len(damped) == 1
        assert damped[0]["features"]["audio_only"] is True

    def test_speech_corroboration_avoids_the_penalty(self):
        audio = _audio(120.0, {60.0: 0.97})
        speech = [_speech(58.0, 62.0, 0.8, "what the hell")]
        penalised = find_candidates(120.0, audio, speech, [], self._settings(0.1))
        assert penalised[0]["features"]["audio_only"] is False

    def test_chat_corroboration_avoids_the_penalty(self):
        audio = _audio(120.0, {60.0: 0.97})
        chat = [{"time": t, "count": 5, "ratio": 3.0, "score": 0.9 if abs(t - 60.0) < 2 else 0.0}
                for t in [i * 1.0 for i in range(121)]]
        result = find_candidates(120.0, audio, [], chat, self._settings(0.1))
        assert result[0]["features"]["audio_only"] is False

    def test_default_penalty_changes_nothing(self):
        audio = _audio(120.0, {60.0: 0.97})
        assert Settings().audio_only_penalty == 1.0
        a = find_candidates(120.0, audio, [], [], self._settings(1.0))
        assert a[0]["score"] > 0.0


class TestDuplicateContainment:
    def _pair(self) -> list[dict]:
        # 0.625 containment: the measured near-miss that survived a 0.65 threshold.
        return [
            {"id": "A", "start": 154.5, "end": 190.5, "score": 0.42, "peak_time": 170.0, "features": {}},
            {"id": "B", "start": 142.5, "end": 174.5, "score": 0.40, "peak_time": 170.5, "features": {}},
        ]

    def test_near_twins_survive_the_old_threshold(self):
        kept = deduplicate_candidates(self._pair(), max_candidates=10, min_containment=0.65)
        assert len(kept) == 2

    def test_a_lower_threshold_suppresses_them(self):
        kept = deduplicate_candidates(self._pair(), max_candidates=10, min_containment=0.55)
        assert len(kept) == 1
        assert kept[0]["id"] == "A"  # the stronger of the two

    def test_distinct_events_are_never_merged(self):
        far = [
            {"id": "A", "start": 0.0, "end": 30.0, "score": 0.5, "peak_time": 15.0, "features": {}},
            {"id": "B", "start": 300.0, "end": 330.0, "score": 0.4, "peak_time": 315.0, "features": {}},
        ]
        assert len(deduplicate_candidates(far, max_candidates=10, min_containment=0.55)) == 2

    def test_default_threshold_is_unchanged(self):
        assert Settings().duplicate_containment == 0.65

    def test_suppression_is_counted_on_the_winner(self):
        kept = deduplicate_candidates(self._pair(), max_candidates=10, min_containment=0.55)
        assert kept[0]["features"]["duplicates_suppressed"] == 1


class TestCpuThreads:
    def test_default_is_auto(self):
        assert Settings().cpu_threads == 0

    @pytest.mark.parametrize("value", [-1, 999])
    def test_rejects_absurd_values(self, value):
        with pytest.raises(ValueError, match="cpu_threads"):
            Settings(cpu_threads=value)

    def test_accepts_an_explicit_count(self):
        assert Settings(cpu_threads=11).cpu_threads == 11
