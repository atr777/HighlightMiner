"""Reshape review windows into short-form clips.

The default candidate window is built for judging a moment: generous pre-roll so
you can see what led up to it. That is the wrong shape for a Short, where the
payoff has to arrive in the first few seconds or the viewer is gone.

These are pure functions over times and transcript segments, so they are cheap
to test and can be re-run from cached evidence without another Whisper pass.
"""

from __future__ import annotations

from dataclasses import dataclass

# Openers that almost always refer to something said before the clip started.
# Deliberately small and boring: this is a nudge in ranking, not a language model.
_DANGLING_OPENERS = frozenset({
    "and", "but", "so", "because", "which", "that", "then", "also", "anyway",
    "he", "she", "it", "they", "them", "him", "her", "this", "these", "those",
})

_SENTENCE_ENDINGS = (".", "!", "?")


@dataclass(frozen=True)
class Window:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def align_hook(
    start: float,
    end: float,
    peak_time: float,
    *,
    lead_sec: float = 3.0,
    min_duration_sec: float = 8.0,
    max_duration_sec: float = 45.0,
) -> Window:
    """Trim the window so the peak lands ``lead_sec`` in, rather than deep inside.

    Only ever trims, never extends: the new start cannot precede the original,
    so this can never pull in material the detector did not actually select.
    A candidate whose peak already arrives early is left alone.
    """
    start = float(start)
    end = max(start, float(end))
    peak_time = min(max(float(peak_time), start), end)

    new_start = min(max(start, peak_time - lead_sec), end)
    new_end = min(end, new_start + max_duration_sec)
    # Keep a floor on length, but never past the original end.
    if new_end - new_start < min_duration_sec:
        new_end = min(end, new_start + min_duration_sec)
    if new_end <= new_start:
        new_end = end
    return Window(round(new_start, 3), round(new_end, 3))


def _segment_bounds(segments: list[dict]) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for segment in segments:
        try:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if seg_end >= seg_start:
            bounds.append((seg_start, seg_end))
    return sorted(bounds)


def snap_to_speech(
    start: float,
    end: float,
    segments: list[dict],
    *,
    tolerance_sec: float = 1.5,
) -> Window:
    """Move clip edges onto nearby speech boundaries so it does not open or end mid-word.

    Both edges move outward to the enclosing segment's boundary when that is
    within ``tolerance_sec``. Moving outward rather than inward keeps whole
    words: trimming to the nearest boundary inside the window would cut the
    very speech the candidate was selected for.
    """
    start = float(start)
    end = max(start, float(end))
    bounds = _segment_bounds(segments)
    if not bounds:
        return Window(round(start, 3), round(end, 3))

    new_start = start
    for seg_start, seg_end in bounds:
        if seg_start < start <= seg_end and (start - seg_start) <= tolerance_sec:
            new_start = seg_start
            break

    new_end = end
    for seg_start, seg_end in bounds:
        if seg_start <= end < seg_end and (seg_end - end) <= tolerance_sec:
            new_end = seg_end
            break

    if new_end <= new_start:
        return Window(round(start, 3), round(end, 3))
    return Window(round(new_start, 3), round(new_end, 3))


def _first_words(segments: list[dict], start: float, end: float, count: int = 3) -> list[str]:
    for segment in sorted(segments, key=lambda s: float(s.get("start", 0.0))):
        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        if seg_end < start or seg_start > end:
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            return text.lower().split()[:count]
    return []


def self_containment(
    segments: list[dict],
    start: float,
    end: float,
    *,
    tolerance_sec: float = 0.35,
) -> float:
    """Rough 0..1 estimate of whether a clip stands on its own.

    Three cheap signals, no language model:

    - does the clip begin close to the start of an utterance, or mid-sentence;
    - does it open on a word that points backwards ("and", "so", "he");
    - does the last spoken segment inside it finish a sentence.

    This is a ranking nudge. It cannot tell whether a joke needs setup, and it
    is not meant to.
    """
    bounds = _segment_bounds(segments)
    if not bounds:
        return 0.5

    score = 1.0

    starts_cleanly = any(abs(seg_start - start) <= tolerance_sec for seg_start, _ in bounds)
    if not starts_cleanly:
        inside_an_utterance = any(
            seg_start < start - tolerance_sec < seg_end for seg_start, seg_end in bounds
        )
        score -= 0.35 if inside_an_utterance else 0.1

    opening = _first_words(segments, start, end)
    if opening and opening[0].strip(".,!?'\"") in _DANGLING_OPENERS:
        score -= 0.25

    last_text = ""
    for segment in sorted(segments, key=lambda s: float(s.get("start", 0.0))):
        seg_start = float(segment.get("start", 0.0))
        if start <= seg_start <= end:
            text = str(segment.get("text", "")).strip()
            if text:
                last_text = text
    if last_text and not last_text.endswith(_SENTENCE_ENDINGS):
        score -= 0.15

    return max(0.0, min(1.0, round(score, 4)))


def shape_candidate(
    start: float,
    end: float,
    peak_time: float,
    segments: list[dict],
    *,
    lead_sec: float = 3.0,
    min_duration_sec: float = 8.0,
    max_duration_sec: float = 45.0,
    snap_tolerance_sec: float = 1.5,
    source_duration: float | None = None,
) -> Window:
    """Hook alignment followed by speech snapping, clamped to the source."""
    window = align_hook(
        start, end, peak_time,
        lead_sec=lead_sec,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
    )
    window = snap_to_speech(
        window.start, window.end, segments, tolerance_sec=snap_tolerance_sec
    )
    # Snapping can push the end past the hard duration ceiling, and outward
    # snapping at the start can reach below zero on a clip at the very
    # beginning of a VOD.
    new_start = max(0.0, window.start)
    new_end = min(window.end, new_start + max_duration_sec)
    if source_duration is not None:
        new_end = min(new_end, float(source_duration))
    if new_end <= new_start:
        new_end = min(float(source_duration) if source_duration else end, end)
    return Window(round(new_start, 3), round(new_end, 3))
