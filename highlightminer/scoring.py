from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings
from .shortform import self_containment, shape_candidate
from .timestamps import normalize_clip_bounds
from .util import clamp, format_time


@dataclass
class TimelineSignal:
    time: float
    audio: float = 0.0
    transcript: float = 0.0
    chat: float = 0.0
    combined: float = 0.0


_DUPLICATE_MIN_CONTAINMENT = 0.65
_DUPLICATE_MIN_PEAK_TOLERANCE_SEC = 1.0
_DUPLICATE_MAX_PEAK_TOLERANCE_SEC = 5.0


def _candidate_duration(candidate: dict) -> float:
    return max(0.0, float(candidate["end"]) - float(candidate["start"]))


def _candidate_peak(candidate: dict) -> float:
    start = float(candidate["start"])
    end = float(candidate["end"])
    return float(candidate.get("peak_time", (start + end) / 2.0))


def _represents_same_event(
    left: dict,
    right: dict,
    *,
    peak_tolerance_floor_sec: float = 0.0,
) -> bool:
    """Return whether two padded clip windows still describe one event.

    Clip overlap alone is insufficient because pre/post-roll can make distinct
    nearby events overlap heavily. A duplicate must both contain most of the
    shorter clip and have nearby signal peaks. The peak tolerance scales with
    clip length but is capped so long clips do not swallow adjacent moments.
    Callers may raise its floor to cover the gap that separated seed groups.
    """
    intersection = max(
        0.0,
        min(float(left["end"]), float(right["end"]))
        - max(float(left["start"]), float(right["start"])),
    )
    shorter = min(_candidate_duration(left), _candidate_duration(right))
    if shorter <= 0.0 or intersection / shorter < _DUPLICATE_MIN_CONTAINMENT:
        return False

    peak_tolerance = max(
        max(0.0, float(peak_tolerance_floor_sec)),
        min(
            _DUPLICATE_MAX_PEAK_TOLERANCE_SEC,
            max(_DUPLICATE_MIN_PEAK_TOLERANCE_SEC, shorter * 0.20),
        ),
    )
    return abs(_candidate_peak(left) - _candidate_peak(right)) <= peak_tolerance


def _candidate_strength(candidate: dict) -> tuple[float, int, float, float]:
    features = candidate.get("features") or {}
    return (
        float(candidate.get("score", 0.0)),
        int(features.get("active_signal_count", 0)),
        float(features.get("peak_combined", 0.0)),
        -_candidate_duration(candidate),
    )


def deduplicate_candidates(
    candidates: list[dict],
    *,
    max_candidates: int,
    peak_tolerance_floor_sec: float = 0.0,
) -> list[dict]:
    """Suppress weaker candidates for the same event without merging windows."""
    limit = max(0, int(max_candidates))
    if limit == 0:
        return []
    ordered = sorted(candidates, key=_candidate_strength, reverse=True)
    kept: list[dict] = []
    for candidate in ordered:
        duplicate_of = next(
            (
                winner
                for winner in kept
                if _represents_same_event(
                    candidate,
                    winner,
                    peak_tolerance_floor_sec=peak_tolerance_floor_sec,
                )
            ),
            None,
        )
        if duplicate_of is not None:
            features = duplicate_of.setdefault("features", {})
            features["duplicates_suppressed"] = int(features.get("duplicates_suppressed", 0)) + 1
            continue
        candidate.setdefault("features", {}).setdefault("duplicates_suppressed", 0)
        kept.append(candidate)
    return kept[:limit]


def _nearest_feature(features: list[dict], t: float, key: str = "score") -> float:
    if not features:
        return 0.0
    times = np.fromiter((float(x["time"]) for x in features), dtype=np.float64)
    idx = int(np.searchsorted(times, t))
    candidates = []
    if idx < len(features):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best = min(candidates, key=lambda i: abs(float(features[i]["time"]) - t))
    return float(features[best].get(key, 0.0))


def _transcript_at(segments: list[dict], t: float) -> float:
    best = 0.0
    for seg in segments:
        if float(seg["start"]) - 0.5 <= t <= float(seg["end"]) + 0.5:
            best = max(best, float(seg.get("score", 0.0)))
    return best


def build_timeline(
    duration: float,
    audio_features: list[dict],
    transcript: list[dict],
    chat_features: list[dict],
    settings: Settings,
    *,
    transcript_available: bool = True,
) -> list[TimelineSignal]:
    step = max(0.5, min(1.0, settings.audio_hop_sec))
    weights = settings.normalized_weights(
        bool(chat_features),
        transcript_available=transcript_available,
    )
    timeline: list[TimelineSignal] = []
    t = 0.0
    while t <= duration:
        a = _nearest_feature(audio_features, t)
        tx = _transcript_at(transcript, t) if transcript_available else 0.0
        ch = _nearest_feature(chat_features, t) if chat_features else 0.0
        combined = weights.get("audio", 0) * a + weights.get("transcript", 0) * tx + weights.get("chat", 0) * ch
        active_values = [a]
        if transcript_available:
            active_values.append(tx)
        if chat_features:
            active_values.append(ch)
        active = sum(v >= 0.68 for v in active_values)
        if active >= 2:
            combined += 0.10
        timeline.append(TimelineSignal(t, a, tx, ch, clamp(combined)))
        t += step
    return timeline


def _excerpt(transcript: list[dict], start: float, end: float, max_chars: int = 650) -> str:
    parts = [seg["text"] for seg in transcript if float(seg["end"]) >= start and float(seg["start"]) <= end]
    text = " ".join(x.strip() for x in parts if x.strip())
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


def find_candidates(
    duration: float,
    audio_features: list[dict],
    transcript: list[dict],
    chat_features: list[dict],
    settings: Settings,
    *,
    transcript_available: bool = True,
) -> list[dict]:
    timeline = build_timeline(
        duration,
        audio_features,
        transcript,
        chat_features,
        settings,
        transcript_available=transcript_available,
    )
    weights = settings.normalized_weights(
        bool(chat_features),
        transcript_available=transcript_available,
    )
    seeds = [
        x for x in timeline
        if x.combined >= settings.min_candidate_score
        or x.audio >= 0.94
        or (transcript_available and x.transcript >= 0.78)
        or (chat_features and x.chat >= 0.92)
    ]
    if not seeds:
        return []

    groups: list[list[TimelineSignal]] = [[seeds[0]]]
    for point in seeds[1:]:
        if point.time - groups[-1][-1].time <= settings.merge_gap_sec:
            groups[-1].append(point)
        else:
            groups.append([point])

    candidates: list[dict] = []
    for group in groups:
        peak = max(group, key=lambda x: x.combined)
        raw_start = max(0.0, group[0].time - settings.pre_roll_sec)
        raw_end = min(duration, group[-1].time + settings.post_roll_sec)

        if raw_end - raw_start > settings.max_candidate_sec:
            start = max(0.0, peak.time - settings.pre_roll_sec)
            end = min(duration, start + settings.max_candidate_sec)
            if peak.time > end:
                end = min(duration, peak.time + settings.post_roll_sec)
                start = max(0.0, end - settings.max_candidate_sec)
        else:
            start, end = raw_start, raw_end

        if settings.short_form_mode:
            # Reshape a review window into something postable: payoff near the
            # front, edges on speech boundaries. Done before the window stats
            # below so scores describe the clip that actually ships.
            shaped = shape_candidate(
                start, end, peak.time, transcript,
                lead_sec=settings.hook_lead_sec,
                min_duration_sec=settings.min_candidate_sec,
                max_duration_sec=settings.max_candidate_sec,
                snap_tolerance_sec=settings.speech_snap_sec,
                source_duration=duration,
            )
            start, end = shaped.start, shaped.end

        # Candidate times are displayed/stored at millisecond precision. Clamp
        # after that rounding so the rounded representation can never exceed
        # ffprobe's more precise source duration.
        bounds = normalize_clip_bounds(round(start, 3), round(end, 3), duration)
        start, end = bounds.start, bounds.end

        local = [x for x in timeline if start <= x.time <= end]
        max_audio = max((x.audio for x in local), default=0.0)
        max_tx = max((x.transcript for x in local), default=0.0) if transcript_available else 0.0
        max_chat = max((x.chat for x in local), default=0.0)
        avg_top = sorted((x.combined for x in local), reverse=True)[:5]
        top_mean = sum(avg_top) / max(1, len(avg_top))
        score = clamp(0.72 * peak.combined + 0.28 * top_mean)

        reasons = []
        if transcript_available and max_tx >= 0.55:
            reasons.append("reaction-heavy speech")
        if max_audio >= 0.72:
            reasons.append("audio spike")
        if chat_features and max_chat >= 0.70:
            reasons.append("chat burst")
        if not reasons:
            reasons.append("combined signal spike")

        signal_count = int(max_audio >= 0.68)
        if transcript_available:
            signal_count += int(max_tx >= 0.68)
        if chat_features:
            signal_count += int(max_chat >= 0.68)
        features = {
            "candidate_duration": round(end - start, 3),
            "peak_combined": round(float(peak.combined), 4),
            "top5_combined_mean": round(float(top_mean), 4),
            "active_signal_count": signal_count,
            "has_transcript": bool(transcript_available),
            "has_chat": bool(chat_features),
            "max_audio": round(max_audio, 4),
            "max_transcript": round(max_tx, 4),
            "max_chat": round(max_chat, 4),
            "weight_audio": round(float(weights.get("audio", 0.0)), 6),
            "weight_transcript": round(float(weights.get("transcript", 0.0)), 6),
            "weight_chat": round(float(weights.get("chat", 0.0)), 6),
            "seed_points": len(group),
            "self_containment": (
                self_containment(transcript, start, end) if transcript_available else None
            ),
            "hook_offset": round(max(0.0, float(peak.time) - start), 3),
            "short_form_mode": bool(settings.short_form_mode),
        }

        candidates.append({
            "id": "",
            "rank": 0,
            "score": round(score, 4),
            "peak_time": round(peak.time, 3),
            "start": start,
            "end": end,
            "start_label": format_time(start),
            "end_label": format_time(end),
            "audio_score": round(max_audio, 4),
            "transcript_score": round(max_tx, 4),
            "chat_score": round(max_chat, 4),
            "reason": ", ".join(reasons),
            "transcript": _excerpt(transcript, start, end) if transcript_available else "",
            "features": features,
        })

    timeline_step = max(0.5, min(1.0, settings.audio_hop_sec))
    kept = deduplicate_candidates(
        candidates,
        max_candidates=settings.max_candidates,
        peak_tolerance_floor_sec=settings.merge_gap_sec + timeline_step,
    )

    for rank, cand in enumerate(kept, start=1):
        cand["rank"] = rank
        cand["id"] = f"H{rank:03d}"
    return kept
