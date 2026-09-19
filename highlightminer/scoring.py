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
    min_containment: float = _DUPLICATE_MIN_CONTAINMENT,
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
    if shorter <= 0.0 or intersection / shorter < min_containment:
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
    min_containment: float = _DUPLICATE_MIN_CONTAINMENT,
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
                    min_containment=min_containment,
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
    """Score of the feature nearest ``t``. Kept for callers outside the timeline."""
    if not features:
        return 0.0
    return float(_nearest_feature_series(features, np.asarray([float(t)]), key)[0])


def _nearest_feature_series(
    features: list[dict],
    times: np.ndarray,
    key: str = "score",
) -> np.ndarray:
    """Vectorised nearest-feature lookup for a whole timeline at once.

    The per-point version rebuilt the feature time array on every call, which
    made ranking quadratic in VOD length: measured 0.27s for a 12 minute VOD,
    7.2s for an hour, and an extrapolated 22 minutes for 13.5 hours.

    Ties resolve to the later feature, matching the original implementation,
    which listed the right-hand candidate first and let ``min`` keep it.
    """
    if not features:
        return np.zeros(times.shape, dtype=np.float64)

    feature_times = np.fromiter(
        (float(x["time"]) for x in features), dtype=np.float64, count=len(features)
    )
    values = np.fromiter(
        (float(x.get(key, 0.0)) for x in features), dtype=np.float64, count=len(features)
    )

    last = len(features) - 1
    idx = np.searchsorted(feature_times, times)
    right = np.clip(idx, 0, last)
    left = np.clip(idx - 1, 0, last)

    right_distance = np.where(idx < len(features), np.abs(feature_times[right] - times), np.inf)
    left_distance = np.where(idx > 0, np.abs(feature_times[left] - times), np.inf)

    chosen = np.where(left_distance < right_distance, left, right)
    return values[chosen]


def _transcript_at(segments: list[dict], t: float) -> float:
    """Best transcript score covering ``t``. Kept for callers outside the timeline."""
    return float(_transcript_series(segments, np.asarray([float(t)]))[0])


def _transcript_series(segments: list[dict], times: np.ndarray) -> np.ndarray:
    """Vectorised transcript coverage over a whole timeline.

    The per-point version scanned every segment for every point, which on a long
    VOD is tens of thousands of segments times a hundred thousand points.
    """
    out = np.zeros(times.shape, dtype=np.float64)
    if not segments or times.size == 0:
        return out

    # The half-second slack matches the original inclusive comparison.
    for segment in segments:
        try:
            start = float(segment["start"]) - 0.5
            end = float(segment["end"]) + 0.5
            score = float(segment.get("score", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if score <= 0.0 or end < start:
            continue
        lo = int(np.searchsorted(times, start, side="left"))
        hi = int(np.searchsorted(times, end, side="right"))
        if hi > lo:
            np.maximum(out[lo:hi], score, out=out[lo:hi])
    return out


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
    # Reproduce the original "while t <= duration" grid exactly, including its
    # floating point accumulation, before doing anything vectorised with it.
    count = 0
    t = 0.0
    while t <= duration:
        count += 1
        t += step
    if count == 0:
        return []
    times = np.arange(count, dtype=np.float64) * step

    audio = _nearest_feature_series(audio_features, times)
    if transcript_available:
        speech = _transcript_series(transcript, times)
    else:
        speech = np.zeros(count, dtype=np.float64)
    chat = (
        _nearest_feature_series(chat_features, times)
        if chat_features
        else np.zeros(count, dtype=np.float64)
    )

    combined = (
        weights.get("audio", 0) * audio
        + weights.get("transcript", 0) * speech
        + weights.get("chat", 0) * chat
    )

    # Several independent signals firing together is worth more than one spike.
    active = (audio >= 0.68).astype(np.int8)
    if transcript_available:
        active = active + (speech >= 0.68)
    if chat_features:
        active = active + (chat >= 0.68)
    combined = combined + np.where(active >= 2, 0.10, 0.0)
    combined = np.clip(combined, 0.0, 1.0)

    return [
        TimelineSignal(float(t), float(a), float(s), float(c), float(x))
        for t, a, s, c, x in zip(times, audio, speech, chat, combined)
    ]


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

        # A loud moment nothing else corroborates is usually noise, not an
        # event. Measured on the test VOD: a candidate scored audio 0.97 with
        # a transcript score of exactly 0.00, and the VOD's own transcript
        # explains it as the streamer's PC fans. Penalised rather than
        # dropped, so it stays reviewable and still yields a learning label.
        corroborated = (
            (transcript_available and max_tx >= 0.55)
            or (bool(chat_features) and max_chat >= 0.70)
        )
        audio_only = max_audio >= 0.72 and not corroborated
        if audio_only:
            score = clamp(score * float(settings.audio_only_penalty))

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
            "audio_only": bool(audio_only),
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
        min_containment=settings.duplicate_containment,
    )

    for rank, cand in enumerate(kept, start=1):
        cand["rank"] = rank
        cand["id"] = f"H{rank:03d}"
    return kept
