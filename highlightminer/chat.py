from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .security import MAX_JSON_LINE_BYTES, MAX_JSON_NESTING, validate_chat_file
from .util import parse_time

# Parser is an original permissive implementation for common chat-export shapes.
# TwitchDownloader is compatibility context only; no TwitchDownloader code is used.

_TIME_KEYS = (
    "content_offset_seconds", "offset_seconds", "timestamp_seconds", "seconds",
    "timestamp", "time", "offset", "video_offset",
)
_TEXT_KEYS = ("body", "message", "text", "content")


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in _TEXT_KEYS:
            if key in value:
                found = _message_text(value[key])
                if found:
                    return found
        fragments = value.get("fragments")
        if isinstance(fragments, list):
            return "".join(_message_text(x) for x in fragments).strip()
    if isinstance(value, list):
        return " ".join(filter(None, (_message_text(x) for x in value))).strip()
    return ""


def _record_from_dict(d: dict) -> dict | None:
    t = None
    for key in _TIME_KEYS:
        if key in d:
            t = parse_time(d[key])
            if t is not None:
                break
    if t is None:
        return None

    text = ""
    for key in _TEXT_KEYS:
        if key in d:
            text = _message_text(d[key])
            if text:
                break
    if not text and isinstance(d.get("message"), dict):
        text = _message_text(d["message"])
    return {"time": t, "text": text} if text else None


def _walk_json(obj: Any, depth: int = 0) -> Iterable[dict]:
    if depth > MAX_JSON_NESTING:
        raise ValueError(f"Chat JSON nesting exceeds the safety limit ({MAX_JSON_NESTING}).")
    if isinstance(obj, dict):
        rec = _record_from_dict(obj)
        if rec:
            yield rec
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from _walk_json(value, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_json(item, depth + 1)


def load_chat(path: str | Path) -> list[dict]:
    p = validate_chat_file(path)
    suffix = p.suffix.lower()
    records: list[dict] = []

    if suffix == ".csv":
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rec = _record_from_dict(row)
                if rec:
                    records.append(rec)
    elif suffix in {".jsonl", ".ndjson"}:
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if len(line) > MAX_JSON_LINE_BYTES:
                    raise ValueError(
                        f"Chat JSON line {line_no} exceeds the safety limit ({MAX_JSON_LINE_BYTES:,} characters)."
                    )
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.extend(_walk_json(obj))
    else:
        with p.open("r", encoding="utf-8") as f:
            records.extend(_walk_json(json.load(f)))

    # Recursive JSON parsing can encounter wrapper + nested records. De-dupe exact matches.
    dedup = {(round(float(r["time"]), 3), r["text"]): r for r in records if r["time"] >= 0}
    return sorted(dedup.values(), key=lambda r: r["time"])


def messages_per_minute(records: list[dict], duration: float) -> float:
    """Average chat rate across the VOD."""
    if not records or duration <= 0:
        return 0.0
    return len(records) / (float(duration) / 60.0)


def volume_confidence(rate: float, quiet_rate: float, active_rate: float) -> float:
    """How much a chat this busy deserves to be trusted, from 0.0 to 1.0.

    The burst detector is purely relative: it compares each second against a
    rolling baseline and then percentile-scales the result across the VOD. On a
    quiet chat that scaling collapses, because most seconds hold zero messages,
    so a single message becomes a maximum-confidence "burst". Relative evidence
    needs an absolute sanity check before it can be believed.
    """
    if rate <= quiet_rate:
        return 0.0
    if active_rate <= quiet_rate or rate >= active_rate:
        return 1.0
    return float((rate - quiet_rate) / (active_rate - quiet_rate))


def analyze_chat(
    records: list[dict],
    duration: float,
    bucket_sec: float = 1.0,
    min_burst_messages: float = 3.0,
    quiet_msgs_per_min: float = 15.0,
    active_msgs_per_min: float = 60.0,
) -> list[dict]:
    """Score chat activity per bucket, damped by how busy the chat actually is.

    Returns an empty list when the chat is too quiet to carry signal. Callers
    treat that exactly like "no chat supplied", so the audio and transcript
    weights renormalize instead of being diluted by noise.
    """
    if not records:
        return []

    rate = messages_per_minute(records, duration)
    confidence = volume_confidence(rate, quiet_msgs_per_min, active_msgs_per_min)
    if confidence <= 0.0:
        return []

    bucket_sec = max(0.25, float(bucket_sec))
    n = max(1, int(np.ceil(duration / bucket_sec)))
    counts = np.zeros(n, dtype=np.float32)
    for r in records:
        idx = int(float(r["time"]) // bucket_sec)
        if 0 <= idx < n:
            counts[idx] += 1.0

    # Compare each second against a local ~60s baseline. +1 keeps quiet chats sane.
    window = max(3, int(round(60.0 / bucket_sec)))
    kernel = np.ones(window, dtype=np.float32) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(counts, (left, right), mode="edge")
    baseline = np.convolve(padded, kernel, mode="valid")
    ratio = (counts + 1.0) / (baseline + 1.0)
    p50 = float(np.percentile(ratio, 50))
    p97 = float(np.percentile(ratio, 97))
    span = max(1e-6, p97 - p50)
    relative = np.clip((ratio - p50) / span, 0.0, 1.0)

    # A burst must also be absolutely large, not merely larger than a quiet
    # baseline. Two messages where the baseline is half a message is a ratio
    # spike and nothing else.
    min_burst = max(1e-6, float(min_burst_messages))
    absolute = np.clip((counts - baseline) / min_burst, 0.0, 1.0)

    score = relative * absolute * confidence

    return [
        {
            "time": round((i + 0.5) * bucket_sec, 3),
            "count": int(counts[i]),
            "ratio": round(float(ratio[i]), 3),
            "score": round(float(score[i]), 4),
        }
        for i in range(n)
    ]
