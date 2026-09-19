"""Suggested clip titles and hashtags.

A starting point for the title field, not a copywriter. The transcript already
tells us what was said and which moment scored highest; this picks the most
quotable line in the window and cleans it up. You edit it.

No model is involved. A local LLM would be a different project, and the
heuristic is good enough to save the retyping.
"""

from __future__ import annotations

import re

# Openers that make a title read as a fragment of a longer conversation.
_LEADING_FILLER = (
    "um", "uh", "like", "so", "and", "but", "okay", "ok", "well", "yeah", "yes",
    "no", "i mean", "you know", "just", "then", "actually", "basically",
)

_FILLER_ANYWHERE = re.compile(r"\b(?:um|uh|erm)\b[,\s]*", re.IGNORECASE)
_REPEATED_WORD = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_NON_TAG = re.compile(r"[^a-z0-9]+")

MAX_TITLE_CHARS = 70


def _clean(text: str) -> str:
    text = _FILLER_ANYWHERE.sub("", str(text))
    text = _REPEATED_WORD.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text).strip(" ,.-")
    # Repeat until nothing strips: "so like I ..." only loses "like" once "so"
    # is gone, and a single pass over the filler list misses that.
    changed = True
    while changed:
        changed = False
        lowered = text.lower()
        for filler in sorted(_LEADING_FILLER, key=len, reverse=True):
            if lowered.startswith(filler + " "):
                text = text[len(filler) + 1:].lstrip(" ,")
                changed = True
                break
            # A line that is nothing but filler ("um uh like so") should end up
            # empty so the caller falls back, rather than leaving a title of "so".
            if lowered == filler:
                text = ""
                changed = True
                break
    return text.strip()


def _truncate(text: str, limit: int = MAX_TITLE_CHARS) -> str:
    """Trim to a word boundary rather than mid-word."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(" ,.-")
    return (cut or text[:limit]).strip()


def _segments_in_window(segments: list[dict], start: float, end: float) -> list[dict]:
    inside = []
    for segment in segments:
        try:
            seg_start = float(segment["start"])
            seg_end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if seg_end >= start and seg_start <= end:
            inside.append(segment)
    return inside


def suggest_title(
    segments: list[dict],
    start: float,
    end: float,
    *,
    fallback: str = "",
    limit: int = MAX_TITLE_CHARS,
) -> str:
    """Pick the most quotable line in a clip window.

    Prefers the highest reaction-scoring segment, because that is the line the
    detector reacted to. Falls back to the first substantial line, then to
    ``fallback``.
    """
    inside = _segments_in_window(segments, float(start), float(end))
    if not inside:
        return fallback

    def score(segment: dict) -> float:
        try:
            return float(segment.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    ranked = sorted(inside, key=score, reverse=True)
    for segment in ranked:
        candidate = _clean(str(segment.get("text", "")))
        # Two words is a grunt, not a title.
        if len(candidate.split()) >= 3:
            return _truncate(candidate, limit)

    for segment in inside:
        candidate = _clean(str(segment.get("text", "")))
        if candidate:
            return _truncate(candidate, limit)
    return fallback


def suggest_hashtags(
    content_label: str | None = None,
    *,
    extra: list[str] | None = None,
    limit: int = 6,
) -> list[str]:
    """Build a small hashtag list from the content label.

    Deliberately short and generic. Chasing trending tags is a moving target
    and not something a local tool can know about.
    """
    tags: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        tag = _NON_TAG.sub("", str(value).lower())
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(f"#{tag}")

    for word in str(content_label or "").split():
        add(word)
    if content_label and len(str(content_label).split()) > 1:
        add(str(content_label).replace(" ", ""))
    for value in extra or []:
        add(value)
    for default in ("shorts", "clips", "twitch"):
        add(default)
    return tags[:limit]


def suggest_description(
    title: str,
    content_label: str | None = None,
    *,
    hashtags: list[str] | None = None,
) -> str:
    """A one-line description with hashtags appended."""
    tags = hashtags if hashtags is not None else suggest_hashtags(content_label)
    parts = [part for part in (title.strip(), str(content_label or "").strip()) if part]
    head = " · ".join(dict.fromkeys(parts))
    return f"{head}\n\n{' '.join(tags)}".strip()
