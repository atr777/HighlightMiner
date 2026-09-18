"""Kick ingest.

Video goes through the shared yt-dlp path like the others. Chat is the honest
gap: Kick has no documented public chat replay endpoint, and the community
routes vary and sit behind bot protection. Rather than ship something that
breaks quietly, chat is reported as unavailable and analysis proceeds on audio
and transcript alone, with the chat weight renormalized away.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import ChatUnavailable, IngestError

platform = "kick"

_URL_RE = re.compile(r"kick\.com/(?:video/([0-9a-fA-F-]{8,})|([A-Za-z0-9_-]+)/videos/([0-9a-fA-F-]{8,}))")


def matches(url: str) -> bool:
    return "kick.com" in url


def video_id(url: str) -> str:
    match = _URL_RE.search(url.strip())
    if not match:
        raise IngestError(f"Not a recognizable Kick VOD URL: {url!r}")
    return match.group(1) or match.group(3)


def fetch_chat(url: str, out_path: str | Path) -> Path:
    raise ChatUnavailable(
        "Kick chat replay is not supported. Kick publishes no documented replay "
        "endpoint, so this VOD will be analyzed on audio and speech alone."
    )


def convert_chat_export(records: Any, out_path: str | Path) -> Path:
    """Convert an externally obtained Kick chat export into our shape.

    An escape hatch: if you capture Kick chat with some other tool, hand the
    parsed records here and the pipeline will accept the result. Each record
    needs a time in seconds and message text under any of the usual key names.
    """
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    time_keys = ("content_offset_seconds", "offset_seconds", "seconds", "time", "offset")
    text_keys = ("message", "text", "body", "content")

    converted: list[dict] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        offset = next(
            (record[key] for key in time_keys if isinstance(record.get(key), (int, float))),
            None,
        )
        text = next(
            (str(record[key]).strip() for key in text_keys if str(record.get(key) or "").strip()),
            "",
        )
        if offset is None or not text:
            continue
        converted.append({"content_offset_seconds": float(offset), "message": text})

    if not converted:
        raise ChatUnavailable("No usable messages in the supplied Kick chat export.")

    converted.sort(key=lambda r: r["content_offset_seconds"])
    with out.open("w", encoding="utf-8") as handle:
        json.dump(converted, handle, ensure_ascii=False)
    return out
