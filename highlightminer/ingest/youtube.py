"""YouTube ingest.

Video comes through the shared yt-dlp path. Chat is the live chat replay of a
stream, which yt-dlp can save as a subtitle track. VODs that were never live
have no chat at all, which is fine: the pipeline renormalizes the remaining
weights.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .base import ChatUnavailable, IngestError, _ffmpeg_dir, _import_yt_dlp

platform = "youtube"

_URL_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|live/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def matches(url: str) -> bool:
    return bool(_URL_RE.search(url)) or "youtube.com" in url or "youtu.be" in url


def video_id(url: str) -> str:
    match = _URL_RE.search(url.strip())
    if not match:
        raise IngestError(f"Not a recognizable YouTube URL: {url!r}")
    return match.group(1)


def _offset_seconds(action: dict) -> float | None:
    value = action.get("videoOffsetTimeMsec")
    if value is None:
        return None
    try:
        return max(0.0, float(value) / 1000.0)
    except (TypeError, ValueError):
        return None


def _message_text(renderer: dict) -> str:
    message = renderer.get("message") or {}
    runs = message.get("runs") or []
    parts: list[str] = []
    for run in runs:
        if "text" in run:
            parts.append(str(run["text"]))
        else:
            # Emoji runs carry a shortcut label like ":smile:" instead of text.
            emoji = run.get("emoji") or {}
            shortcuts = emoji.get("shortcuts") or []
            if shortcuts:
                parts.append(str(shortcuts[0]))
    return "".join(parts).strip()


def parse_live_chat(path: str | Path) -> list[dict]:
    """Convert yt-dlp's ``.live_chat.json`` into the shape ``load_chat`` reads.

    The file is JSON Lines of YouTube "replay actions", each wrapping a chat
    item renderer. Only text messages carry usable timing, so superchats,
    memberships and deletions are skipped.
    """
    records: list[dict] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            action = payload.get("replayChatItemAction") or {}
            offset = _offset_seconds(action)
            if offset is None:
                continue
            for item in action.get("actions") or []:
                renderer = (
                    (item.get("addChatItemAction") or {}).get("item") or {}
                ).get("liveChatTextMessageRenderer")
                if not renderer:
                    continue
                text = _message_text(renderer)
                if not text:
                    continue
                author = (renderer.get("authorName") or {}).get("simpleText", "")
                records.append({
                    "content_offset_seconds": offset,
                    "message": text,
                    "commenter": str(author),
                })
    records.sort(key=lambda r: r["content_offset_seconds"])
    return records


def fetch_chat(url: str, out_path: str | Path) -> Path:
    """Download and convert YouTube live chat replay."""
    yt_dlp = _import_yt_dlp()
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
        "outtmpl": str(stem),
    }
    ffmpeg_dir = _ffmpeg_dir()
    if ffmpeg_dir:
        options["ffmpeg_location"] = ffmpeg_dir

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        raise ChatUnavailable(f"YouTube chat replay could not be downloaded: {exc}") from exc

    raw = Path(f"{stem}.live_chat.json")
    if not raw.exists():
        raise ChatUnavailable(
            "No live chat replay for this video. It was probably never a live stream."
        )

    records = parse_live_chat(raw)
    raw.unlink(missing_ok=True)
    if not records:
        raise ChatUnavailable("The live chat replay contained no usable messages.")

    with out.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)
    return out
