"""Twitch VOD chat replay fetching.

Twitch exposes VOD chat replay through its public GQL endpoint using the same
persisted query the web player issues. Fetching it directly keeps HighlightMiner
self-contained; TwitchDownloader remains a perfectly good alternative but is not
required.

Output is written in the shape ``chat.load_chat`` already accepts, so nothing
downstream needs to change:

    [{"content_offset_seconds": 12.4, "message": "LUL"}, ...]
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator

from .base import IngestError

# The public web-client ID the Twitch player sends. This is not a secret and not
# a credential; it identifies the client application, not a user.
_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
_GQL_ENDPOINT = "https://gql.twitch.tv/gql"

# Persisted-query hash for VideoCommentsByOffsetOrCursor. Twitch rotates these
# occasionally; when chat fetching starts returning no video node, this is the
# first thing to re-check.
_COMMENTS_QUERY_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"

_VIDEO_URL_RE = re.compile(r"(?:twitch\.tv/videos/|^)(\d+)")

_MAX_ATTEMPTS = 4
_BACKOFF_SEC = 1.5

Progress = Callable[[int, float], None]


platform = "twitch"


class TwitchIngestError(IngestError):
    """Raised when Twitch chat replay cannot be retrieved."""


def matches(url: str) -> bool:
    return "twitch.tv" in url


def video_id(url: str) -> str:
    """Adapter-interface alias for :func:`parse_video_id`."""
    return parse_video_id(url)


def parse_video_id(url_or_id: str) -> str:
    """Extract the numeric VOD id from a Twitch URL, or pass through a bare id."""
    match = _VIDEO_URL_RE.search(url_or_id.strip())
    if not match:
        raise TwitchIngestError(f"Not a recognizable Twitch VOD URL or id: {url_or_id!r}")
    return match.group(1)


def _post(payload: list[dict[str, Any]]) -> Any:
    request = urllib.request.Request(
        _GQL_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Client-ID": _CLIENT_ID, "Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # 4xx other than rate limiting will not improve by retrying.
            if exc.code not in (429, 500, 502, 503, 504):
                raise TwitchIngestError(f"Twitch GQL returned HTTP {exc.code}") from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(_BACKOFF_SEC * (attempt + 1))
    raise TwitchIngestError(f"Twitch GQL request failed after {_MAX_ATTEMPTS} attempts: {last_error}")


def _comments_page(video_id: str, offset_sec: float) -> dict[str, Any]:
    """Fetch one page of comments anchored at ``offset_sec``.

    This query also advertises a cursor on each edge, but requesting the next
    page by cursor returns an empty video node under the current persisted-query
    hash. Offset paging works reliably, so that is what we use. Pages are
    centered on the requested offset rather than starting at it, which is why
    the caller de-duplicates by comment id.
    """
    variables: dict[str, Any] = {"videoID": video_id, "contentOffsetSeconds": max(0.0, offset_sec)}

    data = _post([{
        "operationName": "VideoCommentsByOffsetOrCursor",
        "variables": variables,
        "extensions": {"persistedQuery": {"version": 1, "sha256Hash": _COMMENTS_QUERY_HASH}},
    }])

    try:
        video = data[0]["data"]["video"]
    except (KeyError, IndexError, TypeError) as exc:
        raise TwitchIngestError(
            "Unexpected Twitch GQL response shape. The persisted-query hash may have rotated."
        ) from exc

    if video is None:
        raise TwitchIngestError(
            f"Twitch returned no video {video_id}. It may be deleted, subscriber-only, or region locked."
        )
    return video.get("comments") or {}


def _message_text(node: dict[str, Any]) -> str:
    message = node.get("message") or {}
    fragments = message.get("fragments") or []
    return "".join(fragment.get("text") or "" for fragment in fragments).strip()


def iter_comments(video_id: str, progress: Progress | None = None) -> Iterator[dict[str, Any]]:
    """Yield ``{content_offset_seconds, message}`` records, walking the VOD forward."""
    seen_ids: set[str] = set()
    offset = 0.0
    while True:
        comments = _comments_page(video_id, offset)
        edges = comments.get("edges") or []
        if not edges:
            return

        furthest = offset
        for edge in edges:
            node = edge.get("node") or {}
            node_id = node.get("id")
            comment_offset = node.get("contentOffsetSeconds")
            if comment_offset is None:
                continue
            furthest = max(furthest, float(comment_offset))
            # Pages are centered on the requested offset, so consecutive pages
            # overlap. Comment id is the only reliable de-duplication key.
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            text = _message_text(node)
            if not text:
                continue
            yield {
                "content_offset_seconds": float(comment_offset),
                "message": text,
                "commenter": ((node.get("commenter") or {}).get("displayName") or ""),
            }

        if progress:
            progress(len(seen_ids), furthest)

        if not (comments.get("pageInfo") or {}).get("hasNextPage"):
            return
        # Always move forward, even if a page produced nothing new, so a dense
        # cluster of messages on one second cannot stall the walk.
        offset = furthest + 1.0 if furthest > offset else offset + 1.0


def fetch_chat(url_or_id: str, out_path: str | Path, progress: Progress | None = None) -> Path:
    """Download full VOD chat replay to ``out_path`` as JSON.

    Returns the written path. Raises TwitchIngestError if the VOD is unavailable.
    A VOD with chat replay disabled simply yields an empty list, which the
    pipeline treats as "no chat signal" and renormalizes the other weights.
    """
    # Local name avoids shadowing the module-level video_id() adapter function.
    vod_id = parse_video_id(url_or_id)
    records = list(iter_comments(vod_id, progress))
    records.sort(key=lambda r: r["content_offset_seconds"])

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)
    return out
