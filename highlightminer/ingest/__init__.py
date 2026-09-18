"""URL ingest adapters for Twitch, Kick, and YouTube.

Each adapter turns a platform URL into local files the existing pipeline
already understands: a video file for ``analyze_vod`` and, where the platform
offers one, a chat export in the shape ``chat.load_chat`` already parses.
"""

from __future__ import annotations

from .base import (
    ChatUnavailable,
    IngestError,
    IngestResult,
    VodInfo,
    download_video,
    probe_url,
)
from .router import SUPPORTED_PLATFORMS, ingest, is_supported_url, resolve

__all__ = [
    "ChatUnavailable",
    "IngestError",
    "IngestResult",
    "SUPPORTED_PLATFORMS",
    "VodInfo",
    "download_video",
    "ingest",
    "is_supported_url",
    "probe_url",
    "resolve",
]
