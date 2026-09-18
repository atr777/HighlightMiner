"""URL ingest adapters for Twitch, Kick, and YouTube.

Each adapter turns a platform URL into local files the existing pipeline
already understands: a video file for ``analyze_vod`` and, where the platform
offers one, a chat export in the shape ``chat.load_chat`` already parses.
"""

from __future__ import annotations

__all__ = ["twitch"]
