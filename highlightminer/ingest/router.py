"""URL to adapter routing, and the one-call ingest entry point.

Named router rather than resolve because the package re-exports the
``resolve()`` function, which would otherwise shadow the submodule and make
``highlightminer.ingest.resolve`` ambiguous.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from . import kick, twitch, youtube
from .base import (
    ChatUnavailable,
    IngestError,
    IngestProgress,
    IngestResult,
    download_video,
    probe_url,
)

_ADAPTERS: tuple[ModuleType, ...] = (twitch, youtube, kick)

SUPPORTED_PLATFORMS = tuple(adapter.platform for adapter in _ADAPTERS)


def resolve(url: str) -> ModuleType:
    """Pick the adapter for a URL."""
    text = str(url).strip()
    if not text:
        raise IngestError("No URL supplied.")
    for adapter in _ADAPTERS:
        if adapter.matches(text):
            return adapter
    raise IngestError(
        f"Unsupported URL. Supported platforms: {', '.join(SUPPORTED_PLATFORMS)}."
    )


def is_supported_url(url: str) -> bool:
    try:
        resolve(url)
    except IngestError:
        return False
    return True


def ingest(
    url: str,
    video_dir: str | Path,
    chat_dir: str | Path | None = None,
    *,
    max_height: int = 1080,
    with_chat: bool = True,
    progress: IngestProgress | None = None,
) -> IngestResult:
    """Fetch a VOD and, where the platform offers one, its chat replay.

    Chat failure is never fatal. A VOD with no chat replay, or a platform we
    cannot read chat from, still analyzes on audio and speech with the chat
    weight renormalized away, so the result carries ``chat_error`` rather than
    raising.
    """
    adapter = resolve(url)
    if progress:
        progress("probe", 0.0, "Reading VOD details")
    info = probe_url(url)
    info.platform = adapter.platform

    if progress:
        progress("download", 0.0, "Downloading VOD")
    video_path = download_video(url, video_dir, max_height=max_height, progress=progress)

    chat_path: Path | None = None
    chat_error: str | None = None
    if with_chat:
        if progress:
            progress("chat", 0.0, "Fetching chat replay")
        target_dir = Path(chat_dir) if chat_dir else video_path.parent
        target = Path(target_dir) / f"{video_path.stem}-chat.json"
        try:
            chat_path = adapter.fetch_chat(url, target)
        except ChatUnavailable as exc:
            chat_error = str(exc)
        except IngestError as exc:
            chat_error = f"Chat fetch failed: {exc}"

    if progress:
        progress("done", 1.0, "Ingest complete")
    return IngestResult(
        video_path=video_path,
        chat_path=chat_path,
        info=info,
        chat_error=chat_error,
    )
