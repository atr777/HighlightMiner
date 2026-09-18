from __future__ import annotations

import json

import pytest

from highlightminer.chat import load_chat
from highlightminer.ingest import twitch


def _node(node_id: str, offset: float, text: str, who: str = "someone") -> dict:
    return {
        "cursor": f"cursor-{node_id}",
        "node": {
            "id": node_id,
            "contentOffsetSeconds": offset,
            "commenter": {"displayName": who},
            "message": {"fragments": [{"text": text}]},
        },
    }


def _page(edges: list[dict], has_next: bool) -> dict:
    return {"edges": edges, "pageInfo": {"hasNextPage": has_next}}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("https://www.twitch.tv/videos/2849127032", "2849127032"),
        ("twitch.tv/videos/123", "123"),
        ("https://www.twitch.tv/videos/456?t=1h2m3s", "456"),
        ("789", "789"),
    ],
)
def test_parse_video_id(value, expected):
    assert twitch.parse_video_id(value) == expected


def test_parse_video_id_rejects_non_vod():
    with pytest.raises(twitch.TwitchIngestError):
        twitch.parse_video_id("https://www.twitch.tv/somechannel")


def test_iter_comments_dedupes_overlapping_pages(monkeypatch):
    # Real pages are centered on the requested offset, so they overlap heavily.
    pages = [
        _page([_node("a", 1.0, "first"), _node("b", 5.0, "second")], True),
        _page([_node("b", 5.0, "second"), _node("c", 9.0, "third")], True),
        _page([_node("c", 9.0, "third")], False),
    ]
    calls: list[float] = []

    def fake_page(video_id: str, offset_sec: float) -> dict:
        calls.append(offset_sec)
        return pages[min(len(calls) - 1, len(pages) - 1)]

    monkeypatch.setattr(twitch, "_comments_page", fake_page)

    records = list(twitch.iter_comments("1"))
    assert [r["message"] for r in records] == ["first", "second", "third"]
    # Offsets must advance past the furthest seen comment, never repeat.
    assert calls == [0.0, 6.0, 10.0]


def test_iter_comments_advances_when_page_yields_nothing_new(monkeypatch):
    # A page that repeats only already-seen comments must not stall the walk.
    repeated = _page([_node("a", 1.0, "only")], True)
    final = _page([], False)
    calls: list[float] = []

    def fake_page(video_id: str, offset_sec: float) -> dict:
        calls.append(offset_sec)
        return repeated if len(calls) < 3 else final

    monkeypatch.setattr(twitch, "_comments_page", fake_page)

    records = list(twitch.iter_comments("1"))
    assert len(records) == 1
    assert calls[1] > calls[0] and calls[2] > calls[1]


def test_iter_comments_skips_empty_message_text(monkeypatch):
    page = _page([_node("a", 1.0, ""), _node("b", 2.0, "kept")], False)
    monkeypatch.setattr(twitch, "_comments_page", lambda v, o: page)
    records = list(twitch.iter_comments("1"))
    assert [r["message"] for r in records] == ["kept"]


def test_fetch_chat_output_is_readable_by_existing_parser(monkeypatch, tmp_path):
    """The whole point of the output shape: chat.py needs no changes."""
    page = _page([_node("b", 12.5, "LUL"), _node("a", 3.0, "first")], False)
    monkeypatch.setattr(twitch, "_comments_page", lambda v, o: page)

    out = twitch.fetch_chat("999", tmp_path / "chat.json")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert [r["content_offset_seconds"] for r in written] == [3.0, 12.5]

    parsed = load_chat(out)
    assert parsed == [{"time": 3.0, "text": "first"}, {"time": 12.5, "text": "LUL"}]


def test_missing_video_raises(monkeypatch):
    monkeypatch.setattr(twitch, "_post", lambda payload: [{"data": {"video": None}}])
    with pytest.raises(twitch.TwitchIngestError, match="no video"):
        twitch._comments_page("1", 0.0)


def test_unexpected_response_shape_raises(monkeypatch):
    monkeypatch.setattr(twitch, "_post", lambda payload: [{"errors": [{"message": "nope"}]}])
    with pytest.raises(twitch.TwitchIngestError, match="persisted-query hash"):
        twitch._comments_page("1", 0.0)
