from __future__ import annotations

import json

import pytest

from highlightminer.ingest import kick, twitch, youtube
from highlightminer.ingest import router as resolve_module
from highlightminer.ingest.base import ChatUnavailable, IngestError, IngestResult, VodInfo
from highlightminer.ingest.router import ingest, is_supported_url, resolve


class TestRouting:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.twitch.tv/videos/2849127032", "twitch"),
            ("twitch.tv/videos/1", "twitch"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
            ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
            ("https://www.youtube.com/live/dQw4w9WgXcQ", "youtube"),
            ("https://kick.com/video/abc12345", "kick"),
            ("https://kick.com/someone/videos/abc12345", "kick"),
        ],
    )
    def test_routes_to_the_right_adapter(self, url, expected):
        assert resolve(url).platform == expected

    @pytest.mark.parametrize("url", ["https://vimeo.com/1", "not a url", ""])
    def test_rejects_unsupported(self, url):
        with pytest.raises(IngestError):
            resolve(url)

    def test_is_supported_url(self):
        assert is_supported_url("https://kick.com/video/abc12345")
        assert not is_supported_url("https://example.com/x")


class TestVideoIds:
    def test_twitch(self):
        assert twitch.video_id("https://www.twitch.tv/videos/2849127032") == "2849127032"

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/watch?list=x&v=dQw4w9WgXcQ",
        ],
    )
    def test_youtube(self, url):
        assert youtube.video_id(url) == "dQw4w9WgXcQ"

    def test_kick_both_url_shapes(self):
        assert kick.video_id("https://kick.com/video/abcdef12") == "abcdef12"
        assert kick.video_id("https://kick.com/someone/videos/abcdef12") == "abcdef12"

    def test_unrecognizable_raises(self):
        with pytest.raises(IngestError):
            youtube.video_id("https://www.youtube.com/@channel")


class TestYouTubeLiveChatParsing:
    def _action(self, offset_ms, text, author="viewer", emoji=None):
        runs = [{"text": text}] if text else []
        if emoji:
            runs.append({"emoji": {"shortcuts": [emoji]}})
        return {
            "replayChatItemAction": {
                "videoOffsetTimeMsec": str(offset_ms),
                "actions": [{
                    "addChatItemAction": {
                        "item": {
                            "liveChatTextMessageRenderer": {
                                "message": {"runs": runs},
                                "authorName": {"simpleText": author},
                            }
                        }
                    }
                }],
            }
        }

    def _write(self, tmp_path, actions):
        path = tmp_path / "chat.live_chat.json"
        path.write_text(
            "\n".join(json.dumps(a) for a in actions), encoding="utf-8"
        )
        return path

    def test_converts_offsets_to_seconds(self, tmp_path):
        path = self._write(tmp_path, [self._action(12500, "hello")])
        records = youtube.parse_live_chat(path)
        assert records == [
            {"content_offset_seconds": 12.5, "message": "hello", "commenter": "viewer"}
        ]

    def test_emoji_runs_become_shortcuts(self, tmp_path):
        path = self._write(tmp_path, [self._action(1000, "nice ", emoji=":smile:")])
        assert youtube.parse_live_chat(path)[0]["message"] == "nice :smile:"

    def test_sorts_by_offset(self, tmp_path):
        path = self._write(tmp_path, [self._action(5000, "b"), self._action(1000, "a")])
        assert [r["message"] for r in youtube.parse_live_chat(path)] == ["a", "b"]

    def test_skips_non_text_items(self, tmp_path):
        superchat = {
            "replayChatItemAction": {
                "videoOffsetTimeMsec": "1000",
                "actions": [{"addChatItemAction": {"item": {"liveChatPaidMessageRenderer": {}}}}],
            }
        }
        path = self._write(tmp_path, [superchat, self._action(2000, "real")])
        assert [r["message"] for r in youtube.parse_live_chat(path)] == ["real"]

    def test_skips_malformed_lines(self, tmp_path):
        path = tmp_path / "chat.live_chat.json"
        path.write_text(
            "not json\n" + json.dumps(self._action(1000, "kept")) + "\n\n",
            encoding="utf-8",
        )
        assert [r["message"] for r in youtube.parse_live_chat(path)] == ["kept"]

    def test_output_is_readable_by_the_existing_chat_parser(self, tmp_path):
        from highlightminer.chat import load_chat

        raw = self._write(tmp_path, [self._action(2500, "hello"), self._action(500, "first")])
        records = youtube.parse_live_chat(raw)
        out = tmp_path / "chat.json"
        out.write_text(json.dumps(records), encoding="utf-8")
        assert load_chat(out) == [
            {"time": 0.5, "text": "first"},
            {"time": 2.5, "text": "hello"},
        ]


class TestKick:
    def test_chat_is_explicitly_unavailable(self):
        with pytest.raises(ChatUnavailable, match="not supported"):
            kick.fetch_chat("https://kick.com/video/abc12345", "out.json")

    def test_external_export_can_be_converted(self, tmp_path):
        out = kick.convert_chat_export(
            [{"time": 10.0, "message": "b"}, {"time": 2.0, "text": "a"}],
            tmp_path / "chat.json",
        )
        records = json.loads(out.read_text(encoding="utf-8"))
        assert [r["message"] for r in records] == ["a", "b"]

    def test_empty_export_is_rejected(self, tmp_path):
        with pytest.raises(ChatUnavailable):
            kick.convert_chat_export([{"nonsense": 1}], tmp_path / "chat.json")


class TestIngest:
    def _patch(self, monkeypatch, tmp_path, *, chat_error=None):
        video = tmp_path / "twitch-1.mp4"
        video.write_bytes(b"x")
        monkeypatch.setattr(
            resolve_module, "probe_url",
            lambda url: VodInfo(platform="", video_id="1", title="T", duration=60.0),
        )
        monkeypatch.setattr(
            resolve_module, "download_video",
            lambda url, dest, **kw: video,
        )

        def fake_chat(url, out_path):
            if chat_error:
                raise ChatUnavailable(chat_error)
            path = tmp_path / "chat.json"
            path.write_text("[]", encoding="utf-8")
            return path

        monkeypatch.setattr(twitch, "fetch_chat", fake_chat)
        return video

    def test_returns_video_and_chat(self, monkeypatch, tmp_path):
        video = self._patch(monkeypatch, tmp_path)
        result = ingest("https://twitch.tv/videos/1", tmp_path)
        assert isinstance(result, IngestResult)
        assert result.video_path == video
        assert result.chat_path is not None
        assert result.chat_error is None
        assert result.info.platform == "twitch"

    def test_chat_failure_is_not_fatal(self, monkeypatch, tmp_path):
        """No chat replay still analyzes on audio and speech."""
        self._patch(monkeypatch, tmp_path, chat_error="no replay here")
        result = ingest("https://twitch.tv/videos/1", tmp_path)
        assert result.chat_path is None
        assert "no replay here" in result.chat_error

    def test_chat_can_be_skipped(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path)
        result = ingest("https://twitch.tv/videos/1", tmp_path, with_chat=False)
        assert result.chat_path is None
        assert result.chat_error is None

    def test_progress_is_reported(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, tmp_path)
        stages: list[str] = []
        ingest(
            "https://twitch.tv/videos/1", tmp_path,
            progress=lambda stage, frac, msg: stages.append(stage),
        )
        assert stages[0] == "probe" and stages[-1] == "done"
