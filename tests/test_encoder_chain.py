from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from highlightminer import export


@pytest.fixture
def chain(monkeypatch):
    """Record which encoders get attempted, and let tests choose which fail."""
    attempts: list[str] = []
    failing: set[str] = set()
    commands: dict[str, list[str]] = {}

    def fake_run_encode(command: list[str], *, encoder: str) -> None:
        attempts.append(encoder)
        commands[encoder] = command
        if encoder in failing:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(export, "_run_encode", fake_run_encode)
    monkeypatch.setattr(export, "has_encoder", lambda name: True)
    return type("Chain", (), {"attempts": attempts, "failing": failing, "commands": commands})


def _encode(tmp_path: Path, **kwargs) -> None:
    export._run_h264_encode(
        "ffmpeg", Path("src.mp4"), tmp_path / "out.mp4", 0.0, 10.0, **kwargs
    )


def test_prefers_nvenc_when_it_works(chain, tmp_path):
    _encode(tmp_path)
    assert chain.attempts == ["h264_nvenc"]


def test_falls_through_nvenc_to_amf(chain, tmp_path):
    """The measured case on this machine: ffmpeg lists NVENC, no NVIDIA card."""
    chain.failing.add("h264_nvenc")
    _encode(tmp_path)
    assert chain.attempts == ["h264_nvenc", "h264_amf"]


def test_falls_all_the_way_to_libx264(chain, tmp_path):
    chain.failing.update({"h264_nvenc", "h264_amf", "h264_qsv"})
    _encode(tmp_path)
    assert chain.attempts == ["h264_nvenc", "h264_amf", "h264_qsv", "libx264"]


def test_software_failure_propagates(chain, tmp_path):
    chain.failing.update({"h264_nvenc", "h264_amf", "h264_qsv", "libx264"})
    with pytest.raises(subprocess.CalledProcessError):
        _encode(tmp_path)


def test_skips_encoders_ffmpeg_does_not_have(chain, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "has_encoder", lambda name: name == "h264_qsv")
    _encode(tmp_path)
    assert chain.attempts == ["h264_qsv"]


def test_no_hardware_encoders_goes_straight_to_software(chain, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "has_encoder", lambda name: False)
    _encode(tmp_path)
    assert chain.attempts == ["libx264"]


def test_preview_uses_downscale_filter_by_default(chain, tmp_path):
    _encode(tmp_path, preview=True)
    command = chain.commands["h264_nvenc"]
    assert "-vf" in command
    assert command[command.index("-vf") + 1] == export._PREVIEW_SCALE_FILTER


def test_final_export_has_no_filter_by_default(chain, tmp_path):
    _encode(tmp_path)
    assert "-vf" not in chain.commands["h264_nvenc"]


def test_custom_filters_override_the_preview_default(chain, tmp_path):
    _encode(tmp_path, preview=True, video_filters="scale=1080:1920")
    command = chain.commands["h264_nvenc"]
    assert command[command.index("-vf") + 1] == "scale=1080:1920"


def test_custom_filters_apply_to_final_exports(chain, tmp_path):
    _encode(tmp_path, video_filters="crop=100:100")
    command = chain.commands["h264_nvenc"]
    assert command[command.index("-vf") + 1] == "crop=100:100"


def test_empty_filter_string_disables_filtering(chain, tmp_path):
    _encode(tmp_path, preview=True, video_filters="")
    assert "-vf" not in chain.commands["h264_nvenc"]


def test_extra_inputs_are_placed_after_the_source(chain, tmp_path):
    _encode(tmp_path, extra_inputs=["-i", "overlay.png"])
    command = chain.commands["h264_nvenc"]
    assert command[command.index("src.mp4") + 1 : command.index("src.mp4") + 3] == ["-i", "overlay.png"]


def test_preview_and_final_use_different_audio_bitrates(chain, tmp_path):
    _encode(tmp_path, preview=True)
    preview_command = chain.commands["h264_nvenc"]
    chain.attempts.clear()
    _encode(tmp_path)
    final_command = chain.commands["h264_nvenc"]
    assert preview_command[preview_command.index("-b:a") + 1] == "128k"
    assert final_command[final_command.index("-b:a") + 1] == "192k"


class TestExportQuality:
    """One quantizer drives every encoder, on roughly the same 0..51 scale."""

    def test_quality_reaches_every_encoder(self):
        from highlightminer.export import _encoder_variants, _software_variant

        for name, args in _encoder_variants(23):
            assert "23" in args, f"{name} ignored the quality setting"
        assert "23" in _software_variant(23)[1]

    def test_default_is_the_measured_balance(self):
        from highlightminer.config import Settings
        from highlightminer.export import DEFAULT_EXPORT_QUALITY

        # 23 measured 7.6 Mbps at SSIM 0.9887 against 11.3 Mbps at 0.9914 for 20.
        assert DEFAULT_EXPORT_QUALITY == 23
        assert Settings().export_quality == 23

    def test_export_passes_quality_through(self, chain, tmp_path):
        _encode(tmp_path, quality=29)
        assert "29" in chain.commands["h264_nvenc"]

    def test_previews_use_their_own_quality(self, chain, tmp_path):
        from highlightminer.export import PREVIEW_QUALITY

        _encode(tmp_path, preview=True)
        assert str(PREVIEW_QUALITY) in chain.commands["h264_nvenc"]

    def test_preview_quality_is_not_overridden_by_export_quality(self, chain, tmp_path):
        from highlightminer.export import PREVIEW_QUALITY

        _encode(tmp_path, preview=True, quality=15)
        assert str(PREVIEW_QUALITY) in chain.commands["h264_nvenc"]

    @pytest.mark.parametrize("bad", [-1, 52])
    def test_settings_reject_out_of_range_quality(self, bad):
        from highlightminer.config import Settings

        with pytest.raises(ValueError, match="export_quality"):
            Settings(export_quality=bad)
