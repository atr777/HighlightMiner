"""Vertical (9:16) reframing for short-form export.

Layouts are data, not code paths. A layout describes which rectangles of the
source frame go where in the output frame, so gameplay-only VODs, webcam
streams and "just make it fit" all run through one filter builder.

Rectangles are fractions of the source frame (0.0 to 1.0) rather than pixels,
so a template drawn once on a 1080p frame still applies to a 720p VOD from the
same channel.

Everything here is pure string construction. ffmpeg is never invoked, which
keeps it cheap to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SHORT_FORM_WIDTH = 1080
SHORT_FORM_HEIGHT = 1920

LayoutKind = Literal["letterbox", "crop", "webcam"]

# Cheaper than gblur and visually indistinguishable once it is this soft.
_DEFAULT_BLUR = "boxblur=20:2"


class LayoutError(ValueError):
    """A layout is missing a rectangle it needs, or a rectangle is out of bounds."""


@dataclass(frozen=True)
class Rect:
    """A region of the source frame, as fractions of its width and height."""

    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "w", "h"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise LayoutError(f"Rect.{name} must be between 0.0 and 1.0, got {value}")
        if self.w <= 0.0 or self.h <= 0.0:
            raise LayoutError("Rect width and height must be greater than zero.")
        if self.x + self.w > 1.0 + 1e-9:
            raise LayoutError("Rect extends past the right edge of the frame.")
        if self.y + self.h > 1.0 + 1e-9:
            raise LayoutError("Rect extends past the bottom edge of the frame.")

    def is_full_frame(self) -> bool:
        return (
            abs(self.x) < 1e-9
            and abs(self.y) < 1e-9
            and abs(self.w - 1.0) < 1e-9
            and abs(self.h - 1.0) < 1e-9
        )

    def crop_filter(self) -> str:
        """An ffmpeg crop expression in terms of the input dimensions."""
        return (
            f"crop=iw*{self.w:.6g}:ih*{self.h:.6g}"
            f":iw*{self.x:.6g}:ih*{self.y:.6g}"
        )

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, data: dict) -> "Rect":
        return cls(
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            w=float(data.get("w", 1.0)),
            h=float(data.get("h", 1.0)),
        )


def centered_vertical_slice(aspect_w: int = 9, aspect_h: int = 16) -> Rect:
    """The centered 9:16 slice of a 16:9 frame.

    This keeps only about 28% of the source width, which is why the crop layout
    is worth checking against real footage before trusting it: minimaps, kill
    feeds and health bars all live near the edges.
    """
    source_aspect = 16 / 9
    target_aspect = aspect_w / aspect_h
    width_fraction = min(1.0, target_aspect / source_aspect)
    return Rect(x=(1.0 - width_fraction) / 2.0, y=0.0, w=width_fraction, h=1.0)


@dataclass(frozen=True)
class Layout:
    """How to build one vertical frame out of one source frame."""

    kind: LayoutKind = "letterbox"
    gameplay: Rect | None = None
    webcam: Rect | None = None
    blur: str = _DEFAULT_BLUR
    # Share of the output height given to the webcam band in a webcam layout.
    webcam_fraction: float = 0.3

    def __post_init__(self) -> None:
        if self.kind not in ("letterbox", "crop", "webcam"):
            raise LayoutError(f"Unknown layout kind: {self.kind!r}")
        if self.kind == "webcam" and self.webcam is None:
            raise LayoutError("A webcam layout needs a webcam rectangle.")
        if not 0.05 <= float(self.webcam_fraction) <= 0.95:
            raise LayoutError("webcam_fraction must be between 0.05 and 0.95.")

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "gameplay": self.gameplay.as_dict() if self.gameplay else None,
            "webcam": self.webcam.as_dict() if self.webcam else None,
            "blur": self.blur,
            "webcam_fraction": self.webcam_fraction,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Layout":
        gameplay = data.get("gameplay")
        webcam = data.get("webcam")
        return cls(
            kind=data.get("kind", "letterbox"),
            gameplay=Rect.from_dict(gameplay) if gameplay else None,
            webcam=Rect.from_dict(webcam) if webcam else None,
            blur=data.get("blur") or _DEFAULT_BLUR,
            webcam_fraction=float(data.get("webcam_fraction", 0.3)),
        )


def _even(value: float) -> int:
    """H.264 needs even dimensions."""
    return max(2, int(round(value / 2)) * 2)


def _cover(width: int, height: int) -> str:
    """Scale to fill the target and trim the overflow, preserving aspect."""
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )


def _contain(width: int, height: int) -> str:
    """Scale to fit inside the target, preserving aspect."""
    return f"scale={width}:{height}:force_original_aspect_ratio=decrease"


def _letterbox_filter(layout: Layout, width: int, height: int) -> str:
    source = layout.gameplay
    prefix = f"{source.crop_filter()}," if source and not source.is_full_frame() else ""
    # Blur the background at a quarter scale and enlarge afterwards. Blurring a
    # full 1080x1920 frame is expensive enough to dominate the encode (measured
    # 53.6s against 14.8s for an unblurred layout on the same 44s clip), and the
    # result is indistinguishable once it is this soft.
    blur_width, blur_height = _even(width / 4), _even(height / 4)
    return (
        f"{prefix}split=2[bg][fg];"
        f"[bg]{_cover(blur_width, blur_height)},{layout.blur},"
        f"scale={width}:{height}[bgb];"
        f"[fg]{_contain(width, height)}[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2"
    )


def _crop_filter(layout: Layout, width: int, height: int) -> str:
    region = layout.gameplay or centered_vertical_slice()
    prefix = f"{region.crop_filter()}," if not region.is_full_frame() else ""
    return f"{prefix}{_cover(width, height)}"


def _webcam_filter(layout: Layout, width: int, height: int) -> str:
    """Webcam band above gameplay, each given an explicit share of the height.

    Both bands are cover-fitted into their own band rather than stacked at their
    natural heights, so the result is exactly ``width`` x ``height`` regardless
    of the source rectangles and nothing is trimmed unpredictably afterwards.
    """
    assert layout.webcam is not None  # guaranteed by Layout.__post_init__
    gameplay = layout.gameplay or centered_vertical_slice()
    cam_height = _even(height * float(layout.webcam_fraction))
    game_height = _even(height - cam_height)
    # Rounding both bands to even can drift a pixel or two off the target.
    total = cam_height + game_height
    if total != height:
        game_height = _even(game_height + (height - total))
    return (
        f"split=2[cam][game];"
        f"[cam]{layout.webcam.crop_filter()},{_cover(width, cam_height)}[camS];"
        f"[game]{gameplay.crop_filter()},{_cover(width, game_height)}[gameS];"
        f"[camS][gameS]vstack=2,{_cover(width, height)}"
    )


_BUILDERS = {
    "letterbox": _letterbox_filter,
    "crop": _crop_filter,
    "webcam": _webcam_filter,
}


def build_filter(
    layout: Layout | None = None,
    *,
    width: int = SHORT_FORM_WIDTH,
    height: int = SHORT_FORM_HEIGHT,
    subtitles: str | None = None,
) -> str:
    """Build the ffmpeg -vf filtergraph for one vertical clip.

    ``subtitles`` is an ffmpeg-escaped path to an ASS file, appended last so
    captions are burned on top of the finished frame rather than being scaled
    or cropped along with the source.
    """
    layout = layout or Layout()
    chain = _BUILDERS[layout.kind](layout, width, height)
    chain = f"{chain},setsar=1"
    if subtitles:
        chain = f"{chain},ass={subtitles}"
    return chain


def escape_filter_path(path: str) -> str:
    r"""Quote and escape a path for use as an ffmpeg filter argument value.

    ``C:\work\clip.ass`` has to reach the filter parser as
    ``'C\:/work/clip.ass'``. Three things are going on:

    - backslashes become forward slashes, so they are not read as escapes;
    - the drive colon is escaped, since a bare colon separates filter options;
    - the whole value is single quoted, because escaping the colon alone is not
      enough. ffmpeg still splits on it and reads the remainder of the path as
      the filter's second positional option, failing with a confusing complaint
      about ``original_size``. Quoting also covers spaces and commas.
    """
    normalized = str(path).replace("\\", "/")
    escaped = normalized.replace(":", r"\:")
    # Close the quote, emit an escaped literal quote, reopen. Rare in paths,
    # but a stray apostrophe would otherwise end the quoted value early.
    escaped = escaped.replace("'", r"'\''")
    return f"'{escaped}'"
