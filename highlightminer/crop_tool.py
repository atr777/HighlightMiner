"""Frame sampling and overlay rendering for positioning a crop window.

A 9:16 crop of a 16:9 frame keeps only about a third of the width, so where you
put it matters. On gameplay with no facecam that choice is usually "centre, but
nudged off the stream overlay", and the only way to judge it is to look at real
frames from across the VOD rather than one.

Everything here is pure image work with no Streamlit involved, so it can be
tested without a browser and reused from the CLI.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .media import require_executable, require_ffmpeg
from .render import SHORT_FORM_HEIGHT, SHORT_FORM_WIDTH, Rect

# Sampling one frame tells you nothing about a 13 hour VOD: menus, cutscenes and
# loading screens all look different. Spread the samples out instead.
DEFAULT_SAMPLE_COUNT = 6

# Ignore the first and last few percent, which are usually starting soon screens
# and outros rather than representative gameplay.
_EDGE_MARGIN = 0.04

_DIM_ALPHA = 140
_OUTLINE_WIDTH = 4
_OUTLINE_COLOUR = (255, 209, 102)


class CropToolError(RuntimeError):
    """A frame could not be extracted."""


@dataclass(frozen=True)
class FrameSample:
    time: float
    path: Path


def sample_times(duration: float, count: int = DEFAULT_SAMPLE_COUNT) -> list[float]:
    """Evenly spaced timestamps across the usable middle of a VOD."""
    duration = max(0.0, float(duration))
    count = max(1, int(count))
    if duration <= 0:
        return [0.0] * count
    lo = duration * _EDGE_MARGIN
    hi = duration * (1.0 - _EDGE_MARGIN)
    if count == 1:
        return [round((lo + hi) / 2, 3)]
    step = (hi - lo) / (count - 1)
    return [round(lo + step * i, 3) for i in range(count)]


def extract_frame(video_path: str | Path, time_sec: float, out_path: str | Path, width: int = 960) -> Path:
    """Pull a single frame, scaled down for display."""
    require_ffmpeg()
    ffmpeg = require_executable("ffmpeg")
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        # -ss before -i seeks by keyframe, which is what makes this fast enough
        # to sample a 13 hour file interactively.
        "-ss", f"{max(0.0, float(time_sec)):.3f}",
        "-i", str(Path(video_path).expanduser().resolve()),
        "-frames:v", "1",
        "-vf", f"scale={int(width)}:-2",
        str(out),
    ]
    try:
        subprocess.run(command, check=True, shell=False, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise CropToolError(
            f"Could not read a frame at {time_sec:.1f}s. The VOD may be shorter than that."
        ) from exc
    if not out.exists():
        raise CropToolError(f"ffmpeg reported success but wrote no frame at {time_sec:.1f}s.")
    return out


def sample_frames(
    video_path: str | Path,
    duration: float,
    out_dir: str | Path,
    count: int = DEFAULT_SAMPLE_COUNT,
    width: int = 960,
) -> list[FrameSample]:
    """Extract evenly spaced frames, skipping any that cannot be read."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[FrameSample] = []
    for index, when in enumerate(sample_times(duration, count)):
        target = out_dir / f"frame_{index:02d}.png"
        try:
            samples.append(FrameSample(when, extract_frame(video_path, when, target, width)))
        except CropToolError:
            # A seek past the end or onto a corrupt region should not abandon
            # the whole sampling pass.
            continue
    if not samples:
        raise CropToolError("No frames could be extracted from this VOD.")
    return samples


def rect_for_aspect(
    x: float,
    y: float = 0.0,
    height: float = 1.0,
    source_aspect: float = 16 / 9,
    target_aspect: float = 9 / 16,
) -> Rect:
    """Build a Rect of the requested output aspect from a left edge and height.

    Working in fractions means the width needed for 9:16 depends on the source
    aspect, so this derives it rather than making the caller do the arithmetic.
    """
    height = min(max(float(height), 1e-6), 1.0)
    width = min(1.0, (target_aspect / source_aspect) * height)
    x = min(max(float(x), 0.0), 1.0 - width)
    y = min(max(float(y), 0.0), 1.0 - height)
    return Rect(x=round(x, 6), y=round(y, 6), w=round(width, 6), h=round(height, 6))


def annotate_frame(frame_path: str | Path, rect: Rect, out_path: str | Path) -> Path:
    """Dim everything outside the crop and outline what is kept."""
    source = Image.open(frame_path).convert("RGB")
    width, height = source.size
    left = int(rect.x * width)
    top = int(rect.y * height)
    right = int((rect.x + rect.w) * width)
    bottom = int((rect.y + rect.h) * height)

    overlay = Image.new("RGBA", source.size, (0, 0, 0, _DIM_ALPHA))
    # Punch the kept region back out of the dimming layer.
    ImageDraw.Draw(overlay).rectangle([left, top, right, bottom], fill=(0, 0, 0, 0))
    annotated = Image.alpha_composite(source.convert("RGBA"), overlay)

    ImageDraw.Draw(annotated).rectangle(
        [left, top, right - 1, bottom - 1],
        outline=_OUTLINE_COLOUR,
        width=_OUTLINE_WIDTH,
    )

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    annotated.convert("RGB").save(out)
    return out


def crop_preview(
    frame_path: str | Path,
    rect: Rect,
    out_path: str | Path,
    height: int = 640,
) -> Path:
    """Render what the crop actually produces, at the output aspect ratio."""
    source = Image.open(frame_path).convert("RGB")
    width, source_height = source.size
    box = (
        int(rect.x * width),
        int(rect.y * source_height),
        max(int((rect.x + rect.w) * width), int(rect.x * width) + 1),
        max(int((rect.y + rect.h) * source_height), int(rect.y * source_height) + 1),
    )
    cropped = source.crop(box)
    target_width = max(1, int(height * SHORT_FORM_WIDTH / SHORT_FORM_HEIGHT))
    cropped = cropped.resize((target_width, int(height)), Image.LANCZOS)

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(out)
    return out
