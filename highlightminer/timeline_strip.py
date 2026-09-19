"""A visual strip of what surrounds a clip, for setting its in and out points.

Retiming by typing seconds into a box means guessing, then waiting for a
re-encode to find out whether the guess was right. This draws the audio energy
and the actual words around the candidate so the boundaries can be seen rather
than estimated.

Pure image work, no ffmpeg, so it redraws instantly while dragging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

WIDTH = 900
HEIGHT = 150
_PAD_SECONDS = 8.0

# The kept span is the lighter panel. Trimmed material sits on the darker
# base, so "bright is kept" holds for the background as well as the waveform.
_BACKGROUND = (12, 12, 14)
_KEPT_PANEL = (26, 26, 31)
_WAVE = (90, 140, 200)
_WAVE_INSIDE = (130, 190, 255)
_MARKER = (255, 209, 102)
_WORD = (150, 150, 158)
_WORD_INSIDE = (235, 235, 240)
_AXIS = (70, 70, 78)

_WAVE_TOP = 22
_WAVE_BOTTOM = 96
_WORD_ROW = 104


@dataclass(frozen=True)
class StripWindow:
    """The span drawn, and the clip boundaries inside it."""

    view_start: float
    view_end: float
    clip_start: float
    clip_end: float

    @property
    def span(self) -> float:
        return max(1e-6, self.view_end - self.view_start)

    def x(self, when: float) -> int:
        fraction = (when - self.view_start) / self.span
        return int(min(max(fraction, 0.0), 1.0) * (WIDTH - 1))


def window_for(clip_start: float, clip_end: float, duration: float, pad: float = _PAD_SECONDS) -> StripWindow:
    """Show the clip plus some context either side, clamped to the source."""
    view_start = max(0.0, clip_start - pad)
    view_end = min(duration, clip_end + pad) if duration > 0 else clip_end + pad
    if view_end <= view_start:
        view_end = view_start + 1.0
    return StripWindow(view_start, view_end, clip_start, clip_end)


def _energy_at(audio_features: list[dict], when: float) -> float:
    """Nearest loudness value. Features sit on a regular grid, so this is a lookup.

    Prefers ``energy`` over ``score``: the excitement score saturates near 1.0
    through anything lively, which draws as a solid block. Loudness keeps the
    shape that makes speech and pauses distinguishable.
    """
    if not audio_features:
        return 0.0

    def value(row: dict) -> float:
        for key in ("energy", "score"):
            if key in row:
                try:
                    return float(row[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    first = float(audio_features[0].get("time", 0.0))
    if len(audio_features) < 2:
        return value(audio_features[0])
    step = float(audio_features[1].get("time", 0.5)) - first
    if step <= 0:
        return 0.0
    index = int(round((when - first) / step))
    index = min(max(index, 0), len(audio_features) - 1)
    return value(audio_features[index])


def flatten_words(segments: list[dict]) -> list[tuple[float, float, str]]:
    """All word timings from a transcript window, in order."""
    words: list[tuple[float, float, str]] = []
    for segment in segments or []:
        for word in segment.get("words") or []:
            try:
                start = float(word["s"])
                end = float(word["e"])
            except (KeyError, TypeError, ValueError):
                continue
            text = str(word.get("w", "")).strip()
            if text:
                words.append((start, end, text))
    return sorted(words)


def snap_start(words: list[tuple[float, float, str]], when: float, direction: int) -> float | None:
    """The previous or next word start, for exact boundaries.

    Word timings are already stored for captions, so a clip can begin exactly
    where someone starts speaking instead of a guessed number of seconds.
    """
    starts = sorted({round(w[0], 3) for w in words})
    if not starts:
        return None
    if direction < 0:
        earlier = [s for s in starts if s < when - 0.01]
        return earlier[-1] if earlier else None
    later = [s for s in starts if s > when + 0.01]
    return later[0] if later else None


def snap_end(words: list[tuple[float, float, str]], when: float, direction: int) -> float | None:
    """The previous or next word end."""
    ends = sorted({round(w[1], 3) for w in words})
    if not ends:
        return None
    if direction < 0:
        earlier = [e for e in ends if e < when - 0.01]
        return earlier[-1] if earlier else None
    later = [e for e in ends if e > when + 0.01]
    return later[0] if later else None


def render_strip(
    out_path: str | Path,
    window: StripWindow,
    audio_features: list[dict],
    words: list[tuple[float, float, str]],
) -> Path:
    """Draw energy, words and the current in/out points."""
    image = Image.new("RGB", (WIDTH, HEIGHT), _BACKGROUND)
    draw = ImageDraw.Draw(image)

    start_x = window.x(window.clip_start)
    end_x = window.x(window.clip_end)

    # The kept span is lifted out of the darker base, so it reads at a glance.
    draw.rectangle([start_x, 0, end_x, HEIGHT], fill=_KEPT_PANEL)

    height = _WAVE_BOTTOM - _WAVE_TOP
    middle = _WAVE_TOP + height // 2
    for x in range(WIDTH):
        when = window.view_start + (x / max(1, WIDTH - 1)) * window.span
        amplitude = max(0.02, _energy_at(audio_features, when))
        half = int(amplitude * height / 2)
        colour = _WAVE_INSIDE if start_x <= x <= end_x else _WAVE
        draw.line([(x, middle - half), (x, middle + half)], fill=colour)

    draw.line([(0, middle), (WIDTH, middle)], fill=_AXIS)

    # Word ticks, with text where there is room for it.
    last_text_x = -999
    for word_start, word_end, text in words:
        if word_end < window.view_start or word_start > window.view_end:
            continue
        wx = window.x(word_start)
        inside = start_x <= wx <= end_x
        draw.line([(wx, _WORD_ROW), (wx, _WORD_ROW + 6)], fill=_WORD_INSIDE if inside else _WORD)
        if wx - last_text_x > 34:
            draw.text(
                (wx + 2, _WORD_ROW + 8),
                text[:12],
                fill=_WORD_INSIDE if inside else _WORD,
            )
            last_text_x = wx

    for x, label in ((start_x, "in"), (end_x, "out")):
        draw.line([(x, 0), (x, HEIGHT)], fill=_MARKER, width=2)
        draw.text((min(x + 4, WIDTH - 24), 4), label, fill=_MARKER)

    draw.text((4, HEIGHT - 14), f"{window.view_start:.1f}s", fill=_AXIS)
    draw.text((WIDTH - 54, HEIGHT - 14), f"{window.view_end:.1f}s", fill=_AXIS)
    draw.text(
        (max(4, start_x + 4), HEIGHT - 14),
        f"{window.clip_end - window.clip_start:.1f}s clip",
        fill=_MARKER,
    )

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out
