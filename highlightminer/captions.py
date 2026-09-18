"""Burned-in captions for short-form clips.

Builds an ASS subtitle file from Whisper word timings. ASS rather than chained
``drawtext`` filters, because a per-word drawtext graph for a 45 second clip
becomes hundreds of filters and is effectively unmaintainable.

Captions are grouped into short phrases of a few words. Each phrase is one
dialogue event, and words fill in as they are spoken using ASS karaoke timing,
which keeps one event per phrase rather than one event per word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ASS colours are &HAABBGGRR: alpha, then blue, green, red. Not RGB.
_WHITE = "&H00FFFFFF"
_BLACK = "&H00000000"
_ACCENT = "&H0000D7FF"  # amber, high contrast on almost any footage

_MAX_WORDS_PER_PHRASE = 4
_MAX_PHRASE_SECONDS = 2.5
# A gap longer than this ends a phrase: the speaker paused.
_PHRASE_GAP_SECONDS = 0.6
_MIN_EVENT_SECONDS = 0.2


@dataclass(frozen=True)
class CaptionStyle:
    font: str = "Arial"
    size: int = 96
    primary: str = _WHITE
    highlight: str = _ACCENT
    outline_colour: str = _BLACK
    outline: int = 6
    shadow: int = 2
    bold: bool = True
    # Distance from the bottom of the frame, in pixels.
    margin_vertical: int = 420
    margin_horizontal: int = 60
    highlight_words: bool = True
    uppercase: bool = False


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Phrase:
    words: list[Word] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


def _ass_timestamp(seconds: float) -> str:
    """ASS wants H:MM:SS.cc with centisecond precision."""
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis == 100:  # rounding can tip over a whole second
        centis = 0
        secs += 1
        if secs == 60:
            secs = 0
            minutes += 1
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _escape(text: str) -> str:
    r"""Escape ASS markup. A literal ``{`` would open an override block."""
    return text.replace("\\", "\\\\").replace("{", r"\{").replace("}", r"\}")


def words_in_window(segments: list[dict], start: float, end: float) -> list[Word]:
    """Collect word timings overlapping a clip window, rebased to clip time.

    Falls back to whole segments when word timings are unavailable, so a
    transcript produced before word timestamps were enabled still yields
    readable, if less lively, captions.
    """
    words: list[Word] = []
    for segment in segments:
        raw_words = segment.get("words") or []
        if raw_words:
            for word in raw_words:
                word_start = float(word.get("s", 0.0))
                word_end = float(word.get("e", word_start))
                if word_end < start or word_start > end:
                    continue
                text = str(word.get("w", "")).strip()
                if text:
                    words.append(
                        Word(text, max(0.0, word_start - start), max(0.0, word_end - start))
                    )
            continue

        seg_start = float(segment.get("start", 0.0))
        seg_end = float(segment.get("end", seg_start))
        if seg_end < start or seg_start > end:
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            words.append(Word(text, max(0.0, seg_start - start), max(0.0, seg_end - start)))
    return sorted(words, key=lambda w: w.start)


def group_phrases(
    words: list[Word],
    max_words: int = _MAX_WORDS_PER_PHRASE,
    max_seconds: float = _MAX_PHRASE_SECONDS,
    gap_seconds: float = _PHRASE_GAP_SECONDS,
) -> list[Phrase]:
    """Group words into short on-screen phrases."""
    phrases: list[Phrase] = []
    current = Phrase()
    for word in words:
        if current.words:
            too_many = len(current.words) >= max_words
            too_long = (word.end - current.start) > max_seconds
            paused = (word.start - current.words[-1].end) > gap_seconds
            if too_many or too_long or paused:
                phrases.append(current)
                current = Phrase()
        current.words.append(word)
    if current.words:
        phrases.append(current)
    return phrases


def _phrase_text(phrase: Phrase, style: CaptionStyle) -> str:
    """Render one phrase, optionally with karaoke word timing.

    ASS karaoke draws text in SecondaryColour until ``\\k`` reaches it, then
    switches it to PrimaryColour for the rest of the event. So the style sets
    PrimaryColour to the highlight and SecondaryColour to the base colour, and
    the phrase fills in word by word as it is spoken.
    """
    if not style.highlight_words:
        return " ".join(
            _escape(word.text.upper() if style.uppercase else word.text)
            for word in phrase.words
        )

    parts: list[str] = []
    # Karaoke timings are relative to the event start and run consecutively, so
    # silence between words has to be spent explicitly or the highlight drifts
    # ahead of the audio.
    cursor = phrase.start
    for index, word in enumerate(phrase.words):
        text = word.text.upper() if style.uppercase else word.text
        escaped = _escape(text)
        gap_cs = int(round((word.start - cursor) * 100))
        duration_cs = max(1, int(round((word.end - word.start) * 100)))
        separator = " " if index else ""
        gap = f"{{\\k{gap_cs}}}" if gap_cs > 0 else ""
        parts.append(f"{separator}{gap}{{\\k{duration_cs}}}{escaped}")
        cursor = word.end
    return "".join(parts)


def build_ass(
    words: list[Word],
    style: CaptionStyle | None = None,
    width: int = 1080,
    height: int = 1920,
) -> str:
    """Render an ASS subtitle document for one clip."""
    style = style or CaptionStyle()
    phrases = group_phrases(words)

    # With karaoke, words start in SecondaryColour and flip to PrimaryColour as
    # they are spoken. Without it, everything simply sits in the base colour.
    if style.highlight_words:
        primary, secondary = style.highlight, style.primary
    else:
        primary, secondary = style.primary, style.highlight

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{style.font},{style.size},{primary},{secondary},{style.outline_colour},{style.outline_colour},{-1 if style.bold else 0},0,0,0,100,100,0,0,1,{style.outline},{style.shadow},2,{style.margin_horizontal},{style.margin_horizontal},{style.margin_vertical},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines: list[str] = []
    for phrase in phrases:
        start = phrase.start
        end = max(phrase.end, start + _MIN_EVENT_SECONDS)
        text = _phrase_text(phrase, style)
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},Caption,,0,0,0,,{text}"
        )

    return header + "\n".join(lines) + ("\n" if lines else "")


def write_ass(
    path: str | Path,
    words: list[Word],
    style: CaptionStyle | None = None,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 with BOM: libass is more reliable about detecting encoding with one.
    out.write_text(build_ass(words, style, width, height), encoding="utf-8-sig")
    return out
