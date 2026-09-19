"""A/B benchmarks for the choices that actually cost time or quality.

Transcription dominates wall time and the encoder dominates export time, so
those are where a wrong default is expensive. Everything here measures against
real media rather than synthetic input, and reports enough to judge the
quality cost of any speedup rather than just the speed.

Run it on an otherwise idle machine. Results are meaningless while an analysis
is running.

    python tools/benchmark.py transcribe --audio sample.wav
    python tools/benchmark.py encode --video vod.mp4 --start 300
    python tools/benchmark.py chat --chat chat.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from highlightminer.config import Settings  # noqa: E402
from highlightminer.media import require_executable  # noqa: E402


def _table(rows: list[dict], columns: list[str]) -> str:
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    head = " | ".join(c.ljust(widths[c]) for c in columns)
    rule = "-|-".join("-" * widths[c] for c in columns)
    body = "\n".join(
        " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows
    )
    return f"{head}\n{rule}\n{body}"


def _similarity(left: str, right: str) -> float:
    """Rough agreement between two transcripts, 0..1."""
    return SequenceMatcher(None, left.split(), right.split()).ratio()


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

def bench_transcribe(args: argparse.Namespace) -> int:
    from highlightminer.transcribe import transcribe_audio

    audio = Path(args.audio).expanduser().resolve()
    if not audio.exists():
        print(f"No such audio file: {audio}")
        return 2

    duration = args.duration
    base = Settings(whisper_model=args.model, cpu_threads=args.threads)

    variants: list[tuple[str, dict]] = [
        ("turbo int8 beam5 vad", {}),
        ("turbo int8 beam1 vad", {"beam_size": 1}),
        ("turbo int8 beam5 no-vad", {"vad_filter": False}),
        ("turbo int8_float32 beam5", {"compute_type": "int8_float32"}),
        ("turbo float32 beam5", {"compute_type": "float32"}),
        ("turbo int8 beam5 no-words", {"word_timestamps": False}),
        ("medium int8 beam5", {"whisper_model": "medium"}),
        ("small int8 beam5", {"whisper_model": "small"}),
    ]
    if args.only:
        wanted = set(args.only)
        variants = [v for v in variants if v[0] in wanted]

    reference_text = ""
    rows: list[dict] = []
    for label, overrides in variants:
        settings = dataclasses.replace(base, **overrides)
        try:
            started = time.perf_counter()
            segments, meta = transcribe_audio(audio, settings, audio_duration=duration)
            elapsed = time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001 - a failing variant is a result
            rows.append({"variant": label, "seconds": "FAILED", "note": str(exc)[:60]})
            print(f"  {label}: FAILED {exc}")
            continue

        text = " ".join(s["text"] for s in segments)
        if not reference_text:
            reference_text = text
        agreement = _similarity(reference_text, text)
        row = {
            "variant": label,
            "seconds": f"{elapsed:.1f}",
            "xRT": f"{duration / elapsed:.2f}" if elapsed else "-",
            "segments": len(segments),
            "words": sum(len(s.get("words") or []) for s in segments),
            "agree": f"{agreement * 100:.1f}%",
        }
        rows.append(row)
        print(f"  {label}: {elapsed:.1f}s ({duration / elapsed:.2f}x) agree {agreement * 100:.1f}%")

    print()
    print(_table(rows, ["variant", "seconds", "xRT", "segments", "words", "agree"]))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

_ENCODERS = [
    ("h264_amf quality", ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20"]),
    ("h264_amf speed", ["-c:v", "h264_amf", "-quality", "speed", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20"]),
    ("h264_qsv medium", ["-c:v", "h264_qsv", "-preset", "medium", "-global_quality", "20"]),
    ("libx264 medium", ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]),
    ("libx264 veryfast", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]),
]


def _ssim(ffmpeg: str, reference: Path, candidate: Path) -> float | None:
    """Mean SSIM of candidate against reference, or None if it cannot be read."""
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "info",
            "-i", str(candidate), "-i", str(reference),
            "-lavfi", "ssim", "-f", "null", "-",
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    for line in reversed((result.stderr or "").splitlines()):
        if "All:" in line:
            try:
                return float(line.split("All:")[1].split()[0])
            except (IndexError, ValueError):
                return None
    return None


def bench_encode(args: argparse.Namespace) -> int:
    from highlightminer.render import Layout, build_filter

    ffmpeg = require_executable("ffmpeg")
    video = Path(args.video).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vf = build_filter(Layout(kind=args.layout))
    reference = out_dir / "reference.mp4"

    # Lossless reference for the quality comparison.
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(args.start), "-i", str(video), "-t", str(args.seconds),
            "-map", "0:v:0?", "-vf", vf, "-c:v", "libx264", "-qp", "0", "-an",
            str(reference),
        ],
        check=True,
    )

    rows: list[dict] = []
    for label, encoder_args in _ENCODERS:
        out = out_dir / f"{label.replace(' ', '_')}.mp4"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(args.start), "-i", str(video), "-t", str(args.seconds),
            "-map", "0:v:0?", "-map", "0:a:0?", "-vf", vf,
            *encoder_args, "-c:a", "aac", "-b:a", "192k", str(out),
        ]
        started = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        if result.returncode != 0 or not out.exists():
            rows.append({"encoder": label, "seconds": "FAILED"})
            print(f"  {label}: FAILED")
            continue
        size_mb = out.stat().st_size / 1048576
        ssim = _ssim(ffmpeg, reference, out)
        rows.append({
            "encoder": label,
            "seconds": f"{elapsed:.1f}",
            "xRT": f"{args.seconds / elapsed:.2f}",
            "MB": f"{size_mb:.1f}",
            "SSIM": f"{ssim:.4f}" if ssim else "-",
        })
        print(f"  {label}: {elapsed:.1f}s  {size_mb:.1f} MB  SSIM {ssim}")

    print()
    print(_table(rows, ["encoder", "seconds", "xRT", "MB", "SSIM"]))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------
# Chat parsing
# --------------------------------------------------------------------------

def bench_chat(args: argparse.Namespace) -> int:
    """stdlib json against orjson, if it happens to be installed."""
    from highlightminer.chat import load_chat

    path = Path(args.chat).expanduser().resolve()
    rows: list[dict] = []

    def timed(label: str, fn, repeats: int = 3) -> None:
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            result = fn()
            samples.append(time.perf_counter() - started)
        rows.append({
            "parser": label,
            "median_s": f"{statistics.median(samples):.3f}",
            "records": len(result) if hasattr(result, "__len__") else "-",
        })
        print(f"  {label}: {statistics.median(samples):.3f}s")

    timed("load_chat (stdlib json)", lambda: load_chat(path))

    try:
        import orjson  # noqa: PLC0415

        timed("orjson.loads", lambda: orjson.loads(path.read_bytes()))
    except ImportError:
        print("  orjson not installed, skipping")

    timed("json.loads", lambda: json.loads(path.read_text(encoding="utf-8")))

    print()
    print(_table(rows, ["parser", "median_s", "records"]))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="Compare Whisper model and inference settings")
    t.add_argument("--audio", required=True, help="16 kHz mono WAV sample")
    t.add_argument("--duration", type=float, required=True, help="Sample length in seconds")
    t.add_argument("--model", default="turbo")
    t.add_argument("--threads", type=int, default=11)
    t.add_argument("--only", nargs="*", help="Run only these variant labels")
    t.add_argument("--out", default=None)
    t.set_defaults(func=bench_transcribe)

    e = sub.add_parser("encode", help="Compare encoders for speed, size and SSIM")
    e.add_argument("--video", required=True)
    e.add_argument("--start", type=float, default=300.0)
    e.add_argument("--seconds", type=float, default=30.0)
    e.add_argument("--layout", default="crop", choices=("letterbox", "crop"))
    e.add_argument("--out-dir", default="bench-encode")
    e.add_argument("--out", default=None)
    e.set_defaults(func=bench_encode)

    c = sub.add_parser("chat", help="Compare chat parsing")
    c.add_argument("--chat", required=True)
    c.set_defaults(func=bench_chat)

    args = parser.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
