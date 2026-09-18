from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from .config import Settings
from .desktop import (
    UI_URL,
    desktop_runtime_probe,
    open_system_browser,
    resolve_ui_mode,
    run_desktop_shell,
    show_native_error,
    wait_for_server,
)
from .doctor import run_doctor
from .export import export_clip
from .model_access import ModelDecisionRequired
from .pipeline import analyze_vod
from .review import load_review
from .runtime import app_root, bundled_path, is_frozen
from .security import validate_local_video
from .settings_store import load_app_settings
from .shutdown import clear_shutdown_admission, request_shutdown_admission
from .storage import (
    default_db_path,
    import_legacy_analysis,
    learning_summary,
    list_analyses,
    load_analysis,
    record_export,
    transcript_window,
)
from .transcription_status import transcription_status

_STREAMLIT_CHILD_ARG = "__streamlit_child__"
_DESKTOP_PROBE_ARG = "__desktop_probe__"
_STREAMLIT_FLAG_OPTIONS = {
    "global_developmentMode": False,
    "server_headless": True,
    "server_address": "127.0.0.1",
    "server_port": 8501,
    "server_showEmailPrompt": False,
    "browser_serverAddress": "127.0.0.1",
    "browser_serverPort": 8501,
    "browser_gatherUsageStats": False,
    "client_toolbarMode": "minimal",
}


def _progress(message: str, value: float) -> None:
    print(f"[{value * 100:5.1f}%] {message}", flush=True)


def cmd_analyze(args: argparse.Namespace) -> int:
    settings = Settings.from_file(args.settings) if args.settings else load_app_settings(args.db)
    try:
        analysis_id = analyze_vod(
            args.video,
            args.work_dir,
            settings,
            args.chat,
            _progress,
            content_label=args.content,
            db_path=args.db,
            reuse_features=not args.no_reuse,
            allow_model_download=bool(args.allow_model_download),
            skip_transcription=bool(args.no_transcription),
        )
    except ModelDecisionRequired as exc:
        print(f"Cannot start analysis: {exc}", file=sys.stderr)
        print(
            "The CLI is non-interactive. Use --allow-model-download for this command, "
            "use --no-transcription, or choose a local model in the desktop Settings page.",
            file=sys.stderr,
        )
        return 2

    analysis = load_analysis(args.db, analysis_id)
    print(f"Analysis ID: {analysis_id}")
    print(f"Source run: {analysis.get('run_number', 1)}")
    reused = analysis.get("cache", {}).get("reused_stages", [])
    print(f"Reused stages: {', '.join(reused) if reused else 'none'}")
    print(f"Speech recognition: {transcription_status(analysis.get('transcription'))}")
    print(f"Settings: {'JSON override ' + str(Path(args.settings).expanduser().resolve()) if args.settings else 'active database profile'}")
    print(f"Database: {Path(args.db).expanduser().resolve()}")
    return 0


def _streamlit_app_path() -> Path:
    if is_frozen():
        return bundled_path("highlightminer", "app.py")
    return Path(__file__).resolve().with_name("app.py")


def _run_streamlit_child(app_path: str) -> int:
    path = Path(app_path).resolve()
    if not path.is_file():
        print(f"Bundled Streamlit app not found: {path}", file=sys.stderr, flush=True)
        return 2

    from streamlit import config as streamlit_config
    from streamlit.runtime.credentials import check_credentials
    from streamlit.web import bootstrap

    main_script_path = os.path.abspath(path)
    streamlit_config._main_script_path = main_script_path
    flag_options = dict(_STREAMLIT_FLAG_OPTIONS)
    bootstrap.load_config_options(flag_options=flag_options)
    check_credentials()
    print(f"Starting embedded Streamlit server for {main_script_path}", flush=True)
    bootstrap.run(main_script_path, False, [], flag_options)
    return 0


def _stop_ui_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.terminate()
        process.wait(timeout=3)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _monitor_ui_process(
    process: subprocess.Popen,
    shutdown_file: Path,
    *,
    shutdown_blocker: Callable[[], str | None] | None = None,
) -> bool:
    shutdown_requested = False
    while process.poll() is None:
        if shutdown_file.exists():
            block_reason = None
            if shutdown_blocker is not None:
                try:
                    block_reason = shutdown_blocker()
                except Exception as exc:
                    block_reason = (
                        "HighlightMiner could not verify whether background work is active "
                        f"({type(exc).__name__}: {exc}). Wait and try exiting again."
                    )
            if block_reason:
                print(f"Shutdown blocked: {block_reason}", flush=True)
                shutdown_file.unlink(missing_ok=True)
                continue
            shutdown_requested = True
            print("Shutdown requested from HighlightMiner UI...", flush=True)
            time.sleep(0.75)
            _stop_ui_process(process)
            break
        time.sleep(0.25)
    return shutdown_requested


def cmd_ui(args: argparse.Namespace | None = None) -> int:
    app = _streamlit_app_path()
    if not app.is_file():
        raise RuntimeError(f"Streamlit application file is missing: {app}")

    browser_requested = bool(getattr(args, "browser", False))
    mode = resolve_ui_mode(browser_requested=browser_requested)

    shutdown_file = Path(tempfile.gettempdir()) / f"highlightminer-shutdown-{os.getpid()}.flag"
    shutdown_file.unlink(missing_ok=True)
    db_path = default_db_path()
    clear_shutdown_admission(db_path)
    shutdown_blocker = lambda: request_shutdown_admission(db_path)
    env = os.environ.copy()
    env["HIGHLIGHTMINER_SHUTDOWN_FILE"] = str(shutdown_file)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    command = [sys.executable, _STREAMLIT_CHILD_ARG, str(app)] if is_frozen() else [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.headless=true",
        "--server.address=127.0.0.1",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        creationflags=creationflags,
        cwd=str(app_root()),
        shell=False,
    )

    shutdown_requested = False
    desktop_closed_normally = False
    try:
        wait_for_server(process, url=UI_URL, timeout=60.0)
        if mode == "desktop":
            try:
                run_desktop_shell(
                    process,
                    shutdown_file,
                    url=UI_URL,
                    stop_backend=lambda: _stop_ui_process(process),
                    shutdown_blocker=shutdown_blocker,
                )
                shutdown_requested = shutdown_file.exists()
                desktop_closed_normally = process.poll() is None or shutdown_requested
            except Exception as exc:
                message = (
                    "HighlightMiner could not open its embedded desktop window.\n\n"
                    "The Microsoft Edge WebView2 Runtime may be missing or the pywebview runtime may have failed.\n\n"
                    "You can still launch the fallback from a terminal with:\n"
                    "HighlightMiner.exe ui --browser\n\n"
                    f"Technical detail: {type(exc).__name__}: {exc}"
                )
                show_native_error("HighlightMiner desktop UI failed", message)
                raise RuntimeError(message) from exc
        elif mode == "browser":
            open_system_browser(UI_URL)
            shutdown_requested = _monitor_ui_process(
                process,
                shutdown_file,
                shutdown_blocker=shutdown_blocker,
            )
        elif mode == "server":
            shutdown_requested = _monitor_ui_process(
                process,
                shutdown_file,
                shutdown_blocker=shutdown_blocker,
            )
        else:
            raise RuntimeError(f"Unsupported UI mode: {mode}")
    except KeyboardInterrupt:
        print("\nStopping HighlightMiner UI...", flush=True)
    finally:
        if process.poll() is None:
            _stop_ui_process(process)
        shutdown_file.unlink(missing_ok=True)
        if process.poll() is not None:
            try:
                clear_shutdown_admission(db_path)
            except Exception as exc:
                print(
                    "Warning: could not clear the shutdown admission marker "
                    f"({type(exc).__name__}: {exc}).",
                    file=sys.stderr,
                    flush=True,
                )

    return_code = process.wait()
    if shutdown_requested or desktop_closed_normally:
        print("HighlightMiner UI stopped.", flush=True)
        return 0
    return return_code


def cmd_history(args: argparse.Namespace) -> int:
    rows = list_analyses(args.db, args.limit)
    if not rows:
        print("No analyses in the database.")
        return 0
    for row in rows:
        print(
            f"{row['id']}  run={row['run_number']}  {row['created_at']}  {row['content_label']}  "
            f"{row['video_name']}  candidates={row['candidates']} "
            f"kept={row['kept']} rejected={row['rejected']} unreviewed={row['unreviewed']}"
        )
    return 0


def cmd_learning_stats(args: argparse.Namespace) -> int:
    stats = learning_summary(args.db)
    print("HighlightMiner learning dataset")
    print(f"Total candidates: {stats['total']}")
    print(f"Keep:             {stats['kept']}")
    print(f"Reject:           {stats['rejected']}")
    print(f"Unreviewed:       {stats['unreviewed']}")
    print(f"Exported:         {stats['exported']}")
    print("Unreviewed candidates are preserved as unlabeled examples, not treated as rejects.")
    return 0


def cmd_import_legacy(args: argparse.Namespace) -> int:
    analysis_id = import_legacy_analysis(args.analysis_json, args.db)
    print(f"Imported analysis ID: {analysis_id}")
    print(f"Database: {Path(args.db).expanduser().resolve()}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    analysis = load_analysis(args.db, args.analysis_id)
    review = load_review(args.db, args.analysis_id, analysis)
    out_dir = args.output or str(Path(analysis["work_dir"]) / "clips")
    source_video = validate_local_video(analysis["video_path"])

    chosen = []
    for c in analysis.get("candidates", []):
        r = review["items"][c["id"]]
        if args.all or r.get("status") == "keep":
            chosen.append((c, r))
    if not chosen:
        print("No clips selected. Mark clips Keep in the UI or pass --all.")
        return 2

    layout = None
    if getattr(args, "layout", "source") != "source":
        from .render import Layout

        layout = Layout(kind=args.layout)

    captions = bool(getattr(args, "captions", False))
    if captions and layout is None:
        print("--captions needs a vertical --layout; captions are burned during reframing.")
        return 2

    for c, r in chosen:
        category = c.get("content_label") or analysis.get("content_label")
        out = export_clip(
            source_video,
            out_dir,
            c["id"],
            r["start"],
            r["end"],
            r.get("title") or None,
            category=category,
            layout=layout,
            # Fetched per clip rather than all at once: captions only need the
            # segments overlapping this window.
            transcript=(
                transcript_window(args.db, args.analysis_id, r["start"], r["end"])
                if captions else None
            ),
        )
        record_export(args.db, args.analysis_id, c["id"], out)
        print(out)
    return 0



def cmd_ingest(args: argparse.Namespace) -> int:
    """Download a VOD (and chat) from a URL, then analyze it."""
    from .ingest import IngestError, ingest

    try:
        result = ingest(
            args.url,
            args.video_dir,
            with_chat=not args.no_chat,
            progress=lambda stage, frac, msg: print(f"[{stage}] {msg}"),
        )
    except IngestError as exc:
        print(f"Ingest failed: {exc}")
        return 2

    print(f"Video: {result.video_path}")
    if result.chat_path:
        print(f"Chat:  {result.chat_path}")
    elif result.chat_error:
        print(f"Chat:  unavailable ({result.chat_error})")

    if args.download_only:
        return 0

    args.video = str(result.video_path)
    args.chat = str(result.chat_path) if result.chat_path else None
    args.content = args.content or result.info.uploader or None
    return cmd_analyze(args)


def cmd_batch(args: argparse.Namespace) -> int:
    """Ingest and analyze several sources unattended."""
    from .batch import run_batch

    settings = Settings.from_file(args.settings) if args.settings else load_app_settings(args.db)
    result = run_batch(
        args.sources,
        args.work_dir,
        settings,
        db_path=args.db,
        video_dir=args.video_dir,
        content_label=args.content,
        allow_model_download=not args.no_model_download,
        progress=lambda label, message: print(f"  {label}: {message}"),
    )

    print()
    print(result.summary())
    for job in result.jobs:
        marker = "ok  " if job.status == "done" else "FAIL"
        detail = job.analysis_id or job.error or ""
        print(f"  {marker} {job.seconds:>7.1f}s  {job.label}  {detail}")
        if job.chat_note:
            print(f"         chat: {job.chat_note}")
    return 0 if not result.failed else 1


def build_parser() -> argparse.ArgumentParser:
    program_name = "HighlightMiner.exe" if is_frozen() else "highlightminer"
    p = argparse.ArgumentParser(prog=program_name)
    sub = p.add_subparsers(dest="command", required=True)
    default_db = str(default_db_path())

    doctor = sub.add_parser("doctor", help="Check FFmpeg, desktop UI, faster-whisper, CUDA, and NVENC")
    doctor.set_defaults(func=lambda a: run_doctor())

    analyze = sub.add_parser("analyze", help="Analyze a local VOD and store a new SQLite analysis run")
    analyze.add_argument("video")
    analyze.add_argument("--chat", default=None, help="Optional chat JSON/JSONL/CSV")
    analyze.add_argument("--content", default=None, help="Content/game label")
    analyze.add_argument("--work-dir", default=str(app_root() / "highlightminer_work"))
    analyze.add_argument(
        "--settings",
        default=None,
        help="Optional settings JSON override; otherwise use the active profile in --db",
    )
    analyze.add_argument("--db", default=default_db, help="SQLite database path")
    analyze.add_argument("--no-reuse", action="store_true", help="Force fresh audio/transcript/chat processing")
    model_mode = analyze.add_mutually_exclusive_group()
    model_mode.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Explicitly allow a missing recognition model to download for this command only",
    )
    model_mode.add_argument(
        "--no-transcription",
        action="store_true",
        help="Run this analysis without speech recognition even if a transcript/model is available",
    )
    analyze.set_defaults(func=cmd_analyze)


    ingest_cmd = sub.add_parser("ingest", help="Download a VOD from a Twitch/Kick/YouTube URL and analyze it")
    ingest_cmd.add_argument("url")
    ingest_cmd.add_argument("--video-dir", default=str(app_root() / "highlightminer_work" / "vods"))
    ingest_cmd.add_argument("--no-chat", action="store_true", help="Skip chat replay entirely")
    ingest_cmd.add_argument("--download-only", action="store_true", help="Fetch the files without analyzing")
    ingest_cmd.add_argument("--content", default=None, help="Content/game label")
    ingest_cmd.add_argument("--work-dir", default=str(app_root() / "highlightminer_work"))
    ingest_cmd.add_argument("--settings", default=None)
    ingest_cmd.add_argument("--db", default=default_db, help="SQLite database path")
    ingest_cmd.add_argument("--no-reuse", action="store_true")
    ingest_model = ingest_cmd.add_mutually_exclusive_group()
    ingest_model.add_argument("--allow-model-download", action="store_true")
    ingest_model.add_argument("--no-transcription", action="store_true")
    ingest_cmd.set_defaults(func=cmd_ingest)

    batch_cmd = sub.add_parser(
        "batch",
        help="Ingest and analyze several URLs or files unattended, continuing past failures",
    )
    batch_cmd.add_argument(
        "sources",
        nargs="+",
        help="URLs, local video paths, or a .txt file listing one source per line",
    )
    batch_cmd.add_argument("--work-dir", default=str(app_root() / "highlightminer_work"))
    batch_cmd.add_argument("--video-dir", default=None)
    batch_cmd.add_argument("--content", default=None, help="Content/game label for every source")
    batch_cmd.add_argument("--settings", default=None)
    batch_cmd.add_argument("--db", default=default_db, help="SQLite database path")
    batch_cmd.add_argument(
        "--no-model-download",
        action="store_true",
        help="Fail rather than download a missing recognition model",
    )
    batch_cmd.set_defaults(func=cmd_batch)

    ui = sub.add_parser("ui", help="Launch the local review UI")
    ui.add_argument("--browser", action="store_true", help="Use system browser instead of Windows desktop shell")
    ui.set_defaults(func=cmd_ui)

    history = sub.add_parser("history", help="List analyses stored in SQLite")
    history.add_argument("--db", default=default_db)
    history.add_argument("--limit", type=int, default=50)
    history.set_defaults(func=cmd_history)

    learning = sub.add_parser("learning-stats", help="Show Keep/Reject/Unreviewed dataset counts")
    learning.add_argument("--db", default=default_db)
    learning.set_defaults(func=cmd_learning_stats)

    legacy = sub.add_parser("import-legacy", help="Import a v0.1 analysis.json into SQLite")
    legacy.add_argument("analysis_json")
    legacy.add_argument("--db", default=default_db)
    legacy.set_defaults(func=cmd_import_legacy)

    export = sub.add_parser("export", help="Export kept candidates from a database analysis")
    export.add_argument("analysis_id")
    export.add_argument("--db", default=default_db)
    export.add_argument("--output", default=None)
    export.add_argument("--all", action="store_true", help="Export every ranked candidate")
    export.add_argument(
        "--layout",
        choices=("source", "letterbox", "crop", "webcam"),
        default="source",
        help="Vertical 9:16 layout for short-form output; source keeps the original aspect",
    )
    export.add_argument(
        "--captions",
        action="store_true",
        help="Burn word-timed captions into the exported clips",
    )
    export.set_defaults(func=cmd_export)
    return p


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == _STREAMLIT_CHILD_ARG:
        if len(sys.argv) != 3:
            raise SystemExit(2)
        raise SystemExit(_run_streamlit_child(sys.argv[2]))

    if len(sys.argv) == 2 and sys.argv[1] == _DESKTOP_PROBE_ARG:
        desktop_runtime_probe()
        print("pywebview desktop shell: importable", flush=True)
        raise SystemExit(0)

    if is_frozen() and len(sys.argv) == 1:
        raise SystemExit(cmd_ui())

    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
