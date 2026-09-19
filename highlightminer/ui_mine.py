from __future__ import annotations

import dataclasses

import math
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from .analysis_jobs import (
    AnalysisJobStateError,
    AnalysisJobTerminalError,
    cancel_analysis_job,
    create_analysis_job,
    find_active_analysis_job,
    list_analysis_job_events,
    load_analysis_job,
)
from .analysis_history import AnalysisDeletionBlocked, analysis_deletion_impact, delete_analysis
from .analysis_identity import load_analysis_identities, load_analysis_identity, save_analysis_title
from .categorization import normalize_content_label
from .config import Settings
from .captions import style_from_settings
from .export import PreviewFileLockError, create_preview_clip, export_clip
from .ingest import SUPPORTED_PLATFORMS, IngestError, ingest
from .render import layout_from_settings
from .export_queue import (
    ExportBatchAlreadyRunning,
    ExportBatchHeartbeat,
    ExportQueueStateError,
    clear_export_queue,
    complete_export_queue_item,
    enqueue_export_items,
    fail_export_queue_item,
    finish_export_batch,
    interrupt_export_batch,
    list_export_queue,
    load_active_export_batch,
    recover_stale_export_batches,
    remove_export_queue_item,
    retry_failed_export_items,
    start_export_batch,
    update_export_queue_title,
)
from .model_access import (
    ModelAccessPreferences,
    ModelDecisionRequired,
    huggingface_cache_directory,
    load_model_access,
    models_root,
    resolve_model_reference,
    save_model_access,
    set_model_download_consent,
    validate_local_model_directory,
)
from .pipeline import analyze_vod, snapshot_analysis_config
from .preview_marks import marked_bounds
from .preview_player import preview_player
from .review import load_review, save_review
from .security import validate_chat_file, validate_local_video
from .settings_presets import detect_weight_preset, normalize_weights
from .settings_store import load_app_settings, save_app_settings
from .storage import (
    find_source_runs,
    import_legacy_analysis,
    learning_summary,
    list_analyses,
    load_analysis,
    analysis_crop_rect,
    audio_feature_window,
    candidate_crop_rect,
    resolve_crop_rect,
    save_candidate_crop_rect,
    transcript_window,
)
from .shutdown import active_work_shutdown_block_reason
from .timeline_strip import flatten_words, render_strip, window_for
from .timeline_strip import snap_end as timeline_snap_end
from .timeline_strip import snap_start as timeline_snap_start
from .timestamps import ClipBounds, normalize_clip_bounds
from .titles import suggest_title
from .transcription_status import (
    SKIP_REASON_MODEL_DOWNLOADS_DISABLED,
    is_transcription_skipped,
)
from .ui_common import (
    _CHAT_FILTER,
    _JSON_FILTER,
    _VIDEO_FILTER,
    choose_folder,
    default_work_dir,
    path_picker,
    persistent_text_input,
    render_shutdown,
)
from .ui_style import MODEL_ACCESS_CHOICES_KEY, model_access_choices_css
from .util import format_editable_time, format_time, parse_editable_time

_PENDING_MODEL_ANALYSIS_KEY = "pending_model_analysis"
_PENDING_RERUN_KEY = "pending_rerun"
_QUEUED_ANALYSIS_KEY = "queued_analysis"
_ANALYSIS_RUNNING_KEY = "analysis_running"
_MODEL_DECISION_RESUME_KEY = "_resume_model_decision"
_PREVIEW_ACTIVE_KEY = "preview_active_candidate"
_PREVIEW_CLOSED_KEY = "preview_closed"
_PENDING_DELETE_ANALYSIS_KEY = "pending_delete_analysis_id"
_CONTINUE_WITHOUT_SPEECH_LABEL = "Continue without\nspeech"


def _clip_editor_value(
    value: str, original_seconds: float, original_text: str | None = None
) -> float:
    """Parse an editor value while preserving an untouched precise boundary."""
    text = str(value).strip()
    original = float(original_seconds)
    previous_text = format_editable_time(original) if original_text is None else original_text
    if text == previous_text.strip():
        return original
    parsed = parse_editable_time(text)
    if parsed is None:
        raise ValueError("Use MM:SS or HH:MM:SS, with optional fractional seconds.")
    return parsed


def _clip_editor_bounds(
    start_value: str,
    end_value: str,
    *,
    original_start: float,
    original_end: float,
    source_duration: float,
    original_start_text: str | None = None,
    original_end_text: str | None = None,
) -> ClipBounds:
    """Validate edited clip timestamps against the source timeline."""
    message = (
        "Enter a valid clip range using MM:SS or HH:MM:SS. "
        "The end must be at least 0.1 seconds after the start, and both must stay within the VOD."
    )
    try:
        start = _clip_editor_value(start_value, original_start, original_start_text)
        end = _clip_editor_value(end_value, original_end, original_end_text)
        if end <= start or end - start < 0.1 - 1e-9:
            raise ValueError
        bounds = normalize_clip_bounds(start, end, source_duration)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if bounds.meaningfully_invalid:
        raise ValueError(message)
    return bounds


def analysis_is_running(db_path: Path | None = None) -> bool:
    """True while this session owns work or SQLite has a persistent active job."""
    session_owned = bool(
        st.session_state.get(_ANALYSIS_RUNNING_KEY)
        and st.session_state.get(_QUEUED_ANALYSIS_KEY)
    )
    if session_owned or db_path is None:
        return session_owned
    return find_active_analysis_job(db_path) is not None


def _job_analysis_args(job: dict) -> dict:
    config = dict(job.get("config") or {})
    return {
        "video_path": str(config["video_path"]),
        "chat_path": str(config.get("chat_path") or ""),
        "content_label": str(config.get("content_label") or ""),
        "analysis_title": str(config.get("analysis_title") or ""),
        "work_dir": str(config["work_dir"]),
        "source_info": {
            "id": str(job["source_id"]),
            "fingerprint": str(job["source_fingerprint"]),
        },
        "reuse_features": bool(config.get("reuse_features", True)),
        "analysis_job_id": str(job["id"]),
    }


def _restore_persistent_analysis_state(db_path: Path) -> dict | None:
    active = find_active_analysis_job(db_path)
    if active is None:
        return None
    queued = st.session_state.get(_QUEUED_ANALYSIS_KEY) or {}
    matching_model_resume = bool(
        queued.get(_MODEL_DECISION_RESUME_KEY)
        and str(queued.get("analysis_job_id")) == str(active["id"])
    )
    if active["status"] == "queued" and not st.session_state.get(_QUEUED_ANALYSIS_KEY):
        _queue_analysis(**_job_analysis_args(active))
    elif (
        active["status"] == "awaiting_input"
        and not st.session_state.get(_PENDING_MODEL_ANALYSIS_KEY)
        and not matching_model_resume
    ):
        st.session_state[_PENDING_MODEL_ANALYSIS_KEY] = {
            **_job_analysis_args(active),
            "message": str(active.get("message") or "Choose how to continue this analysis."),
        }
    return active


def _parse_job_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _job_elapsed_seconds(job: dict, *, now: datetime | None = None) -> float:
    started = _parse_job_time(job.get("started_at") or job.get("created_at"))
    if started is None:
        return 0.0
    finished = _parse_job_time(job.get("finished_at"))
    endpoint = finished or now or datetime.now(timezone.utc)
    return max(0.0, (endpoint - started).total_seconds())


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _timings_caption(timings: dict) -> str:
    if not timings:
        return "Stage timings will appear as work completes."
    values = []
    for name, seconds in timings.items():
        label = str(name).removesuffix("_seconds").replace("_", " ")
        seconds = float(seconds)
        if not math.isfinite(seconds):
            duration = "unavailable"
        elif seconds < 60:
            duration = f"{seconds:.1f}s"
        else:
            duration = _format_elapsed(seconds)
            if seconds < 3600:
                duration = duration[3:]
        values.append(f"{label} {duration}")
    return "Completed timings · " + " · ".join(values)


def _queue_analysis(**analysis_args) -> None:
    st.session_state[_QUEUED_ANALYSIS_KEY] = dict(analysis_args)
    st.session_state[_ANALYSIS_RUNNING_KEY] = True


def _clear_queued_analysis_state() -> None:
    st.session_state.pop(_QUEUED_ANALYSIS_KEY, None)
    st.session_state.pop(_ANALYSIS_RUNNING_KEY, None)


def _clear_pending_analysis_state() -> None:
    """Clear transient state for an analysis attempt without unloading history."""
    st.session_state.pop(_PENDING_MODEL_ANALYSIS_KEY, None)
    st.session_state.pop(_PENDING_RERUN_KEY, None)


def _cancel_pending_analysis_job(db_path: Path, job_id: str | None) -> tuple[str, dict | None]:
    """Cancel a waiting job and report the state observed after any race."""
    if not job_id:
        return "cancelled", None
    try:
        if cancel_analysis_job(db_path, str(job_id)):
            return "cancelled", None
        job = load_analysis_job(db_path, str(job_id))
    except KeyError:
        return "missing", None
    return str(job["status"]), job


def _create_queued_analysis_job(
    db_path: Path,
    *,
    source: dict,
    video_path: str,
    chat_path: str,
    content_label: str,
    analysis_title: str,
    work_dir: str,
    reuse_features: bool,
) -> dict:
    video = validate_local_video(video_path)
    chat = validate_chat_file(chat_path) if chat_path else None
    work = Path(work_dir).expanduser().resolve()
    settings = load_app_settings(db_path)
    config = snapshot_analysis_config(
        video=video,
        work=work,
        settings=settings,
        chat_path=chat,
        content_label=normalize_content_label(content_label),
        source_fingerprint=str(source["fingerprint"]),
        reuse_features=reuse_features,
        transcription_requested=True,
    )
    config["analysis_title"] = str(analysis_title).strip()
    return create_analysis_job(db_path, source, config)



# Review status colours. Kept clips are what you are hunting for, so they get
# the positive colour; rejected ones are dimmed rather than alarming, because a
# reject is a normal outcome and not an error.
_STATUS_STYLES = {
    "keep": "background-color: rgba(46, 160, 67, 0.28)",
    "reject": "background-color: rgba(210, 153, 34, 0.24)",
}


def _style_candidate_rows(rows: list[dict]):
    """Tint each row by review status, so progress is visible at a glance."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    def row_style(row):
        style = _STATUS_STYLES.get(str(row.get("Status", "")), "")
        return [style] * len(row)

    return frame.style.apply(row_style, axis=1)


def _candidate_rows(analysis: dict, review: dict) -> list[dict]:
    rows = []
    for candidate in analysis.get("candidates", []):
        item = review["items"].get(candidate["id"], {})
        rows.append({
            "#": candidate["rank"],
            "ID": candidate["id"],
            "Score": round(candidate["score"] * 10, 1),
            "Start": format_time(item.get("start", candidate["start"])),
            "End": format_time(item.get("end", candidate["end"])),
            "Why": candidate["reason"],
            "Status": item.get("status", "unreviewed"),
        })
    return rows


def _history_label(row: dict, identity: dict[str, str]) -> str:
    created = str(row.get("created_at", "")).replace("T", " ").replace("+00:00", " UTC")
    title = identity.get("analysis_title", "").strip()
    title_text = f" · {title}" if title else ""
    return (
        f"{identity.get('analysis_name', 'Custom')}{title_text} · {row['content_label']} · "
        f"{row['video_name']} · run {row.get('run_number', 1)} · {created} · "
        f"{row['candidates']} candidates · {row['kept']} kept"
    )


def _latest_compatible_run(runs: list[dict]) -> dict | None:
    """Runs arrive newest-first; incompatible future formats are skipped."""
    return next((run for run in runs if bool(run.get("compatible", True))), None)


def _rerun_source_matches_video(pending: dict, video_path: str) -> bool:
    """Keep a rerun decision tied to the VOD that produced its history lookup."""
    stored_path = str(pending.get("video_path") or "").strip()
    current_path = str(video_path or "").strip()
    if not stored_path or not current_path:
        return False
    try:
        return Path(stored_path).expanduser().resolve() == Path(current_path).expanduser().resolve()
    except (OSError, RuntimeError):
        return False


def _rerun_model_access_blocks_speech(
    settings: Settings,
    model_access: ModelAccessPreferences,
) -> bool:
    """Return whether the saved policy leaves no usable model for a fresh transcript."""
    if model_access.download_consent != "deny":
        return False
    try:
        return resolve_model_reference(settings, model_access) is None
    except (OSError, ValueError):
        return True


def _analysis_delete_confirmation_matches(analysis_id: str, entered: str) -> bool:
    return bool(analysis_id) and str(entered).strip() == str(analysis_id)


def _transcription_skip_notice(metadata: dict) -> str:
    if str(metadata.get("reason") or "") == SKIP_REASON_MODEL_DOWNLOADS_DISABLED:
        return (
            "Speech recognition was disabled because model downloads are set to "
            "**Never download models** and no compatible local or cached model was found. "
            "To enable speech, open **Settings → Analysis engine → Model access**, choose "
            "**Ask before any download** or **Allow model downloads**, save, then run "
            "**Force full reprocess**."
        )
    return (
        "Speech recognition was disabled for this run. Candidate scoring was "
        "renormalized across audio and available chat signals."
    )


def _run_analysis_ui(
    *,
    db_path: Path,
    video_path: str,
    chat_path: str,
    content_label: str,
    analysis_title: str,
    work_dir: str,
    source_info: dict | None = None,
    reuse_features: bool = True,
    allow_model_download: bool = False,
    skip_transcription: bool = False,
    analysis_job_id: str | None = None,
) -> str:
    if analysis_job_id:
        job = load_analysis_job(db_path, analysis_job_id)
        settings = Settings(**dict(job["config"]["settings"]))
    else:
        settings = load_app_settings(db_path)
    with st.status("Analyzing…", expanded=True) as status:
        label = st.empty()
        metadata = st.empty()
        timing_details = st.empty()
        bar = st.progress(0.0)

        def progress(message: str, value: float) -> None:
            label.write(message)
            bar.progress(min(1.0, max(0.0, value)))
            if analysis_job_id:
                current = load_analysis_job(db_path, analysis_job_id)
                metadata.caption(
                    f"Stage: **{str(current['stage']).replace('_', ' ')}** · "
                    f"overall elapsed **{_format_elapsed(_job_elapsed_seconds(current))}**"
                )
                timing_details.caption(_timings_caption(current.get("timings") or {}))

        analysis_id = analyze_vod(
            video_path,
            work_dir,
            settings,
            chat_path or None,
            progress,
            content_label=content_label,
            db_path=db_path,
            source_info=source_info,
            reuse_features=reuse_features,
            allow_model_download=allow_model_download,
            skip_transcription=skip_transcription,
            analysis_job_id=analysis_job_id,
        )
        save_analysis_title(db_path, analysis_id, analysis_title)
    analysis = load_analysis(db_path, analysis_id)
    reused = analysis.get("cache", {}).get("reused_stages", [])
    if is_transcription_skipped(analysis.get("transcription")):
        st.session_state["analysis_notice"] = (
            "Speech recognition was skipped. This run used audio"
            + (" and chat" if analysis.get("chat", {}).get("path") else "")
            + " signals only."
        )
    else:
        st.session_state["analysis_notice"] = ("Reused cached " + ", ".join(reused) + ".") if reused else "Fresh source features generated."
    status.update(label="Analysis complete", state="complete", expanded=False)
    return analysis_id


def _run_queued_analysis(db_path: Path) -> None:
    queued = dict(st.session_state.get(_QUEUED_ANALYSIS_KEY) or {})
    if not queued:
        st.session_state.pop(_ANALYSIS_RUNNING_KEY, None)
        return
    resume_model_decision = bool(queued.pop(_MODEL_DECISION_RESUME_KEY, False))

    job_id = queued.get("analysis_job_id")
    if job_id:
        try:
            job = load_analysis_job(db_path, str(job_id))
        except KeyError as exc:
            st.session_state["analysis_error"] = str(exc)
            _clear_queued_analysis_state()
            return
        if job["status"] == "completed":
            st.session_state.analysis_id = job["analysis_id"]
            _clear_queued_analysis_state()
            st.rerun()
            return
        if job["status"] == "awaiting_input" and not resume_model_decision:
            st.session_state[_PENDING_MODEL_ANALYSIS_KEY] = {
                **_job_analysis_args(job),
                "message": str(job.get("message") or "Choose how to continue this analysis."),
            }
            _clear_queued_analysis_state()
            st.rerun()
            return
        if job["status"] == "running":
            # A Streamlit rerun must not launch a second worker for the same job.
            _clear_queued_analysis_state()
            return
        if job["status"] in {"failed", "cancelled", "interrupted"}:
            st.session_state["analysis_error"] = str(
                job.get("error_message") or job.get("message") or f"Analysis {job['status']}"
            )
            _clear_queued_analysis_state()
            return

    try:
        st.session_state.analysis_id = _run_analysis_ui(db_path=db_path, **queued)
    except ModelDecisionRequired as exc:
        _queue_model_decision(exc, **queued)
    except AnalysisJobTerminalError as exc:
        _clear_pending_analysis_state()
        job = exc.job
        if job["status"] == "completed" and job.get("analysis_id"):
            st.session_state.analysis_id = job["analysis_id"]
            st.session_state["analysis_notice"] = "Analysis had already completed."
        elif job["status"] == "cancelled":
            st.session_state["analysis_notice"] = "Analysis was cancelled."
        else:
            st.session_state["analysis_error"] = str(
                job.get("error_message")
                or job.get("message")
                or f"Analysis {job['status']}."
            )
    except AnalysisJobStateError as exc:
        current_job = None
        job_id = queued.get("analysis_job_id")
        if job_id:
            try:
                current_job = load_analysis_job(db_path, str(job_id))
            except KeyError:
                pass
        if current_job and current_job["status"] == "running":
            st.session_state["analysis_notice"] = "Analysis is already running in another session."
        else:
            st.session_state["analysis_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        st.session_state["analysis_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _clear_queued_analysis_state()
    st.rerun()


def _queue_model_decision(exc: ModelDecisionRequired, **analysis_args) -> None:
    pending = {
        **analysis_args,
        "message": str(exc),
    }
    analysis_job_id = getattr(exc, "analysis_job_id", None)
    if analysis_job_id:
        pending["analysis_job_id"] = str(analysis_job_id)
    st.session_state[_PENDING_MODEL_ANALYSIS_KEY] = pending


def _resume_pending_model_analysis(
    db_path: Path,
    *,
    allow_model_download: bool = False,
    skip_transcription: bool = False,
) -> None:
    del db_path
    pending = dict(st.session_state.get(_PENDING_MODEL_ANALYSIS_KEY) or {})
    if not pending:
        return
    pending.pop("message", None)
    pending.pop(_MODEL_DECISION_RESUME_KEY, None)
    pending.pop("allow_model_download", None)
    pending.pop("skip_transcription", None)
    _clear_pending_analysis_state()
    _queue_analysis(
        **{_MODEL_DECISION_RESUME_KEY: True},
        allow_model_download=allow_model_download,
        skip_transcription=skip_transcription,
        **pending,
    )
    st.rerun()


def _render_model_decision(db_path: Path) -> bool:
    pending = st.session_state.get(_PENDING_MODEL_ANALYSIS_KEY)
    if not pending:
        return False

    job_id = pending.get("analysis_job_id")
    if job_id:
        try:
            job = load_analysis_job(db_path, str(job_id))
        except KeyError:
            _clear_pending_analysis_state()
            _clear_queued_analysis_state()
            st.session_state["analysis_notice"] = "The analysis job no longer exists."
            st.rerun()
            return True
        if job["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            _clear_pending_analysis_state()
            _clear_queued_analysis_state()
            if job["status"] == "completed" and job.get("analysis_id"):
                st.session_state["analysis_id"] = job["analysis_id"]
                st.session_state["analysis_notice"] = "Analysis had already completed."
            elif job["status"] == "failed":
                st.session_state["analysis_error"] = str(
                    job.get("error_message") or job.get("message") or "Analysis failed."
                )
            else:
                st.session_state["analysis_notice"] = str(
                    job.get("message") or f"Analysis {job['status']}."
                )
            st.rerun()
            return True
        if job["status"] == "running":
            _clear_pending_analysis_state()
            st.session_state["analysis_notice"] = "Analysis is already running in another session."
            st.rerun()
            return True
        settings = Settings(**dict(job["config"]["settings"]))
    else:
        settings = load_app_settings(db_path)
    with st.container(border=True):
        st.subheader("Speech-recognition model required", anchor=False)
        st.write(pending.get("message") or f"The model {settings.whisper_model!r} is not installed.")
        st.caption(
            "Your VOD stays local. Downloading only retrieves the speech-recognition model from Hugging Face. "
            f"Downloaded models normally use `{huggingface_cache_directory()}`."
        )
        st.caption(
            "You can choose a complete local CTranslate2 Whisper model, or continue without speech recognition. "
            "Continuing remembers that model downloads are disabled and renormalizes scoring across the signals that remain available."
        )
        st.markdown(model_access_choices_css(), unsafe_allow_html=True)
        with st.container(key=MODEL_ACCESS_CHOICES_KEY):
            download, local, no_speech = st.columns(
                3,
                vertical_alignment="center",
            )
            if download.button(
                "Download model",
                type="primary",
                width="stretch",
            ):
                set_model_download_consent("allow", db_path)
                _resume_pending_model_analysis(db_path, allow_model_download=True)
            if local.button(
                "Choose local model",
                width="stretch",
            ):
                try:
                    selected = choose_folder(
                        "Choose a CTranslate2 Whisper model folder",
                        str(models_root()),
                    )
                    if selected:
                        validated = validate_local_model_directory(selected)
                        current = load_model_access(db_path)
                        save_model_access(
                            ModelAccessPreferences(current.download_consent, str(validated)),
                            db_path,
                        )
                        _resume_pending_model_analysis(db_path)
                except Exception as exc:
                    st.exception(exc)
            if no_speech.button(
                _CONTINUE_WITHOUT_SPEECH_LABEL,
                width="stretch",
            ):
                set_model_download_consent("deny", db_path)
                _resume_pending_model_analysis(db_path, skip_transcription=True)
        st.caption(
            "Cancel analysis stops only this run. It does not change model download permission."
        )
        if st.button("Cancel analysis", width="stretch"):
            outcome, current_job = _cancel_pending_analysis_job(
                db_path,
                pending.get("analysis_job_id"),
            )
            if outcome == "running":
                st.error("This analysis is already running and cannot be cancelled safely.")
            else:
                _clear_pending_analysis_state()
                _clear_queued_analysis_state()
                if outcome == "completed" and current_job and current_job.get("analysis_id"):
                    st.session_state.analysis_id = current_job["analysis_id"]
                    st.session_state["analysis_notice"] = "Analysis had already completed."
                elif outcome == "failed" and current_job:
                    st.session_state["analysis_error"] = str(
                        current_job.get("error_message")
                        or current_job.get("message")
                        or "Analysis failed."
                    )
                elif outcome == "missing":
                    st.session_state["analysis_notice"] = "The analysis job no longer exists."
                elif outcome in {"cancelled", "interrupted"}:
                    st.session_state["analysis_notice"] = (
                        "Analysis cancelled."
                        if outcome == "cancelled"
                        else "Analysis had already been interrupted."
                    )
                st.rerun()
    return True


def _render_analysis_job_status(db_path: Path, job: dict) -> None:
    status_label = str(job["status"]).replace("_", " ").title()
    stage_label = str(job["stage"]).replace("_", " ").title()
    with st.container(border=True):
        st.subheader(f"⏱️ Analysis job · {status_label}", anchor=False)
        st.progress(float(job.get("progress", 0.0)))
        st.write(str(job.get("message") or stage_label))
        st.caption(
            f"Stage: **{stage_label}** · overall elapsed "
            f"**{_format_elapsed(_job_elapsed_seconds(job))}** · job `{str(job['id'])[:12]}`"
        )
        st.caption(_timings_caption(job.get("timings") or {}))

        notable = [
            event
            for event in list_analysis_job_events(db_path, str(job["id"]))
            if event["level"] in {"warning", "error"}
        ]
        if notable:
            with st.expander("Warnings and recovery events"):
                for event in notable[-8:]:
                    st.write(f"**{event['event']}** · {event['message']}")
        if job["status"] == "running":
            st.caption(
                "Controls stay locked while the worker is active. A Stop button is not shown because "
                "the current processing stages cannot yet be cancelled safely."
            )
            if st.button("Refresh analysis status", key=f"refresh_analysis_job_{job['id']}"):
                st.rerun()




_NUDGE_STEPS = (-2.0, -0.5, 0.5, 2.0)


def _adjust_time_field(key: str, delta: float, duration: float) -> None:
    """Shift one boundary by a fixed amount, without retyping a timestamp."""
    current = parse_editable_time(st.session_state.get(key, "")) or 0.0
    st.session_state[key] = format_editable_time(
        min(max(current + delta, 0.0), max(0.0, duration))
    )


def _snap_time_field(
    key: str,
    words: list[tuple[float, float, str]],
    direction: int,
    *,
    to_end: bool,
) -> None:
    """Move one boundary onto the next or previous word edge.

    Word timings already exist for captions, so a clip can start exactly where
    someone begins speaking rather than at a guessed offset.
    """
    current = parse_editable_time(st.session_state.get(key, "")) or 0.0
    target = (
        timeline_snap_end(words, current, direction)
        if to_end
        else timeline_snap_start(words, current, direction)
    )
    if target is not None:
        st.session_state[key] = format_editable_time(target)


def _render_timing_controls(
    label: str,
    key: str,
    words: list[tuple[float, float, str]],
    duration: float,
    token: str,
    *,
    to_end: bool,
) -> None:
    """Nudge and snap buttons beneath a timestamp box."""
    columns = st.columns(6)
    columns[0].button(
        "⏮ word", key=f"snapback_{token}_{key}", width="stretch",
        help="Previous word boundary",
        on_click=_snap_time_field, args=(key, words, -1), kwargs={"to_end": to_end},
    )
    for index, step in enumerate(_NUDGE_STEPS, start=1):
        columns[index].button(
            f"{step:+g}s", key=f"nudge_{token}_{key}_{step}", width="stretch",
            on_click=_adjust_time_field, args=(key, step, duration),
        )
    columns[5].button(
        "word ⏭", key=f"snapfwd_{token}_{key}", width="stretch",
        help="Next word boundary",
        on_click=_snap_time_field, args=(key, words, 1), kwargs={"to_end": to_end},
    )



def _render_crop_controls(
    db_path: Path,
    analysis: dict,
    analysis_id: str,
    candidate: dict,
    settings,
) -> None:
    """Adjust the 9:16 window against a real frame from this clip.

    A still is pulled once and then cropped with PIL, so moving the window is
    instant. Re-encoding a preview to see whether the framing is right is the
    slow way to answer a question a single frame already answers.
    """
    from .crop_tool import CropToolError, annotate_frame, crop_preview, extract_frame, rect_for_aspect

    if settings.render_layout != "crop":
        st.caption(
            f"Layout is **{settings.render_layout}**. Switch it to **crop** in Settings "
            "to use a framing window."
        )
        return

    # Keyed per clip: framing is an editorial choice about one moment, so
    # adjusting one candidate must not silently reframe the rest of the queue.
    x_key = f"crop_x_{analysis_id}_{candidate['id']}"
    y_key = f"crop_y_{analysis_id}_{candidate['id']}"
    h_key = f"crop_h_{analysis_id}_{candidate['id']}"
    if x_key not in st.session_state:
        saved = (
            resolve_crop_rect(db_path, analysis_id, candidate["id"]) or settings.gameplay_rect
        )
        if saved:
            st.session_state[x_key] = float(saved.get("x", 0.341797))
            st.session_state[y_key] = float(saved.get("y", 0.0))
            st.session_state[h_key] = float(saved.get("h", 1.0))
        else:
            centred = rect_for_aspect(x=0.0, height=1.0)
            st.session_state[x_key] = round((1.0 - centred.w) / 2, 6)
            st.session_state[y_key] = 0.0
            st.session_state[h_key] = 1.0

    frame_key = f"crop_frame_{analysis_id}_{candidate['id']}"
    frame_path = st.session_state.get(frame_key)
    if not frame_path or not Path(frame_path).exists():
        # Sample where the moment actually peaks, not the padded start.
        when = float(candidate.get("peak_time") or candidate["start"])
        target = (
            Path(analysis["work_dir"]) / ".previews" / analysis_id / f"crop_{candidate['id']}.png"
        )
        try:
            source = validate_local_video(analysis["video_path"])
            frame_path = str(extract_frame(source, when, target, width=720))
            st.session_state[frame_key] = frame_path
        except (CropToolError, Exception) as exc:  # noqa: BLE001 - shown to the user
            st.caption(f"Could not read a frame for framing: {exc}")
            return

    rect = rect_for_aspect(
        x=float(st.session_state[x_key]),
        y=float(st.session_state[y_key]),
        height=float(st.session_state[h_key]),
    )

    sliders, marked, result = st.columns([2, 2, 1])
    with sliders:
        st.slider("Horizontal", 0.0, 1.0, step=0.005, key=x_key)
        st.slider("Vertical", 0.0, 1.0, step=0.005, key=y_key)
        st.slider("Height kept", 0.3, 1.0, step=0.01, key=h_key)
        st.caption(f"x={rect.x:.3f} w={rect.w:.3f} · keeps {rect.w * 100:.0f}% of the width")

    work = Path(analysis["work_dir"]) / ".previews" / analysis_id
    try:
        marked.image(
            str(annotate_frame(frame_path, rect, work / f"cropmark_{candidate['id']}.png")),
            caption="Kept region",
            width="stretch",
        )
        result.image(
            str(crop_preview(frame_path, rect, work / f"cropout_{candidate['id']}.png", height=420)),
            caption="Result",
            width="stretch",
        )
    except Exception as exc:  # noqa: BLE001 - framing aid, not the feature
        st.caption(f"Could not render the framing preview: {exc}")
        return

    source_rect = analysis_crop_rect(db_path, analysis_id)
    clip_rect = candidate_crop_rect(db_path, analysis_id, candidate["id"])
    if clip_rect:
        st.caption("This clip has its own framing.")
    elif source_rect:
        st.caption("Using the framing saved for this source.")

    this_clip, whole_source, clear = st.columns(3)
    if this_clip.button(
        "Save for this clip",
        key=f"save_crop_clip_{analysis_id}_{candidate['id']}",
        type="primary",
        width="stretch",
    ):
        if save_candidate_crop_rect(db_path, analysis_id, candidate["id"], rect.as_dict()):
            st.success("Saved for this clip. Press **Update preview** to see it in motion.")
        else:
            st.warning("Could not store framing for this clip.")

    if whole_source.button(
        "Apply to whole source",
        key=f"save_crop_source_{analysis_id}_{candidate['id']}",
        width="stretch",
        help="Use this window for every clip from this channel that has no framing of its own.",
    ):
        from .identity import describe_source
        from .storage import save_source_crop_rect

        try:
            fingerprint = describe_source(analysis["video_path"])["fingerprint"]
            stored = save_source_crop_rect(db_path, fingerprint, rect.as_dict())
        except Exception:  # noqa: BLE001 - fall back to the profile default
            stored = False
        if not stored:
            save_app_settings(
                dataclasses.replace(settings, gameplay_rect=rect.as_dict()), db_path
            )
        st.success("Saved for every clip from this source without its own framing.")

    if clear.button(
        "Clear this clip",
        key=f"clear_crop_{analysis_id}_{candidate['id']}",
        width="stretch",
        disabled=not clip_rect,
        help="Fall back to the source framing.",
    ):
        save_candidate_crop_rect(db_path, analysis_id, candidate["id"], None)
        for key in (x_key, y_key, h_key):
            st.session_state.pop(key, None)
        st.rerun()


def _render_url_ingest(db_path: Path, *, disabled: bool = False) -> None:
    """Fetch a VOD and its chat from a URL, then fill in the local file fields."""
    with st.expander("Fetch from a URL", expanded=False):
        st.caption(
            f"Supported: {', '.join(SUPPORTED_PLATFORMS)}. The VOD and, where the platform "
            "offers one, its chat replay are downloaded to the work folder and then "
            "analyzed like any local file."
        )
        url = persistent_text_input(
            "VOD URL",
            "ingest_url_input",
            placeholder="https://www.twitch.tv/videos/…",
            disabled=disabled,
        )
        target = st.session_state.get("work_dir_input") or default_work_dir()
        if st.button("Download", key="ingest_url_button", disabled=disabled or not url):
            status = st.empty()
            try:
                with st.spinner("Fetching VOD…"):
                    result = ingest(
                        url,
                        Path(target) / "vods",
                        progress=lambda stage, frac, msg: status.write(f"{msg}…"),
                    )
            except IngestError as exc:
                status.empty()
                st.error(f"Could not fetch that URL: {exc}")
            else:
                status.empty()
                # Fill the normal local-file fields, so everything downstream is
                # identical to having picked the files by hand.
                st.session_state["video_path_input"] = str(result.video_path)
                if result.chat_path:
                    st.session_state["chat_path_input"] = str(result.chat_path)
                if result.info.uploader and not st.session_state.get("content_label_input"):
                    st.session_state["content_label_input"] = result.info.uploader
                if result.chat_error:
                    st.info(f"No chat for this VOD: {result.chat_error}")
                st.success(f"Downloaded {result.video_path.name}")
                st.rerun()


def _render_source_sidebar(db_path: Path, *, disabled: bool = False) -> tuple[str, str, str, str, str]:
    st.header("🎬 Source", anchor=False)
    st.caption("Choose local files directly. The VOD is read in place and never uploaded.")

    _render_url_ingest(db_path, disabled=disabled)

    video_path = path_picker(
        "VOD",
        "video_path_input",
        placeholder=r"D:\VODs\stream.mp4",
        file_filter=_VIDEO_FILTER,
        disabled=disabled,
    )
    chat_path = path_picker(
        "Chat file (optional)",
        "chat_path_input",
        placeholder="TwitchDownloader JSON / JSONL / CSV",
        file_filter=_CHAT_FILTER,
        disabled=disabled,
    )
    content_label = persistent_text_input(
        "Content / Game",
        "content_label_input",
        placeholder="Just Chatting / Overwatch 2 / ...",
        help="Stored with every candidate for future preference learning.",
        disabled=disabled,
    )
    analysis_title = persistent_text_input(
        "Analysis title (optional)",
        "analysis_title_input",
        placeholder="Boss fight test / chat experiment / ...",
        help="Optional note for this run. The weighting profile is shown separately as the analysis name.",
        disabled=disabled,
    )
    work_dir = path_picker(
        "Work folder",
        "work_dir_input",
        default=default_work_dir(),
        folder=True,
        disabled=disabled,
    )

    settings = load_app_settings(db_path)
    effective = normalize_weights(settings.weights)
    st.caption(
        f"Settings: **{detect_weight_preset(settings.weights)}** · "
        f"audio {effective['audio']:.0%} / transcript {effective['transcript']:.0%} / chat {effective['chat']:.0%}"
    )

    dialog_error = st.session_state.pop("native_dialog_error", None)
    if dialog_error:
        st.error(dialog_error)
    return video_path, chat_path, content_label, analysis_title, work_dir


def _render_analysis_controls(
    db_path: Path,
    video_path: str,
    chat_path: str,
    content_label: str,
    analysis_title: str,
    work_dir: str,
    *,
    active_job: dict | None = None,
    disabled: bool = False,
) -> None:
    if active_job is not None:
        _render_analysis_job_status(db_path, active_job)
        if active_job["status"] == "awaiting_input":
            _render_model_decision(db_path)
        return

    if disabled:
        st.info("Analysis controls are locked while the export worker is active.")
        return

    if _render_model_decision(db_path):
        return

    required_inputs_ready = bool(str(video_path).strip() and str(work_dir).strip())
    pending = st.session_state.get(_PENDING_RERUN_KEY)
    if pending and not _rerun_source_matches_video(pending, video_path):
        st.session_state.pop(_PENDING_RERUN_KEY, None)
        pending = None
        st.info("The VOD selection changed, so the previous rerun choice was cleared.")

    if not required_inputs_ready:
        st.caption("Choose a VOD and work folder before starting analysis.")

    if st.button(
        "⛏️ Analyze VOD",
        type="primary",
        width="stretch",
        disabled=not required_inputs_ready or pending is not None,
    ):
        source = None
        try:
            source, prior_runs = find_source_runs(db_path, video_path)
            if prior_runs:
                st.session_state[_PENDING_RERUN_KEY] = {
                    "source": source,
                    "runs": prior_runs,
                    "video_path": str(Path(video_path).expanduser().resolve()),
                }
                st.rerun()
            job = _create_queued_analysis_job(
                db_path,
                source=source,
                video_path=video_path,
                chat_path=chat_path,
                content_label=content_label,
                analysis_title=analysis_title,
                work_dir=work_dir,
                reuse_features=True,
            )
            _queue_analysis(**_job_analysis_args(job))
            st.rerun()
        except Exception as exc:
            st.exception(exc)

    if not pending:
        return
    runs = pending.get("runs", [])
    latest = _latest_compatible_run(runs)
    st.warning(
        f"This VOD already has {len(runs)} analysis run(s). Choose whether to load results, "
        "rescore with reusable evidence, or process everything again."
    )
    if latest:
        latest_identity = load_analysis_identity(db_path, latest["id"])
        latest_title = latest_identity.get("analysis_title", "")
        title_text = f" · {latest_title}" if latest_title else ""
        st.caption(
            f"Latest: {latest_identity['analysis_name']}{title_text} · run {latest['run_number']} · "
            f"{latest['candidates']} candidates · {latest['kept']} kept / "
            f"{latest['rejected']} rejected / {latest['unreviewed']} unreviewed"
        )
    else:
        st.caption("No completed analysis in a compatible format can be loaded by this app version.")

    st.caption(
        "**Load latest** does no work. **Analyze again** creates a new run with current settings and "
        "reuses only matching evidence. **Force full reprocess** ignores all prior evidence."
    )
    model_access = load_model_access(db_path)
    if _rerun_model_access_blocks_speech(load_app_settings(db_path), model_access):
        st.warning(
            "Speech recognition may be skipped: Model access is set to **Never download models** "
            "and no usable selected or cached model is available. Change it under **Settings → "
            "Analysis engine → Model access** before reprocessing if you want a fresh transcript."
        )
    r1, r2, r3 = st.columns(3)
    if r1.button("Load latest", width="stretch", disabled=latest is None):
        assert latest is not None
        st.session_state.analysis_id = latest["id"]
        st.session_state.pop(_PENDING_RERUN_KEY, None)
        st.rerun()
    if r2.button(
        "Analyze again",
        type="primary",
        width="stretch",
        disabled=not required_inputs_ready,
    ):
        try:
            job = _create_queued_analysis_job(
                db_path,
                source=pending.get("source"),
                video_path=video_path,
                chat_path=chat_path,
                content_label=content_label,
                analysis_title=analysis_title,
                work_dir=work_dir,
                reuse_features=True,
            )
            _queue_analysis(**_job_analysis_args(job))
            st.session_state.pop(_PENDING_RERUN_KEY, None)
            st.rerun()
        except Exception as exc:
            st.exception(exc)
    if r3.button(
        "Force full reprocess",
        width="stretch",
        disabled=not required_inputs_ready,
    ):
        try:
            job = _create_queued_analysis_job(
                db_path,
                source=pending.get("source"),
                video_path=video_path,
                chat_path=chat_path,
                content_label=content_label,
                analysis_title=analysis_title,
                work_dir=work_dir,
                reuse_features=False,
            )
            _queue_analysis(**_job_analysis_args(job))
            st.session_state.pop(_PENDING_RERUN_KEY, None)
            st.rerun()
        except Exception as exc:
            st.exception(exc)


def _render_history_sidebar(db_path: Path, *, disabled: bool = False) -> None:
    st.divider()
    st.subheader("🗃️ Analysis history", anchor=False)
    history = list_analyses(db_path, limit=50)
    if history:
        identities = load_analysis_identities(db_path, [row["id"] for row in history])
        labels = [_history_label(row, identities.get(row["id"], {})) for row in history]
        ids = [row["id"] for row in history]
        current_id = st.session_state.get("analysis_id")
        default_index = ids.index(current_id) if current_id in ids else 0
        selected = st.selectbox("Recent analyses", labels, index=default_index, disabled=disabled)
        selected_id = ids[labels.index(selected)]
        load_button, delete_button = st.columns(2)
        if load_button.button("Load selected", width="stretch", disabled=disabled):
            st.session_state.analysis_id = selected_id
            st.rerun()
        if delete_button.button("Delete…", width="stretch", disabled=disabled):
            st.session_state[_PENDING_DELETE_ANALYSIS_KEY] = selected_id
            st.rerun()

        pending_delete = st.session_state.get(_PENDING_DELETE_ANALYSIS_KEY)
        if pending_delete:
            try:
                impact = analysis_deletion_impact(db_path, str(pending_delete))
            except KeyError:
                st.session_state.pop(_PENDING_DELETE_ANALYSIS_KEY, None)
                st.warning("That analysis no longer exists.")
            else:
                with st.container(border=True):
                    title = impact["analysis_title"] or f"run {impact['run_number']}"
                    st.warning(
                        f"Permanently delete **{impact['video_name']} · {title}**? "
                        "Completed analyses are otherwise immutable."
                    )
                    st.caption(
                        f"This removes {impact['candidates']} candidates, "
                        f"{impact['kept']} kept / {impact['rejected']} rejected / "
                        f"{impact['unreviewed']} unreviewed labels, "
                        f"{impact['review_events']} review events, {impact['exports']} export-history records, "
                        f"and {impact['queue_items']} export-queue entries."
                    )
                    st.caption(
                        "Transcript/audio/chat evidence belonging to this run is also removed "
                        f"({impact['transcript_segments']} transcript segments, "
                        f"{impact['audio_features']} audio features, "
                        f"{impact['chat_features']} chat features). "
                        "Already exported video files remain on disk. Other runs—even for the same VOD—are untouched."
                    )
                    st.code(str(pending_delete))
                    entered_id = st.text_input(
                        "Type the full analysis ID shown above to confirm",
                        key=f"delete_analysis_id_input_{pending_delete}",
                        disabled=disabled,
                    )
                    understood = st.checkbox(
                        "I understand that this analysis and its review/learning history cannot be recovered.",
                        key=f"confirm_delete_analysis_{pending_delete}",
                        disabled=disabled,
                    )
                    cancel, confirm = st.columns(2)
                    if cancel.button(
                        "Cancel",
                        key=f"cancel_delete_analysis_{pending_delete}",
                        disabled=disabled,
                        width="stretch",
                    ):
                        st.session_state.pop(_PENDING_DELETE_ANALYSIS_KEY, None)
                        st.rerun()
                    if confirm.button(
                        "Delete permanently",
                        key=f"confirm_delete_analysis_button_{pending_delete}",
                        disabled=(
                            disabled
                            or not understood
                            or not _analysis_delete_confirmation_matches(
                                str(pending_delete),
                                entered_id,
                            )
                        ),
                        width="stretch",
                    ):
                        try:
                            deleted = delete_analysis(
                                db_path,
                                str(pending_delete),
                                acknowledged=understood,
                                confirmed_analysis_id=entered_id.strip(),
                            )
                        except (AnalysisDeletionBlocked, KeyError) as exc:
                            st.error(str(exc))
                        else:
                            if st.session_state.get("analysis_id") == pending_delete:
                                st.session_state.pop("analysis_id", None)
                            st.session_state.pop(_PENDING_DELETE_ANALYSIS_KEY, None)
                            st.session_state.pop(_PREVIEW_ACTIVE_KEY, None)
                            st.session_state.pop(_PREVIEW_CLOSED_KEY, None)
                            st.session_state["analysis_notice"] = (
                                f"Deleted {deleted['video_name']} run {deleted['run_number']}. "
                                "Other analyses and exported video files were kept."
                            )
                            st.rerun()
    else:
        st.caption("No SQLite analyses yet.")

    with st.expander("Import v0.1 analysis.json"):
        legacy_path = path_picker(
            "Legacy analysis.json",
            "legacy_analysis_input",
            placeholder=r"D:\HighlightMiner\highlightminer_work\stream\analysis.json",
            file_filter=_JSON_FILTER,
            disabled=disabled,
        )
        if st.button("Import into database", disabled=disabled or not legacy_path, width="stretch"):
            try:
                st.session_state.analysis_id = import_legacy_analysis(legacy_path, db_path)
                st.session_state["analysis_notice"] = "Legacy analysis imported into SQLite."
                st.rerun()
            except Exception as exc:
                st.exception(exc)

    with st.expander("🧠 Learning data"):
        stats = learning_summary(db_path)
        st.caption(f"{stats['kept']} keep · {stats['rejected']} reject · {stats['unreviewed']} unreviewed · {stats['exported']} exported")
        st.caption("Unreviewed stays unlabeled; it is not silently treated as a reject.")


def _render_review(db_path: Path) -> None:
    analysis_id = st.session_state.get("analysis_id")
    if not analysis_id:
        st.info("Analyze a VOD, choose an item from **Analysis history**, or import a v0.1 `analysis.json`.")
        return
    try:
        analysis = load_analysis(db_path, analysis_id)
    except KeyError:
        st.session_state.pop("analysis_id", None)
        st.warning("That analysis no longer exists in the database.")
        return

    identity = load_analysis_identity(db_path, analysis_id)
    candidates = analysis.get("candidates", [])
    review = load_review(db_path, analysis_id, analysis)
    content_label = normalize_content_label(analysis.get("content_label"))
    transcription = analysis.get("transcription", {})
    transcription_skipped = is_transcription_skipped(transcription)
    st.subheader("📊 Analysis overview", anchor=False)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Candidates", len(candidates))
    c2.metric("Kept", sum(x.get("status") == "keep" for x in review["items"].values()))
    c3.metric("Rejected", sum(x.get("status") == "reject" for x in review["items"].values()))
    c4.metric("Unreviewed", sum(x.get("status") == "unreviewed" for x in review["items"].values()))
    c5.metric("Speech recognition", "Off" if transcription_skipped else (transcription.get("language") or "On"))
    reused = analysis.get("cache", {}).get("reused_stages", [])
    cache_text = f" · reused: {', '.join(reused)}" if reused else ""
    title_text = f" · Title: **{identity['analysis_title']}**" if identity.get("analysis_title") else ""
    st.caption(
        f"Analysis: **{identity['analysis_name']}**{title_text} · Content / Game: **{content_label}** · "
        f"source run **{analysis.get('run_number', 1)}** · Analysis ID: `{analysis_id[:12]}`{cache_text}"
    )
    if transcription_skipped:
        st.info(_transcription_skip_notice(transcription))
    if not candidates:
        st.warning("No candidates cleared the current threshold. Adjust Settings and analyze again.")
        return

    st.subheader("⛏️ Ranked candidates", anchor=False)
    # The table already lists everything a dropdown would repeat, so it is the
    # selector. Rows are in the same order as `candidates`, so the selected
    # position indexes straight into it.
    selection_key = f"candidate_table_{analysis_id}"
    last_row_key = f"candidate_row_{analysis_id}"
    table = st.dataframe(
        _style_candidate_rows(_candidate_rows(analysis, review)),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=selection_key,
    )
    st.caption("Click a row to review that clip. Green is kept, amber is rejected.")

    chosen_rows: list[int] = []
    try:
        chosen_rows = list(table.selection.rows)
    except AttributeError:
        chosen_rows = []
    row_index = chosen_rows[0] if chosen_rows else int(st.session_state.get(last_row_key, 0))
    row_index = min(max(row_index, 0), len(candidates) - 1)
    st.session_state[last_row_key] = row_index

    if load_active_export_batch(db_path) is not None:
        st.info(
            "Candidate preview and review controls are locked while the export worker uses the staged queue snapshot."
        )
        return
    candidate = candidates[row_index]
    item = review["items"][candidate["id"]]
    original_start = float(item["start"])
    original_end = float(item["end"])

    preview_token = f"{analysis_id}:{candidate['id']}"
    preview_bounds_key = f"preview_bounds_{analysis_id}_{candidate['id']}"
    candidate_changed = st.session_state.get(_PREVIEW_ACTIVE_KEY) != preview_token
    if candidate_changed:
        st.session_state[_PREVIEW_ACTIVE_KEY] = preview_token
        st.session_state[_PREVIEW_CLOSED_KEY] = False
        st.session_state.setdefault(preview_bounds_key, (original_start, original_end))

    # Apply mark events before instantiating the timestamp widgets on this rerun.
    mark_warning = st.session_state.pop(f"mark_warning_{preview_token}", None)
    if mark_warning:
        st.warning(mark_warning)
    pending_mark = st.session_state.pop(f"pending_mark_{preview_token}", None)
    start_key = f"clip_start_time_{analysis_id}_{candidate['id']}"
    end_key = f"clip_end_time_{analysis_id}_{candidate['id']}"
    precise_bounds_key = f"clip_precise_bounds_{preview_token}"
    precise_start, precise_end = st.session_state.get(
        precise_bounds_key, (original_start, original_end)
    )
    text_key = f"clip_boundary_text_{preview_token}"
    start_text, end_text = st.session_state.get(
        text_key, (format_editable_time(precise_start), format_editable_time(precise_end))
    )
    # Restore candidate-specific edits after Streamlit removes absent widgets.
    st.session_state.setdefault(start_key, start_text)
    st.session_state.setdefault(end_key, end_text)
    st.session_state[precise_bounds_key] = (precise_start, precise_end)
    if pending_mark is not None:
        st.session_state[start_key] = format_editable_time(pending_mark[0])
        st.session_state[end_key] = format_editable_time(pending_mark[1])
        st.session_state[precise_bounds_key] = pending_mark
        start_text = st.session_state[start_key]
        end_text = st.session_state[end_key]
    precise_start, precise_end = st.session_state[precise_bounds_key]

    st.subheader(f"🎞️ {candidate['id']} — {candidate['reason']}", anchor=False)

    source_duration = float(analysis["duration"])
    # The strip is drawn from stored features, so it redraws instantly on every
    # adjustment. Re-encoding a preview just to find out where the speech
    # starts is the slow way to answer that question.
    pending_start = parse_editable_time(st.session_state.get(start_key, "")) or original_start
    pending_end = parse_editable_time(st.session_state.get(end_key, "")) or original_end
    strip_window = window_for(pending_start, pending_end, source_duration)
    strip_words = flatten_words(
        transcript_window(db_path, analysis_id, strip_window.view_start, strip_window.view_end)
    )
    try:
        strip = render_strip(
            Path(analysis["work_dir"]) / ".previews" / analysis_id / f"strip_{candidate['id']}.png",
            strip_window,
            audio_feature_window(
                db_path, analysis_id, strip_window.view_start, strip_window.view_end
            ),
            strip_words,
        )
        st.image(str(strip), width="stretch")
        st.caption(
            "Loudness and words around the clip. Bright is kept, dim is trimmed. "
            "Adjust below, then update the preview once you are happy."
        )
    except Exception as exc:  # noqa: BLE001 - the strip is an aid, not the feature
        st.caption(f"Timeline strip unavailable: {exc}")

    # Marks arrive outside a form; keep these widgets in the same live state.
    left, right = st.columns(2)
    with left:
        start_value = st.text_input(
            "Clip start",
            key=start_key,
            help="Use MM:SS or HH:MM:SS. Fractional seconds are optional.",
        )
        _render_timing_controls(
            "start", start_key, strip_words, source_duration, preview_token, to_end=False
        )
    with right:
        end_value = st.text_input(
            "Clip end",
            key=end_key,
            help="Use MM:SS or HH:MM:SS. Fractional seconds are optional.",
        )
        _render_timing_controls(
            "end", end_key, strip_words, source_duration, preview_token, to_end=True
        )
    update_preview = st.button(
        "Update preview", type="primary", width="stretch",
        key=f"update_preview_{preview_token}",
    )

    try:
        edited_bounds = _clip_editor_bounds(
            start_value,
            end_value,
            original_start=precise_start,
            original_end=precise_end,
            original_start_text=start_text,
            original_end_text=end_text,
            source_duration=float(analysis["duration"]),
        )
    except ValueError as exc:
        timing_valid = False
        edited_start = original_start
        edited_end = original_end
        st.error(str(exc))
        st.caption("Preview was not updated; it remains at the last valid range.")
    else:
        timing_valid = True
        edited_start = edited_bounds.start
        edited_end = edited_bounds.end
        st.session_state[precise_bounds_key] = (edited_start, edited_end)
        st.session_state[text_key] = (start_value, end_value)

    if update_preview and timing_valid:
        st.session_state[_PREVIEW_CLOSED_KEY] = False
        st.session_state[preview_bounds_key] = (edited_start, edited_end)

    preview_start, preview_end = st.session_state.get(
        preview_bounds_key,
        (original_start, original_end),
    )
    preview_start = float(preview_start)
    preview_end = float(preview_end)

    preview_dir = Path(analysis["work_dir"]) / ".previews" / analysis_id
    preview_slot = st.empty()
    if candidate_changed or (update_preview and timing_valid):
        preview_slot.empty()

    preview_closed = bool(st.session_state.get(_PREVIEW_CLOSED_KEY, False))
    if not preview_closed:
        try:
            source_video = validate_local_video(analysis["video_path"])
        except Exception as exc:
            st.error("Could not open the source VOD. Check that the local file still exists and is readable.")
            st.exception(exc)
        else:
            try:
                render_settings = load_app_settings(db_path)
                preview_layout = layout_from_settings(
                    render_settings,
                    resolve_crop_rect(db_path, analysis_id, candidate["id"]),
                )
                preview_transcript = (
                    transcript_window(db_path, analysis_id, preview_start, preview_end)
                    if (preview_layout is not None and render_settings.burn_captions)
                    else None
                )
                # Only pass render arguments when short-form output is on, so a
                # plain source-aspect preview keeps the original call shape.
                render_kwargs = {}
                if preview_layout is not None:
                    render_kwargs = {
                        "layout": preview_layout,
                        "transcript": preview_transcript,
                        "caption_style": style_from_settings(render_settings),
                    }
                with st.spinner("Preparing lightweight preview…"):
                    preview = create_preview_clip(
                        source_video,
                        preview_dir,
                        candidate["id"],
                        preview_start,
                        preview_end,
                        **render_kwargs,
                    )
                player_token = f"{preview_token}:{preview_start!r}:{preview_end!r}"
                last_mark_key = f"last_mark_{preview_token}"
                with preview_slot.container():
                    mark = preview_player(
                        preview.path, token=player_token,
                        key=f"mark_player_{player_token}", disabled=not timing_valid,
                        duration=preview_end - preview_start,
                        ack=st.session_state.get(last_mark_key),
                    )
                if isinstance(mark, dict) and mark.get("id") and mark["id"] != st.session_state.get(last_mark_key):
                    st.session_state[last_mark_key] = mark["id"]
                    try:
                        if not timing_valid:
                            raise ValueError("Correct the clip range before setting a mark.")
                        marked = marked_bounds(
                            mark, token=player_token, preview_start=preview_start,
                            preview_end=preview_end, start=edited_start, end=edited_end,
                            source_duration=float(analysis["duration"]),
                        )
                    except ValueError as exc:
                        st.session_state[f"mark_warning_{preview_token}"] = str(exc)
                    else:
                        st.session_state[f"pending_mark_{preview_token}"] = marked
                    st.rerun()
                st.caption(
                    f"Local preview only: {format_time(preview_start)} → {format_time(preview_end)}. "
                    "The full source VOD is never sent to the UI player."
                )
                if preview.cleanup_failures:
                    st.warning(
                        "Preview ready, but Windows is still holding "
                        f"{preview.cleanup_failures} older temporary preview file(s). "
                        "Cleanup will be retried the next time this preview is updated."
                    )
            except PreviewFileLockError as exc:
                st.error(
                    "Windows could not remove or replace a temporary preview file. "
                    "Close any program using the preview folder and try **Update preview** again."
                )
                st.exception(exc)
            except Exception as exc:
                st.error("Could not build the lightweight preview clip. Check the local preview folder and FFmpeg setup.")
                st.exception(exc)
    else:
        preview_slot.empty()
        st.caption("Preview closed. Use **Update preview** to load it again.")

    with st.expander("🎯 Crop framing", expanded=False):
        # Loaded here rather than reusing the preview branch's copy, which only
        # exists when a preview was rendered this run.
        _render_crop_controls(
            db_path, analysis, analysis_id, candidate, load_app_settings(db_path)
        )

    if st.button("Close preview", disabled=preview_closed, key=f"close_preview_{analysis_id}_{candidate['id']}"):
        st.session_state[_PREVIEW_CLOSED_KEY] = True
        preview_slot.empty()
        st.rerun()

    # A suggestion, not a default: filling the field would make every clip look
    # deliberately titled when it is really just the loudest line in the window.
    suggested = suggest_title(
        transcript_window(db_path, analysis_id, candidate["start"], candidate["end"]),
        candidate["start"],
        candidate["end"],
        fallback="",
    )
    title = st.text_input(
        "Optional clip title",
        value=item.get("title", ""),
        placeholder=suggested or "Untitled clips keep their candidate ID",
        key=f"title_{analysis_id}_{candidate['id']}",
    )
    st.caption(f"Signals — audio {candidate['audio_score']:.2f} · transcript {candidate['transcript_score']:.2f} · chat {candidate['chat_score']:.2f}")
    if candidate.get("transcript"):
        with st.expander("Transcript around this moment", expanded=True):
            st.write(candidate["transcript"])

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("✅ Keep", width="stretch", disabled=not timing_valid):
        item.update(status="keep", start=edited_start, end=edited_end, title=title); save_review(db_path, analysis_id, review); st.rerun()
    if b2.button("❌ Reject", width="stretch", disabled=not timing_valid):
        item.update(status="reject", start=edited_start, end=edited_end, title=title); save_review(db_path, analysis_id, review); st.rerun()
    if b3.button("↩ Unreview", width="stretch", disabled=not timing_valid):
        item.update(status="unreviewed", start=edited_start, end=edited_end, title=title); save_review(db_path, analysis_id, review); st.rerun()
    if b4.button("💾 Save timing", width="stretch", disabled=not timing_valid):
        item.update(start=edited_start, end=edited_end, title=title); save_review(db_path, analysis_id, review); st.success("Saved")

    st.divider()
    st.subheader("📦 Add to export queue", anchor=False)
    export_running = load_active_export_batch(db_path) is not None
    export_dir = path_picker(
        "Export folder",
        "export_dir_input",
        default=str(Path(analysis["work_dir"]) / "clips"),
        folder=True,
        disabled=export_running,
    )
    st.caption(
        f"Kept clips from this analysis will be queued under **{content_label}**. "
        "Queueing does not start FFmpeg."
    )
    kept = [(c, review["items"][c["id"]]) for c in candidates if review["items"][c["id"]].get("status") == "keep"]
    if st.button(
        f"Add {len(kept)} kept clip(s) to queue",
        disabled=not kept or export_running,
        type="primary",
    ):
        try:
            result = enqueue_export_items(
                db_path,
                analysis,
                [
                    {
                        "candidate_id": candidate["id"],
                        "start": review_item["start"],
                        "end": review_item["end"],
                        "title": review_item.get("title") or "",
                        "content_label": candidate.get("content_label") or content_label,
                    }
                    for candidate, review_item in kept
                ],
                export_dir,
            )
            message = f"Added {result['added']} clip(s) to the export queue."
            if result["skipped"]:
                message += f" Skipped {result['skipped']} duplicate(s)."
            st.session_state["export_notice"] = message
            st.rerun()
        except Exception as exc:
            st.exception(exc)


def _execute_export_queue(db_path: Path) -> None:
    try:
        batch, items = start_export_batch(db_path)
    except (ExportBatchAlreadyRunning, ExportQueueStateError) as exc:
        st.session_state["export_error"] = str(exc)
        st.rerun()
        return

    batch_id = str(batch["id"])
    heartbeat = ExportBatchHeartbeat(db_path, batch_id)
    overall = st.progress(0.0)
    current = st.empty()
    failures = 0
    try:
        heartbeat.start()
        for index, item in enumerate(items, start=1):
            current.write(
                f"Exporting **{item['source_name']} · {item['candidate_id']}** "
                f"({index}/{len(items)})"
            )
            try:
                source_video = validate_local_video(item["source_path"])
                export_settings = load_app_settings(db_path)
                export_layout = layout_from_settings(
                    export_settings,
                    resolve_crop_rect(db_path, item["analysis_id"], item.get("candidate_id")),
                )
                export_kwargs = {}
                if export_layout is not None:
                    export_kwargs = {
                        "layout": export_layout,
                        "transcript": (
                            transcript_window(
                                db_path, item["analysis_id"], item["start"], item["end"]
                            )
                            if export_settings.burn_captions else None
                        ),
                        "caption_style": style_from_settings(export_settings),
                        "quality": export_settings.export_quality,
                    }
                output = export_clip(
                    source_video,
                    item["export_dir"],
                    item["candidate_id"],
                    item["start"],
                    item["end"],
                    item.get("title") or None,
                    category=item.get("content_label"),
                    **export_kwargs,
                )
                complete_export_queue_item(db_path, batch_id, item["id"], output)
            except Exception as exc:
                failures += 1
                fail_export_queue_item(db_path, batch_id, item["id"], exc)
            overall.progress(index / len(items))
        finished = finish_export_batch(db_path, batch_id)
    except Exception as exc:
        interrupt_export_batch(db_path, batch_id, exc)
        st.session_state["export_error"] = f"{type(exc).__name__}: {exc}"
        st.rerun()
        return
    finally:
        heartbeat.stop()

    if failures:
        st.session_state["export_error"] = str(finished["message"])
    else:
        st.session_state["export_notice"] = str(finished["message"])
    st.rerun()


def _render_export_queue(db_path: Path) -> None:
    st.divider()
    st.subheader("📦 Export queue", anchor=False)
    active = load_active_export_batch(db_path)
    items = list_export_queue(db_path)
    if not items:
        st.caption("The queue is empty. Keep clips in an analysis, then add them here before exporting.")
        return

    completed = sum(item["status"] == "completed" for item in items)
    failed = sum(item["status"] == "failed" for item in items)
    queued = sum(item["status"] == "queued" for item in items)
    exporting = sum(item["status"] == "exporting" for item in items)
    st.caption(
        f"{queued} queued · {exporting} exporting · {completed} completed · {failed} failed"
    )
    finished_count = completed + failed
    st.progress(finished_count / len(items) if items else 0.0)

    locked = active is not None
    if locked:
        st.info(
            "An export batch is running. Queue controls stay locked until it finishes or its worker heartbeat expires. "
            "No cosmetic Stop button is shown."
        )

    for item in items:
        with st.container(border=True):
            info, title_column, actions = st.columns([3, 4, 2], vertical_alignment="bottom")
            info.markdown(f"**{item['source_name']} · {item['candidate_id']}**")
            info.caption(
                f"{format_time(float(item['start']))} → {format_time(float(item['end']))} · "
                f"{str(item['status']).title()}"
            )
            edited_title = title_column.text_input(
                "Output title",
                value=str(item.get("title") or ""),
                key=f"export_queue_title_{item['id']}",
                disabled=locked or item["status"] == "completed",
                placeholder=str(item["candidate_id"]),
            )
            save, remove = actions.columns(2)
            if save.button(
                "Save",
                key=f"save_export_queue_{item['id']}",
                disabled=locked or item["status"] == "completed" or edited_title == str(item.get("title") or ""),
                width="stretch",
            ):
                update_export_queue_title(db_path, item["id"], edited_title)
                st.rerun()
            if remove.button(
                "Remove",
                key=f"remove_export_queue_{item['id']}",
                disabled=locked,
                width="stretch",
            ):
                remove_export_queue_item(db_path, item["id"])
                st.rerun()
            if item.get("error_message"):
                st.error(str(item["error_message"]))
            if item.get("output_path"):
                st.code(str(item["output_path"]))

    start, retry, clear, refresh = st.columns(4)
    if start.button(
        f"Export {queued} queued clip(s)",
        type="primary",
        disabled=locked or queued == 0,
        width="stretch",
    ):
        _execute_export_queue(db_path)
    if retry.button(
        f"Retry {failed} failed",
        disabled=locked or failed == 0,
        width="stretch",
    ):
        count = retry_failed_export_items(db_path)
        st.session_state["export_notice"] = f"Returned {count} failed clip(s) to the queue."
        st.rerun()
    if clear.button("Clear queue", disabled=locked, width="stretch"):
        count = clear_export_queue(db_path)
        st.session_state["export_notice"] = f"Cleared {count} queue item(s). Exported files and history were kept."
        st.rerun()
    if refresh.button("Refresh", disabled=not locked, width="stretch"):
        st.rerun()


def render_mine_page(db_path: Path) -> None:
    recovered_exports = recover_stale_export_batches(db_path)
    if recovered_exports:
        st.session_state.setdefault(
            "export_error",
            f"Recovered {len(recovered_exports)} export batch(es) whose worker heartbeat expired. "
            "Failed items are still in the queue for retry.",
        )
    _restore_persistent_analysis_state(db_path)
    _run_queued_analysis(db_path)
    active_job = find_active_analysis_job(db_path)
    active_export = load_active_export_batch(db_path)
    controls_locked = active_job is not None or active_export is not None

    with st.sidebar:
        video_path, chat_path, content_label, analysis_title, work_dir = _render_source_sidebar(
            db_path,
            disabled=controls_locked,
        )
        _render_analysis_controls(
            db_path,
            video_path,
            chat_path,
            content_label,
            analysis_title,
            work_dir,
            active_job=active_job,
            disabled=active_export is not None,
        )
        _render_history_sidebar(db_path, disabled=controls_locked)
        st.caption(f"Database: `{db_path}`")
        render_shutdown(block_reason=active_work_shutdown_block_reason(db_path))

    error = st.session_state.pop("analysis_error", None)
    if error:
        st.error(f"Analysis stopped: {error}")
    notice = st.session_state.pop("analysis_notice", None)
    if notice:
        st.success(notice)
    _render_review(db_path)
    export_error = st.session_state.pop("export_error", None)
    if export_error:
        st.error(f"Export queue: {export_error}")
    export_notice = st.session_state.pop("export_notice", None)
    if export_notice:
        st.success(export_notice)
    _render_export_queue(db_path)
