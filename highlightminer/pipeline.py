from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .analysis_jobs import (
    ANALYSIS_JOB_HEARTBEAT_INTERVAL_SECONDS,
    TERMINAL_ANALYSIS_JOB_STATUSES,
    AnalysisJobStateError,
    AnalysisJobTerminalError,
    create_analysis_job,
    complete_analysis_job,
    fail_analysis_job,
    find_active_analysis_job,
    heartbeat_analysis_job,
    load_analysis_job,
    mark_analysis_job_awaiting_input,
    record_analysis_job_event,
    recover_stale_analysis_jobs,
    start_analysis_job,
    update_analysis_job_progress,
)
from .audio import analyze_audio
from .categorization import normalize_content_label
from .chat import analyze_chat, load_chat, messages_per_minute
from .config import Settings
from .diagnostic_preferences import (
    consume_detailed_diagnostics_next_run,
    detailed_diagnostics_next_run,
)
from .diagnostics import (
    diagnostic_stage,
    log_detailed,
    log_event,
    log_exception,
    media_summary,
    redacted_settings,
    safe_model_name,
    signal_statistics,
    start_detailed_run,
    stop_detailed_run,
)
from .identity import describe_source, full_file_sha256, stable_signature
from .media import extract_analysis_audio, probe_media
from .model_access import (
    ModelAccessPreferences,
    ModelDecisionRequired,
    PreparedModelReference,
    load_model_access,
    model_signature_payload,
    resolve_model_reference,
)
from .scoring import find_candidates
from .security import validate_chat_file, validate_local_video
from .storage import (
    ALGORITHM_VERSION,
    FEATURE_SCHEMA_VERSION,
    connect,
    load_reusable_features,
    register_source,
    save_analysis,
)
from .transcribe import score_text, transcribe_audio
from .transcription_status import (
    SKIP_REASON_MODEL_DOWNLOADS_DISABLED,
    SKIP_REASON_USER_REQUESTED,
    TRANSCRIPTION_AVAILABLE,
    is_transcription_skipped,
    skipped_transcription_metadata,
)
from .util import clamp, ensure_dir

Progress = Callable[[str, float], None]

_TRANSCRIPTION_PROGRESS_START = 0.32
_TRANSCRIPTION_PROGRESS_END = 0.76
REUSE_COMPATIBLE_EVIDENCE = "reuse_compatible_evidence"
FORCE_FULL_REPROCESS = "force_full_reprocess"


def _processing_mode(reuse_features: bool) -> str:
    return REUSE_COMPATIBLE_EVIDENCE if reuse_features else FORCE_FULL_REPROCESS


class _AnalysisJobHeartbeat:
    """Keep a running job's lease fresh until this worker exits the run."""

    def __init__(self, db_path: str | Path | None, job_id: str):
        self._db_path = db_path
        self._job_id = job_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not heartbeat_analysis_job(self._db_path, self._job_id):
            current = load_analysis_job(self._db_path, self._job_id)
            raise AnalysisJobStateError(
                f"Cannot heartbeat analysis job {self._job_id} while it is {current['status']}."
            )
        self._thread = threading.Thread(
            target=self._run,
            name=f"analysis-heartbeat-{self._job_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(ANALYSIS_JOB_HEARTBEAT_INTERVAL_SECONDS):
            try:
                if not heartbeat_analysis_job(self._db_path, self._job_id):
                    return
            except Exception as exc:
                log_exception("analysis.heartbeat_error", exc, job_id=self._job_id)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _noop(_: str, __: float) -> None:
    pass


def _stage_signatures(
    settings: Settings,
    chat_path: str | Path | None,
    model_access: ModelAccessPreferences,
) -> dict[str, str]:
    audio = stable_signature(
        "highlightminer-audio-features-v1",
        {
            "window": settings.audio_window_sec,
            "hop": settings.audio_hop_sec,
            "analysis_audio": "pcm_s16le-16khz-mono",
        },
    )
    # Reaction phrase scoring is deliberately excluded: cached transcript text
    # can be cheaply rescored without re-running Whisper.
    transcript = stable_signature(
        "highlightminer-whisper-transcript-v1",
        {
            "model": model_signature_payload(settings, model_access),
            "device": settings.device,
            "compute_type": settings.compute_type,
            "language": settings.language,
            "beam_size": settings.beam_size,
            "vad_filter": settings.vad_filter,
            # A transcript without word timings cannot produce captions, so it
            # must not satisfy a run that needs them.
            "word_timestamps": settings.word_timestamps,
        },
    )
    # The chat signature covers the scoring parameters as well as the file,
    # because changing the volume gate changes the features it produces.
    chat_scoring = {
        "min_burst_messages": settings.chat_min_burst_messages,
        "quiet_msgs_per_min": settings.chat_quiet_msgs_per_min,
        "active_msgs_per_min": settings.chat_active_msgs_per_min,
    }
    if chat_path:
        validated_chat = validate_chat_file(chat_path)
        chat = stable_signature(
            "highlightminer-chat-features-v2",
            {"sha256": full_file_sha256(validated_chat), "scoring": chat_scoring},
        )
    else:
        chat = stable_signature(
            "highlightminer-chat-features-v2",
            {"chat": None, "scoring": chat_scoring},
        )
    return {"audio": audio, "transcript": transcript, "chat": chat}


def _rescore_transcript(rows: list[dict], settings: Settings) -> list[dict]:
    rescored: list[dict] = []
    for row in rows:
        score, reasons = score_text(str(row.get("text", "")), settings.reaction_phrases)
        updated = dict(row)
        updated["score"] = round(score, 4)
        updated["reasons"] = reasons
        rescored.append(updated)
    return rescored


def _elapsed_since(started_at: float) -> float:
    """Measure rounded elapsed seconds for persisted timing metadata."""
    return round(max(0.0, time.perf_counter() - started_at), 3)


def _map_transcription_progress(fraction: float) -> float:
    """Map Whisper's 0..1 progress into its reserved pipeline progress range."""
    bounded = clamp(float(fraction))
    span = _TRANSCRIPTION_PROGRESS_END - _TRANSCRIPTION_PROGRESS_START
    return _TRANSCRIPTION_PROGRESS_START + (span * bounded)


def _log_model_metadata(metadata: dict | None, settings: Settings) -> None:
    data = dict(metadata or {})
    source = str(data.get("model_source") or "configured")
    model_name = safe_model_name(str(data.get("model") or settings.whisper_model), source)
    log_event(
        "model.runtime",
        model_name=model_name,
        model_source=source,
        device=str(data.get("device") or settings.device),
        compute_type=str(data.get("compute_type") or settings.compute_type),
        fallback=bool(data.get("fallback_reason")),
    )


def snapshot_analysis_config(
    *,
    video: Path,
    work: Path,
    settings: Settings,
    chat_path: str | Path | None,
    content_label: str,
    source_fingerprint: str,
    reuse_features: bool,
    transcription_requested: bool,
) -> dict:
    """Capture the inputs that must remain fixed for the lifetime of a run."""
    return {
        "version": 1,
        "video_path": str(video),
        "work_dir": str(work),
        "chat_path": str(chat_path) if chat_path else None,
        "content_label": content_label,
        "source_fingerprint": source_fingerprint,
        "reuse_features": bool(reuse_features),
        "processing_mode": _processing_mode(reuse_features),
        "transcription_requested": bool(transcription_requested),
        "settings": asdict(settings),
    }


def _restore_analysis_job_config(job: dict) -> tuple[dict, Settings]:
    config = dict(job.get("config") or {})
    if int(config.get("version", 0)) != 1:
        raise ValueError(f"Unsupported analysis job configuration version for {job['id']}.")
    stored_settings = config.get("settings")
    if not isinstance(stored_settings, dict):
        raise ValueError(f"Analysis job {job['id']} has no valid settings snapshot.")
    return config, Settings(**stored_settings)


def _job_configs_match(left: dict, right: dict) -> bool:
    invariant_keys = {
        "version",
        "video_path",
        "work_dir",
        "chat_path",
        "content_label",
        "source_fingerprint",
        "reuse_features",
        "processing_mode",
        "settings",
    }
    return {key: left.get(key) for key in invariant_keys} == {
        key: right.get(key) for key in invariant_keys
    }


def analyze_vod(
    video_path: str | Path,
    work_dir: str | Path,
    settings: Settings,
    chat_path: str | Path | None = None,
    progress: Progress | None = None,
    content_label: str | None = None,
    db_path: str | Path | None = None,
    *,
    source_info: dict | None = None,
    reuse_features: bool = True,
    allow_model_download: bool = False,
    skip_transcription: bool = False,
    analysis_job_id: str | None = None,
) -> str:
    """Analyze a VOD and persist a new analysis run in SQLite.

    Compatible feature stages are reused from prior runs. A fresh Whisper pass
    is resolved only when it is actually needed. If model downloads were
    explicitly denied and no local/cached model exists, analysis continues with
    audio plus optional chat instead of failing.
    """
    user_progress = progress or _noop
    job_id = analysis_job_id
    stored_job_config: dict | None = None
    job_created = False
    job_started = False
    resumable_job_loaded = False
    pipeline_started_at = time.perf_counter()
    timings: dict[str, float] = {}
    wav: Path | None = None
    detailed_started = False
    job_heartbeat: _AnalysisJobHeartbeat | None = None

    try:
        recovered_jobs = recover_stale_analysis_jobs(db_path)
        if recovered_jobs:
            log_event(
                "analysis.stale_jobs_recovered",
                level=logging.WARNING,
                job_ids=recovered_jobs,
            )
        if job_id is not None:
            existing_job = load_analysis_job(db_path, job_id)
            resumable_job_loaded = existing_job["status"] in {"queued", "awaiting_input"}
            stored_job_config, settings = _restore_analysis_job_config(existing_job)
            video_path = str(stored_job_config["video_path"])
            work_dir = str(stored_job_config["work_dir"])
            chat_path = stored_job_config.get("chat_path")
            content_label = str(stored_job_config.get("content_label") or "")
            reuse_features = bool(stored_job_config.get("reuse_features", True))
            skip_transcription = skip_transcription or not bool(
                stored_job_config.get("transcription_requested", True)
            )
            source_info = {"fingerprint": str(stored_job_config["source_fingerprint"])}

        detailed_requested = detailed_diagnostics_next_run(db_path)
        if detailed_requested:
            start_detailed_run()
            detailed_started = True
            log_detailed("settings.redacted", settings=redacted_settings(settings))

        user_progress("Identifying source VOD", 0.01)
        video = validate_local_video(video_path)
        work = ensure_dir(work_dir)
        normalized_content_label = normalize_content_label(content_label)
        if chat_path:
            chat_path = validate_chat_file(chat_path)
        model_access = load_model_access(db_path)
        log_detailed("database.operation", operation="load_model_access")

        source_started_at = time.perf_counter()
        with diagnostic_stage("source_setup"):
            actual_source = describe_source(video)
            if source_info and source_info.get("fingerprint") == actual_source["fingerprint"]:
                actual_source["fingerprint"] = str(source_info["fingerprint"])
            log_detailed("database.operation", operation="register_source")
            source = register_source(db_path, actual_source)

            if stored_job_config is None:
                job_config = snapshot_analysis_config(
                    video=video,
                    work=work,
                    settings=settings,
                    chat_path=chat_path,
                    content_label=normalized_content_label,
                    source_fingerprint=str(source["fingerprint"]),
                    reuse_features=reuse_features,
                    transcription_requested=not skip_transcription,
                )
                active = find_active_analysis_job(
                    db_path,
                    source_id=str(source["id"]),
                )
                decision_supplied = bool(
                    allow_model_download
                    or skip_transcription
                    or model_access.download_consent != "unset"
                    or model_access.local_model_path
                )
                if (
                    active is not None
                    and active["status"] == "awaiting_input"
                    and decision_supplied
                    and _job_configs_match(active["config"], job_config)
                ):
                    job_id = str(active["id"])
                    stored_job_config = dict(active["config"])
                else:
                    job_id = str(create_analysis_job(db_path, source, job_config)["id"])
                    stored_job_config = job_config
                    job_created = True
            elif str(source["fingerprint"]) != str(stored_job_config["source_fingerprint"]):
                raise ValueError("The source VOD no longer matches this analysis job.")

            assert job_id is not None
            start_analysis_job(db_path, job_id)
            job_started = True
            job_heartbeat = _AnalysisJobHeartbeat(db_path, job_id)
            job_heartbeat.start()

            if allow_model_download:
                record_analysis_job_event(
                    db_path,
                    job_id,
                    "model.download_allowed",
                    stage="model_resolution",
                    message="Model download approved for this run",
                )
            if skip_transcription:
                record_analysis_job_event(
                    db_path,
                    job_id,
                    "transcription.skip_requested",
                    level="warning",
                    stage="model_resolution",
                    message="Continuing without speech recognition",
                )

            def report(stage: str, message: str, value: float) -> None:
                update_analysis_job_progress(
                    db_path,
                    job_id,
                    stage=stage,
                    progress=min(1.0, max(0.0, float(value))),
                    message=message,
                    timings=timings,
                )
                user_progress(message, value)

            report("source_setup", "Preparing source evidence", 0.01)
            signatures = _stage_signatures(settings, chat_path, model_access)
            if reuse_features:
                log_detailed("database.operation", operation="load_reusable_features")
                cached = load_reusable_features(
                    db_path,
                    source["id"],
                    audio_signature=signatures["audio"],
                    transcript_signature=signatures["transcript"],
                    chat_signature=signatures["chat"],
                )
            else:
                cached = {
                    "audio": None,
                    "transcript": None,
                    "transcription": None,
                    "chat": None,
                    "chat_info": None,
                    "from": {},
                }
        timings["source_setup_seconds"] = _elapsed_since(source_started_at)

        log_event(
            "analysis.start",
            job_id=job_id,
            reuse_enabled=bool(reuse_features),
            processing_mode=_processing_mode(reuse_features),
            chat_available=bool(chat_path),
            transcription_requested=bool(stored_job_config.get("transcription_requested", True)),
        )

        probe_started_at = time.perf_counter()
        report("media_probe", "Probing media", 0.03)
        with diagnostic_stage("media_probe"):
            media = probe_media(video)
        timings["media_probe_seconds"] = _elapsed_since(probe_started_at)
        duration = float(media["duration"])
        if duration <= 0:
            raise ValueError("The VOD duration reported by ffprobe is invalid.")
        log_detailed("media.info", media=media_summary(media))

        audio_features = cached.get("audio")
        transcript = cached.get("transcript")
        transcript_meta = cached.get("transcription")
        chat_features = cached.get("chat")
        chat_info = cached.get("chat_info")
        cache_from = dict(cached.get("from") or {})
        log_event(
            "cache.reuse",
            reuse_enabled=bool(reuse_features),
            reused_stages=sorted(cache_from),
        )
        record_analysis_job_event(
            db_path,
            job_id,
            "cache.evaluated",
            stage="source_setup",
            message=(
                "Reusing compatible evidence: " + ", ".join(sorted(cache_from))
                if cache_from
                else "No compatible evidence reused"
            ),
            details={
                "processing_mode": _processing_mode(reuse_features),
                "reuse_enabled": bool(reuse_features),
                "reused_stages": sorted(cache_from),
            },
        )

        need_audio = audio_features is None
        need_transcript = transcript is None
        prepared_model: PreparedModelReference | None = None

        report("model_resolution", "Resolving speech-recognition model", 0.06)
        model_resolution_started_at = time.perf_counter()
        with diagnostic_stage("model_resolution"):
            if skip_transcription:
                transcript = []
                transcript_meta = skipped_transcription_metadata(
                    settings.whisper_model,
                    SKIP_REASON_USER_REQUESTED,
                )
                cache_from.pop("transcript", None)
                need_transcript = False
                log_detailed("model.resolution", decision="user_skipped")
                record_analysis_job_event(
                    db_path,
                    job_id,
                    "transcription.skipped",
                    level="warning",
                    stage="model_resolution",
                    message="Speech recognition was skipped by request",
                    details={"reason": SKIP_REASON_USER_REQUESTED},
                )
            elif need_transcript:
                prepared_model = resolve_model_reference(
                    settings,
                    model_access,
                    allow_download_override=allow_model_download,
                )
                if prepared_model is None:
                    transcript = []
                    transcript_meta = skipped_transcription_metadata(
                        settings.whisper_model,
                        SKIP_REASON_MODEL_DOWNLOADS_DISABLED,
                    )
                    need_transcript = False
                    log_detailed("model.resolution", decision="downloads_disabled")
                    record_analysis_job_event(
                        db_path,
                        job_id,
                        "transcription.fallback",
                        level="warning",
                        stage="model_resolution",
                        message="Speech recognition unavailable; continuing with available signals",
                        details={"reason": SKIP_REASON_MODEL_DOWNLOADS_DISABLED},
                    )
                else:
                    log_detailed(
                        "model.resolution",
                        decision="resolved",
                        model_name=safe_model_name(prepared_model.display_name, prepared_model.source),
                        model_source=prepared_model.source,
                        local_files_only=prepared_model.local_files_only,
                    )
            else:
                log_detailed("model.resolution", decision="cached_transcript")
        timings["model_resolution_seconds"] = _elapsed_since(model_resolution_started_at)

        # The run is now committed to proceed. Consume the one-shot flag only here,
        # after any ModelDecisionRequired interaction has been resolved.
        if detailed_requested:
            consume_detailed_diagnostics_next_run(db_path)
            log_detailed("diagnostics.one_shot_consumed", next_mode="standard")

        transcription_skipped = is_transcription_skipped(transcript_meta)
        skip_reason = str((transcript_meta or {}).get("reason") or "") if transcription_skipped else None
        need_wav = need_audio or need_transcript

        if need_wav:
            temp_handle = tempfile.NamedTemporaryFile(
                prefix="highlightminer-",
                suffix=".wav",
                dir=work,
                delete=False,
            )
            wav = Path(temp_handle.name)
            temp_handle.close()
            report("audio_extract", "Extracting 16 kHz analysis audio", 0.10)
            stage_started_at = time.perf_counter()
            with diagnostic_stage("audio_extract"):
                extract_analysis_audio(video, wav)
            timings["audio_extract_seconds"] = _elapsed_since(stage_started_at)
        elif transcription_skipped:
            report("audio_extract", "Speech recognition disabled; using available signals", 0.16)
        else:
            report("audio_extract", "Reusing cached analysis audio features + transcript", 0.16)

        if need_audio:
            report("audio_analysis", "Analyzing audio energy", 0.20)
            assert wav is not None
            stage_started_at = time.perf_counter()
            with diagnostic_stage("audio_analysis"):
                audio_features = analyze_audio(wav, settings.audio_window_sec, settings.audio_hop_sec)
            timings["audio_analysis_seconds"] = _elapsed_since(stage_started_at)
        else:
            report("audio_analysis", "Reusing cached audio features", 0.24)
        log_detailed("signal.statistics", signal="audio", statistics=signal_statistics(list(audio_features or [])))

        if need_transcript:
            report("transcription", "Preparing faster-whisper transcription", _TRANSCRIPTION_PROGRESS_START)
            assert wav is not None
            assert prepared_model is not None
            stage_started_at = time.perf_counter()

            def transcription_progress(message: str, fraction: float) -> None:
                report("transcription", message, _map_transcription_progress(fraction))

            with diagnostic_stage("transcription"):
                transcript, transcript_meta = transcribe_audio(
                    wav,
                    settings,
                    model_access=model_access,
                    prepared_model=prepared_model,
                    audio_duration=duration,
                    progress=transcription_progress,
                )
            timings["transcription_seconds"] = _elapsed_since(stage_started_at)
        elif transcription_skipped:
            report("transcription", "Skipping speech recognition", 0.60)
        else:
            report("transcription", "Reusing cached Whisper transcript", 0.60)

        transcript = _rescore_transcript(list(transcript or []), settings)
        transcript_meta = dict(transcript_meta or {})
        if not transcription_skipped:
            transcript_meta.setdefault("status", TRANSCRIPTION_AVAILABLE)
            transcript_meta["reaction_scoring"] = "current-settings"
        _log_model_metadata(transcript_meta, settings)
        log_detailed("signal.statistics", signal="transcript", statistics=signal_statistics(transcript))

        if chat_path:
            validated_chat = validate_chat_file(chat_path)
            if chat_features is None:
                report("chat_analysis", "Parsing chat", _TRANSCRIPTION_PROGRESS_END)
                stage_started_at = time.perf_counter()
                with diagnostic_stage("chat_analysis"):
                    records = load_chat(validated_chat)
                    chat_features = analyze_chat(
                        records,
                        duration,
                        min_burst_messages=settings.chat_min_burst_messages,
                        quiet_msgs_per_min=settings.chat_quiet_msgs_per_min,
                        active_msgs_per_min=settings.chat_active_msgs_per_min,
                    )
                timings["chat_analysis_seconds"] = _elapsed_since(stage_started_at)
                rate = messages_per_minute(records, duration)
                chat_info = {
                    "path": str(validated_chat),
                    "messages": len(records),
                    "messages_per_minute": round(rate, 2),
                    # Empty features mean the chat was too quiet to trust, which
                    # downstream treats exactly like "no chat supplied".
                    "volume_gated": bool(records) and not chat_features,
                }
                if chat_info["volume_gated"]:
                    report(
                        "chat_analysis",
                        f"Chat too quiet to score ({rate:.1f} msgs/min), continuing without it",
                        _TRANSCRIPTION_PROGRESS_END,
                    )
            else:
                report("chat_analysis", "Reusing cached chat features", 0.78)
                chat_info = dict(chat_info or {})
                chat_info["path"] = str(validated_chat)
        else:
            chat_features = []
            chat_info = {"path": None, "messages": 0}
        log_detailed("signal.statistics", signal="chat", statistics=signal_statistics(list(chat_features or [])))

        report("candidate_ranking", "Ranking candidate moments", 0.86)
        stage_started_at = time.perf_counter()
        with diagnostic_stage("candidate_ranking"):
            candidates = find_candidates(
                duration,
                list(audio_features or []),
                transcript,
                list(chat_features or []),
                settings,
                transcript_available=not transcription_skipped,
            )
        timings["candidate_ranking_seconds"] = _elapsed_since(stage_started_at)
        for candidate in candidates:
            candidate["content_label"] = normalized_content_label
        log_event("candidates.generated", count=len(candidates))

        timings["pipeline_elapsed_seconds"] = _elapsed_since(pipeline_started_at)
        cache_info = {
            "reused": cache_from,
            "reused_stages": sorted(cache_from),
            "reuse_enabled": bool(reuse_features),
            "processing_mode": _processing_mode(reuse_features),
            "transcription_skipped": transcription_skipped,
            "transcription_skip_reason": skip_reason,
            "timings": timings,
        }
        analysis = {
            "version": 2,
            "video_path": str(video),
            "content_label": normalized_content_label,
            "duration": duration,
            "media": media,
            "transcription": transcript_meta,
            "chat": chat_info,
            "settings": settings.__dict__,
            "algorithm_version": ALGORITHM_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "candidates": candidates,
        }
        report("database_save", "Saving analysis database", 0.96)
        save_signatures: dict[str, str | None] = dict(signatures)
        if transcription_skipped:
            save_signatures["transcript"] = None
        reused = ", ".join(sorted(cache_from))
        suffix = f" · reused {reused}" if reused else ""
        if transcription_skipped:
            suffix += " · no transcript"
        completion_message = f"Done — {len(candidates)} candidates{suffix}"
        log_detailed("database.operation", operation="save_analysis")
        database_save_started_at = time.perf_counter()
        try:
            with connect(db_path) as transaction:
                with diagnostic_stage("database_save"):
                    analysis_id = save_analysis(
                        db_path,
                        analysis,
                        transcript,
                        list(audio_features or []),
                        list(chat_features or []),
                        work_dir=work,
                        source=source,
                        signatures=save_signatures,
                        cache_info=cache_info,
                        connection=transaction,
                    )
                timings["database_save_seconds"] = _elapsed_since(database_save_started_at)
                timings["pipeline_elapsed_seconds"] = _elapsed_since(pipeline_started_at)
                complete_analysis_job(
                    db_path,
                    job_id,
                    analysis_id,
                    message=completion_message,
                    timings=timings,
                    connection=transaction,
                )
        except AnalysisJobStateError:
            current_job = load_analysis_job(db_path, job_id)
            if current_job["status"] in TERMINAL_ANALYSIS_JOB_STATUSES:
                raise AnalysisJobTerminalError(current_job) from None
            raise
        try:
            user_progress(completion_message, 1.0)
        except Exception as exc:
            log_exception(
                "analysis.completion_callback_error",
                exc,
                job_id=job_id,
                analysis_id=analysis_id,
            )
        log_event(
            "analysis.complete",
            duration_seconds=_elapsed_since(pipeline_started_at),
            candidate_count=len(candidates),
            reused_stages=sorted(cache_from),
        )
        return analysis_id
    except ModelDecisionRequired as exc:
        if job_id is not None and job_started:
            try:
                mark_analysis_job_awaiting_input(db_path, job_id, message=str(exc))
            except AnalysisJobStateError:
                current_job = load_analysis_job(db_path, job_id)
                if current_job["status"] in TERMINAL_ANALYSIS_JOB_STATUSES:
                    log_event(
                        "analysis.model_decision_discarded",
                        level=logging.WARNING,
                        job_id=job_id,
                        terminal_status=current_job["status"],
                    )
                    raise AnalysisJobTerminalError(current_job) from None
                if current_job["status"] != "awaiting_input":
                    raise
            setattr(exc, "analysis_job_id", job_id)
        log_event("analysis.model_decision_required", level=logging.WARNING, job_id=job_id)
        raise
    except AnalysisJobTerminalError as exc:
        log_event(
            "analysis.terminal_race",
            level=logging.WARNING,
            job_id=job_id,
            terminal_status=exc.job["status"],
        )
        raise
    except Exception as exc:
        if job_id is not None and (job_created or job_started or resumable_job_loaded):
            timings["pipeline_elapsed_seconds"] = _elapsed_since(pipeline_started_at)
            failed = fail_analysis_job(db_path, job_id, exc, timings=timings)
            if not failed:
                current_job = load_analysis_job(db_path, job_id)
                if current_job["status"] in TERMINAL_ANALYSIS_JOB_STATUSES:
                    log_event(
                        "analysis.failure_discarded",
                        level=logging.WARNING,
                        job_id=job_id,
                        terminal_status=current_job["status"],
                        stale_error_type=type(exc).__name__,
                    )
                    raise AnalysisJobTerminalError(current_job) from exc
        log_exception(
            "analysis.error",
            exc,
            job_id=job_id,
            duration_seconds=_elapsed_since(pipeline_started_at),
        )
        raise
    finally:
        if job_heartbeat is not None:
            job_heartbeat.stop()
        if wav is not None:
            try:
                wav.unlink(missing_ok=True)
            except OSError:
                pass
        if detailed_started:
            stop_detailed_run()
