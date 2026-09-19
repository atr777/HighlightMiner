from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .categorization import normalize_content_label
from .identity import describe_source
from .runtime import app_root
from .security import validate_legacy_analysis_file, validate_local_video
from .util import load_json

SCHEMA_VERSION = 5
FEATURE_SCHEMA_VERSION = 2
ALGORITHM_VERSION = "heuristic-v1"
DATABASE_FILENAME = "highlightminer.db"
SUPPORTED_ANALYSIS_SOURCE_VERSIONS = frozenset({1, 2})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


DATABASE_PATH_ENV = "HIGHLIGHTMINER_DB"


def default_db_path() -> Path:
    """Where the database lives, unless told otherwise.

    Working files often belong on a different drive from the application, so
    the location is overridable. Without this the desktop UI could only ever
    open the database beside the executable, while the CLI could be pointed
    anywhere with --db, which meant an analysis run on another drive was
    invisible in the app.
    """
    override = os.environ.get(DATABASE_PATH_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return app_root() / DATABASE_FILENAME


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _unjson(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_column(conn: sqlite3.Connection, table: str, definition: str) -> bool:
    name = definition.split()[0]
    if name not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")
        return True
    return False


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or default_db_path()).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    initialize(conn)
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL UNIQUE,
            current_path TEXT NOT NULL,
            video_name TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            last_run_number INTEGER NOT NULL DEFAULT 0,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            video_path TEXT NOT NULL,
            video_name TEXT NOT NULL,
            content_label TEXT NOT NULL,
            duration REAL NOT NULL,
            work_dir TEXT NOT NULL,
            media_json TEXT NOT NULL,
            transcription_json TEXT NOT NULL,
            chat_json TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            source_version INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS candidates (
            analysis_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            score REAL NOT NULL,
            peak_time REAL NOT NULL,
            start REAL NOT NULL,
            end REAL NOT NULL,
            audio_score REAL NOT NULL,
            transcript_score REAL NOT NULL,
            chat_score REAL NOT NULL,
            reason TEXT NOT NULL,
            transcript TEXT NOT NULL,
            content_label TEXT NOT NULL,
            PRIMARY KEY (analysis_id, candidate_id),
            FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS reviews (
            analysis_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'unreviewed'
                CHECK (status IN ('unreviewed', 'keep', 'reject')),
            start REAL NOT NULL,
            end REAL NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            reviewed_at TEXT,
            updated_at TEXT NOT NULL,
            exported_at TEXT,
            export_path TEXT,
            PRIMARY KEY (analysis_id, candidate_id),
            FOREIGN KEY (analysis_id, candidate_id)
                REFERENCES candidates(analysis_id, candidate_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS review_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            from_status TEXT NOT NULL,
            to_status TEXT NOT NULL,
            start REAL NOT NULL,
            end REAL NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (analysis_id, candidate_id)
                REFERENCES candidates(analysis_id, candidate_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS exports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            output_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (analysis_id, candidate_id)
                REFERENCES candidates(analysis_id, candidate_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS export_batches (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL
                CHECK (status IN ('running', 'completed', 'failed', 'interrupted')),
            total_items INTEGER NOT NULL DEFAULT 0,
            completed_items INTEGER NOT NULL DEFAULT 0,
            failed_items INTEGER NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT
        );

        CREATE TABLE IF NOT EXISTS export_queue_items (
            id TEXT PRIMARY KEY,
            analysis_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_name TEXT NOT NULL,
            content_label TEXT NOT NULL,
            start REAL NOT NULL,
            end REAL NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            export_dir TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK (status IN ('queued', 'exporting', 'completed', 'failed')),
            batch_id TEXT,
            output_path TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            UNIQUE (analysis_id, candidate_id),
            FOREIGN KEY (analysis_id, candidate_id)
                REFERENCES candidates(analysis_id, candidate_id) ON DELETE CASCADE,
            FOREIGN KEY (batch_id) REFERENCES export_batches(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS transcript_segments (
            analysis_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            start REAL NOT NULL,
            end REAL NOT NULL,
            text TEXT NOT NULL,
            score REAL NOT NULL,
            reasons_json TEXT NOT NULL,
            PRIMARY KEY (analysis_id, seq),
            FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS audio_features (
            analysis_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            time REAL NOT NULL,
            dbfs REAL NOT NULL,
            energy REAL NOT NULL,
            onset REAL NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (analysis_id, seq),
            FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS chat_features (
            analysis_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            time REAL NOT NULL,
            count INTEGER NOT NULL,
            ratio REAL NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (analysis_id, seq),
            FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS analysis_jobs (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK (status IN (
                    'queued', 'running', 'awaiting_input',
                    'completed', 'failed', 'cancelled', 'interrupted'
                )),
            stage TEXT NOT NULL,
            progress REAL NOT NULL DEFAULT 0.0
                CHECK (progress >= 0.0 AND progress <= 1.0),
            message TEXT NOT NULL DEFAULT '',
            config_json TEXT NOT NULL,
            timings_json TEXT NOT NULL DEFAULT '{}',
            analysis_id TEXT,
            error_type TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            updated_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS analysis_job_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            level TEXT NOT NULL CHECK (level IN ('info', 'warning', 'error')),
            event TEXT NOT NULL,
            stage TEXT,
            message TEXT NOT NULL DEFAULT '',
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (job_id) REFERENCES analysis_jobs(id) ON DELETE CASCADE
        );
        """
    )

    for definition in (
        "source_id TEXT",
        "source_fingerprint TEXT",
        "run_number INTEGER NOT NULL DEFAULT 1",
        f"algorithm_version TEXT NOT NULL DEFAULT '{ALGORITHM_VERSION}'",
        f"feature_schema_version INTEGER NOT NULL DEFAULT {FEATURE_SCHEMA_VERSION}",
        "audio_signature TEXT",
        "transcript_signature TEXT",
        "chat_signature TEXT",
        "cache_json TEXT NOT NULL DEFAULT '{}'",
    ):
        _ensure_column(conn, "analyses", definition)
    source_counter_added = _ensure_column(
        conn,
        "sources",
        "last_run_number INTEGER NOT NULL DEFAULT 0",
    )
    _ensure_column(conn, "candidates", "features_json TEXT NOT NULL DEFAULT '{}'")
    _ensure_column(conn, "transcript_segments", "words_json TEXT NOT NULL DEFAULT '[]'")
    _ensure_column(conn, "sources", "crop_rect_json TEXT")
    if source_counter_added:
        conn.execute(
            """
            UPDATE sources
            SET last_run_number = MAX(
                last_run_number,
                COALESCE((
                    SELECT MAX(a.run_number) FROM analyses a WHERE a.source_id = sources.id
                ), 0)
            )
            """
        )

    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_candidates_analysis_rank
            ON candidates(analysis_id, rank);
        CREATE INDEX IF NOT EXISTS idx_reviews_status
            ON reviews(status);
        CREATE INDEX IF NOT EXISTS idx_review_events_candidate
            ON review_events(analysis_id, candidate_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_exports_candidate
            ON exports(analysis_id, candidate_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_export_queue_status_created
            ON export_queue_items(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_export_queue_batch
            ON export_queue_items(batch_id, status);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_export_batches_active
            ON export_batches((1))
            WHERE status = 'running';
        CREATE INDEX IF NOT EXISTS idx_transcript_analysis_time
            ON transcript_segments(analysis_id, start, end);
        CREATE INDEX IF NOT EXISTS idx_analyses_created
            ON analyses(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_analyses_content
            ON analyses(content_label);
        CREATE INDEX IF NOT EXISTS idx_analyses_source
            ON analyses(source_id, run_number);
        CREATE INDEX IF NOT EXISTS idx_analyses_audio_cache
            ON analyses(source_id, audio_signature, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_analyses_transcript_cache
            ON analyses(source_id, transcript_signature, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_analyses_chat_cache
            ON analyses(source_id, chat_signature, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_analysis_jobs_updated
            ON analysis_jobs(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status_updated
            ON analysis_jobs(status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_analysis_job_events_job
            ON analysis_job_events(job_id, id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_analysis_jobs_active_source_id
            ON analysis_jobs(source_id)
            WHERE status IN ('queued', 'running', 'awaiting_input');

        CREATE TRIGGER IF NOT EXISTS prevent_analysis_job_config_mutation
        BEFORE UPDATE OF source_id, source_fingerprint, config_json ON analysis_jobs
        WHEN NEW.source_id <> OLD.source_id
          OR NEW.source_fingerprint <> OLD.source_fingerprint
          OR NEW.config_json <> OLD.config_json
        BEGIN
            SELECT RAISE(ABORT, 'analysis job configuration is immutable');
        END;
        """
    )
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()



def source_crop_rect(db_path: str | Path | None, fingerprint: str) -> dict | None:
    """The crop region remembered for one source, if any.

    Where the 9:16 window belongs depends on the channel's overlay layout, not
    on global preference, so it is stored against the source rather than the
    settings profile.
    """
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT crop_rect_json FROM sources WHERE fingerprint = ?", (fingerprint,)
        ).fetchone()
    if not row or not row["crop_rect_json"]:
        return None
    rect = _unjson(row["crop_rect_json"], None)
    return rect if isinstance(rect, dict) else None


def save_source_crop_rect(
    db_path: str | Path | None,
    fingerprint: str,
    rect: dict | None,
) -> bool:
    """Remember (or forget) the crop region for one source. False if unknown."""
    with connect(db_path) as conn:
        cursor = conn.execute(
            "UPDATE sources SET crop_rect_json = ? WHERE fingerprint = ?",
            (_json(rect) if rect else None, fingerprint),
        )
        conn.commit()
        return cursor.rowcount > 0


def analysis_crop_rect(db_path: str | Path | None, analysis_id: str) -> dict | None:
    """The crop region for the source behind an analysis."""
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT s.crop_rect_json
            FROM analyses a JOIN sources s ON s.id = a.source_id
            WHERE a.id = ?
            """,
            (analysis_id,),
        ).fetchone()
    if not row or not row["crop_rect_json"]:
        return None
    rect = _unjson(row["crop_rect_json"], None)
    return rect if isinstance(rect, dict) else None


def register_source(db_path: str | Path | None, source: dict[str, Any]) -> dict[str, Any]:
    fingerprint = str(source["fingerprint"])
    current_path = str(Path(source["path"]).expanduser().resolve())
    video_name = str(source.get("video_name") or Path(current_path).name)
    file_size = int(source.get("file_size", Path(current_path).stat().st_size))
    now = utc_now()

    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM sources WHERE fingerprint = ?", (fingerprint,)).fetchone()
        if row is None:
            source_id = uuid.uuid4().hex
            conn.execute(
                """
                INSERT INTO sources(id, fingerprint, current_path, video_name, file_size, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, fingerprint, current_path, video_name, file_size, now, now),
            )
        else:
            source_id = str(row["id"])
            conn.execute(
                """
                UPDATE sources
                SET current_path = ?, video_name = ?, file_size = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (current_path, video_name, file_size, now, source_id),
            )

        legacy_rows = conn.execute(
            """
            SELECT id FROM analyses
            WHERE source_id IS NULL AND video_path = ?
            ORDER BY created_at, id
            """,
            (current_path,),
        ).fetchall()
        next_run = int(
            conn.execute(
                "SELECT COALESCE(MAX(run_number), 0) FROM analyses WHERE source_id = ?",
                (source_id,),
            ).fetchone()[0]
        )
        for legacy in legacy_rows:
            next_run += 1
            conn.execute(
                """
                UPDATE analyses
                SET source_id = ?, source_fingerprint = ?, run_number = ?
                WHERE id = ?
                """,
                (source_id, fingerprint, next_run, legacy["id"]),
            )

        conn.execute(
            """
            UPDATE sources
            SET last_run_number = MAX(last_run_number, ?)
            WHERE id = ?
            """,
            (next_run, source_id),
        )

        conn.execute(
            """
            UPDATE analyses
            SET video_path = ?, video_name = ?, source_fingerprint = ?
            WHERE source_id = ?
            """,
            (current_path, video_name, fingerprint, source_id),
        )
        conn.commit()

    return {
        "id": source_id,
        "fingerprint": fingerprint,
        "path": current_path,
        "video_name": video_name,
        "file_size": file_size,
    }


def find_source_runs(
    db_path: str | Path | None,
    video_path: str | Path,
    *,
    source: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    video = validate_local_video(video_path)
    source_row = register_source(db_path, source or describe_source(video))
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                a.id, a.created_at, a.run_number, a.content_label, a.video_name,
                a.source_fingerprint, a.source_version, a.feature_schema_version,
                COUNT(c.candidate_id) AS candidates,
                SUM(CASE WHEN r.status = 'keep' THEN 1 ELSE 0 END) AS kept,
                SUM(CASE WHEN r.status = 'reject' THEN 1 ELSE 0 END) AS rejected,
                SUM(CASE WHEN r.status = 'unreviewed' THEN 1 ELSE 0 END) AS unreviewed
            FROM analyses a
            LEFT JOIN candidates c ON c.analysis_id = a.id
            LEFT JOIN reviews r
                ON r.analysis_id = c.analysis_id AND r.candidate_id = c.candidate_id
            WHERE a.source_id = ?
            GROUP BY a.id
            ORDER BY a.run_number DESC, a.created_at DESC
            """,
            (source_row["id"],),
        ).fetchall()
    runs = [dict(row) for row in rows]
    for run in runs:
        run["compatible"] = analysis_run_is_compatible(
            run,
            expected_source_fingerprint=str(source_row["fingerprint"]),
        )
    return source_row, runs


def analysis_run_is_compatible(
    run: dict[str, Any],
    *,
    expected_source_fingerprint: str | None = None,
) -> bool:
    """Return whether this completed analysis can be loaded by this app build."""
    if expected_source_fingerprint is not None:
        if str(run.get("source_fingerprint") or "") != expected_source_fingerprint:
            return False
    try:
        source_version = int(run.get("source_version", 0))
        feature_schema_version = int(run.get("feature_schema_version", 0))
    except (TypeError, ValueError):
        return False
    return (
        source_version in SUPPORTED_ANALYSIS_SOURCE_VERSIONS
        and 0 < feature_schema_version <= FEATURE_SCHEMA_VERSION
    )


def _feature_rows(conn: sqlite3.Connection, table: str, analysis_id: str) -> list[dict[str, Any]]:
    if table == "audio_features":
        rows = conn.execute(
            "SELECT time, dbfs, energy, onset, score FROM audio_features WHERE analysis_id = ? ORDER BY seq",
            (analysis_id,),
        ).fetchall()
    elif table == "transcript_segments":
        rows = conn.execute(
            "SELECT start, end, text, score, reasons_json, words_json FROM transcript_segments WHERE analysis_id = ? ORDER BY seq",
            (analysis_id,),
        ).fetchall()
        return [
            {
                "start": float(row["start"]),
                "end": float(row["end"]),
                "text": row["text"],
                "score": float(row["score"]),
                "reasons": _unjson(row["reasons_json"], []),
                "words": _unjson(row["words_json"], []),
            }
            for row in rows
        ]
    elif table == "chat_features":
        rows = conn.execute(
            "SELECT time, count, ratio, score FROM chat_features WHERE analysis_id = ? ORDER BY seq",
            (analysis_id,),
        ).fetchall()
    else:
        raise ValueError(f"Unsupported feature table: {table}")
    return [dict(row) for row in rows]


def load_reusable_features(
    db_path: str | Path | None,
    source_id: str,
    *,
    audio_signature: str,
    transcript_signature: str,
    chat_signature: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "audio": None,
        "transcript": None,
        "transcription": None,
        "chat": None,
        "chat_info": None,
        "from": {},
    }
    with connect(db_path) as conn:
        stage_specs = (
            ("audio", "audio_signature", audio_signature, "audio_features"),
            ("transcript", "transcript_signature", transcript_signature, "transcript_segments"),
            ("chat", "chat_signature", chat_signature, "chat_features"),
        )
        for stage, column, signature, table in stage_specs:
            row = conn.execute(
                f"""
                SELECT id, transcription_json, chat_json
                FROM analyses
                WHERE source_id = ? AND {column} = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (source_id, signature),
            ).fetchone()
            if row is None:
                continue
            analysis_id = str(row["id"])
            values = _feature_rows(conn, table, analysis_id)
            if stage != "chat" and not values:
                continue
            result[stage] = values
            result["from"][stage] = analysis_id
            if stage == "transcript":
                result["transcription"] = _unjson(row["transcription_json"], {})
            elif stage == "chat":
                result["chat_info"] = _unjson(row["chat_json"], {})
    return result


def save_analysis(
    db_path: str | Path | None,
    analysis: dict,
    transcript: Iterable[dict],
    audio_features: Iterable[dict],
    chat_features: Iterable[dict],
    *,
    work_dir: str | Path,
    analysis_id: str | None = None,
    source: dict[str, Any] | None = None,
    signatures: dict[str, str] | None = None,
    cache_info: dict[str, Any] | None = None,
    connection: sqlite3.Connection | None = None,
) -> str:
    analysis_id = analysis_id or uuid.uuid4().hex
    video = validate_local_video(analysis["video_path"])
    owns_connection = connection is None
    if owns_connection:
        source_row = register_source(db_path, source or describe_source(video))
    else:
        if source is None or not source.get("id"):
            raise ValueError("An existing source is required for transactional analysis saves.")
        persisted_source = connection.execute(
            "SELECT id, fingerprint FROM sources WHERE id = ?",
            (str(source["id"]),),
        ).fetchone()
        if persisted_source is None:
            raise KeyError(f"Source not found: {source['id']}")
        if str(persisted_source["fingerprint"]) != str(source["fingerprint"]):
            raise ValueError("The source ID and fingerprint do not identify the same source.")
        source_row = {
            "id": str(persisted_source["id"]),
            "fingerprint": str(persisted_source["fingerprint"]),
        }
    content_label = normalize_content_label(analysis.get("content_label"))
    created_at = utc_now()
    signatures = signatures or {}
    cache_info = cache_info or {}

    candidates = list(analysis.get("candidates", []))
    transcript_rows = list(transcript)
    audio_rows = list(audio_features)
    chat_rows = list(chat_features)

    connection_scope = connect(db_path) if owns_connection else nullcontext(connection)
    with connection_scope as conn:
        cursor = conn.execute(
            """
            UPDATE sources
            SET last_run_number = last_run_number + 1
            WHERE id = ?
            """,
            (source_row["id"],),
        )
        if not cursor.rowcount:
            raise KeyError(f"Source not found: {source_row['id']}")
        run_number = int(
            conn.execute(
                "SELECT last_run_number FROM sources WHERE id = ?",
                (source_row["id"],),
            ).fetchone()[0]
        )
        conn.execute(
            """
            INSERT INTO analyses(
                id, created_at, video_path, video_name, content_label, duration,
                work_dir, media_json, transcription_json, chat_json, settings_json,
                source_version, source_id, source_fingerprint, run_number,
                algorithm_version, feature_schema_version, audio_signature,
                transcript_signature, chat_signature, cache_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                created_at,
                str(video),
                video.name,
                content_label,
                float(analysis.get("duration", 0.0)),
                str(Path(work_dir).expanduser().resolve()),
                _json(analysis.get("media", {})),
                _json(analysis.get("transcription", {})),
                _json(analysis.get("chat", {})),
                _json(analysis.get("settings", {})),
                int(analysis.get("version", 2)),
                source_row["id"],
                source_row["fingerprint"],
                run_number,
                str(analysis.get("algorithm_version", ALGORITHM_VERSION)),
                int(analysis.get("feature_schema_version", FEATURE_SCHEMA_VERSION)),
                signatures.get("audio"),
                signatures.get("transcript"),
                signatures.get("chat"),
                _json(cache_info),
            ),
        )

        for c in candidates:
            cid = str(c["id"])
            conn.execute(
                """
                INSERT INTO candidates(
                    analysis_id, candidate_id, rank, score, peak_time, start, end,
                    audio_score, transcript_score, chat_score, reason, transcript,
                    content_label, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    cid,
                    int(c.get("rank", 0)),
                    float(c.get("score", 0.0)),
                    float(c.get("peak_time", 0.0)),
                    float(c.get("start", 0.0)),
                    float(c.get("end", 0.0)),
                    float(c.get("audio_score", 0.0)),
                    float(c.get("transcript_score", 0.0)),
                    float(c.get("chat_score", 0.0)),
                    str(c.get("reason", "")),
                    str(c.get("transcript", "")),
                    normalize_content_label(c.get("content_label") or content_label),
                    _json(c.get("features", {})),
                ),
            )
            conn.execute(
                """
                INSERT INTO reviews(
                    analysis_id, candidate_id, status, start, end, title, updated_at
                ) VALUES (?, ?, 'unreviewed', ?, ?, '', ?)
                """,
                (analysis_id, cid, float(c["start"]), float(c["end"]), created_at),
            )

        conn.executemany(
            """
            INSERT INTO transcript_segments(
                analysis_id, seq, start, end, text, score, reasons_json, words_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    analysis_id,
                    i,
                    float(row.get("start", 0.0)),
                    float(row.get("end", 0.0)),
                    str(row.get("text", "")),
                    float(row.get("score", 0.0)),
                    _json(row.get("reasons", [])),
                    _json(row.get("words", [])),
                )
                for i, row in enumerate(transcript_rows)
            ],
        )
        conn.executemany(
            """
            INSERT INTO audio_features(
                analysis_id, seq, time, dbfs, energy, onset, score
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    analysis_id,
                    i,
                    float(row.get("time", 0.0)),
                    float(row.get("dbfs", 0.0)),
                    float(row.get("energy", 0.0)),
                    float(row.get("onset", 0.0)),
                    float(row.get("score", 0.0)),
                )
                for i, row in enumerate(audio_rows)
            ],
        )
        conn.executemany(
            """
            INSERT INTO chat_features(
                analysis_id, seq, time, count, ratio, score
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    analysis_id,
                    i,
                    float(row.get("time", 0.0)),
                    int(row.get("count", 0)),
                    float(row.get("ratio", 0.0)),
                    float(row.get("score", 0.0)),
                )
                for i, row in enumerate(chat_rows)
            ],
        )
        if owns_connection:
            conn.commit()
    return analysis_id


def load_analysis(db_path: str | Path | None, analysis_id: str) -> dict:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if row is None:
            raise KeyError(f"Analysis not found: {analysis_id}")
        candidates = conn.execute(
            "SELECT * FROM candidates WHERE analysis_id = ? ORDER BY rank",
            (analysis_id,),
        ).fetchall()

    result = {
        "version": int(row["source_version"]),
        "analysis_id": row["id"],
        "source_id": row["source_id"],
        "source_fingerprint": row["source_fingerprint"],
        "run_number": int(row["run_number"] or 1),
        "algorithm_version": row["algorithm_version"],
        "feature_schema_version": int(row["feature_schema_version"] or 1),
        "cache": _unjson(row["cache_json"], {}),
        "created_at": row["created_at"],
        "video_path": row["video_path"],
        "video_name": row["video_name"],
        "content_label": row["content_label"],
        "duration": float(row["duration"]),
        "work_dir": row["work_dir"],
        "media": _unjson(row["media_json"], {}),
        "transcription": _unjson(row["transcription_json"], {}),
        "chat": _unjson(row["chat_json"], {}),
        "settings": _unjson(row["settings_json"], {}),
        "candidates": [],
    }
    for c in candidates:
        result["candidates"].append(
            {
                "id": c["candidate_id"],
                "rank": int(c["rank"]),
                "score": float(c["score"]),
                "peak_time": float(c["peak_time"]),
                "start": float(c["start"]),
                "end": float(c["end"]),
                "audio_score": float(c["audio_score"]),
                "transcript_score": float(c["transcript_score"]),
                "chat_score": float(c["chat_score"]),
                "reason": c["reason"],
                "transcript": c["transcript"],
                "content_label": c["content_label"],
                "features": _unjson(c["features_json"], {}),
            }
        )
    return result


def list_analyses(db_path: str | Path | None, limit: int = 100) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                a.id, a.created_at, a.video_path, a.video_name, a.content_label,
                a.duration, a.source_id, a.run_number, a.cache_json,
                COUNT(c.candidate_id) AS candidates,
                SUM(CASE WHEN r.status = 'keep' THEN 1 ELSE 0 END) AS kept,
                SUM(CASE WHEN r.status = 'reject' THEN 1 ELSE 0 END) AS rejected,
                SUM(CASE WHEN r.status = 'unreviewed' THEN 1 ELSE 0 END) AS unreviewed
            FROM analyses a
            LEFT JOIN candidates c ON c.analysis_id = a.id
            LEFT JOIN reviews r
                ON r.analysis_id = c.analysis_id AND r.candidate_id = c.candidate_id
            GROUP BY a.id
            ORDER BY a.created_at DESC, a.rowid DESC
            LIMIT ?
            """,
            (max(1, min(int(limit), 1000)),),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "video_path": r["video_path"],
            "video_name": r["video_name"],
            "content_label": r["content_label"],
            "duration": float(r["duration"]),
            "source_id": r["source_id"],
            "run_number": int(r["run_number"] or 1),
            "cache": _unjson(r["cache_json"], {}),
            "candidates": int(r["candidates"] or 0),
            "kept": int(r["kept"] or 0),
            "rejected": int(r["rejected"] or 0),
            "unreviewed": int(r["unreviewed"] or 0),
        }
        for r in rows
    ]


def load_review(db_path: str | Path | None, analysis_id: str, analysis: dict | None = None) -> dict:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE analysis_id = ? ORDER BY candidate_id",
            (analysis_id,),
        ).fetchall()
    items = {
        r["candidate_id"]: {
            "status": r["status"],
            "start": float(r["start"]),
            "end": float(r["end"]),
            "title": r["title"],
            "reviewed_at": r["reviewed_at"],
            "exported_at": r["exported_at"],
            "export_path": r["export_path"],
        }
        for r in rows
    }
    if analysis:
        for c in analysis.get("candidates", []):
            items.setdefault(
                c["id"],
                {
                    "status": "unreviewed",
                    "start": c["start"],
                    "end": c["end"],
                    "title": "",
                    "reviewed_at": None,
                    "exported_at": None,
                    "export_path": None,
                },
            )
    return {"version": 2, "items": items}


def save_review(db_path: str | Path | None, analysis_id: str, review: dict) -> None:
    now = utc_now()
    with connect(db_path) as conn:
        for candidate_id, item in review.get("items", {}).items():
            status = str(item.get("status", "unreviewed"))
            if status not in {"unreviewed", "keep", "reject"}:
                raise ValueError(f"Invalid review status: {status}")
            old = conn.execute(
                "SELECT status, start, end, title, reviewed_at FROM reviews WHERE analysis_id = ? AND candidate_id = ?",
                (analysis_id, candidate_id),
            ).fetchone()
            if old is None:
                continue

            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
            title = str(item.get("title", ""))[:500]
            changed = (
                status != old["status"]
                or abs(start - float(old["start"])) > 1e-6
                or abs(end - float(old["end"])) > 1e-6
                or title != old["title"]
            )
            if not changed:
                continue

            reviewed_at = old["reviewed_at"]
            if status != old["status"]:
                reviewed_at = now if status in {"keep", "reject"} else None

            conn.execute(
                """
                UPDATE reviews
                SET status = ?, start = ?, end = ?, title = ?, reviewed_at = ?, updated_at = ?
                WHERE analysis_id = ? AND candidate_id = ?
                """,
                (status, start, end, title, reviewed_at, now, analysis_id, candidate_id),
            )
            conn.execute(
                """
                INSERT INTO review_events(
                    analysis_id, candidate_id, from_status, to_status, start, end, title, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (analysis_id, candidate_id, old["status"], status, start, end, title, now),
            )
        conn.commit()


def record_export(
    db_path: str | Path | None,
    analysis_id: str,
    candidate_id: str,
    output_path: str | Path,
) -> None:
    now = utc_now()
    path = str(Path(output_path).expanduser().resolve())
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO exports(analysis_id, candidate_id, output_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (analysis_id, candidate_id, path, now),
        )
        conn.execute(
            """
            UPDATE reviews
            SET exported_at = ?, export_path = ?, updated_at = ?
            WHERE analysis_id = ? AND candidate_id = ?
            """,
            (now, path, now, analysis_id, candidate_id),
        )
        conn.commit()


def transcript_window(
    db_path: str | Path | None,
    analysis_id: str,
    start: float,
    end: float,
) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT start, end, text, score, reasons_json, words_json
            FROM transcript_segments
            WHERE analysis_id = ? AND end >= ? AND start <= ?
            ORDER BY seq
            """,
            (analysis_id, float(start), float(end)),
        ).fetchall()
    return [
        {
            "start": float(r["start"]),
            "end": float(r["end"]),
            "text": r["text"],
            "score": float(r["score"]),
            "reasons": _unjson(r["reasons_json"], []),
            "words": _unjson(r["words_json"], []),
        }
        for r in rows
    ]


def learning_examples(
    db_path: str | Path | None,
    *,
    include_unreviewed: bool = True,
) -> list[dict[str, Any]]:
    where = "" if include_unreviewed else "WHERE r.status IN ('keep', 'reject')"
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                a.id AS analysis_id, a.source_id, a.run_number, a.created_at,
                a.content_label, a.algorithm_version, a.feature_schema_version,
                c.candidate_id, c.rank, c.score, c.peak_time, c.start AS original_start,
                c.end AS original_end, c.audio_score, c.transcript_score, c.chat_score,
                c.reason, c.features_json,
                r.status, r.start AS reviewed_start, r.end AS reviewed_end, r.title,
                r.reviewed_at, r.exported_at,
                (SELECT COUNT(*) FROM exports e
                 WHERE e.analysis_id = c.analysis_id AND e.candidate_id = c.candidate_id) AS export_count,
                (SELECT COUNT(*) FROM review_events re
                 WHERE re.analysis_id = c.analysis_id AND re.candidate_id = c.candidate_id) AS review_event_count
            FROM candidates c
            JOIN analyses a ON a.id = c.analysis_id
            JOIN reviews r ON r.analysis_id = c.analysis_id AND r.candidate_id = c.candidate_id
            {where}
            ORDER BY a.created_at, c.rank
            """
        ).fetchall()

    examples: list[dict[str, Any]] = []
    for row in rows:
        status = str(row["status"])
        label = 1 if status == "keep" else 0 if status == "reject" else None
        original_start = float(row["original_start"])
        original_end = float(row["original_end"])
        reviewed_start = float(row["reviewed_start"])
        reviewed_end = float(row["reviewed_end"])
        examples.append(
            {
                "analysis_id": row["analysis_id"],
                "source_id": row["source_id"],
                "run_number": int(row["run_number"] or 1),
                "created_at": row["created_at"],
                "content_label": row["content_label"],
                "algorithm_version": row["algorithm_version"],
                "feature_schema_version": int(row["feature_schema_version"] or 1),
                "candidate_id": row["candidate_id"],
                "rank": int(row["rank"]),
                "base_score": float(row["score"]),
                "peak_time": float(row["peak_time"]),
                "audio_score": float(row["audio_score"]),
                "transcript_score": float(row["transcript_score"]),
                "chat_score": float(row["chat_score"]),
                "reason": row["reason"],
                "features": _unjson(row["features_json"], {}),
                "review_status": status,
                "label": label,
                "exported": int(row["export_count"] or 0) > 0,
                "export_count": int(row["export_count"] or 0),
                "review_event_count": int(row["review_event_count"] or 0),
                "title": row["title"],
                "reviewed_at": row["reviewed_at"],
                "exported_at": row["exported_at"],
                "original_start": original_start,
                "original_end": original_end,
                "reviewed_start": reviewed_start,
                "reviewed_end": reviewed_end,
                "start_adjustment": reviewed_start - original_start,
                "end_adjustment": reviewed_end - original_end,
                "original_duration": original_end - original_start,
                "reviewed_duration": reviewed_end - reviewed_start,
            }
        )
    return examples


def learning_summary(db_path: str | Path | None) -> dict[str, int]:
    with connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'keep' THEN 1 ELSE 0 END) AS kept,
                SUM(CASE WHEN status = 'reject' THEN 1 ELSE 0 END) AS rejected,
                SUM(CASE WHEN status = 'unreviewed' THEN 1 ELSE 0 END) AS unreviewed,
                SUM(CASE WHEN exported_at IS NOT NULL THEN 1 ELSE 0 END) AS exported
            FROM reviews
            """
        ).fetchone()
    return {
        "total": int(row["total"] or 0),
        "kept": int(row["kept"] or 0),
        "rejected": int(row["rejected"] or 0),
        "unreviewed": int(row["unreviewed"] or 0),
        "exported": int(row["exported"] or 0),
    }


def import_legacy_analysis(
    analysis_json: str | Path,
    db_path: str | Path | None = None,
) -> str:
    """Import a v0.1 analysis folder into SQLite."""
    analysis_path = validate_legacy_analysis_file(analysis_json)
    analysis = load_json(analysis_path)
    video = validate_local_video(analysis.get("video_path", ""))

    folder = analysis_path.parent
    transcript = load_json(folder / "transcript.json") if (folder / "transcript.json").is_file() else []
    audio_features = load_json(folder / "audio_features.json") if (folder / "audio_features.json").is_file() else []
    chat_features = load_json(folder / "chat_features.json") if (folder / "chat_features.json").is_file() else []

    analysis_id = save_analysis(
        db_path,
        analysis,
        transcript,
        audio_features,
        chat_features,
        work_dir=folder,
        source=describe_source(video),
        cache_info={"legacy_import": True},
    )

    review_path = folder / "review.json"
    if review_path.is_file():
        legacy_review = load_json(review_path)
        save_review(db_path, analysis_id, legacy_review)
    return analysis_id
