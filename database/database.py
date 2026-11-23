"""SQLite persistence: processed videos, jobs (status/checkpoints) and generated Shorts.

The ``shorts`` table also tracks YouTube uploads (youtube_video_id, upload_status,
scheduled_at, uploaded_at, upload_error, privacy) and ``youtube_quota`` keeps a
local per-day estimate of API quota units spent.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    title        TEXT,
    duration     REAL,
    width        INTEGER,
    height       INTEGER,
    source_path  TEXT,
    info_json    TEXT,
    created_at   TEXT,
    updated_at   TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    video_id      TEXT,
    status        TEXT NOT NULL,
    stage         TEXT,
    progress      REAL DEFAULT 0,
    message       TEXT,
    error         TEXT,
    settings_json TEXT,
    output_dir    TEXT,
    created_at    TEXT,
    updated_at    TEXT,
    finished_at   TEXT
);
CREATE TABLE IF NOT EXISTS shorts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL,
    video_id         TEXT,
    rank             INTEGER,
    clip_start       REAL,
    hook_start       REAL,
    clip_end         REAL,
    score            REAL,
    category         TEXT,
    title            TEXT,
    description      TEXT,
    hashtags         TEXT,
    output_path      TEXT,
    status           TEXT,
    qc_json          TEXT,
    youtube_video_id TEXT,
    upload_status    TEXT,
    scheduled_at     TEXT,
    uploaded_at      TEXT,
    created_at       TEXT,
    UNIQUE(job_id, rank)
);
CREATE TABLE IF NOT EXISTS youtube_quota (
    day    TEXT PRIMARY KEY,
    units  INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_video ON jobs(video_id);
CREATE INDEX IF NOT EXISTS idx_shorts_job ON shorts(job_id);
"""

ACTIVE_STATUSES = ("queued", "running")
FINAL_STATUSES = ("completed", "completed_with_errors", "failed", "cancelled", "interrupted")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(shorts)").fetchall()}
        for name, typ in (("upload_error", "TEXT"), ("privacy", "TEXT"), ("tags", "TEXT")):
            if name not in cols:
                self._conn.execute(f"ALTER TABLE shorts ADD COLUMN {name} {typ}")

    # ------------------------------------------------------------- helpers
    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------- videos
    def upsert_video(self, info: Dict[str, Any]) -> None:
        now = _now()
        self._execute(
            """INSERT INTO videos (video_id, source, title, duration, width, height, source_path, info_json,
                                   created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(video_id) DO UPDATE SET
                 source=excluded.source, title=excluded.title, duration=excluded.duration,
                 width=excluded.width, height=excluded.height, source_path=excluded.source_path,
                 info_json=excluded.info_json, updated_at=excluded.updated_at""",
            (
                info["video_id"],
                info.get("source", ""),
                info.get("title", ""),
                info.get("duration", 0),
                info.get("width", 0),
                info.get("height", 0),
                info.get("source_path", ""),
                json.dumps(info, default=str),
                now,
                now,
            ),
        )

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT * FROM videos WHERE video_id=?", (video_id,))
        return rows[0] if rows else None

    # ------------------------------------------------------------- jobs
    def create_job(self, source: str, settings: Dict[str, Any]) -> str:
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        now = _now()
        self._execute(
            """INSERT INTO jobs (id, source, status, stage, progress, message, settings_json, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (job_id, source, "queued", "queued", 0.0, "Waiting to start", json.dumps(settings, default=str), now, now),
        )
        return job_id

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {"source", "video_id", "status", "stage", "progress", "message", "error", "settings_json",
                   "output_dir", "finished_at"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Unknown job fields: {bad}")
        fields["updated_at"] = _now()
        if fields.get("status") in FINAL_STATUSES and "finished_at" not in fields:
            fields["finished_at"] = fields["updated_at"]
        cols = ", ".join(f"{k}=?" for k in fields)
        self._execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT * FROM jobs WHERE id=?", (job_id,))
        return rows[0] if rows else None

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))

    def latest_completed_job(self, video_id: str) -> Optional[Dict[str, Any]]:
        rows = self._query(
            """SELECT * FROM jobs WHERE video_id=? AND status IN ('completed','completed_with_errors')
               ORDER BY finished_at DESC LIMIT 1""",
            (video_id,),
        )
        return rows[0] if rows else None

    def mark_interrupted(self, exclude_ids: Optional[List[str]] = None) -> int:
        """Jobs left 'running' by a crashed/stopped process become 'interrupted' (resumable)."""
        exclude_ids = exclude_ids or []
        rows = self._query("SELECT id FROM jobs WHERE status IN ('queued','running')")
        count = 0
        for r in rows:
            if r["id"] not in exclude_ids:
                self.update_job(r["id"], status="interrupted", message="Interrupted — can be resumed")
                count += 1
        return count

    # ------------------------------------------------------------- shorts
    def save_short(self, job_id: str, video_id: str, meta: Dict[str, Any]) -> None:
        self._execute(
            """INSERT INTO shorts (job_id, video_id, rank, clip_start, hook_start, clip_end, score, category, title,
                                   description, hashtags, output_path, status, qc_json, created_at, tags)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(job_id, rank) DO UPDATE SET
                 clip_start=excluded.clip_start, hook_start=excluded.hook_start, clip_end=excluded.clip_end,
                 score=excluded.score, category=excluded.category, title=excluded.title,
                 description=excluded.description, hashtags=excluded.hashtags, output_path=excluded.output_path,
                 status=excluded.status, qc_json=excluded.qc_json, tags=excluded.tags""",
            (
                job_id,
                video_id,
                meta["rank"],
                meta["clip_start"],
                meta["hook_start"],
                meta["clip_end"],
                meta["score"],
                meta.get("category", ""),
                meta.get("title", ""),
                meta.get("description", ""),
                " ".join(meta.get("hashtags", [])),
                meta.get("output_path", ""),
                meta.get("status", ""),
                json.dumps(meta.get("qc", {}), default=str),
                _now(),
                json.dumps(meta.get("tags", []), ensure_ascii=False),
            ),
        )

    def update_short_upload(self, job_id: str, rank: int, **fields: Any) -> None:
        allowed = {"youtube_video_id", "upload_status", "scheduled_at", "uploaded_at", "upload_error", "privacy",
                   "title", "description", "hashtags", "tags"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Unknown short fields: {bad}")
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            self._execute(f"UPDATE shorts SET {cols} WHERE job_id=? AND rank=?", (*fields.values(), job_id, rank))

    def get_short(self, job_id: str, rank: int) -> Optional[Dict[str, Any]]:
        rows = self._query("SELECT * FROM shorts WHERE job_id=? AND rank=?", (job_id, rank))
        return rows[0] if rows else None

    def last_scheduled_at(self) -> Optional[str]:
        rows = self._query("SELECT MAX(scheduled_at) AS s FROM shorts WHERE upload_status='scheduled'")
        return rows[0]["s"] if rows and rows[0]["s"] else None

    # ------------------------------------------------------------- quota
    def add_quota(self, day: str, units: int) -> int:
        self._execute("""INSERT INTO youtube_quota (day, units) VALUES (?, ?)
                         ON CONFLICT(day) DO UPDATE SET units = units + excluded.units""", (day, units))
        return self.quota_used(day)

    def quota_used(self, day: str) -> int:
        rows = self._query("SELECT units FROM youtube_quota WHERE day=?", (day,))
        return int(rows[0]["units"]) if rows else 0

    def list_shorts(self, job_id: str) -> List[Dict[str, Any]]:
        return self._query("SELECT * FROM shorts WHERE job_id=? ORDER BY rank", (job_id,))

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_instances: Dict[str, Database] = {}
_instances_lock = threading.Lock()


def get_database(path: Path) -> Database:
    key = str(Path(path).resolve())
    with _instances_lock:
        if key not in _instances:
            _instances[key] = Database(path)
        return _instances[key]
