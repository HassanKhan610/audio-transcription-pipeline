"""Job storage.

SQLite keeps the demo dependency-free. The schema maps one-to-one onto the
Postgres design in the README (jobs table + result stored per job).
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    sha256      TEXT NOT NULL,
    options_key TEXT NOT NULL,
    options_json TEXT NOT NULL,
    filename    TEXT,
    status      TEXT NOT NULL,          -- queued | processing | completed | failed
    attempts    INTEGER NOT NULL DEFAULT 0,
    progress    TEXT,                   -- e.g. "7/20"
    error       TEXT,
    result      TEXT,                   -- JSON transcript
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_dedupe ON jobs (sha256, options_key);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)

    def find_reusable(self, sha256: str, options_key: str) -> dict | None:
        """Same file + same options already queued, running, or done -> reuse it."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE sha256=? AND options_key=? AND status != 'failed' "
                "ORDER BY created_at DESC LIMIT 1",
                (sha256, options_key),
            ).fetchone()
        return self._to_dict(row)

    def create(self, sha256: str, options_key: str, options_json: str, filename: str) -> dict:
        job_id = uuid.uuid4().hex
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, sha256, options_key, options_json, filename, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
                (job_id, sha256, options_key, options_json, filename, now, now),
            )
            self._conn.commit()
        return self.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        if "result" in fields and fields["result"] is not None:
            fields["result"] = json.dumps(fields["result"])
        fields["updated_at"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))
            self._conn.commit()

    def unfinished(self) -> list[dict]:
        """Jobs left queued/processing, e.g. after the server was killed mid-job."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'processing') ORDER BY created_at"
            ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._to_dict(row)

    @staticmethod
    def _to_dict(row) -> dict | None:
        if row is None:
            return None
        job = dict(row)
        job["result"] = json.loads(job["result"]) if job["result"] else None
        return job
