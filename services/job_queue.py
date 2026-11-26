"""
SQL Server–backed background job queue.

Jobs are claimed with UPDLOCK/READPAST so multiple API workers behind a load
balancer can share one queue without double-processing. Handlers register by
job_type string.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import traceback
from contextlib import closing
from typing import Any, Callable

from core.config import JOB_QUEUE_MAX_ATTEMPTS, JOB_QUEUE_WORKER_ID

logger = logging.getLogger("pathology.job_queue")

Handler = Callable[[dict[str, Any]], dict[str, Any] | None]
_HANDLERS: dict[str, Handler] = {}


def worker_id() -> str:
    if JOB_QUEUE_WORKER_ID.strip():
        return JOB_QUEUE_WORKER_ID.strip()[:120]
    return f"{socket.gethostname()}:{os.getpid()}"[:120]


def register_handler(job_type: str, handler: Handler) -> None:
    _HANDLERS[job_type] = handler


def init_job_queue_schema(connect_fn) -> None:
    with closing(connect_fn()) as db:
        db.execute(
            """
            IF OBJECT_ID('background_jobs','U') IS NULL
            CREATE TABLE background_jobs (
                id BIGINT IDENTITY(1,1) PRIMARY KEY,
                job_type VARCHAR(80) NOT NULL,
                payload_json NVARCHAR(MAX) NULL,
                status VARCHAR(30) NOT NULL CONSTRAINT DF_background_jobs_status DEFAULT 'queued',
                priority INT NOT NULL CONSTRAINT DF_background_jobs_priority DEFAULT 100,
                attempts INT NOT NULL CONSTRAINT DF_background_jobs_attempts DEFAULT 0,
                max_attempts INT NOT NULL CONSTRAINT DF_background_jobs_max_attempts DEFAULT 3,
                run_after DATETIME2 NULL,
                locked_by VARCHAR(120) NULL,
                locked_at DATETIME2 NULL,
                started_at DATETIME2 NULL,
                completed_at DATETIME2 NULL,
                result_json NVARCHAR(MAX) NULL,
                error_detail NVARCHAR(MAX) NULL,
                created_by VARCHAR(255) NULL,
                created_at DATETIME2 NOT NULL CONSTRAINT DF_background_jobs_created DEFAULT SYSUTCDATETIME(),
                updated_at DATETIME2 NOT NULL CONSTRAINT DF_background_jobs_updated DEFAULT SYSUTCDATETIME()
            )
            """
        )
        db.execute(
            """
            IF NOT EXISTS (
                SELECT 1 FROM sys.indexes WHERE name = 'IX_background_jobs_claim' AND object_id = OBJECT_ID('background_jobs')
            )
            CREATE INDEX IX_background_jobs_claim
            ON background_jobs (status, priority, id)
            INCLUDE (run_after, attempts, max_attempts)
            """
        )
        db.commit()


def enqueue(
    connect_fn,
    job_type: str,
    payload: dict[str, Any] | None = None,
    *,
    priority: int = 100,
    run_after=None,
    created_by: str | None = None,
    max_attempts: int | None = None,
) -> int:
    if job_type not in _HANDLERS and job_type not in {"ping", "ingestion.process"}:
        # Allow enqueue before handler registration for known built-ins; unknown types still store.
        logger.debug("enqueue job_type=%s with no in-process handler yet", job_type)
    body = json.dumps(payload or {}, default=str)
    attempts_cap = max_attempts if max_attempts is not None else JOB_QUEUE_MAX_ATTEMPTS
    with closing(connect_fn()) as db:
        cur = db.execute(
            """
            INSERT INTO background_jobs (job_type, payload_json, status, priority, run_after, created_by, max_attempts)
            VALUES (?, ?, 'queued', ?, ?, ?, ?)
            """,
            (job_type[:80], body, int(priority), run_after, (created_by or "")[:255] or None, int(attempts_cap)),
        )
        job_id = int(cur.lastrowid)
        db.commit()
    logger.info("job.enqueued id=%s type=%s", job_id, job_type)
    return job_id


def _row_to_job(row) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        data = dict(row)
    else:
        keys = [
            "id", "job_type", "payload_json", "status", "priority", "attempts",
            "max_attempts", "run_after", "locked_by", "locked_at", "started_at",
            "completed_at", "result_json", "error_detail", "created_by", "created_at", "updated_at",
        ]
        data = {keys[i]: row[i] for i in range(min(len(keys), len(row)))}
    payload = data.get("payload_json")
    if isinstance(payload, str) and payload:
        try:
            data["payload"] = json.loads(payload)
        except json.JSONDecodeError:
            data["payload"] = {}
    else:
        data["payload"] = payload if isinstance(payload, dict) else {}
    return data


def claim_next(connect_fn, *, owner: str | None = None) -> dict[str, Any] | None:
    owner = (owner or worker_id())[:120]
    with closing(connect_fn()) as db:
        # Claim one queued row that is due; READPAST skips rows locked by other workers.
        cur = db.execute(
            """
            ;WITH cte AS (
                SELECT TOP (1) id
                FROM background_jobs WITH (UPDLOCK, READPAST, ROWLOCK)
                WHERE status = 'queued'
                  AND attempts < max_attempts
                  AND (run_after IS NULL OR run_after <= SYSUTCDATETIME())
                ORDER BY priority ASC, id ASC
            )
            UPDATE j
            SET status = 'running',
                locked_by = ?,
                locked_at = SYSUTCDATETIME(),
                started_at = COALESCE(started_at, SYSUTCDATETIME()),
                attempts = attempts + 1,
                updated_at = SYSUTCDATETIME(),
                error_detail = NULL
            OUTPUT INSERTED.id, INSERTED.job_type, INSERTED.payload_json, INSERTED.status,
                   INSERTED.priority, INSERTED.attempts, INSERTED.max_attempts, INSERTED.run_after,
                   INSERTED.locked_by, INSERTED.locked_at, INSERTED.started_at, INSERTED.completed_at,
                   INSERTED.result_json, INSERTED.error_detail, INSERTED.created_by,
                   INSERTED.created_at, INSERTED.updated_at
            FROM background_jobs j
            INNER JOIN cte ON cte.id = j.id
            """,
            (owner,),
        )
        row = cur.fetchone()
        db.commit()
    return _row_to_job(row) if row else None


def complete(connect_fn, job_id: int, result: dict[str, Any] | None = None) -> None:
    with closing(connect_fn()) as db:
        db.execute(
            """
            UPDATE background_jobs
            SET status = 'completed',
                result_json = ?,
                completed_at = SYSUTCDATETIME(),
                updated_at = SYSUTCDATETIME(),
                locked_by = NULL,
                locked_at = NULL
            WHERE id = ?
            """,
            (json.dumps(result or {}, default=str), int(job_id)),
        )
        db.commit()


def fail(connect_fn, job_id: int, error: str, *, requeue: bool = True) -> None:
    with closing(connect_fn()) as db:
        row = db.execute(
            "SELECT attempts, max_attempts FROM background_jobs WHERE id=?",
            (int(job_id),),
        ).fetchone()
        attempts = int(row["attempts"] if isinstance(row, dict) else row[0]) if row else 1
        max_attempts = int(row["max_attempts"] if isinstance(row, dict) else row[1]) if row else JOB_QUEUE_MAX_ATTEMPTS
        status = "queued" if requeue and attempts < max_attempts else "failed"
        db.execute(
            """
            UPDATE background_jobs
            SET status = ?,
                error_detail = ?,
                completed_at = CASE WHEN ? = 'failed' THEN SYSUTCDATETIME() ELSE completed_at END,
                updated_at = SYSUTCDATETIME(),
                locked_by = NULL,
                locked_at = NULL,
                run_after = CASE WHEN ? = 'queued' THEN DATEADD(second, 15, SYSUTCDATETIME()) ELSE run_after END
            WHERE id = ?
            """,
            (status, (error or "")[:4000], status, status, int(job_id)),
        )
        db.commit()


def process_one(connect_fn, job: dict[str, Any]) -> None:
    job_id = int(job["id"])
    job_type = str(job.get("job_type") or "")
    handler = _HANDLERS.get(job_type)
    if handler is None and job_type == "ping":
        complete(connect_fn, job_id, {"pong": True, "worker": worker_id()})
        return
    if handler is None:
        fail(connect_fn, job_id, f"No handler registered for job_type={job_type}", requeue=False)
        return
    try:
        result = handler(job.get("payload") or {}) or {}
        complete(connect_fn, job_id, result if isinstance(result, dict) else {"ok": True})
    except Exception as exc:
        logger.exception("job.failed id=%s type=%s", job_id, job_type)
        fail(connect_fn, job_id, f"{exc}\n{traceback.format_exc()[-1500:]}")


def list_jobs(connect_fn, *, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    with closing(connect_fn()) as db:
        if status:
            rows = db.execute(
                """
                SELECT TOP (?) id, job_type, status, priority, attempts, max_attempts, locked_by,
                       created_by, created_at, started_at, completed_at, error_detail, updated_at
                FROM background_jobs
                WHERE status = ?
                ORDER BY id DESC
                """,
                (limit, status),
            ).fetchall()
        else:
            rows = db.execute(
                """
                SELECT TOP (?) id, job_type, status, priority, attempts, max_attempts, locked_by,
                       created_by, created_at, started_at, completed_at, error_detail, updated_at
                FROM background_jobs
                ORDER BY id DESC
                """,
                (limit,),
            ).fetchall()
    return [_row_to_job(row) for row in rows]


def register_builtin_handlers(_connect_fn=None) -> None:
    def ping_handler(_payload: dict[str, Any]) -> dict[str, Any]:
        return {"pong": True, "worker": worker_id()}

    def echo_handler(payload: dict[str, Any]) -> dict[str, Any]:
        return {"echo": payload, "worker": worker_id()}

    register_handler("ping", ping_handler)
    register_handler("echo", echo_handler)
