"""Background job queue administration and enqueue API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from services.job_queue import enqueue, list_jobs


class EnqueueBody(BaseModel):
    job_type: str = Field(..., min_length=1, max_length=80)
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=100, ge=1, le=1000)


def create_queue_router(connect, require_roles, get_current_user):
    router = APIRouter(prefix="/api/queue", tags=["queue"])

    @router.get("/jobs")
    def get_jobs(
        status: str | None = None,
        limit: int = 50,
        _user=Depends(require_roles(["Super Administrator", "System Administrator", "Laboratory Manager"])),
    ):
        return {"jobs": list_jobs(connect, limit=limit, status=status)}

    @router.post("/jobs", status_code=202)
    def post_job(
        body: EnqueueBody,
        user=Depends(require_roles(["Super Administrator", "System Administrator", "Laboratory Manager"])),
    ):
        job_type = body.job_type.strip().lower()
        if not job_type:
            raise HTTPException(400, "job_type is required")
        job_id = enqueue(
            connect,
            job_type,
            body.payload,
            priority=body.priority,
            created_by=user.get("username"),
        )
        return {"id": job_id, "status": "queued", "job_type": job_type}

    @router.post("/ping", status_code=202)
    def ping_queue(user=Depends(get_current_user)):
        job_id = enqueue(connect, "ping", {"from": user.get("username")}, priority=10, created_by=user.get("username"))
        return {"id": job_id, "status": "queued", "job_type": "ping"}

    return router
