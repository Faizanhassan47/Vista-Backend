"""HTTP API for the VISTA AI Evidence Map (Phase 1)."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from core import AI_READ_ROLES, AI_RUN_ROLES
from services.ai_evidence_service import (
    get_evidence_finding,
    list_evidence_findings,
    list_review_history,
    project_slide_evidence,
    submit_review_event,
)


class EvidenceReviewRequest(BaseModel):
    action: str = Field(..., description="Accept, Reject, or Correct")
    reason: Optional[str] = None
    correction: Optional[dict[str, Any]] = None


def create_ai_evidence_router(
    connect_func,
    require_roles_func,
    get_current_user_func,
    get_slide_func,
    get_case_func,
    require_assigned_case_func,
):
    router = APIRouter(tags=["AI Evidence Map"])

    def _authorized_slide(slide_id: int, current_user: dict):
        slide_row = get_slide_func(slide_id)
        case_row = get_case_func(slide_row["case_id"])
        require_assigned_case_func(case_row, current_user)
        return slide_row, case_row

    def _authorized_finding(finding_id: int, current_user: dict):
        finding = get_evidence_finding(connect_func, finding_id)
        if not finding:
            raise HTTPException(404, f"Evidence finding #{finding_id} not found")
        _authorized_slide(finding["slide_id"], current_user)
        return finding

    @router.get(
        "/api/slides/{slide_id}/ai/evidence",
        dependencies=[Depends(require_roles_func(AI_READ_ROLES))],
    )
    def list_slide_evidence(
        slide_id: int,
        feature_code: Optional[str] = None,
        finding_type: Optional[str] = None,
        review_status: Optional[str] = None,
        source_analysis_id: Optional[int] = None,
        source_kind: Optional[str] = None,
        limit: int = Query(200, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        current_user: dict = Depends(get_current_user_func),
    ):
        _authorized_slide(slide_id, current_user)
        return list_evidence_findings(
            connect_func,
            slide_id,
            feature_code=feature_code,
            finding_type=finding_type,
            review_status=review_status,
            source_analysis_id=source_analysis_id,
            source_kind=source_kind,
            limit=limit,
            offset=offset,
        )

    @router.post(
        "/api/slides/{slide_id}/ai/evidence/project",
        dependencies=[Depends(require_roles_func(AI_RUN_ROLES))],
    )
    def project_slide(
        slide_id: int,
        current_user: dict = Depends(get_current_user_func),
    ):
        """Backfill Evidence Map rows from existing completed analyses (idempotent)."""
        _authorized_slide(slide_id, current_user)
        return project_slide_evidence(connect_func, slide_id)

    @router.get(
        "/api/ai/evidence/{finding_id}",
        dependencies=[Depends(require_roles_func(AI_READ_ROLES))],
    )
    def get_finding(
        finding_id: int,
        current_user: dict = Depends(get_current_user_func),
    ):
        return _authorized_finding(finding_id, current_user)

    @router.get(
        "/api/ai/evidence/{finding_id}/history",
        dependencies=[Depends(require_roles_func(AI_READ_ROLES))],
    )
    def get_history(
        finding_id: int,
        current_user: dict = Depends(get_current_user_func),
    ):
        _authorized_finding(finding_id, current_user)
        history = list_review_history(connect_func, finding_id)
        if not history:
            raise HTTPException(404, f"Evidence finding #{finding_id} not found")
        return history

    @router.post(
        "/api/ai/evidence/{finding_id}/review",
        dependencies=[Depends(require_roles_func(AI_RUN_ROLES))],
    )
    def review_finding(
        finding_id: int,
        item: EvidenceReviewRequest,
        current_user: dict = Depends(get_current_user_func),
    ):
        _authorized_finding(finding_id, current_user)
        try:
            return submit_review_event(
                connect_func,
                finding_id,
                action=item.action,
                actor=current_user["username"],
                reason=item.reason,
                correction=item.correction,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, f"Evidence finding #{finding_id} not found") from exc

    return router
