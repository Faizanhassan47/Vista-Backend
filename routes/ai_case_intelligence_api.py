"""HTTP API for Phase 3 Case Intelligence."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core import AI_READ_ROLES, AI_RUN_ROLES
from services.ai_case_intelligence import (
    build_case_fusion_summary,
    build_verified_report_draft,
    compare_fusion_runs,
    list_report_evidence_catalog,
)
from services.ai_fusion_service import get_fusion_run


class VerifiedReportDraftRequest(BaseModel):
    finding_ids: Optional[list[int]] = None
    fusion_group_ids: Optional[list[int]] = None


def create_ai_case_intelligence_router(
    connect_func,
    require_roles_func,
    get_current_user_func,
    get_case_func,
    require_assigned_case_func,
    get_slide_func=None,
):
    router = APIRouter(tags=["AI Case Intelligence"])

    def _authorized_case(case_pk: int, current_user: dict):
        case = get_case_func(case_pk)
        require_assigned_case_func(case, current_user)
        return case

    @router.get(
        "/api/cases/{case_pk}/ai/fusion/summary",
        dependencies=[Depends(require_roles_func(AI_READ_ROLES))],
    )
    def case_fusion_summary(case_pk: int, current_user: dict = Depends(get_current_user_func)):
        _authorized_case(case_pk, current_user)
        return build_case_fusion_summary(connect_func, case_pk)

    @router.get(
        "/api/cases/{case_pk}/ai/report-evidence",
        dependencies=[Depends(require_roles_func(AI_READ_ROLES))],
    )
    def report_evidence_catalog(case_pk: int, current_user: dict = Depends(get_current_user_func)):
        _authorized_case(case_pk, current_user)
        return list_report_evidence_catalog(connect_func, case_pk)

    @router.post(
        "/api/cases/{case_pk}/ai-report-draft",
        dependencies=[Depends(require_roles_func(AI_RUN_ROLES))],
    )
    def verified_report_draft(
        case_pk: int,
        item: Optional[VerifiedReportDraftRequest] = None,
        current_user: dict = Depends(get_current_user_func),
    ):
        _authorized_case(case_pk, current_user)
        payload = item or VerifiedReportDraftRequest()
        try:
            return build_verified_report_draft(
                connect_func,
                case_pk,
                finding_ids=payload.finding_ids,
                fusion_group_ids=payload.fusion_group_ids,
            )
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.get(
        "/api/ai/fusion/runs/{from_run_id}/compare/{to_run_id}",
        dependencies=[Depends(require_roles_func(AI_READ_ROLES))],
    )
    def compare_runs(
        from_run_id: int,
        to_run_id: int,
        current_user: dict = Depends(get_current_user_func),
    ):
        run = get_fusion_run(connect_func, from_run_id)
        if not run:
            raise HTTPException(404, f"Fusion run #{from_run_id} not found")
        if get_slide_func is not None:
            slide = get_slide_func(run["slide_id"])
            case = get_case_func(slide["case_id"])
            require_assigned_case_func(case, current_user)
        else:
            _authorized_case(run["case_id"], current_user)
        try:
            return compare_fusion_runs(connect_func, from_run_id, to_run_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return router
