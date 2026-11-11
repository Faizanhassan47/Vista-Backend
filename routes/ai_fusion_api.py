"""HTTP API for VISTA AI Fusion v1."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core import AI_READ_ROLES, AI_RUN_ROLES
from services.ai_fusion_rules import (
    DEFAULT_MITOTIC_SUPPORT_THRESHOLD,
    DEFAULT_TUMOR_CONFLICT_THRESHOLD,
    DEFAULT_TUMOR_SUPPORT_THRESHOLD,
)
from services.ai_fusion_service import (
    DEFAULT_CELL_SIZE,
    DEFAULT_PROXIMITY_PX,
    get_fusion_group,
    get_fusion_run,
    list_fusion_runs,
    run_fusion_for_slide,
)


class FusionRunRequest(BaseModel):
    cell_size_px: float = Field(DEFAULT_CELL_SIZE, ge=32, le=4096)
    proximity_px: float = Field(DEFAULT_PROXIMITY_PX, ge=0, le=256)
    tumor_support_threshold: float = Field(DEFAULT_TUMOR_SUPPORT_THRESHOLD, ge=0, le=1)
    tumor_conflict_threshold: float = Field(DEFAULT_TUMOR_CONFLICT_THRESHOLD, ge=0, le=1)
    mitotic_support_threshold: float = Field(DEFAULT_MITOTIC_SUPPORT_THRESHOLD, ge=0, le=1)


def create_ai_fusion_router(
    connect_func,
    require_roles_func,
    get_current_user_func,
    get_slide_func,
    get_case_func,
    require_assigned_case_func,
):
    router = APIRouter(tags=["AI Fusion"])

    def _authorized_slide(slide_id: int, current_user: dict):
        slide_row = get_slide_func(slide_id)
        case_row = get_case_func(slide_row["case_id"])
        require_assigned_case_func(case_row, current_user)
        return slide_row, case_row

    def _authorized_run(run_id: int, current_user: dict):
        run = get_fusion_run(connect_func, run_id)
        if not run:
            raise HTTPException(404, f"Fusion run #{run_id} not found")
        _authorized_slide(run["slide_id"], current_user)
        return run

    @router.get(
        "/api/slides/{slide_id}/ai/fusion/runs",
        dependencies=[Depends(require_roles_func(AI_READ_ROLES))],
    )
    def list_runs(slide_id: int, current_user: dict = Depends(get_current_user_func)):
        _authorized_slide(slide_id, current_user)
        return {"slide_id": slide_id, "runs": list_fusion_runs(connect_func, slide_id)}

    @router.post(
        "/api/slides/{slide_id}/ai/fusion/runs",
        dependencies=[Depends(require_roles_func(AI_RUN_ROLES))],
    )
    def create_run(
        slide_id: int,
        item: Optional[FusionRunRequest] = None,
        current_user: dict = Depends(get_current_user_func),
    ):
        slide_row, case_row = _authorized_slide(slide_id, current_user)
        params = item or FusionRunRequest()
        return run_fusion_for_slide(
            connect_func,
            slide_id=int(slide_row["id"]),
            case_id=int(case_row["id"]),
            requested_by=current_user["username"],
            cell_size=params.cell_size_px,
            proximity_px=params.proximity_px,
            tumor_support=params.tumor_support_threshold,
            tumor_conflict=params.tumor_conflict_threshold,
            mitotic_support=params.mitotic_support_threshold,
        )

    @router.get(
        "/api/ai/fusion/runs/{run_id}",
        dependencies=[Depends(require_roles_func(AI_READ_ROLES))],
    )
    def get_run(run_id: int, current_user: dict = Depends(get_current_user_func)):
        return _authorized_run(run_id, current_user)

    @router.get(
        "/api/ai/fusion/groups/{group_id}",
        dependencies=[Depends(require_roles_func(AI_READ_ROLES))],
    )
    def get_group(group_id: int, current_user: dict = Depends(get_current_user_func)):
        group = get_fusion_group(connect_func, group_id)
        if not group:
            raise HTTPException(404, f"Fusion group #{group_id} not found")
        if group.get("slide_id") is not None:
            _authorized_slide(group["slide_id"], current_user)
        return group

    return router
