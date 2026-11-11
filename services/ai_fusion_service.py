"""VISTA AI Fusion v1 — evidence-driven spatial correlation.

Creates append-only fusion runs. Never fabricates combined diagnostic confidence.
"""

from __future__ import annotations

from contextlib import closing
from collections import defaultdict
import json
import logging
from typing import Any, Optional

from services.ai_fusion_geometry import (
    UnionFind,
    bbox_as_polygon,
    bbox_union,
    geometries_overlap,
    geometry_bbox,
    grid_bins,
)
from services.ai_fusion_rules import (
    DEFAULT_MITOTIC_SUPPORT_THRESHOLD,
    DEFAULT_TUMOR_CONFLICT_THRESHOLD,
    DEFAULT_TUMOR_SUPPORT_THRESHOLD,
    classify_group,
    classify_pair,
)
from services.ai_evidence_service import _json_loads, _serialize_finding

logger = logging.getLogger("pathology.ai_fusion")

DEFAULT_CELL_SIZE = 256.0
DEFAULT_PROXIMITY_PX = 8.0


def init_ai_fusion_schema(connect_func) -> None:
    with closing(connect_func()) as db:
        db.execute(
            """
            IF OBJECT_ID(N'dbo.ai_fusion_runs', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.ai_fusion_runs (
                    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_ai_fusion_runs PRIMARY KEY,
                    slide_id INT NOT NULL CONSTRAINT FK_ai_fusion_runs_slide
                        REFERENCES dbo.slides(id) ON DELETE CASCADE,
                    case_id INT NOT NULL CONSTRAINT FK_ai_fusion_runs_case
                        REFERENCES dbo.cases(id),
                    requested_by VARCHAR(255) NOT NULL,
                    status VARCHAR(30) NOT NULL CONSTRAINT DF_ai_fusion_runs_status DEFAULT ('Completed'),
                    parameters_json NVARCHAR(MAX) NOT NULL,
                    summary_json NVARCHAR(MAX) NULL,
                    evidence_count INT NOT NULL CONSTRAINT DF_ai_fusion_runs_ev DEFAULT (0),
                    group_count INT NOT NULL CONSTRAINT DF_ai_fusion_runs_grp DEFAULT (0),
                    created_at DATETIME2 NOT NULL CONSTRAINT DF_ai_fusion_runs_created DEFAULT (SYSUTCDATETIME()),
                    completed_at DATETIME2 NULL,
                    CONSTRAINT CK_ai_fusion_runs_parameters CHECK (ISJSON(parameters_json) = 1),
                    CONSTRAINT CK_ai_fusion_runs_summary CHECK (summary_json IS NULL OR ISJSON(summary_json) = 1)
                );
                CREATE INDEX IX_ai_fusion_runs_slide ON dbo.ai_fusion_runs(slide_id, id DESC);
            END
            """
        )
        db.execute(
            """
            IF OBJECT_ID(N'dbo.ai_fusion_groups', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.ai_fusion_groups (
                    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_ai_fusion_groups PRIMARY KEY,
                    run_id INT NOT NULL CONSTRAINT FK_ai_fusion_groups_run
                        REFERENCES dbo.ai_fusion_runs(id) ON DELETE CASCADE,
                    group_key VARCHAR(80) NOT NULL,
                    classification VARCHAR(40) NOT NULL,
                    geometry_json NVARCHAR(MAX) NOT NULL,
                    rationale_json NVARCHAR(MAX) NOT NULL,
                    uncertainty_json NVARCHAR(MAX) NOT NULL,
                    member_count INT NOT NULL CONSTRAINT DF_ai_fusion_groups_members DEFAULT (0),
                    created_at DATETIME2 NOT NULL CONSTRAINT DF_ai_fusion_groups_created DEFAULT (SYSUTCDATETIME()),
                    CONSTRAINT CK_ai_fusion_groups_geometry CHECK (ISJSON(geometry_json) = 1),
                    CONSTRAINT CK_ai_fusion_groups_rationale CHECK (ISJSON(rationale_json) = 1),
                    CONSTRAINT CK_ai_fusion_groups_uncertainty CHECK (ISJSON(uncertainty_json) = 1)
                );
                CREATE INDEX IX_ai_fusion_groups_run ON dbo.ai_fusion_groups(run_id, id);
            END
            """
        )
        db.execute(
            """
            IF OBJECT_ID(N'dbo.ai_fusion_group_members', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.ai_fusion_group_members (
                    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_ai_fusion_group_members PRIMARY KEY,
                    group_id INT NOT NULL CONSTRAINT FK_ai_fusion_group_members_group
                        REFERENCES dbo.ai_fusion_groups(id) ON DELETE CASCADE,
                    evidence_finding_id INT NOT NULL,
                    role VARCHAR(40) NOT NULL CONSTRAINT DF_ai_fusion_member_role DEFAULT ('participant'),
                    relationship_tags_json NVARCHAR(MAX) NULL,
                    CONSTRAINT UQ_ai_fusion_group_member UNIQUE (group_id, evidence_finding_id),
                    CONSTRAINT CK_ai_fusion_member_tags
                        CHECK (relationship_tags_json IS NULL OR ISJSON(relationship_tags_json) = 1)
                );
                CREATE INDEX IX_ai_fusion_members_finding
                    ON dbo.ai_fusion_group_members(evidence_finding_id);
            END
            """
        )
        db.commit()


def correlate_findings(
    findings: list[dict],
    *,
    cell_size: float = DEFAULT_CELL_SIZE,
    proximity_px: float = DEFAULT_PROXIMITY_PX,
    tumor_support: float = DEFAULT_TUMOR_SUPPORT_THRESHOLD,
    tumor_conflict: float = DEFAULT_TUMOR_CONFLICT_THRESHOLD,
    mitotic_support: float = DEFAULT_MITOTIC_SUPPORT_THRESHOLD,
) -> dict:
    """Pure spatial fusion over in-memory evidence findings (no DB writes)."""
    prepared = []
    for item in findings:
        geometry = item.get("geometry")
        if isinstance(geometry, str):
            geometry = _json_loads(geometry, None)
        bbox = geometry_bbox(geometry) if geometry else None
        if bbox is None:
            continue
        prepared.append({**item, "geometry": geometry, "_bbox": bbox})

    if not prepared:
        return {
            "groups": [],
            "parameters": _parameters_dict(cell_size, proximity_px, tumor_support, tumor_conflict, mitotic_support),
            "summary": {"evidence_count": 0, "group_count": 0, "by_classification": {}},
        }

    ids = [f["id"] for f in prepared]
    by_id = {f["id"]: f for f in prepared}
    uf = UnionFind(ids)
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    for finding in prepared:
        for cell in grid_bins(finding["_bbox"], cell_size):
            bins[cell].append(finding["id"])

    candidate_pairs: set[tuple[int, int]] = set()
    for cell, members in bins.items():
        # include neighboring cells to catch boundary overlaps
        cx, cy = cell
        neighborhood_ids = set(members)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighborhood_ids.update(bins.get((cx + dx, cy + dy), []))
        local = sorted(neighborhood_ids)
        for i, left in enumerate(local):
            for right in local[i + 1 :]:
                a, b = (left, right) if left < right else (right, left)
                candidate_pairs.add((a, b))

    pair_relations: dict[tuple[int, int], list[dict]] = {}
    for left, right in candidate_pairs:
        fa, fb = by_id[left], by_id[right]
        if not geometries_overlap(fa["geometry"], fb["geometry"], proximity_px=proximity_px):
            continue
        uf.union(left, right)
        pair_relations[(left, right)] = classify_pair(
            fa,
            fb,
            tumor_support=tumor_support,
            tumor_conflict=tumor_conflict,
            mitotic_support=mitotic_support,
        )

    groups_out = []
    for index, (_root, member_ids) in enumerate(sorted(uf.groups().items(), key=lambda item: min(item[1])), start=1):
        members = [by_id[mid] for mid in sorted(member_ids)]
        relations = []
        for i, left in enumerate(member_ids):
            for right in member_ids[i + 1 :]:
                key = (left, right) if left < right else (right, left)
                relations.extend(pair_relations.get(key, []))
        rollup = classify_group(relations, members)
        union = bbox_union(m["_bbox"] for m in members)
        geometry = bbox_as_polygon(union) if union else members[0]["geometry"]
        groups_out.append({
            "group_key": f"G{index}",
            "classification": rollup["classification"],
            "geometry": geometry,
            "bbox": union,
            "member_ids": [m["id"] for m in members],
            "members": [
                {
                    "id": m["id"],
                    "finding_type": m.get("finding_type"),
                    "label": m.get("label"),
                    "feature_code": m.get("feature_code"),
                    "model_name": m.get("model_name"),
                    "model_version": m.get("model_version"),
                    "score": m.get("score"),
                    "score_calibration_status": m.get("score_calibration_status") or "uncalibrated",
                    "review_status": m.get("review_status") or "Unreviewed",
                    "geometry": m.get("geometry"),
                    "role": "qc_overlay" if m.get("finding_type") == "qc_polygon" else "participant",
                }
                for m in members
            ],
            "rationale": rollup,
            "uncertainty": rollup["uncertainty"],
            "combined_diagnostic_confidence": None,
        })

    by_class: dict[str, int] = defaultdict(int)
    for group in groups_out:
        by_class[group["classification"]] += 1

    return {
        "groups": groups_out,
        "parameters": _parameters_dict(cell_size, proximity_px, tumor_support, tumor_conflict, mitotic_support),
        "summary": {
            "evidence_count": len(prepared),
            "group_count": len(groups_out),
            "by_classification": dict(by_class),
            "disclaimer": (
                "Research use only. Fusion correlates evidence spatially and does not produce "
                "a combined diagnostic confidence or diagnosis."
            ),
        },
    }


def _parameters_dict(cell_size, proximity_px, tumor_support, tumor_conflict, mitotic_support) -> dict:
    return {
        "cell_size_px": float(cell_size),
        "proximity_px": float(proximity_px),
        "tumor_support_threshold": float(tumor_support),
        "tumor_conflict_threshold": float(tumor_conflict),
        "mitotic_support_threshold": float(mitotic_support),
        "coordinate_system": "level_0_pixels",
        "engine": "ai_fusion_v1",
        "produces_combined_diagnostic_confidence": False,
    }


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _load_slide_findings(db, slide_id: int) -> list[dict]:
    rows = db.execute(
        """
        SELECT * FROM dbo.ai_evidence_findings
        WHERE slide_id = ?
        ORDER BY id
        """,
        (slide_id,),
    ).fetchall()
    findings = []
    for row in rows:
        item = _serialize_finding(dict(row))
        findings.append(item)
    return findings


def run_fusion_for_slide(
    connect_func,
    *,
    slide_id: int,
    case_id: int,
    requested_by: str,
    cell_size: float = DEFAULT_CELL_SIZE,
    proximity_px: float = DEFAULT_PROXIMITY_PX,
    tumor_support: float = DEFAULT_TUMOR_SUPPORT_THRESHOLD,
    tumor_conflict: float = DEFAULT_TUMOR_CONFLICT_THRESHOLD,
    mitotic_support: float = DEFAULT_MITOTIC_SUPPORT_THRESHOLD,
) -> dict:
    """Create a new immutable fusion run for the slide (never overwrites prior runs)."""
    with closing(connect_func()) as db:
        findings = _load_slide_findings(db, slide_id)
        result = correlate_findings(
            findings,
            cell_size=cell_size,
            proximity_px=proximity_px,
            tumor_support=tumor_support,
            tumor_conflict=tumor_conflict,
            mitotic_support=mitotic_support,
        )
        params = result["parameters"]
        summary = result["summary"]
        cursor = db.execute(
            """
            INSERT INTO dbo.ai_fusion_runs (
                slide_id, case_id, requested_by, status,
                parameters_json, summary_json, evidence_count, group_count, completed_at
            ) VALUES (?, ?, ?, 'Completed', ?, ?, ?, ?, SYSUTCDATETIME())
            """,
            (
                slide_id,
                case_id,
                requested_by,
                _dumps(params),
                _dumps(summary),
                summary.get("evidence_count", 0),
                summary.get("group_count", 0),
            ),
        )
        run_id = cursor.lastrowid
        if run_id is None:
            saved = db.execute(
                """
                SELECT TOP 1 id FROM dbo.ai_fusion_runs
                WHERE slide_id = ? AND requested_by = ?
                ORDER BY id DESC
                """,
                (slide_id, requested_by),
            ).fetchone()
            run_id = int(saved["id"])

        for group in result["groups"]:
            gcur = db.execute(
                """
                INSERT INTO dbo.ai_fusion_groups (
                    run_id, group_key, classification, geometry_json,
                    rationale_json, uncertainty_json, member_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    group["group_key"],
                    group["classification"],
                    _dumps(group["geometry"]),
                    _dumps(group["rationale"]),
                    _dumps(group["uncertainty"]),
                    len(group["member_ids"]),
                ),
            )
            group_id = gcur.lastrowid
            if group_id is None:
                grow = db.execute(
                    """
                    SELECT TOP 1 id FROM dbo.ai_fusion_groups
                    WHERE run_id = ? AND group_key = ?
                    ORDER BY id DESC
                    """,
                    (run_id, group["group_key"]),
                ).fetchone()
                group_id = int(grow["id"])
            for member in group["members"]:
                tags = [
                    rel.get("code")
                    for rel in group["rationale"].get("relations", [])
                    if member["id"] in (rel.get("evidence_ids") or [])
                ]
                db.execute(
                    """
                    INSERT INTO dbo.ai_fusion_group_members (
                        group_id, evidence_finding_id, role, relationship_tags_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        group_id,
                        member["id"],
                        member.get("role") or "participant",
                        _dumps(tags),
                    ),
                )
        db.commit()
    return get_fusion_run(connect_func, run_id)


def list_fusion_runs(connect_func, slide_id: int) -> list[dict]:
    with closing(connect_func()) as db:
        rows = db.execute(
            """
            SELECT id, slide_id, case_id, requested_by, status, parameters_json, summary_json,
                   evidence_count, group_count, created_at, completed_at
            FROM dbo.ai_fusion_runs
            WHERE slide_id = ?
            ORDER BY id DESC
            """,
            (slide_id,),
        ).fetchall()
    return [_serialize_run(dict(row), include_groups=False) for row in rows]


def get_fusion_run(connect_func, run_id: int) -> Optional[dict]:
    with closing(connect_func()) as db:
        row = db.execute(
            "SELECT * FROM dbo.ai_fusion_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        groups = db.execute(
            """
            SELECT * FROM dbo.ai_fusion_groups
            WHERE run_id = ?
            ORDER BY id
            """,
            (run_id,),
        ).fetchall()
        serialized_groups = []
        for grow in groups:
            members = db.execute(
                """
                SELECT m.*, f.finding_type, f.label, f.feature_code, f.model_name, f.model_version,
                       f.score, f.score_calibration_status, f.review_status, f.geometry_json,
                       f.source_kind, f.source_analysis_id, f.source_finding_key
                FROM dbo.ai_fusion_group_members m
                JOIN dbo.ai_evidence_findings f ON f.id = m.evidence_finding_id
                WHERE m.group_id = ?
                ORDER BY m.id
                """,
                (grow["id"],),
            ).fetchall()
            serialized_groups.append(_serialize_group(dict(grow), [dict(m) for m in members]))
    return _serialize_run(dict(row), include_groups=True, groups=serialized_groups)


def get_fusion_group(connect_func, group_id: int) -> Optional[dict]:
    with closing(connect_func()) as db:
        grow = db.execute(
            "SELECT * FROM dbo.ai_fusion_groups WHERE id = ?",
            (group_id,),
        ).fetchone()
        if not grow:
            return None
        run = db.execute(
            "SELECT id, slide_id, case_id FROM dbo.ai_fusion_runs WHERE id = ?",
            (grow["run_id"],),
        ).fetchone()
        members = db.execute(
            """
            SELECT m.*, f.finding_type, f.label, f.feature_code, f.model_name, f.model_version,
                   f.score, f.score_calibration_status, f.review_status, f.geometry_json,
                   f.source_kind, f.source_analysis_id, f.source_finding_key, f.provenance_json
            FROM dbo.ai_fusion_group_members m
            JOIN dbo.ai_evidence_findings f ON f.id = m.evidence_finding_id
            WHERE m.group_id = ?
            ORDER BY m.id
            """,
            (group_id,),
        ).fetchall()
    payload = _serialize_group(dict(grow), [dict(m) for m in members])
    payload["run_id"] = int(grow["run_id"])
    payload["slide_id"] = int(run["slide_id"]) if run else None
    payload["case_id"] = int(run["case_id"]) if run else None
    return payload


def _serialize_run(row: dict, *, include_groups: bool, groups: Optional[list] = None) -> dict:
    return {
        "id": int(row["id"]),
        "slide_id": int(row["slide_id"]),
        "case_id": int(row["case_id"]),
        "requested_by": row["requested_by"],
        "status": row["status"],
        "parameters": _json_loads(row.get("parameters_json"), {}),
        "summary": _json_loads(row.get("summary_json"), {}),
        "evidence_count": int(row.get("evidence_count") or 0),
        "group_count": int(row.get("group_count") or 0),
        "created_at": str(row["created_at"]) if row.get("created_at") else None,
        "completed_at": str(row["completed_at"]) if row.get("completed_at") else None,
        "groups": groups if include_groups else None,
        "combined_diagnostic_confidence": None,
        "disclaimer": (
            "Research use only. Fusion correlates evidence spatially and does not produce "
            "a combined diagnostic confidence or diagnosis."
        ),
    }


def _serialize_group(row: dict, members: list[dict]) -> dict:
    rationale = _json_loads(row.get("rationale_json"), {}) or {}
    return {
        "id": int(row["id"]),
        "run_id": int(row["run_id"]),
        "group_key": row["group_key"],
        "classification": row["classification"],
        "geometry": _json_loads(row.get("geometry_json"), {}),
        "rationale": rationale,
        "uncertainty": _json_loads(row.get("uncertainty_json"), {}),
        "member_count": int(row.get("member_count") or len(members)),
        "combined_diagnostic_confidence": None,
        "members": [
            {
                "evidence_finding_id": int(m["evidence_finding_id"]),
                "role": m.get("role") or "participant",
                "relationship_tags": _json_loads(m.get("relationship_tags_json"), []),
                "finding_type": m.get("finding_type"),
                "label": m.get("label"),
                "feature_code": m.get("feature_code"),
                "model_name": m.get("model_name"),
                "model_version": m.get("model_version"),
                "score": m.get("score"),
                "score_calibration_status": m.get("score_calibration_status") or "uncalibrated",
                "review_status": m.get("review_status") or "Unreviewed",
                "geometry": _json_loads(m.get("geometry_json"), {}),
                "source_kind": m.get("source_kind"),
                "source_analysis_id": m.get("source_analysis_id"),
                "source_finding_key": m.get("source_finding_key"),
                "provenance": _json_loads(m.get("provenance_json"), None),
            }
            for m in members
        ],
        "why": {
            "classification": row["classification"],
            "rationale": rationale.get("rationale") or [],
            "relation_codes": rationale.get("relation_codes") or [],
            "qc_affected": rationale.get("qc_affected"),
            "has_agreement": rationale.get("has_agreement"),
            "has_conflict": rationale.get("has_conflict"),
        },
    }
