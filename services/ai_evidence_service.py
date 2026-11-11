"""VISTA AI Evidence Map — Phase 1.

Normalized, reviewable findings projected from immutable source analyses.
Original model outputs are never overwritten. Scores are labelled uncalibrated.
"""

from __future__ import annotations

from contextlib import closing
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("pathology.ai_evidence")

SOURCE_AI = "ai_analyses"
SOURCE_QC = "qc_analyses"
CALIBRATION_UNCALIBRATED = "uncalibrated"
REVIEW_UNREVIEWED = "Unreviewed"
REVIEW_ACTIONS = {"Accept": "Accepted", "Reject": "Rejected", "Correct": "Corrected"}
FINDING_TYPES = {
    "nucleus",
    "mitotic_candidate",
    "tumor_patch",
    "qc_polygon",
}

# Cap extreme QC feature dumps in a single projection pass to keep inserts bounded.
MAX_QC_FEATURES_PER_ANALYSIS = 5000
MAX_NUCLEI_PER_ANALYSIS = 20000
MAX_PATCHES_PER_ANALYSIS = 20000


def init_ai_evidence_schema(connect_func) -> None:
    """Create Evidence Map tables idempotently."""
    with closing(connect_func()) as db:
        db.execute(
            """
            IF OBJECT_ID(N'dbo.ai_evidence_findings', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.ai_evidence_findings (
                    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_ai_evidence_findings PRIMARY KEY,
                    slide_id INT NOT NULL CONSTRAINT FK_ai_evidence_findings_slide
                        REFERENCES dbo.slides(id) ON DELETE CASCADE,
                    case_id INT NOT NULL CONSTRAINT FK_ai_evidence_findings_case
                        REFERENCES dbo.cases(id),
                    source_kind VARCHAR(30) NOT NULL,
                    source_analysis_id INT NOT NULL,
                    source_finding_key VARCHAR(160) NOT NULL,
                    feature_code VARCHAR(80) NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    model_version VARCHAR(120) NOT NULL,
                    finding_type VARCHAR(40) NOT NULL,
                    label VARCHAR(160) NULL,
                    geometry_json NVARCHAR(MAX) NOT NULL,
                    coordinate_system VARCHAR(40) NOT NULL
                        CONSTRAINT DF_ai_evidence_coord DEFAULT ('level_0_pixels'),
                    score FLOAT NULL,
                    score_calibration_status VARCHAR(40) NOT NULL
                        CONSTRAINT DF_ai_evidence_calib DEFAULT ('uncalibrated'),
                    qc_context_json NVARCHAR(MAX) NULL,
                    provenance_json NVARCHAR(MAX) NULL,
                    review_status VARCHAR(30) NOT NULL
                        CONSTRAINT DF_ai_evidence_review DEFAULT ('Unreviewed'),
                    latest_review_event_id INT NULL,
                    created_at DATETIME2 NOT NULL
                        CONSTRAINT DF_ai_evidence_created DEFAULT (SYSUTCDATETIME()),
                    CONSTRAINT UQ_ai_evidence_source
                        UNIQUE (source_kind, source_analysis_id, source_finding_key),
                    CONSTRAINT CK_ai_evidence_source_kind
                        CHECK (source_kind IN ('ai_analyses', 'qc_analyses')),
                    CONSTRAINT CK_ai_evidence_review_status
                        CHECK (review_status IN ('Unreviewed', 'Accepted', 'Rejected', 'Corrected')),
                    CONSTRAINT CK_ai_evidence_geometry_json
                        CHECK (ISJSON(geometry_json) = 1),
                    CONSTRAINT CK_ai_evidence_qc_context_json
                        CHECK (qc_context_json IS NULL OR ISJSON(qc_context_json) = 1),
                    CONSTRAINT CK_ai_evidence_provenance_json
                        CHECK (provenance_json IS NULL OR ISJSON(provenance_json) = 1)
                );
                CREATE INDEX IX_ai_evidence_findings_slide
                    ON dbo.ai_evidence_findings(slide_id, id DESC);
                CREATE INDEX IX_ai_evidence_findings_case
                    ON dbo.ai_evidence_findings(case_id, id DESC);
                CREATE INDEX IX_ai_evidence_findings_review
                    ON dbo.ai_evidence_findings(slide_id, review_status, finding_type);
            END
            """
        )
        db.execute(
            """
            IF OBJECT_ID(N'dbo.ai_evidence_review_events', N'U') IS NULL
            BEGIN
                CREATE TABLE dbo.ai_evidence_review_events (
                    id INT IDENTITY(1,1) NOT NULL CONSTRAINT PK_ai_evidence_review_events PRIMARY KEY,
                    finding_id INT NOT NULL CONSTRAINT FK_ai_evidence_review_finding
                        REFERENCES dbo.ai_evidence_findings(id) ON DELETE CASCADE,
                    action VARCHAR(30) NOT NULL,
                    actor VARCHAR(255) NOT NULL,
                    reason NVARCHAR(MAX) NULL,
                    correction_json NVARCHAR(MAX) NULL,
                    created_at DATETIME2 NOT NULL
                        CONSTRAINT DF_ai_evidence_review_created DEFAULT (SYSUTCDATETIME()),
                    CONSTRAINT CK_ai_evidence_review_action
                        CHECK (action IN ('Accept', 'Reject', 'Correct')),
                    CONSTRAINT CK_ai_evidence_correction_json
                        CHECK (correction_json IS NULL OR ISJSON(correction_json) = 1)
                );
                CREATE INDEX IX_ai_evidence_review_finding
                    ON dbo.ai_evidence_review_events(finding_id, id ASC);
            END
            """
        )
        db.commit()


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError, ValueError):
        return default


def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _point(x: float, y: float) -> dict:
    return {"type": "Point", "coordinates": [float(x), float(y)]}


def _polygon_from_contour(contour: list) -> Optional[dict]:
    if not contour or len(contour) < 3:
        return None
    ring = []
    for pt in contour:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        ring.append([float(pt[0]), float(pt[1])])
    if len(ring) < 3:
        return None
    if ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    return {"type": "Polygon", "coordinates": [ring]}


def _bbox_polygon(x: float, y: float, width: float, height: float) -> dict:
    x0, y0 = float(x), float(y)
    x1, y1 = x0 + float(width), y0 + float(height)
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _latest_qc_context(db, slide_id: int) -> Optional[dict]:
    row = db.execute(
        """
        SELECT TOP 1 id, model_name, model_version, polygon_count, status, completed_at
        FROM dbo.qc_analyses
        WHERE slide_id = ? AND status = 'Completed'
        ORDER BY id DESC
        """,
        (slide_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "qc_analysis_id": int(row["id"]),
        "model_name": row["model_name"],
        "model_version": row["model_version"],
        "polygon_count": row["polygon_count"],
        "status": row["status"],
        "completed_at": str(row["completed_at"]) if row["completed_at"] else None,
        "note": "Latest completed GrandQC run on this slide; not a per-finding spatial intersection.",
    }


def adapt_hovernet_findings(row: dict, predictions: dict, qc_context: Optional[dict]) -> list[dict]:
    findings = []
    items = list(predictions.items()) if isinstance(predictions, dict) else []
    for index, (key, nuc) in enumerate(items[:MAX_NUCLEI_PER_ANALYSIS]):
        if not isinstance(nuc, dict):
            continue
        contour = nuc.get("contour") or []
        geometry = _polygon_from_contour(contour)
        if geometry is None:
            centroid = nuc.get("centroid") or []
            if len(centroid) >= 2:
                geometry = _point(centroid[0], centroid[1])
            else:
                continue
        nid = str(nuc.get("id") or key)
        label = nuc.get("type_name") or nuc.get("type") or "nucleus"
        score = nuc.get("type_prob")
        findings.append(
            {
                "source_finding_key": f"nucleus:{nid}",
                "feature_code": "HOVERNET",
                "model_name": row.get("model_name") or "HoVer-Net FAST",
                "model_version": row.get("model_version") or "PanNuke-Fold1",
                "finding_type": "nucleus",
                "label": str(label),
                "geometry": geometry,
                "score": float(score) if score is not None else None,
                "qc_context": qc_context,
                "provenance": {
                    "source_kind": SOURCE_AI,
                    "source_analysis_id": int(row["id"]),
                    "nucleus_id": nid,
                    "type_id": nuc.get("type_id"),
                    "roi": {
                        "x": row.get("roi_x"),
                        "y": row.get("roi_y"),
                        "width": row.get("roi_width"),
                        "height": row.get("roi_height"),
                        "level": row.get("roi_level"),
                    },
                    "score_note": "Uncalibrated HoVer-Net type probability; not a clinical probability.",
                    "immutable_source": True,
                },
            }
        )
        if index >= MAX_NUCLEI_PER_ANALYSIS - 1:
            break
    return findings


def adapt_kongnet_findings(row: dict, predictions: dict, qc_context: Optional[dict]) -> list[dict]:
    detections = predictions.get("detections") if isinstance(predictions, dict) else None
    if not isinstance(detections, list):
        return []
    findings = []
    for index, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        center = det.get("center_level0") or {}
        x = center.get("x") if isinstance(center, dict) else None
        y = center.get("y") if isinstance(center, dict) else None
        if x is None or y is None:
            continue
        det_id = str(det.get("id") or index + 1)
        score = det.get("score")
        findings.append(
            {
                "source_finding_key": f"mitotic:{det_id}",
                "feature_code": "MITOTIC_DETECTION",
                "model_name": row.get("model_name") or "KongNet-Det",
                "model_version": row.get("model_version") or "",
                "finding_type": "mitotic_candidate",
                "label": "Mitotic candidate",
                "geometry": _point(x, y),
                "score": float(score) if score is not None else None,
                "qc_context": qc_context,
                "provenance": {
                    "source_kind": SOURCE_AI,
                    "source_analysis_id": int(row["id"]),
                    "detection_id": det_id,
                    "roi_coordinates": det.get("roi_coordinates"),
                    "roi": {
                        "x": row.get("roi_x"),
                        "y": row.get("roi_y"),
                        "width": row.get("roi_width"),
                        "height": row.get("roi_height"),
                    },
                    "score_note": "Uncalibrated KongNet-Det score; candidate only, not a mitosis diagnosis.",
                    "immutable_source": True,
                },
            }
        )
    return findings


def adapt_camelyon_findings(row: dict, predictions: dict, summary: dict, qc_context: Optional[dict]) -> list[dict]:
    patches = predictions.get("patches") if isinstance(predictions, dict) else None
    if not isinstance(patches, list):
        return []
    patch_size = int(predictions.get("patch_size") or 224)
    findings = []
    for index, patch in enumerate(patches[:MAX_PATCHES_PER_ANALYSIS]):
        if not isinstance(patch, dict):
            continue
        x = patch.get("x")
        y = patch.get("y")
        if x is None or y is None:
            continue
        width = float(patch.get("width") or patch_size)
        height = float(patch.get("height") or patch_size)
        score = patch.get("score")
        findings.append(
            {
                "source_finding_key": f"patch:{index}:{int(x)}:{int(y)}",
                "feature_code": "CAMELYON_TUMOR_DETECTION",
                "model_name": row.get("model_name") or summary.get("model_name") or "CAMELYON",
                "model_version": row.get("model_version") or summary.get("model_version") or "",
                "finding_type": "tumor_patch",
                "label": "Tumor probability patch",
                "geometry": _bbox_polygon(x, y, width, height),
                "score": float(score) if score is not None else None,
                "qc_context": qc_context,
                "provenance": {
                    "source_kind": SOURCE_AI,
                    "source_analysis_id": int(row["id"]),
                    "patch_index": index,
                    "checkpoint_sha256": summary.get("checkpoint_sha256"),
                    "coordinate_system": summary.get("coordinate_system") or "level_0_pixels",
                    "score_note": "Uncalibrated CAMELYON patch score; research tumor probability only.",
                    "immutable_source": True,
                },
            }
        )
    return findings


def adapt_grandqc_findings(row: dict, result: dict, geojson: Optional[dict], qc_context: Optional[dict]) -> list[dict]:
    features = (geojson or {}).get("features") if isinstance(geojson, dict) else None
    if not isinstance(features, list):
        return []
    findings = []
    for index, feature in enumerate(features[:MAX_QC_FEATURES_PER_ANALYSIS]):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict) or not geometry.get("type"):
            continue
        props = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        label = props.get("classification") or props.get("class_name") or f"QC class {props.get('class_id', '?')}"
        findings.append(
            {
                "source_finding_key": f"qc:{index}",
                "feature_code": "SLIDE_QUALITY_CONTROL",
                "model_name": row.get("model_name") or "GrandQC",
                "model_version": row.get("model_version") or "MPP1.5",
                "finding_type": "qc_polygon",
                "label": str(label),
                "geometry": geometry,
                "score": None,
                "qc_context": qc_context
                or {
                    "qc_analysis_id": int(row["id"]),
                    "job_id": row.get("job_id"),
                    "polygon_count": row.get("polygon_count"),
                },
                "provenance": {
                    "source_kind": SOURCE_QC,
                    "source_analysis_id": int(row["id"]),
                    "feature_index": index,
                    "class_id": props.get("class_id"),
                    "area": props.get("area"),
                    "job_id": row.get("job_id"),
                    "score_note": "GrandQC artifact region; qualitative QC context only.",
                    "immutable_source": True,
                },
            }
        )
    return findings


def _feature_code_for_ai_row(row: dict, summary: dict) -> str:
    code = (summary or {}).get("feature_code") or ""
    if code:
        return str(code)
    name = str(row.get("model_name") or "")
    if "kongnet" in name.lower() or "mitotic" in name.lower():
        return "MITOTIC_DETECTION"
    if "camelyon" in name.lower() or "tumor" in name.lower():
        return "CAMELYON_TUMOR_DETECTION"
    return "HOVERNET"


def _load_grandqc_geojson(result: dict) -> Optional[dict]:
    path_value = (result.get("output_files") or {}).get("geojson")
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        logger.warning("GrandQC geojson missing for evidence projection: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read GrandQC geojson %s: %s", path, exc)
        return None


def _findings_from_ai_row(row: dict, qc_context: Optional[dict]) -> list[dict]:
    summary = _json_loads(row.get("summary_json"), {}) or {}
    predictions = _json_loads(row.get("predictions_json"), {}) or {}
    feature_code = _feature_code_for_ai_row(row, summary)
    if feature_code == "MITOTIC_DETECTION":
        return adapt_kongnet_findings(row, predictions, qc_context)
    if feature_code == "CAMELYON_TUMOR_DETECTION":
        return adapt_camelyon_findings(row, predictions, summary, qc_context)
    return adapt_hovernet_findings(row, predictions, qc_context)


def _findings_from_qc_row(row: dict) -> list[dict]:
    result = _json_loads(row.get("result_json"), {}) or {}
    geojson = _load_grandqc_geojson(result)
    qc_context = {
        "qc_analysis_id": int(row["id"]),
        "job_id": row.get("job_id"),
        "polygon_count": row.get("polygon_count"),
        "elapsed_seconds": row.get("elapsed_seconds"),
    }
    return adapt_grandqc_findings(row, result, geojson, qc_context)


def _insert_findings(db, slide_id: int, case_id: int, source_kind: str, source_analysis_id: int, findings: list[dict]) -> int:
    inserted = 0
    for finding in findings:
        key = finding["source_finding_key"]
        try:
            exists = db.execute(
                """
                SELECT 1 AS present FROM dbo.ai_evidence_findings
                WHERE source_kind = ? AND source_analysis_id = ? AND source_finding_key = ?
                """,
                (source_kind, source_analysis_id, key),
            ).fetchone()
            if exists:
                continue
            db.execute(
                """
                INSERT INTO dbo.ai_evidence_findings (
                    slide_id, case_id, source_kind, source_analysis_id, source_finding_key,
                    feature_code, model_name, model_version, finding_type, label,
                    geometry_json, coordinate_system, score, score_calibration_status,
                    qc_context_json, provenance_json, review_status
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, 'level_0_pixels', ?, ?,
                    ?, ?, 'Unreviewed'
                )
                """,
                (
                    slide_id,
                    case_id,
                    source_kind,
                    source_analysis_id,
                    key,
                    finding["feature_code"],
                    finding["model_name"],
                    finding["model_version"],
                    finding["finding_type"],
                    finding.get("label"),
                    _dumps(finding["geometry"]),
                    finding.get("score"),
                    CALIBRATION_UNCALIBRATED,
                    _dumps(finding.get("qc_context")),
                    _dumps(finding.get("provenance")),
                ),
            )
            inserted += 1
        except Exception:
            logger.exception(
                "Failed to insert evidence finding %s for %s#%s",
                key,
                source_kind,
                source_analysis_id,
            )
    return inserted


def project_ai_analysis(connect_func, analysis_id: int) -> dict:
    """Project one ai_analyses row into evidence findings without mutating source output."""
    with closing(connect_func()) as db:
        row = db.execute("SELECT * FROM dbo.ai_analyses WHERE id = ?", (analysis_id,)).fetchone()
        if not row:
            return {"projected": 0, "skipped": True, "reason": "analysis_not_found"}
        if str(row.get("status") or "").lower() not in {"completed", ""}:
            return {"projected": 0, "skipped": True, "reason": "not_completed"}
        existing = db.execute(
            """
            SELECT COUNT(1) AS n FROM dbo.ai_evidence_findings
            WHERE source_kind = ? AND source_analysis_id = ?
            """,
            (SOURCE_AI, analysis_id),
        ).fetchone()
        if existing and int(existing["n"] or 0) > 0:
            return {"projected": 0, "skipped": True, "reason": "already_projected", "existing": int(existing["n"])}
        qc_context = _latest_qc_context(db, int(row["slide_id"]))
        findings = _findings_from_ai_row(dict(row), qc_context)
        inserted = _insert_findings(db, int(row["slide_id"]), int(row["case_id"]), SOURCE_AI, analysis_id, findings)
        db.commit()
        return {"projected": inserted, "skipped": False, "source_kind": SOURCE_AI, "source_analysis_id": analysis_id}


def project_qc_analysis(connect_func, analysis_id: int) -> dict:
    """Project one qc_analyses row into evidence findings."""
    with closing(connect_func()) as db:
        row = db.execute("SELECT * FROM dbo.qc_analyses WHERE id = ?", (analysis_id,)).fetchone()
        if not row:
            return {"projected": 0, "skipped": True, "reason": "analysis_not_found"}
        if str(row.get("status") or "").lower() not in {"completed", ""}:
            return {"projected": 0, "skipped": True, "reason": "not_completed"}
        existing = db.execute(
            """
            SELECT COUNT(1) AS n FROM dbo.ai_evidence_findings
            WHERE source_kind = ? AND source_analysis_id = ?
            """,
            (SOURCE_QC, analysis_id),
        ).fetchone()
        if existing and int(existing["n"] or 0) > 0:
            return {"projected": 0, "skipped": True, "reason": "already_projected", "existing": int(existing["n"])}
        findings = _findings_from_qc_row(dict(row))
        inserted = _insert_findings(db, int(row["slide_id"]), int(row["case_id"]), SOURCE_QC, analysis_id, findings)
        db.commit()
        return {"projected": inserted, "skipped": False, "source_kind": SOURCE_QC, "source_analysis_id": analysis_id}


def project_slide_evidence(connect_func, slide_id: int) -> dict:
    """Backfill evidence findings for all completed analyses on a slide."""
    totals = {"ai_analyses": 0, "qc_analyses": 0, "projected": 0, "details": []}
    with closing(connect_func()) as db:
        ai_rows = db.execute(
            "SELECT id FROM dbo.ai_analyses WHERE slide_id = ? AND status = 'Completed' ORDER BY id",
            (slide_id,),
        ).fetchall()
        qc_rows = db.execute(
            "SELECT id FROM dbo.qc_analyses WHERE slide_id = ? AND status = 'Completed' ORDER BY id",
            (slide_id,),
        ).fetchall()
    for row in ai_rows:
        result = project_ai_analysis(connect_func, int(row["id"]))
        totals["ai_analyses"] += 1
        totals["projected"] += int(result.get("projected") or 0)
        totals["details"].append(result)
    for row in qc_rows:
        result = project_qc_analysis(connect_func, int(row["id"]))
        totals["qc_analyses"] += 1
        totals["projected"] += int(result.get("projected") or 0)
        totals["details"].append(result)
    return totals


def _serialize_finding(row: dict, include_history: bool = False, history: Optional[list] = None) -> dict:
    provenance = _json_loads(row.get("provenance_json"), {}) or {}
    return {
        "id": int(row["id"]),
        "slide_id": int(row["slide_id"]),
        "case_id": int(row["case_id"]),
        "source_kind": row["source_kind"],
        "source_analysis_id": int(row["source_analysis_id"]),
        "source_finding_key": row["source_finding_key"],
        "feature_code": row["feature_code"],
        "model_name": row["model_name"],
        "model_version": row["model_version"],
        "finding_type": row["finding_type"],
        "label": row.get("label"),
        "geometry": _json_loads(row.get("geometry_json"), {}),
        "coordinate_system": row.get("coordinate_system") or "level_0_pixels",
        "score": row.get("score"),
        "score_calibration_status": row.get("score_calibration_status") or CALIBRATION_UNCALIBRATED,
        "qc_context": _json_loads(row.get("qc_context_json"), None),
        "provenance": provenance,
        "review_status": row.get("review_status") or REVIEW_UNREVIEWED,
        "latest_review_event_id": row.get("latest_review_event_id"),
        "created_at": str(row["created_at"]) if row.get("created_at") else None,
        "report_eligible": (row.get("review_status") in {"Accepted", "Corrected"}),
        "history": history if include_history else None,
        "disclaimer": "Research use only. Model scores are uncalibrated and are not clinical probabilities or diagnoses.",
    }


def list_evidence_findings(
    connect_func,
    slide_id: int,
    *,
    feature_code: Optional[str] = None,
    finding_type: Optional[str] = None,
    review_status: Optional[str] = None,
    source_analysis_id: Optional[int] = None,
    source_kind: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> dict:
    clauses = ["slide_id = ?"]
    params: list[Any] = [slide_id]
    if feature_code:
        clauses.append("feature_code = ?")
        params.append(feature_code)
    if finding_type:
        clauses.append("finding_type = ?")
        params.append(finding_type)
    if review_status:
        clauses.append("review_status = ?")
        params.append(review_status)
    if source_analysis_id is not None:
        clauses.append("source_analysis_id = ?")
        params.append(source_analysis_id)
    if source_kind:
        clauses.append("source_kind = ?")
        params.append(source_kind)
    where = " AND ".join(clauses)
    limit = max(1, min(int(limit or 200), 1000))
    offset = max(0, int(offset or 0))
    with closing(connect_func()) as db:
        total_row = db.execute(
            f"SELECT COUNT(1) AS n FROM dbo.ai_evidence_findings WHERE {where}",
            tuple(params),
        ).fetchone()
        rows = db.execute(
            f"""
            SELECT * FROM dbo.ai_evidence_findings
            WHERE {where}
            ORDER BY id DESC
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
            """,
            tuple(params + [offset, limit]),
        ).fetchall()
        summary_rows = db.execute(
            f"""
            SELECT finding_type, review_status, COUNT(1) AS n
            FROM dbo.ai_evidence_findings
            WHERE {where}
            GROUP BY finding_type, review_status
            """,
            tuple(params),
        ).fetchall()
    return {
        "slide_id": slide_id,
        "total": int(total_row["n"] if total_row else 0),
        "limit": limit,
        "offset": offset,
        "items": [_serialize_finding(dict(row)) for row in rows],
        "summary": [
            {
                "finding_type": row["finding_type"],
                "review_status": row["review_status"],
                "count": int(row["n"]),
            }
            for row in summary_rows
        ],
        "score_calibration_status": CALIBRATION_UNCALIBRATED,
        "disclaimer": "Research use only. Findings reference immutable source analyses; scores are uncalibrated.",
    }


def get_evidence_finding(connect_func, finding_id: int) -> Optional[dict]:
    with closing(connect_func()) as db:
        row = db.execute(
            "SELECT * FROM dbo.ai_evidence_findings WHERE id = ?",
            (finding_id,),
        ).fetchone()
        if not row:
            return None
        history = _list_review_events(db, finding_id)
    return _serialize_finding(dict(row), include_history=True, history=history)


def _list_review_events(db, finding_id: int) -> list[dict]:
    rows = db.execute(
        """
        SELECT id, finding_id, action, actor, reason, correction_json, created_at
        FROM dbo.ai_evidence_review_events
        WHERE finding_id = ?
        ORDER BY id ASC
        """,
        (finding_id,),
    ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "finding_id": int(row["finding_id"]),
            "action": row["action"],
            "actor": row["actor"],
            "reason": row.get("reason"),
            "correction": _json_loads(row.get("correction_json"), None),
            "created_at": str(row["created_at"]) if row.get("created_at") else None,
        }
        for row in rows
    ]


def list_review_history(connect_func, finding_id: int) -> Optional[dict]:
    with closing(connect_func()) as db:
        row = db.execute(
            "SELECT id, slide_id FROM dbo.ai_evidence_findings WHERE id = ?",
            (finding_id,),
        ).fetchone()
        if not row:
            return None
        events = _list_review_events(db, finding_id)
    return {"finding_id": finding_id, "slide_id": int(row["slide_id"]), "events": events}


def submit_review_event(
    connect_func,
    finding_id: int,
    *,
    action: str,
    actor: str,
    reason: Optional[str] = None,
    correction: Optional[dict] = None,
) -> dict:
    if action not in REVIEW_ACTIONS:
        raise ValueError("action must be Accept, Reject, or Correct")
    if action == "Correct" and not (reason or correction):
        raise ValueError("Correct requires a reason or correction payload")
    new_status = REVIEW_ACTIONS[action]
    with closing(connect_func()) as db:
        row = db.execute(
            "SELECT * FROM dbo.ai_evidence_findings WHERE id = ?",
            (finding_id,),
        ).fetchone()
        if not row:
            raise KeyError("finding_not_found")
        cursor = db.execute(
            """
            INSERT INTO dbo.ai_evidence_review_events (
                finding_id, action, actor, reason, correction_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (finding_id, action, actor, reason, _dumps(correction)),
        )
        event_id = cursor.lastrowid
        if event_id is None:
            saved = db.execute(
                """
                SELECT TOP 1 id FROM dbo.ai_evidence_review_events
                WHERE finding_id = ? AND actor = ?
                ORDER BY id DESC
                """,
                (finding_id, actor),
            ).fetchone()
            event_id = int(saved["id"]) if saved else None
        db.execute(
            """
            UPDATE dbo.ai_evidence_findings
            SET review_status = ?, latest_review_event_id = ?
            WHERE id = ?
            """,
            (new_status, event_id, finding_id),
        )
        db.commit()
        updated = db.execute(
            "SELECT * FROM dbo.ai_evidence_findings WHERE id = ?",
            (finding_id,),
        ).fetchone()
        history = _list_review_events(db, finding_id)
    return _serialize_finding(dict(updated), include_history=True, history=history)


def list_verified_findings_for_case(connect_func, case_id: int) -> list[dict]:
    """Accepted/Corrected findings only — eligible for AI-assisted report drafts."""
    with closing(connect_func()) as db:
        rows = db.execute(
            """
            SELECT f.*, s.label AS slide_label, s.stain
            FROM dbo.ai_evidence_findings f
            JOIN dbo.slides s ON s.id = f.slide_id
            WHERE f.case_id = ? AND f.review_status IN ('Accepted', 'Corrected')
            ORDER BY f.slide_id, f.id
            """,
            (case_id,),
        ).fetchall()
    items = []
    for row in rows:
        item = _serialize_finding(dict(row))
        item["slide_label"] = row.get("slide_label")
        item["stain"] = row.get("stain")
        items.append(item)
    return items


def safe_project_ai_analysis(connect_func, analysis_id: int) -> None:
    try:
        result = project_ai_analysis(connect_func, analysis_id)
        logger.info("Evidence projection for ai_analyses#%s: %s", analysis_id, result)
    except Exception:
        logger.exception("Evidence projection failed for ai_analyses#%s", analysis_id)


def safe_project_qc_analysis(connect_func, analysis_id: int) -> None:
    try:
        result = project_qc_analysis(connect_func, analysis_id)
        logger.info("Evidence projection for qc_analyses#%s: %s", analysis_id, result)
    except Exception:
        logger.exception("Evidence projection failed for qc_analyses#%s", analysis_id)
