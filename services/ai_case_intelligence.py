"""Phase 3 — Case Intelligence over Evidence Map + Fusion runs.

Aggregates slide-level evidence without claiming cross-slide tissue identity
unless validated registration is present (not assumed here).
"""

from __future__ import annotations

from contextlib import closing
from collections import defaultdict
from typing import Optional

from services.ai_evidence_service import (
    _json_loads,
    list_verified_findings_for_case,
)
from services.ai_fusion_service import get_fusion_run


def build_case_fusion_summary(connect_func, case_id: int) -> dict:
    """Aggregate latest Fusion run + evidence review status per slide."""
    with closing(connect_func()) as db:
        slides = db.execute(
            """
            SELECT id, label, stain, file_type, qc_status
            FROM dbo.slides
            WHERE case_id = ?
            ORDER BY id
            """,
            (case_id,),
        ).fetchall()

        evidence_rows = db.execute(
            """
            SELECT slide_id, review_status, finding_type, COUNT(1) AS n
            FROM dbo.ai_evidence_findings
            WHERE case_id = ?
            GROUP BY slide_id, review_status, finding_type
            """,
            (case_id,),
        ).fetchall()

        latest_runs = db.execute(
            """
            SELECT r.*
            FROM dbo.ai_fusion_runs r
            INNER JOIN (
                SELECT slide_id, MAX(id) AS max_id
                FROM dbo.ai_fusion_runs
                WHERE case_id = ?
                GROUP BY slide_id
            ) latest ON latest.max_id = r.id
            ORDER BY r.slide_id
            """,
            (case_id,),
        ).fetchall()

        group_class_rows = []
        if latest_runs:
            run_ids = [int(r["id"]) for r in latest_runs]
            marks = ",".join("?" for _ in run_ids)
            group_class_rows = db.execute(
                f"""
                SELECT run_id, classification, COUNT(1) AS n
                FROM dbo.ai_fusion_groups
                WHERE run_id IN ({marks})
                GROUP BY run_id, classification
                """,
                tuple(run_ids),
            ).fetchall()

        registration_count = 0
        try:
            reg = db.execute(
                """
                SELECT COUNT(1) AS n
                FROM dbo.slide_registrations
                WHERE case_id = ? AND status IN ('Reviewed', 'Calculated')
                """,
                (case_id,),
            ).fetchone()
            registration_count = int(reg["n"] or 0) if reg else 0
        except Exception:
            registration_count = 0

    evidence_by_slide: dict[int, dict] = defaultdict(lambda: {
        "total": 0,
        "by_review": defaultdict(int),
        "by_type": defaultdict(int),
        "unresolved": 0,
        "verified": 0,
    })
    for row in evidence_rows:
        slide_id = int(row["slide_id"])
        bucket = evidence_by_slide[slide_id]
        count = int(row["n"] or 0)
        status = row["review_status"] or "Unreviewed"
        ftype = row["finding_type"] or "unknown"
        bucket["total"] += count
        bucket["by_review"][status] += count
        bucket["by_type"][ftype] += count
        if status == "Unreviewed":
            bucket["unresolved"] += count
        if status in {"Accepted", "Corrected"}:
            bucket["verified"] += count

    classes_by_run: dict[int, dict[str, int]] = defaultdict(dict)
    for row in group_class_rows:
        classes_by_run[int(row["run_id"])][row["classification"]] = int(row["n"] or 0)

    runs_by_slide = {int(r["slide_id"]): dict(r) for r in latest_runs}

    slide_summaries = []
    case_totals = {
        "slides": len(slides),
        "evidence_total": 0,
        "unresolved_reviews": 0,
        "verified_findings": 0,
        "fusion_runs_present": 0,
        "groups_by_classification": defaultdict(int),
        "qc_affected_groups": 0,
        "conflicts": 0,
        "agreements": 0,
    }

    for slide in slides:
        slide_id = int(slide["id"])
        ev = evidence_by_slide[slide_id]
        by_review = dict(ev["by_review"])
        by_type = dict(ev["by_type"])
        unresolved = int(ev["unresolved"])
        verified = int(ev["verified"])
        total = int(ev["total"])
        run = runs_by_slide.get(slide_id)
        classification = classes_by_run.get(int(run["id"]), {}) if run else {}
        if run:
            case_totals["fusion_runs_present"] += 1
            for name, count in classification.items():
                case_totals["groups_by_classification"][name] += count
                if "QC" in name or name == "QCAffected":
                    case_totals["qc_affected_groups"] += count
                if "Conflict" in name:
                    case_totals["conflicts"] += count
                if name.startswith("Agreement"):
                    case_totals["agreements"] += count

        case_totals["evidence_total"] += total
        case_totals["unresolved_reviews"] += unresolved
        case_totals["verified_findings"] += verified

        slide_summaries.append({
            "slide_id": slide_id,
            "label": slide.get("label"),
            "stain": slide.get("stain"),
            "file_type": slide.get("file_type"),
            "qc_status": slide.get("qc_status"),
            "evidence": {
                "total": total,
                "unresolved": unresolved,
                "verified": verified,
                "by_review": by_review,
                "by_type": by_type,
            },
            "latest_fusion_run": None if not run else {
                "id": int(run["id"]),
                "created_at": str(run["created_at"]) if run.get("created_at") else None,
                "group_count": int(run.get("group_count") or 0),
                "evidence_count": int(run.get("evidence_count") or 0),
                "by_classification": classification,
                "parameters": _json_loads(run.get("parameters_json"), {}),
            },
        })

    return {
        "case_id": case_id,
        "totals": {
            **{k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in case_totals.items()},
            "validated_registrations": registration_count,
        },
        "slides": slide_summaries,
        "cross_slide_correlation": {
            "enabled": False,
            "reason": (
                "Findings on different slides are summarized side-by-side only. "
                "VISTA does not treat them as the same tissue region without validated registration."
            ),
            "validated_registrations": registration_count,
        },
        "disclaimer": (
            "Research use only. Case summary aggregates slide-level Evidence Map and Fusion results. "
            "It does not produce a diagnosis or combined clinical confidence."
        ),
    }


def list_report_evidence_catalog(connect_func, case_id: int) -> dict:
    """Selectable verified findings and fusion groups for report drafting."""
    findings = list_verified_findings_for_case(connect_func, case_id)
    with closing(connect_func()) as db:
        groups = db.execute(
            """
            SELECT g.*, r.slide_id, r.id AS run_id, s.label AS slide_label, s.stain
            FROM dbo.ai_fusion_groups g
            JOIN dbo.ai_fusion_runs r ON r.id = g.run_id
            JOIN dbo.slides s ON s.id = r.slide_id
            WHERE r.case_id = ?
              AND r.id IN (
                  SELECT MAX(id) FROM dbo.ai_fusion_runs WHERE case_id = ? GROUP BY slide_id
              )
            ORDER BY s.id, g.id
            """,
            (case_id, case_id),
        ).fetchall()

        catalog_groups = []
        for grow in groups:
            members = db.execute(
                """
                SELECT m.evidence_finding_id, m.role, f.review_status, f.finding_type, f.label,
                       f.model_name, f.model_version, f.score, f.score_calibration_status
                FROM dbo.ai_fusion_group_members m
                JOIN dbo.ai_evidence_findings f ON f.id = m.evidence_finding_id
                WHERE m.group_id = ?
                """,
                (grow["id"],),
            ).fetchall()
            member_list = [dict(m) for m in members]
            verified_members = [
                m for m in member_list
                if (m.get("review_status") in {"Accepted", "Corrected"})
                or m.get("finding_type") == "qc_polygon"
            ]
            # Group is selectable when every non-QC participant is verified
            participants = [m for m in member_list if m.get("finding_type") != "qc_polygon"]
            selectable = bool(participants) and all(
                m.get("review_status") in {"Accepted", "Corrected"} for m in participants
            )
            rationale = _json_loads(grow.get("rationale_json"), {}) or {}
            catalog_groups.append({
                "id": int(grow["id"]),
                "run_id": int(grow["run_id"]),
                "slide_id": int(grow["slide_id"]),
                "slide_label": grow.get("slide_label"),
                "stain": grow.get("stain"),
                "group_key": grow["group_key"],
                "classification": grow["classification"],
                "member_count": int(grow.get("member_count") or len(member_list)),
                "verified_member_count": len(verified_members),
                "selectable": selectable,
                "why": rationale.get("rationale") or [],
                "members": [
                    {
                        "evidence_finding_id": int(m["evidence_finding_id"]),
                        "finding_type": m.get("finding_type"),
                        "label": m.get("label"),
                        "review_status": m.get("review_status"),
                        "model_name": m.get("model_name"),
                        "model_version": m.get("model_version"),
                        "score": m.get("score"),
                        "score_calibration_status": m.get("score_calibration_status") or "uncalibrated",
                        "role": m.get("role"),
                    }
                    for m in member_list
                ],
            })

    return {
        "case_id": case_id,
        "findings": findings,
        "fusion_groups": catalog_groups,
        "disclaimer": (
            "Only pathologist-verified findings and fusion groups with verified participants "
            "may be inserted into AI-assisted report drafts. Uncalibrated scores are not clinical probabilities."
        ),
    }


def build_verified_report_draft(
    connect_func,
    case_id: int,
    *,
    finding_ids: Optional[list[int]] = None,
    fusion_group_ids: Optional[list[int]] = None,
) -> dict:
    """Build microscopic draft from explicitly selected verified evidence."""
    catalog = list_report_evidence_catalog(connect_func, case_id)
    finding_map = {int(f["id"]): f for f in catalog["findings"]}
    group_map = {int(g["id"]): g for g in catalog["fusion_groups"]}

    selected_findings = []
    if finding_ids is None and fusion_group_ids is None:
        selected_findings = list(catalog["findings"])
        selected_groups = [g for g in catalog["fusion_groups"] if g.get("selectable")]
    else:
        for fid in finding_ids or []:
            item = finding_map.get(int(fid))
            if not item:
                raise ValueError(f"Finding #{fid} is not a verified Evidence Map item for this case")
            selected_findings.append(item)
        selected_groups = []
        for gid in fusion_group_ids or []:
            group = group_map.get(int(gid))
            if not group:
                raise ValueError(f"Fusion group #{gid} was not found for this case")
            if not group.get("selectable"):
                raise ValueError(
                    f"Fusion group #{gid} is not eligible: all non-QC participants must be Accepted or Corrected"
                )
            selected_groups.append(group)

    if not selected_findings and not selected_groups:
        raise LookupError(
            "No verified findings or selectable fusion groups were provided for the report draft"
        )

    lines = [
        "AI-ASSISTED MICROSCOPIC FINDINGS DRAFT",
        "Research-use output only. Content limited to pathologist-verified Evidence Map findings and verified fusion context. Model scores are uncalibrated; this draft is not a diagnosis.",
        "",
    ]

    if selected_findings:
        lines.append("Verified evidence findings")
        for item in selected_findings:
            slide = item.get("slide_label") or "Unlabelled slide"
            stain = f" ({item['stain']})" if item.get("stain") else ""
            score = item.get("score")
            score_text = f"; uncalibrated score {float(score):.3f}" if score is not None else ""
            lines.append(
                f"- {slide}{stain}: {item.get('finding_type')} ({item.get('label') or 'unlabelled'}) "
                f"via {item.get('model_name')} {item.get('model_version') or ''} "
                f"[{item.get('review_status')}]{score_text}; evidence #{item.get('id')}."
            )

    if selected_groups:
        lines.append("")
        lines.append("Verified fusion context (slide-scoped; not cross-slide tissue identity)")
        for group in selected_groups:
            slide = group.get("slide_label") or "Unlabelled slide"
            stain = f" ({group['stain']})" if group.get("stain") else ""
            why = "; ".join(group.get("why") or []) or "No rationale recorded"
            lines.append(
                f"- {slide}{stain}: fusion {group.get('group_key')} classified as "
                f"{group.get('classification')} (run #{group.get('run_id')}, group #{group.get('id')})."
            )
            lines.append(f"  Why: {why}")
            for member in group.get("members") or []:
                lines.append(
                    f"  · member evidence #{member.get('evidence_finding_id')}: "
                    f"{member.get('finding_type')} / {member.get('label')} "
                    f"[{member.get('review_status')}] score={member.get('score')} "
                    f"({member.get('score_calibration_status')})"
                )

    lines.append("")
    lines.append(
        "Cross-slide note: Findings from different slides are listed separately and are not "
        "asserted to represent the same tissue region without validated registration."
    )

    draft = "\n".join(lines)
    return {
        "microscopic_findings": draft[:8000],
        "result_count": len(selected_findings) + len(selected_groups),
        "finding_count": len(selected_findings),
        "fusion_group_count": len(selected_groups),
        "disclaimer": lines[1],
        "source": "verified_evidence_report_builder",
        "review_required": True,
        "selected_finding_ids": [int(f["id"]) for f in selected_findings],
        "selected_fusion_group_ids": [int(g["id"]) for g in selected_groups],
    }


def _group_signature(group: dict) -> frozenset:
    members = group.get("members") or []
    ids = []
    for member in members:
        eid = member.get("evidence_finding_id")
        if eid is None:
            eid = member.get("id")
        if eid is not None:
            ids.append(int(eid))
    return frozenset(ids)


def compare_fusion_runs(connect_func, run_a_id: int, run_b_id: int) -> dict:
    """Compare two fusion runs on the same slide (append-only history)."""
    run_a = get_fusion_run(connect_func, run_a_id)
    run_b = get_fusion_run(connect_func, run_b_id)
    if not run_a or not run_b:
        raise LookupError("One or both fusion runs were not found")
    if int(run_a["slide_id"]) != int(run_b["slide_id"]):
        raise ValueError("Fusion run comparison requires both runs on the same slide")

    groups_a = { _group_signature(g): g for g in (run_a.get("groups") or []) if _group_signature(g) }
    groups_b = { _group_signature(g): g for g in (run_b.get("groups") or []) if _group_signature(g) }

    shared = set(groups_a) & set(groups_b)
    only_a = set(groups_a) - set(groups_b)
    only_b = set(groups_b) - set(groups_a)

    classification_changes = []
    for sig in sorted(shared, key=lambda s: min(s) if s else 0):
        ga, gb = groups_a[sig], groups_b[sig]
        if ga.get("classification") != gb.get("classification"):
            classification_changes.append({
                "member_evidence_ids": sorted(sig),
                "from_run_id": run_a_id,
                "to_run_id": run_b_id,
                "from_classification": ga.get("classification"),
                "to_classification": gb.get("classification"),
                "from_group_key": ga.get("group_key"),
                "to_group_key": gb.get("group_key"),
            })

    # Review status deltas for shared members
    review_changes = []
    for sig in shared:
        map_a = {
            int(m.get("evidence_finding_id")): m
            for m in (groups_a[sig].get("members") or [])
            if m.get("evidence_finding_id") is not None
        }
        map_b = {
            int(m.get("evidence_finding_id")): m
            for m in (groups_b[sig].get("members") or [])
            if m.get("evidence_finding_id") is not None
        }
        for eid in sorted(set(map_a) & set(map_b)):
            sa = map_a[eid].get("review_status")
            sb = map_b[eid].get("review_status")
            if sa != sb:
                review_changes.append({
                    "evidence_finding_id": eid,
                    "from_review_status": sa,
                    "to_review_status": sb,
                    "finding_type": map_b[eid].get("finding_type"),
                    "label": map_b[eid].get("label"),
                })

    return {
        "slide_id": int(run_a["slide_id"]),
        "from_run": {
            "id": run_a_id,
            "created_at": run_a.get("created_at"),
            "group_count": run_a.get("group_count"),
            "summary": run_a.get("summary"),
        },
        "to_run": {
            "id": run_b_id,
            "created_at": run_b.get("created_at"),
            "group_count": run_b.get("group_count"),
            "summary": run_b.get("summary"),
        },
        "groups_removed": [
            {
                "member_evidence_ids": sorted(sig),
                "classification": groups_a[sig].get("classification"),
                "group_key": groups_a[sig].get("group_key"),
            }
            for sig in sorted(only_a, key=lambda s: min(s) if s else 0)
        ],
        "groups_added": [
            {
                "member_evidence_ids": sorted(sig),
                "classification": groups_b[sig].get("classification"),
                "group_key": groups_b[sig].get("group_key"),
            }
            for sig in sorted(only_b, key=lambda s: min(s) if s else 0)
        ],
        "classification_changes": classification_changes,
        "review_status_changes": review_changes,
        "unchanged_group_count": len(shared) - len(classification_changes),
        "disclaimer": (
            "Comparison is structural and review-status based. It does not score diagnostic improvement "
            "and does not invent combined confidence."
        ),
    }
