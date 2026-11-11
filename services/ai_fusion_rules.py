"""Semantic agreement / conflict rules for AI Fusion v1.

Different prediction types never automatically count as agreement.
No fabricated combined diagnostic confidence is produced.
"""

from __future__ import annotations

from typing import Any, Optional


NEOPLASTIC_LABELS = {"neoplastic", "tumor", "malignant"}
NON_NEOPLASTIC_LABELS = {"non-neoplastic", "non neoplastic", "epithelial", "normal"}
QC_SEVERE_LABELS = {"fold", "blur", "out of focus", "out-of-focus", "air bubble", "pen mark", "tissue fold"}

DEFAULT_TUMOR_SUPPORT_THRESHOLD = 0.50
DEFAULT_TUMOR_CONFLICT_THRESHOLD = 0.70
DEFAULT_MITOTIC_SUPPORT_THRESHOLD = 0.50


def _norm_label(label: Optional[str]) -> str:
    return str(label or "").strip().lower()


def is_neoplastic_nucleus(finding: dict) -> bool:
    return finding.get("finding_type") == "nucleus" and _norm_label(finding.get("label")) in NEOPLASTIC_LABELS


def is_non_neoplastic_nucleus(finding: dict) -> bool:
    label = _norm_label(finding.get("label"))
    return finding.get("finding_type") == "nucleus" and (
        label in NON_NEOPLASTIC_LABELS or label == "non-neoplastic"
    )


def is_tumor_patch(finding: dict, *, min_score: float) -> bool:
    if finding.get("finding_type") != "tumor_patch":
        return False
    score = finding.get("score")
    return score is not None and float(score) >= min_score


def is_mitotic(finding: dict, *, min_score: float = 0.0) -> bool:
    if finding.get("finding_type") != "mitotic_candidate":
        return False
    score = finding.get("score")
    if score is None:
        return True
    return float(score) >= min_score


def is_qc(finding: dict) -> bool:
    return finding.get("finding_type") == "qc_polygon"


def is_severe_qc(finding: dict) -> bool:
    if not is_qc(finding):
        return False
    return _norm_label(finding.get("label")) in QC_SEVERE_LABELS or True  # all QC polygons flag tissue compromise


def classify_pair(
    a: dict,
    b: dict,
    *,
    tumor_support: float = DEFAULT_TUMOR_SUPPORT_THRESHOLD,
    tumor_conflict: float = DEFAULT_TUMOR_CONFLICT_THRESHOLD,
    mitotic_support: float = DEFAULT_MITOTIC_SUPPORT_THRESHOLD,
) -> list[dict]:
    """Return zero or more relationship descriptors for an overlapping pair."""
    relations: list[dict] = []
    types = {a.get("finding_type"), b.get("finding_type")}

    # QC awareness — not agreement
    if is_qc(a) ^ is_qc(b):
        non_qc = b if is_qc(a) else a
        qc = a if is_qc(a) else b
        relations.append({
            "code": "qc_affected",
            "severity": "qc_limitation",
            "summary": f"{non_qc.get('finding_type')} overlaps QC region '{qc.get('label')}'.",
            "evidence_ids": [a.get("id"), b.get("id")],
        })
        return relations

    if is_qc(a) and is_qc(b):
        relations.append({
            "code": "qc_co_located",
            "severity": "related",
            "summary": "Multiple QC polygons overlap the same tissue region.",
            "evidence_ids": [a.get("id"), b.get("id")],
        })
        return relations

    # Same type co-location is related, not automatic agreement
    if a.get("finding_type") == b.get("finding_type"):
        if a.get("finding_type") == "tumor_patch" and is_tumor_patch(a, min_score=tumor_support) and is_tumor_patch(b, min_score=tumor_support):
            relations.append({
                "code": "tumor_patch_concordance",
                "severity": "agreement",
                "summary": "Overlapping tumor-probability patches both exceed the support threshold (uncalibrated scores preserved separately).",
                "evidence_ids": [a.get("id"), b.get("id")],
            })
        else:
            relations.append({
                "code": "same_type_co_located",
                "severity": "related",
                "summary": f"Co-located {a.get('finding_type')} findings; same type alone is not treated as multi-model agreement.",
                "evidence_ids": [a.get("id"), b.get("id")],
            })
        # Review tension
        statuses = {a.get("review_status"), b.get("review_status")}
        if "Accepted" in statuses and "Rejected" in statuses:
            relations.append({
                "code": "review_tension",
                "severity": "conflict",
                "summary": "Accepted and Rejected reviews for co-located findings of the same type.",
                "evidence_ids": [a.get("id"), b.get("id")],
            })
        return relations

    # Cross-type: only explicit validated rules may produce agreement/conflict
    neo = a if is_neoplastic_nucleus(a) else b if is_neoplastic_nucleus(b) else None
    non = a if is_non_neoplastic_nucleus(a) else b if is_non_neoplastic_nucleus(b) else None
    tumor_high = a if is_tumor_patch(a, min_score=tumor_conflict) else b if is_tumor_patch(b, min_score=tumor_conflict) else None
    tumor_sup = a if is_tumor_patch(a, min_score=tumor_support) else b if is_tumor_patch(b, min_score=tumor_support) else None
    mitotic = a if is_mitotic(a, min_score=mitotic_support) else b if is_mitotic(b, min_score=mitotic_support) else None

    if neo and tumor_sup and {neo.get("id"), tumor_sup.get("id")} == {a.get("id"), b.get("id")}:
        relations.append({
            "code": "neoplastic_in_tumor_patch",
            "severity": "agreement",
            "summary": "Neoplastic nucleus overlaps a tumor-probability patch above the support threshold. Source scores remain uncalibrated and separate.",
            "evidence_ids": [a.get("id"), b.get("id")],
        })

    if mitotic and tumor_sup and {mitotic.get("id"), tumor_sup.get("id")} == {a.get("id"), b.get("id")}:
        relations.append({
            "code": "mitotic_in_tumor_patch",
            "severity": "agreement",
            "summary": "Mitotic candidate overlaps a tumor-probability patch above the support threshold. Candidate only — not a mitosis diagnosis.",
            "evidence_ids": [a.get("id"), b.get("id")],
        })

    if mitotic and neo and {mitotic.get("id"), neo.get("id")} == {a.get("id"), b.get("id")}:
        relations.append({
            "code": "mitotic_near_neoplastic",
            "severity": "related",
            "summary": "Mitotic candidate co-located with neoplastic nucleus. Related observation, not automatic agreement across model semantics.",
            "evidence_ids": [a.get("id"), b.get("id")],
        })

    if non and tumor_high and {non.get("id"), tumor_high.get("id")} == {a.get("id"), b.get("id")}:
        relations.append({
            "code": "nonneoplastic_vs_high_tumor",
            "severity": "conflict",
            "summary": "Non-neoplastic nucleus overlaps a high tumor-probability patch. Contradiction candidate under Fusion rules; pathologist review required.",
            "evidence_ids": [a.get("id"), b.get("id")],
        })

    if not relations and types == {"nucleus", "tumor_patch"}:
        relations.append({
            "code": "nucleus_tumor_co_located",
            "severity": "related",
            "summary": "Nucleus and tumor patch overlap without meeting agreement/conflict thresholds.",
            "evidence_ids": [a.get("id"), b.get("id")],
        })

    if not relations:
        relations.append({
            "code": "cross_type_co_located",
            "severity": "related",
            "summary": f"Co-located {a.get('finding_type')} and {b.get('finding_type')}; no validated agreement/conflict rule matched.",
            "evidence_ids": [a.get("id"), b.get("id")],
        })

    return relations


def classify_group(relations: list[dict], members: list[dict]) -> dict[str, Any]:
    """Roll up pair relations into a group classification with rationale (no fused score)."""
    severities = {r.get("severity") for r in relations}
    codes = [r.get("code") for r in relations]
    qc_affected = "qc_limitation" in severities or any(is_qc(m) for m in members) and len(members) > 1 and any(not is_qc(m) for m in members)
    has_conflict = "conflict" in severities
    has_agreement = "agreement" in severities

    if has_conflict and qc_affected:
        classification = "ConflictQC"
    elif has_conflict:
        classification = "Conflict"
    elif qc_affected and has_agreement:
        classification = "AgreementQC"
    elif qc_affected:
        classification = "QCAffected"
    elif has_agreement:
        classification = "Agreement"
    elif len(members) == 1:
        classification = "Singleton"
    else:
        classification = "Related"

    model_names = sorted({f"{m.get('model_name')} {m.get('model_version')}".strip() for m in members})
    feature_codes = sorted({m.get("feature_code") for m in members if m.get("feature_code")})
    review_statuses = sorted({m.get("review_status") or "Unreviewed" for m in members})

    return {
        "classification": classification,
        "relation_codes": codes,
        "qc_affected": bool(qc_affected),
        "has_agreement": bool(has_agreement),
        "has_conflict": bool(has_conflict),
        "model_versions": model_names,
        "feature_codes": feature_codes,
        "review_statuses": review_statuses,
        "participant_count": len(members),
        "rationale": [r.get("summary") for r in relations if r.get("summary")],
        "uncertainty": {
            "combined_diagnostic_confidence": None,
            "note": "Fusion never fabricates a combined diagnostic confidence. Source-model scores remain uncalibrated and are listed per finding.",
            "score_calibration_status": "uncalibrated",
        },
        "relations": relations,
    }
