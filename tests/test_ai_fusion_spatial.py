"""Deterministic spatial tests for AI Fusion v1 (no model inference)."""

from services.ai_fusion_geometry import geometries_overlap, geometry_bbox, point_in_geometry
from services.ai_fusion_service import correlate_findings
from services.ai_fusion_synthetic import build_synthetic_evidence_findings


def test_synthetic_dataset_coordinate_consistency():
    findings = build_synthetic_evidence_findings()
    assert len(findings) == 11
    for item in findings:
        bbox = geometry_bbox(item["geometry"])
        assert bbox is not None
        assert item["score_calibration_status"] == "uncalibrated"


def test_overlap_neoplastic_inside_tumor_patch():
    findings = {f["id"]: f for f in build_synthetic_evidence_findings()}
    assert geometries_overlap(findings[102]["geometry"], findings[101]["geometry"])
    assert point_in_geometry(160, 160, findings[101]["geometry"])


def test_fusion_classifies_agreement_conflict_qc_and_singleton():
    result = correlate_findings(build_synthetic_evidence_findings())
    assert result["summary"]["evidence_count"] == 11
    assert result["parameters"]["produces_combined_diagnostic_confidence"] is False

    by_members = {
        frozenset(group["member_ids"]): group["classification"]
        for group in result["groups"]
    }

    assert by_members[frozenset({101, 102, 103})] == "Agreement"
    assert by_members[frozenset({201, 202})] == "Conflict"
    assert by_members[frozenset({301, 302})] == "QCAffected"
    assert by_members[frozenset({401})] == "Singleton"

    # Mitotic pair with opposing reviews + connective nucleus → conflict
    mitotic_group = next(
        group for group in result["groups"] if 502 in group["member_ids"] and 503 in group["member_ids"]
    )
    assert mitotic_group["classification"] == "Conflict"
    assert mitotic_group["combined_diagnostic_confidence"] is None
    assert "review_tension" in mitotic_group["rationale"]["relation_codes"]


def test_different_types_are_not_automatic_agreement():
    connective = {
        "id": 1,
        "finding_type": "nucleus",
        "label": "Connective",
        "feature_code": "HOVERNET",
        "model_name": "HoVer-Net",
        "model_version": "t",
        "score": 0.5,
        "review_status": "Unreviewed",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [50, 0], [50, 50], [0, 50], [0, 0]]]},
    }
    mitotic = {
        "id": 2,
        "finding_type": "mitotic_candidate",
        "label": "Mitotic candidate",
        "feature_code": "MITOTIC_DETECTION",
        "model_name": "KongNet",
        "model_version": "t",
        "score": 0.9,
        "review_status": "Unreviewed",
        "geometry": {"type": "Point", "coordinates": [25, 25]},
    }
    result = correlate_findings([connective, mitotic])
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["classification"] == "Related"
    assert group["rationale"]["has_agreement"] is False


def test_provenance_and_source_scores_preserved():
    result = correlate_findings(build_synthetic_evidence_findings())
    agreement = next(g for g in result["groups"] if g["classification"] == "Agreement")
    scores = {m["id"]: m["score"] for m in agreement["members"]}
    assert scores[101] == 0.82
    assert scores[102] == 0.91
    assert scores[103] == 0.77
    assert agreement["uncertainty"]["combined_diagnostic_confidence"] is None
    assert all(m["score_calibration_status"] == "uncalibrated" for m in agreement["members"])
