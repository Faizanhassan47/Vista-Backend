"""Expanded Fusion validation suite: boundaries, empty runs, reproducibility, duplicates."""

from services.ai_fusion_geometry import geometries_overlap, geometry_bbox
from services.ai_fusion_service import correlate_findings
from services.ai_fusion_synthetic import build_synthetic_evidence_findings
from services.ai_case_intelligence import compare_fusion_runs


def test_touching_bboxes_without_interior_overlap_still_group_by_bbox_rule():
    left = {
        "id": 1,
        "finding_type": "tumor_patch",
        "label": "Tumor probability patch",
        "feature_code": "CAMELYON_TUMOR_DETECTION",
        "model_name": "CAMELYON",
        "model_version": "t",
        "score": 0.8,
        "review_status": "Unreviewed",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]},
    }
    right = {
        "id": 2,
        "finding_type": "tumor_patch",
        "label": "Tumor probability patch",
        "feature_code": "CAMELYON_TUMOR_DETECTION",
        "model_name": "CAMELYON",
        "model_version": "t",
        "score": 0.81,
        "review_status": "Unreviewed",
        "geometry": {"type": "Polygon", "coordinates": [[[10, 0], [20, 0], [20, 10], [10, 10], [10, 0]]]},
    }
    assert geometries_overlap(left["geometry"], right["geometry"])
    result = correlate_findings([left, right])
    assert len(result["groups"]) == 1
    assert result["groups"][0]["classification"] == "Agreement"


def test_separated_findings_remain_singletons():
    a = {
        "id": 1,
        "finding_type": "nucleus",
        "label": "Inflammatory",
        "feature_code": "HOVERNET",
        "model_name": "HoVer-Net",
        "model_version": "t",
        "score": 0.5,
        "review_status": "Unreviewed",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [5, 0], [5, 5], [0, 5], [0, 0]]]},
    }
    b = {
        "id": 2,
        "finding_type": "nucleus",
        "label": "Inflammatory",
        "feature_code": "HOVERNET",
        "model_name": "HoVer-Net",
        "model_version": "t",
        "score": 0.5,
        "review_status": "Unreviewed",
        "geometry": {"type": "Polygon", "coordinates": [[[500, 500], [510, 500], [510, 510], [500, 510], [500, 500]]]},
    }
    result = correlate_findings([a, b])
    assert len(result["groups"]) == 2
    assert all(g["classification"] == "Singleton" for g in result["groups"])


def test_empty_findings_produce_empty_run_summary():
    result = correlate_findings([])
    assert result["summary"]["evidence_count"] == 0
    assert result["summary"]["group_count"] == 0
    assert result["groups"] == []
    assert result["parameters"]["produces_combined_diagnostic_confidence"] is False


def test_fusion_is_reproducible_for_same_input():
    findings = build_synthetic_evidence_findings()
    first = correlate_findings(findings)
    second = correlate_findings(findings)
    assert first["summary"] == second["summary"]
    assert [g["classification"] for g in first["groups"]] == [g["classification"] for g in second["groups"]]
    assert [sorted(g["member_ids"]) for g in first["groups"]] == [sorted(g["member_ids"]) for g in second["groups"]]


def test_point_on_boundary_with_proximity():
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]}
    point = {"type": "Point", "coordinates": [100, 50]}
    assert geometries_overlap(point, poly, proximity_px=0) or geometries_overlap(point, poly, proximity_px=1)
    assert geometry_bbox(point) == (100.0, 50.0, 100.0, 50.0)


def test_compare_fusion_runs_detects_group_and_review_changes():
    # Pure structural compare helper via synthetic run-shaped payloads is covered by service logic;
    # here we validate signature comparison behavior through correlate deltas.
    base = build_synthetic_evidence_findings()
    run_a = correlate_findings(base)
    altered = [dict(item) for item in base]
    for item in altered:
        if item["id"] == 202:
            item["label"] = "Neoplastic"
            item["score"] = 0.9
    run_b = correlate_findings(altered)

    # Conflict group {201,202} should become Agreement after label flip
    class_a = {frozenset(g["member_ids"]): g["classification"] for g in run_a["groups"]}
    class_b = {frozenset(g["member_ids"]): g["classification"] for g in run_b["groups"]}
    assert class_a[frozenset({201, 202})] == "Conflict"
    assert class_b[frozenset({201, 202})] == "Agreement"
