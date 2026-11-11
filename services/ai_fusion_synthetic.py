"""Deterministic synthetic Evidence Map findings for Fusion spatial tests."""

from __future__ import annotations


def _poly(x0, y0, x1, y1):
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _point(x, y):
    return {"type": "Point", "coordinates": [x, y]}


def build_synthetic_evidence_findings() -> list[dict]:
    """Overlapping nuclei, mitotic candidates, tumor patches and QC polygons.

    Layout (level-0 pixels):
      Region A (agreement): neoplastic nucleus + high tumor patch + mitotic
      Region B (conflict): non-neoplastic nucleus + high tumor patch
      Region C (QC): neoplastic nucleus inside Fold QC polygon
      Region D (singleton): isolated inflammatory nucleus
      Region E (related only): mitotic outside tumor thresholds near connective nucleus
    """
    findings = [
        {
            "id": 101,
            "finding_type": "tumor_patch",
            "label": "Tumor probability patch",
            "feature_code": "CAMELYON_TUMOR_DETECTION",
            "model_name": "CAMELYON",
            "model_version": "synth-1",
            "score": 0.82,
            "score_calibration_status": "uncalibrated",
            "review_status": "Unreviewed",
            "geometry": _poly(100, 100, 324, 324),
        },
        {
            "id": 102,
            "finding_type": "nucleus",
            "label": "Neoplastic",
            "feature_code": "HOVERNET",
            "model_name": "HoVer-Net FAST",
            "model_version": "synth-1",
            "score": 0.91,
            "score_calibration_status": "uncalibrated",
            "review_status": "Accepted",
            "geometry": _poly(140, 140, 180, 180),
        },
        {
            "id": 103,
            "finding_type": "mitotic_candidate",
            "label": "Mitotic candidate",
            "feature_code": "MITOTIC_DETECTION",
            "model_name": "KongNet-Det",
            "model_version": "synth-1",
            "score": 0.77,
            "score_calibration_status": "uncalibrated",
            "review_status": "Unreviewed",
            "geometry": _point(160, 160),
        },
        {
            "id": 201,
            "finding_type": "tumor_patch",
            "label": "Tumor probability patch",
            "feature_code": "CAMELYON_TUMOR_DETECTION",
            "model_name": "CAMELYON",
            "model_version": "synth-1",
            "score": 0.88,
            "score_calibration_status": "uncalibrated",
            "review_status": "Unreviewed",
            "geometry": _poly(500, 100, 724, 324),
        },
        {
            "id": 202,
            "finding_type": "nucleus",
            "label": "Non-neoplastic",
            "feature_code": "HOVERNET",
            "model_name": "HoVer-Net FAST",
            "model_version": "synth-1",
            "score": 0.86,
            "score_calibration_status": "uncalibrated",
            "review_status": "Accepted",
            "geometry": _poly(560, 160, 600, 200),
        },
        {
            "id": 301,
            "finding_type": "qc_polygon",
            "label": "Fold",
            "feature_code": "SLIDE_QUALITY_CONTROL",
            "model_name": "GrandQC",
            "model_version": "synth-1",
            "score": None,
            "score_calibration_status": "uncalibrated",
            "review_status": "Unreviewed",
            "geometry": _poly(900, 100, 1100, 300),
        },
        {
            "id": 302,
            "finding_type": "nucleus",
            "label": "Neoplastic",
            "feature_code": "HOVERNET",
            "model_name": "HoVer-Net FAST",
            "model_version": "synth-1",
            "score": 0.8,
            "score_calibration_status": "uncalibrated",
            "review_status": "Unreviewed",
            "geometry": _poly(950, 150, 990, 190),
        },
        {
            "id": 401,
            "finding_type": "nucleus",
            "label": "Inflammatory",
            "feature_code": "HOVERNET",
            "model_name": "HoVer-Net FAST",
            "model_version": "synth-1",
            "score": 0.7,
            "score_calibration_status": "uncalibrated",
            "review_status": "Unreviewed",
            "geometry": _poly(50, 800, 90, 840),
        },
        {
            "id": 501,
            "finding_type": "nucleus",
            "label": "Connective",
            "feature_code": "HOVERNET",
            "model_name": "HoVer-Net FAST",
            "model_version": "synth-1",
            "score": 0.65,
            "score_calibration_status": "uncalibrated",
            "review_status": "Unreviewed",
            "geometry": _poly(200, 600, 240, 640),
        },
        {
            "id": 502,
            "finding_type": "mitotic_candidate",
            "label": "Mitotic candidate",
            "feature_code": "MITOTIC_DETECTION",
            "model_name": "KongNet-Det",
            "model_version": "synth-1",
            "score": 0.6,
            "score_calibration_status": "uncalibrated",
            "review_status": "Rejected",
            "geometry": _point(220, 620),
        },
        {
            "id": 503,
            "finding_type": "mitotic_candidate",
            "label": "Mitotic candidate",
            "feature_code": "MITOTIC_DETECTION",
            "model_name": "KongNet-Det",
            "model_version": "synth-1",
            "score": 0.55,
            "score_calibration_status": "uncalibrated",
            "review_status": "Accepted",
            "geometry": _point(222, 622),
        },
    ]
    return findings
