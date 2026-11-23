"""Cached CPU inference for the NeuroMRI EfficientNet-B0 checkpoint.

A missing checkpoint or a missing PyTorch install leaves the rest of VISTA running.
"""

from datetime import datetime, timezone
import hashlib
import threading
import time
from pathlib import Path

from services.neuromri_preprocessing import (
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    eval_tensor,
)

CHECKPOINT = Path(__file__).resolve().parents[2] / "Ai models" / "neuromri" / "checkpoints" / "best_model.pth"
EXPECTED_SHA256 = "25c9ba05c6073d59912a39e596fed0fea173c9b8faeab8be3db123738491c723"
MODEL_VERSION = "epoch-9"

_LOCK = threading.Lock()
_INFER = threading.Semaphore(1)
_STATE = {"model": None, "epoch": None, "error": None, "checksum": None}


class NeuromriUnavailable(RuntimeError):
    pass


FEATURE_CODE = "NEUROMRI_SLICE_CLASSIFICATION"


def init_neuromri_catalog(connect_fn):
    """Register NeuroMRI on the AI catalog and show the CAMELYON card."""
    from contextlib import closing
    with closing(connect_fn()) as db:
        db.execute(
            """IF NOT EXISTS (SELECT 1 FROM ai_features WHERE feature_code = ?)
            INSERT INTO ai_features (
                feature_code, feature_name, description, is_implemented, is_active,
                is_catalog_visible, catalog_category, catalog_summary, catalog_purpose,
                catalog_value, catalog_title, catalog_display_order, catalog_outputs_json
            ) VALUES (
                ?, 'NeuroMRI EfficientNet-B0',
                'Research-only 10-class classification of one selected brain MRI slice.',
                1, 1, 1,
                'Radiology',
                'Classifies one selected 2D brain MRI slice into 10 dataset categories.',
                'Research slice classification for MR DICOM and verified MRI NIfTI.',
                'Scores are uncalibrated model outputs, not a diagnosis.',
                'NeuroMRI EfficientNet-B0',
                5,
                ?
            )""",
            (FEATURE_CODE, FEATURE_CODE, '["Class scores","Selected slice","Research use only"]'),
        )
        db.execute(
            """UPDATE ai_features
               SET is_catalog_visible = 1
               WHERE feature_code = 'CAMELYON_TUMOR_DETECTION' AND is_catalog_deleted = 0"""
        )
        db.commit()


def checkpoint_sha256():
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status():
    present = CHECKPOINT.is_file()
    checksum = None
    intact = False
    if present:
        try:
            checksum = checkpoint_sha256()
            intact = checksum == EXPECTED_SHA256
        except OSError:
            intact = False
    return {
        "model_name": "NeuroMRI EfficientNet-B0",
        "model_version": MODEL_VERSION,
        "available": present and intact,
        "loaded": _STATE["model"] is not None,
        "device": "cpu",
        "checkpoint_intact": intact,
        "classes": list(CLASS_NAMES),
        "research_use_only": True,
        "error": None if present and intact else "NeuroMRI checkpoint is missing or does not match the trained file",
    }


def _build():
    if not CHECKPOINT.is_file():
        raise NeuromriUnavailable("NeuroMRI checkpoint is not installed")
    checksum = checkpoint_sha256()
    if checksum != EXPECTED_SHA256:
        raise NeuromriUnavailable("NeuroMRI checkpoint does not match the trained file")
    try:
        import torch
        from torch import nn
        from torchvision.models import efficientnet_b0
    except ImportError as exc:
        raise NeuromriUnavailable("PyTorch is not available for NeuroMRI") from exc
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "efficientnet_b0":
        raise NeuromriUnavailable("NeuroMRI checkpoint is not EfficientNet-B0")
    if int(checkpoint.get("epoch", -1)) != 9:
        raise NeuromriUnavailable("NeuroMRI checkpoint epoch is not 9")
    if list(checkpoint.get("class_names") or []) != CLASS_NAMES:
        raise NeuromriUnavailable("NeuroMRI class mapping does not match training")
    if int(checkpoint.get("image_size") or 0) != IMAGE_SIZE:
        raise NeuromriUnavailable("NeuroMRI input size is not 224")
    if list(checkpoint.get("normalization_mean") or []) != list(IMAGENET_MEAN):
        raise NeuromriUnavailable("NeuroMRI normalization mean does not match training")
    if list(checkpoint.get("normalization_std") or []) != list(IMAGENET_STD):
        raise NeuromriUnavailable("NeuroMRI normalization std does not match training")
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _STATE["model"] = model
    _STATE["epoch"] = int(checkpoint["epoch"])
    _STATE["checksum"] = checksum
    _STATE["error"] = None
    return model


def get_model():
    if _STATE["model"] is not None:
        return _STATE["model"]
    with _LOCK:
        if _STATE["model"] is None:
            _build()
    return _STATE["model"]


def predict_image(image, source):
    import torch

    started = time.perf_counter()
    model = get_model()
    tensor = eval_tensor(image)
    with _INFER:
        with torch.inference_mode():
            logits = model(tensor)
            if logits.shape[-1] != len(CLASS_NAMES):
                raise NeuromriUnavailable("NeuroMRI did not return 10 class scores")
            scores = torch.softmax(logits.float(), dim=1)[0].cpu()
    label = int(scores.argmax().item())
    elapsed = time.perf_counter() - started
    return {
        "model_name": "NeuroMRI EfficientNet-B0",
        "model_version": MODEL_VERSION,
        "task": "brain_mri_slice_classification",
        "status": "completed",
        "predicted_class": CLASS_NAMES[label],
        "predicted_label": label,
        "top_softmax_score": float(scores[label].item()),
        "class_scores": {name: float(scores[index].item()) for index, name in enumerate(CLASS_NAMES)},
        "device": "cpu",
        "research_use_only": True,
        "score_meaning": "uncalibrated_model_score",
        "inference_seconds": round(elapsed, 4),
        "inferred_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }


def predict_path(path):
    from PIL import Image
    with Image.open(path) as image:
        loaded = image.convert("RGB")
    return predict_image(loaded, {"source_format": "image", "name": Path(path).name})
