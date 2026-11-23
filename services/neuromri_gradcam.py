"""Class-specific Grad-CAM for the cached NeuroMRI EfficientNet-B0.

The heatmap shows regions that raised the chosen class logit. It is not a
lesion segmentation and it does not change classification scores.
"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from services.neuromri_engine import _INFER, get_model
from services.neuromri_preprocessing import CLASS_NAMES, IMAGE_SIZE, eval_tensor

def _jet(values):
    """Map 0..1 to an RGB jet ramp without an extra colormap dependency."""
    stops = np.array([
        [0, 0, 128],
        [0, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 0],
        [128, 0, 0],
    ], dtype=np.float32)
    positions = np.linspace(0, 1, len(stops))
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    index = np.clip(np.searchsorted(positions, flat, side="right") - 1, 0, len(stops) - 2)
    span = positions[index + 1] - positions[index]
    weight = np.where(span > 0, (flat - positions[index]) / span, 0)
    color = stops[index] * (1 - weight[:, None]) + stops[index + 1] * weight[:, None]
    return np.clip(np.round(color), 0, 255).astype(np.uint8).reshape(values.shape + (3,))


def _content_box(width, height):
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    x0 = int(round(left / side * IMAGE_SIZE))
    y0 = int(round(top / side * IMAGE_SIZE))
    x1 = int(round((left + width) / side * IMAGE_SIZE))
    y1 = int(round((top + height) / side * IMAGE_SIZE))
    return max(0, x0), max(0, y0), min(IMAGE_SIZE, max(x0 + 1, x1)), min(IMAGE_SIZE, max(y0 + 1, y1))


def _png(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def explain_image(image, target_class=None):
    """Return a source-aligned Grad-CAM for one class logit."""
    import torch
    import torch.nn.functional as F

    image = image.convert("RGB")
    width, height = image.size
    model = get_model()
    model.eval()
    names = list(CLASS_NAMES)
    tensor = eval_tensor(image).detach().requires_grad_(True)
    layer = model.features[-1]
    captured = {}

    def save_activation(_module, _inputs, output):
        captured["activation"] = output

    def save_gradient(_module, _grad_input, grad_output):
        captured["gradient"] = grad_output[0]

    with _INFER:
        forward = layer.register_forward_hook(save_activation)
        backward = layer.register_full_backward_hook(save_gradient)
        try:
            with torch.enable_grad():
                logits = model(tensor)
                if logits.shape[-1] != len(names):
                    raise RuntimeError("NeuroMRI did not return 10 class scores")
                scores = torch.softmax(logits.detach().float(), dim=1)[0]
                predicted = int(scores.argmax().item())
                if target_class:
                    if target_class not in names:
                        raise ValueError(f"Unknown NeuroMRI class: {target_class}")
                    target = names.index(target_class)
                else:
                    target = predicted
                model.zero_grad(set_to_none=True)
                logits[0, target].backward()
                activation = captured.get("activation")
                gradient = captured.get("gradient")
                if activation is None or gradient is None:
                    raise RuntimeError("Grad-CAM did not receive feature maps")
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
                heat = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        finally:
            forward.remove()
            backward.remove()
            model.zero_grad(set_to_none=True)
            captured.clear()
            tensor.grad = None

    if not np.isfinite(heat).all():
        raise RuntimeError("Grad-CAM produced a non-finite heatmap")
    low = float(heat.min())
    high = float(heat.max())
    if high > low:
        heat = (heat - low) / (high - low)
    else:
        heat = np.zeros_like(heat)
    x0, y0, x1, y1 = _content_box(width, height)
    cropped = heat[y0:y1, x0:x1]
    aligned = np.array(
        Image.fromarray(cropped, mode="F").resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    aligned = np.clip(aligned, 0, 1)
    color = _jet(aligned)
    alpha = np.round(aligned * 255).astype(np.uint8)
    overlay = np.dstack([color, alpha])
    return {
        "explanation_method": "Grad-CAM",
        "model_name": "NeuroMRI EfficientNet-B0",
        "predicted_class": names[predicted],
        "predicted_label": predicted,
        "target_class": names[target],
        "target_class_index": target,
        "target_softmax_score": float(scores[target].item()),
        "class_scores": {name: float(scores[index].item()) for index, name in enumerate(names)},
        "heatmap_png_base64": base64.b64encode(_png(color)).decode("ascii"),
        "overlay_png_base64": base64.b64encode(_png(overlay)).decode("ascii"),
        "width": width,
        "height": height,
        "value_min": 0.0,
        "value_max": 1.0,
        "research_only": True,
        "ground_truth": "Not available",
        "disclaimer": "Grad-CAM highlights image regions influencing the model output. It is not a validated lesion segmentation or diagnosis.",
    }
