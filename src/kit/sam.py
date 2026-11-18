"""The one learned helper, and it is optional: Segment Anything 2.1 (Meta, Apache-2.0) turns a box and a few points
into the exact outline of the thing they indicate. It only selects; it makes no pixels. It runs on this machine, from
weights that setup.sh downloaded, and nothing leaves the machine. Without it select_object() falls back to a colour
graph cut."""
from __future__ import annotations

import hashlib
import os

import numpy as np

_state: dict = {"predictor": None, "device": None, "key": None}


def _weights() -> str:
    root = os.environ.get("KIT_MODELS") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".models")
    return os.path.join(root, "sam2.1_hiera_large.pt")


def available() -> bool:
    if not os.path.exists(_weights()):
        return False
    try:
        import sam2  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _predictor():
    if _state["predictor"] is None:
        import logging

        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        logging.getLogger("root").setLevel(logging.ERROR)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        try:
            model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", _weights(), device=device)
        except Exception:
            if device == "cpu":
                raise
            device = "cpu"
            model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", _weights(), device=device)
        _state["predictor"], _state["device"] = SAM2ImagePredictor(model), device
    return _state["predictor"]


def segment(img, box, fg=(), bg=()):
    """Mask (H, W) float 0..1 of the object in `box` (x0, y0, x1, y1), steered by points on it (fg) and off it (bg)."""
    import torch

    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    # Work on a window around the box, so the object gets most of the model's fixed 1024 px input.
    pad = int(0.35 * max(x1 - x0, y1 - y0)) + 16
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    crop = (np.clip(img[cy0:cy1, cx0:cx1, :3], 0, 1) * 255).astype(np.uint8)
    pred = _predictor()
    key = hashlib.sha1(crop[::8, ::8].tobytes() + repr((crop.shape, cx0, cy0)).encode()).hexdigest()
    with torch.inference_mode():
        if _state["key"] != key:
            pred.set_image(crop)
            _state["key"] = key
        pts = [(x - cx0, y - cy0) for x, y in fg] + [(x - cx0, y - cy0) for x, y in bg]
        logits, scores, _ = pred.predict(
            point_coords=np.array(pts, np.float32) if pts else None,
            point_labels=np.array([1] * len(fg) + [0] * len(bg), np.int32) if pts else None,
            box=np.array([x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0], np.float32),
            multimask_output=False,
            return_logits=True,
        )
    lg = np.asarray(logits[int(np.argmax(scores))], np.float32)
    m = 1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30)))
    full = np.zeros((h, w), np.float32)
    full[cy0:cy1, cx0:cx1] = m
    return full
