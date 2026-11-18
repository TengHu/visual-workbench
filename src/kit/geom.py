"""Moving pixels: lifting something off the picture, placing a layer, and pushing shapes around (liquify)."""
from __future__ import annotations

import cv2
import numpy as np

from .blend import blend
from .masks import mask_box

__all__ = ["cut", "place", "warp"]


def cut(img, mask):
    """Lift the masked part of a picture off as a layer (H, W, 4), to move, copy, scale or keep while you work on
    what is behind it. The picture itself is not changed; fill() the hole if the thing is to be gone from there."""
    a = np.asarray(img, np.float32)[..., :3]
    m = np.clip(np.asarray(mask, np.float32), 0, 1)
    if m.shape != a.shape[:2]:
        raise ValueError(f"mask is {m.shape[1]}x{m.shape[0]} but the picture is {a.shape[1]}x{a.shape[0]}")
    return np.dstack([a, m]).astype(np.float32)


def place(base, layer, at=None, scale=1.0, angle=0.0, flip=False, opacity=1.0, return_layer=False):
    """Put a layer onto a picture, moved, scaled, rotated or mirrored.

    layer  (h, w, 4), any size: something cut from the picture, drawn with skia, rendered with render_html, or loaded.
    at     (x, y) in the picture where the centre of the layer's visible part should land. None: keep its position
           (the layer must then be the picture's size).
    scale  1 = same size (a pair (sx, sy) stretches); angle in degrees, anticlockwise; flip mirrors left-right.
    return_layer=True returns the moved layer at the picture's size instead of compositing, so you can shade it,
    match its colour and grain, or build its shadow from its alpha before blend()-ing it in yourself."""
    b = np.asarray(base, np.float32)[..., :3]
    lay = np.asarray(layer, np.float32)
    if lay.ndim != 3 or lay.shape[2] != 4:
        raise ValueError("place: the layer must have an alpha channel, shape (h, w, 4)")
    H, W = b.shape[:2]
    bb = mask_box(lay[..., 3])
    if bb is None:
        raise ValueError("place: the layer is empty (alpha is 0 everywhere)")
    cx, cy = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
    if at is None:
        if lay.shape[:2] != (H, W) :
            raise ValueError("place: give at=(x, y); the layer is not the picture's size")
        tx, ty = cx, cy
    else:
        tx, ty = float(at[0]), float(at[1])
    sx, sy = (scale if np.ndim(scale) else (scale, scale))
    sx, sy = float(sx) * (-1.0 if flip else 1.0), float(sy)
    th = np.deg2rad(float(angle))
    c, s = np.cos(th), np.sin(th)
    # picture = R(angle) * S * (layer - centre) + target     (y is down, so anticlockwise uses -angle in maths terms)
    A = np.array([[c * sx, s * sy], [-s * sx, c * sy]], np.float64)
    M = np.hstack([A, np.array([[tx], [ty]]) - A @ np.array([[cx], [cy]])]).astype(np.float32)
    shrink = min(abs(sx), abs(sy)) < 0.7
    src = lay.copy()
    src[..., :3] *= src[..., 3:4]  # premultiply so edges do not pick up colour from outside the layer
    if shrink:
        k = max(1.0, 0.5 / min(abs(sx), abs(sy)))
        src = cv2.GaussianBlur(src, (0, 0), k * 0.6)
    out = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = np.clip(out[..., 3], 0, 1)
    rgb = np.clip(out[..., :3] / np.maximum(alpha, 1e-4)[..., None], 0, 1)
    moved = np.dstack([rgb, alpha]).astype(np.float32)
    if return_layer:
        return moved
    return blend(b, moved, opacity=opacity)


def warp(img, moves, radius, mask=None):
    """Liquify: push parts of the picture from one place to another, the rest stretching smoothly around them.

    moves   [((x_from, y_from), (x_to, y_to)), ...]: what is at `from` ends up at `to`.
    radius  how far around each push the picture gives way, in pixels (about the size of the feature being moved).
    mask    only pixels inside it may change (protect what must stay put).
    Small pushes look natural; a push longer than about half the radius smears. Works on a picture or a layer."""
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    for (p0, p1) in moves:
        x0, y0, x1, y1 = float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1])
        g = np.exp(-((xx - x1) ** 2 + (yy - y1) ** 2) / (2.0 * (float(radius) / 2.0) ** 2))
        dx += g * (x0 - x1)  # the output pixel near `to` reads from near `from`
        dy += g * (y0 - y1)
    if mask is not None:
        m = np.clip(np.asarray(mask, np.float32), 0, 1)
        dx, dy = dx * m, dy * m
    out = cv2.remap(a, xx + dx, yy + dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 1).astype(np.float32)
