"""Selections. A mask says where an edit applies: float32 (H, W), 1 inside, 0 outside, soft in between.

Combine masks with plain arithmetic: union np.maximum(a, b), overlap a * b, a without b  a * (1 - b), inverse 1 - a.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from scipy import ndimage as ndi

from .core import as_color, rgb2oklab

__all__ = [
    "rect_mask", "ellipse_mask", "poly_mask", "curve_mask", "stroke_mask",
    "select_object", "select_color", "refine_edge",
    "grow", "feather", "fill_holes", "keep_largest", "mask_box",
]

_SHIFT = 4  # sub-pixel precision of drawn shapes: 1/16 px


def _hw(like):
    if isinstance(like, (tuple, list)) and len(like) == 2 and all(isinstance(v, (int, np.integer)) for v in like):
        return int(like[0]), int(like[1])
    return np.asarray(like).shape[:2]


def _fix(points):
    return np.round(np.asarray(points, np.float64).reshape(-1, 2) * (1 << _SHIFT)).astype(np.int32)


def _soft(m8, feather_px):
    m = m8.astype(np.float32) / 255.0
    return feather(m, feather_px) if feather_px else m


def rect_mask(like, box, feather_px=0):
    """A rectangle. `like` is the picture (or (H, W)) the mask is for; box is (x0, y0, x1, y1)."""
    h, w = _hw(like)
    x0, y0, x1, y1 = box
    return poly_mask((h, w), [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], feather_px)


def ellipse_mask(like, box, feather_px=0, angle=0.0):
    """An ellipse filling the box (x0, y0, x1, y1), optionally rotated by `angle` degrees."""
    h, w = _hw(like)
    x0, y0, x1, y1 = box
    m = np.zeros((h, w), np.uint8)
    c = _fix([((x0 + x1) / 2, (y0 + y1) / 2)])[0]
    ax = _fix([(abs(x1 - x0) / 2, abs(y1 - y0) / 2)])[0]
    cv2.ellipse(m, (int(c[0]), int(c[1])), (int(ax[0]), int(ax[1])), float(angle), 0, 360, 255, -1, cv2.LINE_AA, _SHIFT)
    return _soft(m, feather_px)


def poly_mask(like, points, feather_px=0):
    """A polygon through the points [(x, y), ...], in order. For a curved outline, give many points."""
    h, w = _hw(like)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [_fix(points)], 255, cv2.LINE_AA, _SHIFT)
    return _soft(m, feather_px)


def _curve(points, corners=(), closed=True, per=24):
    """A smooth curve through the points (centripetal Catmull-Rom: no loops, no overshoot between close points);
    at the indices in `corners` it turns sharply instead."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    n = len(p)
    if n < 3:
        return p
    sharp = {int(i) % n for i in corners}
    out = []
    segs = n if closed else n - 1
    for i in range(segs):
        i1, i2 = i, (i + 1) % n
        i0 = (i - 1) % n if (closed or i > 0) else 0
        i3 = (i + 2) % n if (closed or i + 2 < n) else n - 1
        P0, P1, P2, P3 = p[i0], p[i1], p[i2], p[i3]
        if i1 in sharp:
            P0 = P1 - (P2 - P1) * 1e-3  # a corner: the curve leaves it heading straight for the next point
        if i2 in sharp:
            P3 = P2 + (P2 - P1) * 1e-3
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(P1 - P0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(P2 - P1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(P3 - P2), 1e-6) ** 0.5
        for t in np.linspace(t1, t2, per, endpoint=False):
            A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
            A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
            A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3
            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
            out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    if not closed:
        out.append(p[-1])
    return np.asarray(out)


def curve_mask(like, points, corners=(), feather_px=0):
    """A closed shape whose outline runs smoothly through the points [(x, y), ...], in order round the shape.
    The curve passes through every point, so more points follow a more detailed outline. `corners` lists the
    indices of points where the outline turns sharply instead of curving."""
    return poly_mask(like, _curve(points, corners, closed=True), feather_px)


def stroke_mask(like, points, width, feather_px=0):
    """A brush stroke `width` pixels wide along the points [(x, y), ...]: for wires, cracks, straps, edges."""
    h, w = _hw(like)
    m = np.zeros((h, w), np.uint8)
    pts = _fix(points)
    if len(pts) == 1:
        cv2.circle(m, (int(pts[0][0]), int(pts[0][1])), int(round(width / 2 * (1 << _SHIFT))), 255, -1, cv2.LINE_AA, _SHIFT)
    else:
        cv2.polylines(m, [pts], False, 255, max(1, int(round(width))), cv2.LINE_AA, _SHIFT)
    return _soft(m, feather_px)


# ---------------------------------------------------------------------------------------------------------------
# Shaping a mask


def grow(mask, px):
    """Move a mask's edge outward by px pixels (negative: inward). The result has a crisp edge; feather it afterwards."""
    hard = np.asarray(mask, np.float32) > 0.5
    if not hard.any() or hard.all() or px == 0:
        return hard.astype(np.float32) if px else np.asarray(mask, np.float32).copy()
    signed = cv2.distanceTransform((~hard).astype(np.uint8), cv2.DIST_L2, 5) - cv2.distanceTransform(hard.astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(px - signed + 0.5, 0, 1).astype(np.float32)


def feather(mask, px, inward=False):
    """Soften a mask's edge over about px pixels. inward=True keeps the soft edge inside the original outline."""
    m = np.asarray(mask, np.float32)
    if px <= 0:
        return m.copy()
    if inward:
        m = grow(m, -px)
    return np.clip(cv2.GaussianBlur(m, (0, 0), px / 2.0), 0, 1)


def fill_holes(mask):
    """Close the holes inside a mask."""
    m = np.asarray(mask, np.float32)
    return np.maximum(m, ndi.binary_fill_holes(m > 0.5).astype(np.float32))


def keep_largest(mask, n=1):
    """Keep only the n largest connected pieces of a mask."""
    m = np.asarray(mask, np.float32)
    lab, k = ndi.label(m > 0.5)
    if k <= n:
        return m.copy()
    sizes = ndi.sum_labels(np.ones_like(lab), lab, np.arange(1, k + 1))
    keep = np.isin(lab, 1 + np.argsort(sizes)[::-1][:n])
    return m * grow(keep.astype(np.float32), 2)


def mask_box(mask, pad=0):
    """The box (x0, y0, x1, y1) around everything a mask selects, widened by pad pixels. None if it selects nothing."""
    m = np.asarray(mask) > 0.02
    if not m.any():
        return None
    ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
    h, w = m.shape
    return (max(0, int(xs[0]) - pad), max(0, int(ys[0]) - pad), min(w, int(xs[-1]) + 1 + pad), min(h, int(ys[-1]) + 1 + pad))


def refine_edge(img, mask, radius=6, strength=1e-4):
    """Snap a mask's edge to the picture's own edges and give it their softness (hair, fur, motion blur, soft focus).

    radius is how far, in pixels, the true edge may be from the mask's edge. Larger for hair and fur (12 to 30)."""
    m = np.asarray(mask, np.float32)
    g = np.ascontiguousarray(np.asarray(img, np.float32)[..., :3])
    out = cv2.ximgproc.guidedFilter(g, m, int(max(1, radius)), float(strength))
    out = np.clip((out - 0.02) / 0.96, 0, 1)
    # Far from the old edge the mask keeps its value: only the edge zone is redrawn.
    hard = (m > 0.5).astype(np.uint8)
    dist = np.maximum(cv2.distanceTransform(hard, cv2.DIST_L2, 3), cv2.distanceTransform(1 - hard, cv2.DIST_L2, 3))
    t = np.clip((dist - radius) / max(1.0, radius * 0.5), 0, 1)
    return (out * (1 - t) + m * t).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Selecting by colour


def select_color(img, at=None, color=None, tolerance=0.06, shading=0.3, contiguous=True, within=None):
    """Select pixels of one colour (a magic wand). Give a point `at=(x, y)` to sample, or a `color`.

    tolerance  the colour difference (Oklab) still fully selected: 0.03 tight, 0.06 normal, 0.12 loose. It fades
               out to 1.5x that.
    shading    how much a difference in brightness counts. 1: a strict wand, the same tone only. 0: hue and
               saturation only, so one painted surface is selected through its shadows, folds and highlights
               (but so is anything else of that hue). Lower it when a lit surface comes out striped or patchy.
    contiguous=True keeps only the area connected to `at`. within: a mask or box to stay inside."""
    lab = rgb2oklab(img)
    h, w = lab.shape[:2]
    if at is not None:
        x, y = int(round(at[0])), int(round(at[1]))
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"point {at} is outside the {w}x{h} picture")
        ref = np.median(lab[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3), 0)
    elif color is not None:
        ref = rgb2oklab(as_color(color).reshape(1, 1, 3))[0, 0]
    else:
        raise ValueError("give at=(x, y) or color=")
    strict = ((lab - ref) ** 2).sum(-1)
    # Shading scales lightness and chroma together, so colour per unit of lightness is what a surface keeps.
    Lr = max(float(ref[0]), 0.1)
    rel = lab[..., 1:] / np.maximum(lab[..., :1], 0.1) * Lr
    loose = ((rel - ref[1:]) ** 2).sum(-1)
    s = float(np.clip(shading, 0, 1))
    d = np.sqrt(s * strict + (1 - s) * loose)
    m = np.clip((1.5 * tolerance - d) / (0.5 * tolerance), 0, 1).astype(np.float32)
    if within is not None:
        m = m * (rect_mask((h, w), within) if np.ndim(within) == 1 else np.asarray(within, np.float32))
    if contiguous and at is not None:
        lab_, _ = ndi.label(m > 0)
        m = m * (lab_ == lab_[y, x]) if lab_[y, x] else m * 0
    return m.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Selecting an object


def _discs(shape, pts, r):
    m = np.zeros(shape, np.uint8)
    for (x, y) in pts:
        cv2.circle(m, (int(round(x)), int(round(y))), int(r), 1, -1)
    return m > 0


def _grabcut(img, box, fg, bg):
    """Colour-model graph cut seeded by a box and by points. No learned model; works when the object differs in
    colour or brightness from what is around it."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    pad = int(0.3 * max(x1 - x0, y1 - y0)) + 8
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    crop = (np.clip(img[cy0:cy1, cx0:cx1, :3], 0, 1) * 255).astype(np.uint8)[..., ::-1]
    s = min(1.0, 720.0 / max(crop.shape[:2]))
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop
    sh, sw = small.shape[:2]
    to = lambda x, y: ((x - cx0) * s, (y - cy0) * s)
    bx0, by0 = to(x0, y0)
    bx1, by1 = to(x1, y1)
    gc = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    gc[int(max(0, by0)):int(np.ceil(min(sh, by1))), int(max(0, bx0)):int(np.ceil(min(sw, bx1)))] = cv2.GC_PR_FGD
    r = max(3, int(0.02 * max(bx1 - bx0, by1 - by0)))
    if len(fg):
        gc[_discs((sh, sw), [to(*p) for p in fg], r)] = cv2.GC_FGD
    if len(bg):
        gc[_discs((sh, sw), [to(*p) for p in bg], r)] = cv2.GC_BGD
    if not ((gc == cv2.GC_BGD).any()):
        gc[0, :] = gc[-1, :] = cv2.GC_BGD
    cv2.grabCut(np.ascontiguousarray(small), gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
    m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)
    m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((h, w), np.float32)
    full[cy0:cy1, cx0:cx1] = m
    return full


def select_object(img, box, fg=(), bg=(), refine=True):
    """Select the object inside a box, following its outline.

    box  (x0, y0, x1, y1) drawn snugly around the whole object.
    fg   points [(x, y), ...] that are on the object; bg: points that are not. Add them when the first result
         takes too much or too little, and call again.
    show(overlay(img, m), box=...) shows what was selected."""
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        raise ValueError(f"box {tuple(box)} covers no pixels of the {w}x{h} picture")
    fg, bg = [tuple(p) for p in fg], [tuple(p) for p in bg]
    method = "colour graph cut"
    m = None
    if os.environ.get("KIT_SAM", "1") != "0":
        try:
            from . import sam
            if sam.available():
                m = sam.segment(a, (x0, y0, x1, y1), fg, bg)
                method = "SAM 2.1"
        except Exception as e:  # the learned helper is optional; never let it stop an edit
            print(f"(SAM unavailable: {type(e).__name__}: {e}; using the colour graph cut)")
    if m is None:
        m = _grabcut(a, (x0, y0, x1, y1), fg, bg)
    if refine:
        m = refine_edge(a, m, radius=max(2, int(0.004 * max(x1 - x0, y1 - y0)) + 2))
    area = float((m > 0.5).sum())
    bb = mask_box(m)
    print(f"select_object ({method}): {area:.0f} px selected, {100 * area / max(1, (x1 - x0) * (y1 - y0)):.0f}% of the box" + (f", spans x {bb[0]}..{bb[2]}, y {bb[1]}..{bb[3]}" if bb else " (nothing)"))
    return m.astype(np.float32)
