"""Checking: an edit against the original (what changed, where, by how much), and a drawing against the faults
that can be measured."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from .blend import GRAIN_FLOOR, grain_level
from .core import as_color, rgb2oklab
from .masks import mask_box

__all__ = ["changes", "change_map", "edit_report", "drawing_report", "sharpness_level", "tones", "squint", "mirror"]


def _delta(a, b):
    a = np.asarray(a, np.float32)[..., :3]
    b = np.asarray(b, np.float32)[..., :3]
    if a.shape != b.shape:
        raise ValueError(f"the two pictures differ in size: {a.shape[1]}x{a.shape[0]} and {b.shape[1]}x{b.shape[0]}")
    return np.abs(a - b).max(-1)


def changes(before, after, mask=None, quiet=False):
    """Measure what an edit changed. Prints, and returns a dict with:

    changed  share of pixels that differ visibly (by more than 1/255)
    box      (x0, y0, x1, y1) around everything that changed, or None
    areas    boxes of the separate changed areas, largest first (up to 6)
    outside  with a mask of where the edit was meant to be: the largest change outside it (0 means it did not leak)
    A local edit must leave the rest of the picture exactly as it was."""
    d = _delta(before, after)
    ch = d > 1.0 / 255 + 1e-6
    out = {"changed": float(ch.mean()), "box": mask_box(ch.astype(np.float32)), "max": float(d.max()), "areas": []}
    if ch.any():
        k = max(3, int(0.01 * max(ch.shape)) | 1)
        lab, n = ndi.label(cv2.dilate(ch.astype(np.uint8), np.ones((k, k), np.uint8)) > 0)
        sizes = ndi.sum_labels(ch, lab, np.arange(1, n + 1))
        for i in np.argsort(sizes)[::-1][:6]:
            out["areas"].append(mask_box(((lab == i + 1) & ch).astype(np.float32)))
    if mask is not None:
        m = np.asarray(mask, np.float32)
        outside = d * (m <= 0.0)
        out["outside"] = float(outside.max())
        out["outside_pixels"] = int((outside > 1.0 / 255 + 1e-6).sum())
    if not quiet:
        msg = f"changes: {100 * out['changed']:.2f}% of pixels differ"
        if out["box"]:
            b = out["box"]
            msg += f", all within x {b[0]}..{b[2]}, y {b[1]}..{b[3]}, in {len(out['areas'])} area(s); largest difference {out['max']:.3f}"
        if mask is not None:
            msg += f"; outside the mask: {out['outside_pixels']} pixels changed (max {out['outside']:.3f})"
        print(msg)
    return out


def change_map(before, after, gain=4.0):
    """A picture of where and how strongly an edit changed things (black: untouched; bright: changed), to show().
    Faint, wide-spread grey where nothing should have changed means the edit leaked."""
    d = _delta(before, after)
    v = np.clip(d * gain, 0, 1)
    heat = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32) / 255
    heat[d <= 1.0 / 255 + 1e-6] = 0
    return heat


def sharpness_level(img, mask=None):
    """How sharp a picture is in the masked area: the typical strength of its strongest edges (0 = flat or blurred).
    Compare an added or edited area with what surrounds it at the same distance: they should be close."""
    g = np.asarray(img, np.float32)[..., :3].mean(-1)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.8), cv2.CV_32F, ksize=3))
    w = np.asarray(mask, np.float32) > 0.5 if mask is not None else np.ones(g.shape, bool)
    if not w.any():
        raise ValueError("sharpness_level: the mask selects nothing")
    return float(np.percentile(lap[w], 95))


def edit_report(before, after, quiet=False):
    """Numbers on an edit, comparing each changed area with the ring of untouched picture around it: how much
    changed, the grain inside against outside, the sharpness inside against outside, and how strongly the border of
    the change shows. A grain or sharpness ratio far from 1, or a strong border, is an edit that can be seen."""
    d = _delta(before, after)
    ch = d > 1.0 / 255 + 1e-6
    out = {"changed": float(ch.mean())}
    if not ch.any() or ch.mean() > 0.9:
        if not quiet:
            print(f"edit_report: {100 * out['changed']:.1f}% of pixels changed" + ("" if ch.any() else " (nothing)"))
        return out
    k = max(5, int(0.012 * max(ch.shape)) | 1)
    area = cv2.morphologyEx(ch.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
    ring = (cv2.dilate(area.astype(np.uint8), np.ones((6 * k + 1, 6 * k + 1), np.uint8)) > 0) & ~area
    a = np.asarray(after, np.float32)[..., :3]
    if ring.sum() > 200 and area.sum() > 200:
        gi, go = grain_level(a, area.astype(np.float32)), grain_level(a, ring.astype(np.float32))
        si, so = sharpness_level(a, area.astype(np.float32)), sharpness_level(a, ring.astype(np.float32))
        # Floored at the 8-bit rounding level: on a nearly grainless photo the ratio of two tiny numbers would swing
        # with every save.
        out.update(grain_ratio=(gi + GRAIN_FLOOR) / (go + GRAIN_FLOOR), sharpness_ratio=si / max(so, 1e-5))
        edge = (cv2.dilate(area.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~area
        g = a.mean(-1)
        grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        gb = np.hypot(cv2.Sobel(np.asarray(before, np.float32)[..., :3].mean(-1), cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(np.asarray(before, np.float32)[..., :3].mean(-1), cv2.CV_32F, 0, 1, ksize=3))
        out["border"] = float(np.clip(grad[edge].mean() - gb[edge].mean(), 0, None))
    if not quiet:
        msg = f"edit_report: {100 * out['changed']:.1f}% of pixels changed"
        if "grain_ratio" in out:
            msg += f"; grain inside/outside x{out['grain_ratio']:.2f}; sharpness inside/outside x{out['sharpness_ratio']:.2f}; new edge along the border {out['border']:.3f}"
        print(msg)
    return out


def drawing_report(img, paper="#f6f2e9", quiet=False):
    """Numbers on a drawing or painting, for the faults that can be measured. Prints, and returns a dict with:

    bare    share of the sheet that is still untouched paper (right for a pen drawing; a fault in a painting)
    flat    share of the picture lying in areas of one unvarying tone, with no light and shade in them, and
            `flat_boxes`, the largest such areas. A form drawn as a flat silhouette looks cut out of paper: give
            it volume (form()), a gradient, or a shadow falling across it. A flat poster style is the exception.
    tones   the darkest and lightest tones used (0 black, 1 white) and the shares of dark, middle and light. A
            picture squeezed into the middle tones, with no real dark and no real light, looks weak.
    It does not see drawing errors, speckle or bad taste: those take looking (squint, tones, mirror, close-ups)."""
    a = np.asarray(img, np.float32)[..., :3]
    H, W = a.shape[:2]
    s = min(1.0, 800.0 / max(H, W))
    if s < 1:
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = a.shape[:2]
    bare = np.abs(a - as_color(paper)).max(-1) < 3.5 / 255
    try:
        sm = cv2.ximgproc.rollingGuidanceFilter(a, d=-1, sigmaColor=0.12, sigmaSpace=7.0, numOfIter=4)
    except (AttributeError, cv2.error):
        sm = cv2.bilateralFilter(a, -1, 0.12, 7.0)
    lab = rgb2oklab(sm)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 5)
    # Areas: connected parts across which the smoothed colour does not step.
    step = np.abs(cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)).sum(-1) + np.abs(cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)).sum(-1)
    comp, n = ndi.label(step < 0.06)
    flat = np.zeros((h, w), bool)
    boxes = []
    if n:
        sizes = ndi.sum_labels(np.ones((h, w), np.float32), comp, np.arange(1, n + 1))
        for i in np.argsort(sizes)[::-1]:
            if sizes[i] < 0.01 * h * w:
                break
            sel = (comp == i + 1) & ~bare
            if sel.sum() < 0.01 * h * w:
                continue
            v = L[sel]
            if np.percentile(v, 95) - np.percentile(v, 5) < 0.035:
                flat |= sel
                b = mask_box(sel.astype(np.float32))
                boxes.append(tuple(int(round(c / s)) for c in b))
    Lr = rgb2oklab(a)[..., 0][~bare] if (~bare).any() else rgb2oklab(a)[..., 0].ravel()
    out = {"bare": float(bare.mean()), "flat": float(flat.mean()), "flat_boxes": boxes[:4],
           "darkest": float(np.percentile(Lr, 2)), "lightest": float(np.percentile(Lr, 98)),
           "dark": float((Lr < 0.4).mean()), "middle": float(((Lr >= 0.4) & (Lr < 0.7)).mean()), "light": float((Lr >= 0.7).mean())}
    if not quiet:
        msg = f"drawing_report: {100 * out['bare']:.0f}% of the sheet is bare paper; {100 * out['flat']:.0f}% lies in flat areas with no light and shade"
        if boxes:
            msg += " (largest: " + ", ".join(f"box {b}" for b in boxes[:3]) + ")"
        msg += f"; tones run from {out['darkest']:.2f} to {out['lightest']:.2f} (dark {100 * out['dark']:.0f}%, middle {100 * out['middle']:.0f}%, light {100 * out['light']:.0f}%)"
        if out["lightest"] - out["darkest"] < 0.35:
            msg += ": a narrow range, with no real " + ("dark" if out["darkest"] > 0.35 else "light")
        print(msg)
    return out


def tones(img, levels=3):
    """The picture reduced to `levels` flat steps of light and dark (a notan). The value check painters make: if
    the picture does not read in three tones, no colour or detail will save it. show() the result."""
    L = rgb2oklab(np.asarray(img, np.float32)[..., :3])[..., 0]
    L = cv2.GaussianBlur(L, (0, 0), max(1.0, 0.004 * max(L.shape)))
    q = np.round(np.clip(L, 0, 1) * (levels - 1)) / max(1, levels - 1)
    return np.dstack([q] * 3).astype(np.float32)


def squint(img, amount=1.0):
    """The picture blurred as if seen through half-closed eyes: detail gone, masses and the focal point left.
    Look at it to judge composition and where the eye goes. show() the result."""
    a = np.asarray(img, np.float32)[..., :3]
    return cv2.GaussianBlur(a, (0, 0), max(2.0, 0.012 * max(a.shape[:2]) * amount))


def mirror(img):
    """The picture flipped left to right. Errors of drawing that the eye has got used to (a leaning figure, a
    lopsided face, uneven eyes) show at once in the mirror. show() the result."""
    return np.ascontiguousarray(np.asarray(img, np.float32)[:, ::-1])
