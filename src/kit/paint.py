"""The stroke engine: it lays the strokes, you decide what is painted.

paint() renders a picture you have made (a study: flat shapes, gradients, a render, a rough composite) as brush
strokes, from large brushes to small, each stroke following the forms. This is painterly rendering after Hertzmann
(1998). hatch() does the same for pen and ink, and ink_lines() draws tapered line art through points you give.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
from numba import njit

from .core import _state, _to8, as_color, oklab2rgb, rgb2oklab, to_linear, to_srgb

__all__ = ["paint", "wash", "hatch", "ink_lines"]

_KIND = {"round": 0, "bristle": 1, "flat": 2, "pencil": 3, "dab": 0}


@njit(cache=False, fastmath=True)
def _plan(ref, canvas, fx, fy, sy, sx, R, minlen, maxlen, curve):
    """One stroke per start point: follow the direction field while the stroke's colour still suits the study."""
    H, W = ref.shape[0], ref.shape[1]
    n = len(sy)
    paths = np.zeros((n, maxlen + 1, 2), np.float32)
    counts = np.zeros(n, np.int32)
    cols = np.zeros((n, 3), np.float32)
    for i in range(n):
        y, x = float(sy[i]), float(sx[i])
        for c in range(3):
            cols[i, c] = ref[sy[i], sx[i], c]
        paths[i, 0, 0], paths[i, 0, 1] = x, y
        k = 1
        lx, ly = 0.0, 0.0
        while k <= maxlen:
            iy, ix = int(y), int(x)
            if k > minlen:
                d_canvas = 0.0
                d_stroke = 0.0
                for c in range(3):
                    d_canvas += abs(ref[iy, ix, c] - canvas[iy, ix, c])
                    d_stroke += abs(ref[iy, ix, c] - cols[i, c])
                if d_canvas < d_stroke:
                    break
            dx, dy = fx[iy, ix], fy[iy, ix]
            if lx * dx + ly * dy < 0:
                dx, dy = -dx, -dy
            dx, dy = curve * dx + (1 - curve) * lx, curve * dy + (1 - curve) * ly
            m = (dx * dx + dy * dy) ** 0.5
            if m < 1e-6:
                break
            dx, dy = dx / m, dy / m
            x, y = x + R * dx, y + R * dy
            if x < 0 or y < 0 or x >= W - 1 or y >= H - 1:
                # The stroke runs off the sheet (its last point lies outside) instead of stopping short of the edge.
                paths[i, k, 0], paths[i, k, 1] = x, y
                k += 1
                break
            # A stroke ends where the study turns to another colour: it does not run across an edge.
            d_next = 0.0
            for c in range(3):
                d_next += abs(ref[int(y), int(x), c] - cols[i, c])
            if k >= 1 and d_next > 0.30:
                break
            lx, ly = dx, dy
            paths[i, k, 0], paths[i, k, 1] = x, y
            k += 1
        counts[i] = k
    return paths, counts, cols


@njit(cache=False, fastmath=True)
def _render(canvas, hgt, ref, paths, counts, cols, thick, order, R, kind, opacity, mask, streaks, grain, wet, fit, slack, soft):
    """Lay the strokes on the canvas in the given order. A stroke's stamps are merged first (max), then laid once,
    so it does not get denser where its stamps overlap. With `fit`, a stroke takes only where it brings the canvas
    closer to the study (or leaves it as close, within `slack`): strokes stop at the study's edges instead of
    spilling over them, and no stroke makes the picture worse."""
    H, W = canvas.shape[0], canvas.shape[1]
    K = streaks.shape[1]
    step = max(1.0, R / 3.0)
    edge = 1.2 if kind == 2 else (max(1.0, 0.45 * R) if kind == 0 else max(1.0, 0.22 * R))
    laid = 0
    for oi in range(len(order)):
        i = order[oi]
        n = counts[i]
        x0, y0, x1, y1 = W, H, 0, 0
        for k in range(n):
            px, py = paths[i, k, 0], paths[i, k, 1]
            x0, y0 = min(x0, int(px - R - 1)), min(y0, int(py - R - 1))
            x1, y1 = max(x1, int(px + R + 2)), max(y1, int(py + R + 2))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        cov = np.zeros((y1 - y0, x1 - x0), np.float32)
        furrow = np.ones((y1 - y0, x1 - x0), np.float32)
        nseg = max(1, n - 1)
        for s in range(nseg):
            ax, ay = paths[i, s, 0], paths[i, s, 1]
            if n > 1:
                bx, by = paths[i, s + 1, 0], paths[i, s + 1, 1]
            else:
                bx, by = ax, ay
            seg = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
            ux, uy = (1.0, 0.0) if seg < 1e-6 else ((bx - ax) / seg, (by - ay) / seg)
            m = int(seg / step) + 1
            for j in range(m + 1):
                t = j / m
                cx, cy = ax + (bx - ax) * t, ay + (by - ay) * t
                # The brush lifts at both ends: the stroke is narrower there.
                pos = s + t
                r = R if n < 3 else R * min(1.0, 0.55 + 0.45 * min(pos, (n - 1) - pos))
                for yy in range(max(y0, int(cy - r - 1)), min(y1, int(cy + r + 2))):
                    for xx in range(max(x0, int(cx - r - 1)), min(x1, int(cx + r + 2))):
                        ox, oy = xx - cx, yy - cy
                        dist = (ox * ox + oy * oy) ** 0.5
                        a = (r - dist) / edge
                        if a <= 0:
                            continue
                        if a > 1:
                            a = 1.0
                        f = 1.0
                        if kind == 1 or kind == 3:
                            # Bristles: the stroke is streaked across its width, the same streak along its length.
                            across = (-uy * ox + ux * oy) / r * 0.5 + 0.5
                            f = streaks[i % streaks.shape[0], min(K - 1, max(0, int(across * K)))]
                            a *= 0.55 + 0.45 * f if kind == 1 else f
                        if a > cov[yy - y0, xx - x0]:
                            cov[yy - y0, xx - x0] = a
                            furrow[yy - y0, xx - x0] = f
        sx_, sy_ = int(paths[i, 0, 0]), int(paths[i, 0, 1])
        took = 0.0
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                a = cov[yy - y0, xx - x0]
                if a <= 0:
                    continue
                if kind == 3:
                    a *= grain[yy % grain.shape[0], xx % grain.shape[1]]
                a *= opacity * mask[yy, xx]
                if fit:
                    e_canvas = 0.0
                    e_stroke = 0.0
                    for c in range(3):
                        e_canvas += abs(canvas[yy, xx, c] - ref[yy, xx, c])
                        e_stroke += abs(cols[i, c] - ref[yy, xx, c])
                    # In proportion to how much nearer the stroke's colour is to the study than the canvas is: all of
                    # it where the canvas is wrong, none where the stroke would be, and at a soft edge of the study
                    # the same mixture the study has there (so its edges come out as soft as it drew them).
                    w = 0.5 + 0.5 * (e_canvas - e_stroke + slack) / max(e_canvas + e_stroke, soft)
                    if w <= 0:
                        continue
                    if w < 1:
                        a *= w
                took += a
                for c in range(3):
                    col = cols[i, c] * (1 - wet) + canvas[sy_, sx_, c] * wet
                    canvas[yy, xx, c] = canvas[yy, xx, c] * (1 - a) + col * a
                hgt[yy, xx] = hgt[yy, xx] * (1 - a) + thick[i] * (0.35 + 0.65 * furrow[yy - y0, xx - x0]) * cov[yy - y0, xx - x0] * a
        if took > 0.5:
            laid += 1
    return laid


def _reference(tgt, R):
    """The study as a brush of radius R sees it: detail smaller than the brush smoothed away, edges kept where they
    are. (A plain blur would mix the colours on both sides of every edge, and the strokes would carry that mixture
    into the picture as a halo.)"""
    if R < 1.5:
        return tgt
    f = max(1.0, R / 8.0)
    small = tgt if f == 1.0 else cv2.resize(tgt, (max(8, round(tgt.shape[1] / f)), max(8, round(tgt.shape[0] / f))), interpolation=cv2.INTER_AREA)
    try:
        small = cv2.ximgproc.rollingGuidanceFilter(small, d=-1, sigmaColor=0.10, sigmaSpace=float(max(1.0, 0.5 * R / f)), numOfIter=3)
    except (AttributeError, cv2.error):
        small = cv2.bilateralFilter(small, -1, 0.10, float(max(1.0, 0.5 * R / f)))
    return small if f == 1.0 else cv2.resize(small, (tgt.shape[1], tgt.shape[0]), interpolation=cv2.INTER_LINEAR)


def _flow(ref, R):
    """Stroke directions for every pixel. Where the study has an edge or turns from light to shade, strokes run
    along it, so they outline shapes and wrap round shaded volumes; a gentle turn of tone counts as much as a hard
    edge. Where the study is plain, away from any edge, strokes go the way a hand hatches: on a slant, turning a
    little from place to place (without this they would ring the nearest object like ripples)."""
    L = rgb2oklab(ref)[..., 0]
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    k = np.minimum(1.0, mag / 0.0008) / np.maximum(mag, 1e-12)  # direction only, faded out below a faint slope
    ux, uy = gx * k, gy * k
    J = np.dstack([ux * ux, ux * uy, uy * uy])
    h, w = L.shape
    r = max(1.5, R)
    out = cv2.GaussianBlur(J, (0, 0), r) + 0.3 * cv2.GaussianBlur(J, (0, 0), 2.5 * r)
    rng = np.random.default_rng(11)
    drift = cv2.resize(rng.uniform(-1, 1, (4, 5)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    slant = np.deg2rad(-38.0) + np.deg2rad(22.0) * drift  # the hatch's direction; the tensor holds its normal
    nx_, ny_ = -np.sin(slant), np.cos(slant)
    out = out + 0.02 * np.dstack([nx_ * nx_, nx_ * ny_, ny_ * ny_])
    # Smooth once more, relative to what is there, so that where two directions meet they turn into each other.
    out = cv2.GaussianBlur(out, (0, 0), 1.5 * max(2.0, R)) + 0.25 * out
    ang = 0.5 * np.arctan2(2 * out[..., 1], out[..., 0] - out[..., 2] + 1e-12)  # the direction across the edge
    return (-np.sin(ang)).astype(np.float32), np.cos(ang).astype(np.float32)


def _field(ref, direction, shape, R):
    h, w = shape
    if direction is None:
        return _flow(ref, R)
    ang = np.deg2rad(direction) * np.ones((h, w), np.float32) if np.ndim(direction) == 0 else np.asarray(direction, np.float32)
    if ang.shape != (h, w):
        raise ValueError(f"direction must be an angle in degrees or an array of angles in radians shaped ({h}, {w})")
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


_RELIEF = {"round": 0.15, "bristle": 0.45, "flat": 0.06, "pencil": 0.0, "dab": 0.3}
_VARIATION = {"round": 0.35, "bristle": 0.5, "flat": 0.2, "pencil": 0.2, "dab": 0.5}
_last: dict = {}
_SOFT = 0.2  # how gradually a stroke fades out where it is only slightly better than what is there


def _lit(canvas, hgt, relief):
    """The paint's thickness catching light from the upper left: ridges along strokes, furrows between bristles."""
    if relief <= 0 or not float(hgt.max()) > 0:
        return canvas
    hs = cv2.GaussianBlur(hgt, (0, 0), 1.1)
    slope = -(cv2.Sobel(hs, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(hs, cv2.CV_32F, 0, 1, ksize=3)) * 0.125
    slope = slope / (1.0 + 2.5 * np.abs(slope))  # a ridge catches the light; it does not become an outline
    shade = np.clip(1.0 + 1.6 * relief * slope, 0.85, 1.18)
    return canvas * shade[..., None]


def paint(target, onto=None, mask=None, brushes=(48, 24, 12, 6, 3), brush="round", direction=None, threshold=0.04,
          opacity=1.0, wet=0.0, length=(3, 14), curve=0.7, spread=1.0, seed=1, record=None, variation=None,
          relief=None, loose=0.0):
    """Paint a picture in brush strokes, largest brush first. Returns the painting.

    target     the study to paint from: a picture you made (shapes with their light and shade, gradients, a
               render, a composite) or part of the photograph. The strokes take their colours and directions from
               it; nothing in it is copied pixel for pixel.
    onto       the canvas to paint on (default: a ground washed with the study's big masses). Give the current
               picture to paint into it, and `mask` to keep the strokes inside an area.
    brushes    brush radii in pixels, large to small. The first covers the canvas; each further size lays a stroke
               wherever the canvas still differs from the study by more than `threshold`, and only where the stroke
               brings it closer, so edges come out as clean as the study's and no pass can make the picture worse.
               Large sizes give the broad masses, small ones the detail: stop at a larger size for a loose
               painting, go down to 2 and 1 for a tight one. A feature narrower than the smallest brush is lost.
    brush      'round' (soft, opaque), 'bristle' (streaked, like oil or acrylic), 'flat' (hard-edged, like gouache or
               cel painting), 'pencil' (grainy, thin), 'dab' (single touches, no drag).
    direction  None: strokes follow the forms (they run along edges and wrap round shaded volumes; a plain area
               away from any edge is hatched on a slant). A number: all strokes at that angle in degrees. An array
               (H, W) of angles in radians: your own flow field (hair, water, fur, folds).
    threshold  how wrong the canvas must be before a stroke is laid, 0..1. Lower paints more and tighter.
    opacity    under 1 lets strokes build up like glazes. wet (0..1) drags some of the colour already on the canvas
               into each stroke, so neighbouring colours blend as wet paint does.
    length     (shortest, longest) stroke in steps of one brush radius. curve (0..1): how quickly strokes may bend.
    spread     spacing of stroke starts in brush radii (1 = dense; 2 = sparse and sketchy).
    variation  0..1: how much each stroke's colour differs from the study's, as each brush load is mixed a little
               differently (default by brush: bristle and dab 0.5, round 0.35, flat and pencil 0.2). Noise or
               random colour in the study is painted as it is, stroke by stroke.
    relief     0..1: how strongly the paint's thickness catches the light (default by brush: bristle 0.45, dab
               0.3, round 0.15, flat 0.06, pencil 0): thick oil or acrylic has it, gouache, watercolour and ink
               hardly do. 0 gives a perfectly matte surface.
    loose      0..1: how far strokes may spill over the study's edges. 0 keeps every edge; 0.3 is a loose sketch.
    record     a name: the canvas after each brush size is kept as a stage frame for the run's replay.
    It prints how far the painting still is from the study and where it is furthest; `onto=` the painting with
    `mask=` an area and smaller brushes repaints just that area. paint() adds nothing the study does not hold."""
    tgt = np.ascontiguousarray(np.asarray(target, np.float32)[..., :3])
    h, w = tgt.shape[:2]
    if brush not in _KIND:
        raise ValueError(f"brush must be one of {sorted(_KIND)}")
    m = np.ones((h, w), np.float32) if mask is None else np.clip(np.asarray(mask, np.float32), 0, 1)
    if m.shape != (h, w):
        raise ValueError("paint: the mask and the target differ in size")
    first_covers = onto is None
    hgt = np.zeros((h, w), np.float32)
    if onto is None:
        # The ground: a thin wash of the study's big masses, as a painter tones the canvas before the first brush.
        # (On a ground of one flat colour, every gap between strokes would show as a speck of that colour.)
        canvas = cv2.GaussianBlur(tgt, (0, 0), float(max(brushes)))
    elif _last.get("out") is onto:
        # Painting on into the last painting: carry on from its paint, not from the picture of its lit surface.
        canvas, hgt = _last["canvas"].copy(), _last["hgt"].copy()
    else:
        canvas = np.array(np.asarray(onto, np.float32)[..., :3])
    if canvas.shape != tgt.shape:
        raise ValueError("paint: `onto` and the target differ in size")
    relief = _RELIEF[brush] if relief is None else float(relief)
    variation = _VARIATION[brush] if variation is None else float(variation)
    rng = np.random.default_rng(seed)
    streaks = (0.25 + 0.75 * rng.random((64, 12))).astype(np.float32) if brush in ("bristle", "pencil") else np.ones((1, 1), np.float32)
    grain = (0.45 + 0.55 * rng.random((256, 256))).astype(np.float32)
    slack = 3.0 * (0.010 + 0.022 * float(variation) + 0.30 * float(loose))
    total = 0
    for bi, R in enumerate(brushes):
        R = float(R)
        ref = np.ascontiguousarray(_reference(tgt, R))
        fx, fy = _field(ref, direction, (h, w), R)
        err = np.abs(canvas - ref).sum(-1) / 3.0
        g = max(1, int(round(R * spread)))
        gh, gw = -(-h // g), -(-w // g)  # cells cover the whole sheet, the last row and column running past it
        pad = ((0, gh * g - h), (0, gw * g - w))
        cells = np.pad(err, pad, mode="edge").reshape(gh, g, gw, g).transpose(0, 2, 1, 3).reshape(gh, gw, g * g)
        inside = np.pad(m, pad, mode="edge").reshape(gh, g, gw, g).max((1, 3)) > 0.05
        cover = first_covers and bi == 0
        pick = ((cells.mean(-1) >= (0.0 if cover else float(threshold))) & inside)
        # Start each stroke where its cell is furthest from the study; among equals, anywhere (not on a grid).
        arg = (cells + rng.random(cells.shape, dtype=np.float32) * (0.02 * cells.max(-1, keepdims=True) + 1e-6)).argmax(-1)
        cy, cx = np.nonzero(pick)
        if not len(cy):
            continue
        sy = np.clip(cy * g + arg[cy, cx] // g, 0, h - 1).astype(np.int32)
        sx = np.clip(cx * g + arg[cy, cx] % g, 0, w - 1).astype(np.int32)
        lo, hi = (0, 0) if brush == "dab" else (int(length[0]), int(length[1]))
        paths, counts, cols = _plan(ref, canvas, fx, fy, sy, sx, R, lo, max(hi, 1) if brush != "dab" else 0, float(curve))
        if variation > 0:
            lab = rgb2oklab(cols.reshape(-1, 1, 3)).reshape(-1, 3)
            lab[:, 0] += rng.normal(0, 0.016 * variation, len(lab)).astype(np.float32)
            lab[:, 1:] += rng.normal(0, 0.006 * variation, (len(lab), 2)).astype(np.float32)
            cols = np.ascontiguousarray(oklab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3), np.float32)
        thick = ((0.6 + 0.8 * rng.random(len(sy))) * min(1.5, R / 10.0)).astype(np.float32)
        order = rng.permutation(len(sy)).astype(np.int32)
        total += _render(canvas, hgt, ref, paths, counts, cols, thick, order, R, _KIND[brush], float(opacity), m, streaks, grain, float(wet), not cover, slack, _SOFT)
        if record:
            _frame(_lit(canvas, hgt, relief), f"{record}: brush {R:g}")
    canvas = np.clip(canvas, 0, 1).astype(np.float32)
    out = np.clip(_lit(canvas, hgt, relief), 0, 1).astype(np.float32)
    _last.clear()
    _last.update(out=out, canvas=canvas, hgt=hgt)
    # How far the painting is from the study, and where it is furthest.
    off = cv2.GaussianBlur(np.abs(canvas - tgt).mean(-1) * m, (0, 0), max(2.0, float(min(brushes))))
    k = max(8, min(h, w) // 6)
    tiles = cv2.blur(off, (k, k))
    ty, tx = np.unravel_index(int(tiles.argmax()), tiles.shape)
    box = (max(0, tx - k // 2), max(0, ty - k // 2), min(w, tx + k // 2), min(h, ty + k // 2))
    if total == 0:
        print("paint: no stroke was laid: the canvas (`onto`) already matches the study. Give paint() the study you want painted (a different picture from the canvas), or leave `onto` out.")
    print(f"paint: {total} strokes with brushes {tuple(brushes)} ({brush}); the painting is {float(off.sum() / max(m.sum(), 1)):.3f} from the study on average"
          + (f", {float(tiles.max()):.3f} at its furthest, around box {box}" if float(tiles.max()) > 0.03 else ""))
    return out


def wash(target, paper=None, mask=None, layers=4, wet=0.5, edges=0.5, granulation=0.35, bloom=0.0, seed=1, record=None):
    """Watercolour: render a study as transparent washes of pigment on paper. Returns the painting.

    Unlike paint(), nothing here is opaque. Each wash darkens what is under it, as pigment does; the paper shows
    through everywhere and is the only white, so the lightest parts of the study stay paper. The study is built
    up in `layers` washes, light to dark: the first broad and soft, as a wet first wash is, the later ones smaller
    and with crisper edges, as washes laid on dry paper are. Each wash gathers pigment at its rim (a dried edge),
    settles a little unevenly (granulation), and has the soft, slightly wandering outline of water on paper.

    target       the study: shapes with their light and shade and colour, as for paint(). Its colour where it is
                 lighter than the paper is lost (watercolour cannot lighten): keep lights as untouched paper.
    paper        the paper: a colour or a picture (default: the study's lightest colour, or the sheet).
    mask         where to paint (a wash for one part, the rest left as it is).
    layers       how many washes build up the darkest tone (2 to 6).
    wet          0..1: how soft and spread the washes are (0 crisp, dry brush; 1 everything wet-in-wet).
    edges        0..1: how strongly pigment gathers at the rim of each wash.
    granulation  0..1: how unevenly the pigment settles. bloom 0..1: a few cauliflower-like blooms where washes
                 were wet."""
    tgt = np.clip(np.asarray(target, np.float32)[..., :3], 0, 1)
    h, w = tgt.shape[:2]
    rng = np.random.default_rng(seed)
    if paper is None:
        lum = rgb2oklab(tgt)[..., 0]
        pap = np.broadcast_to(np.percentile(tgt.reshape(-1, 3)[lum.ravel() >= np.percentile(lum, 98)], 50, axis=0), tgt.shape).astype(np.float32)
    elif isinstance(paper, str) or np.ndim(paper) == 1:
        pap = np.broadcast_to(as_color(paper), tgt.shape).astype(np.float32)
    else:
        pap = np.asarray(paper, np.float32)[..., :3]
    m = np.ones((h, w), np.float32) if mask is None else np.clip(np.asarray(mask, np.float32), 0, 1)
    side = float(max(h, w))
    # Absorbance (Beer-Lambert, in linear light): how much pigment it takes to turn the paper into the study.
    A = np.clip(-np.log(np.clip(to_linear(tgt), 1e-4, 1) / np.clip(to_linear(pap), 1e-4, 1)), 0, 4.0)
    total = A.max(-1)
    hue = A / np.maximum(total, 1e-4)[..., None]  # the pigment's colour, as a share of absorbance per channel
    hue = cv2.GaussianBlur(hue, (0, 0), 1.0)
    top = float(np.percentile(total, 99.5)) + 1e-4
    acc = np.zeros_like(A)
    low = cv2.resize(rng.random((max(4, h // 40), max(4, w // 40))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.8)
    fine = (fine - fine.mean()) / (fine.std() + 1e-6)
    n = max(1, int(layers))
    for k in range(n):
        lo, hi = top * k / n, top * (k + 1) / n
        part = np.clip((total - lo) / (hi - lo), 0, 1)  # the share of this wash's band of tone
        soft = side * (0.003 + 0.012 * float(wet)) * (1.0 - 0.75 * k / max(1, n - 1))
        # The wash's outline wanders a little, as water does on paper.
        dx = (cv2.resize(rng.standard_normal((max(4, h // 60), max(4, w // 60))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)) * soft * 0.6
        dy = (cv2.resize(rng.standard_normal((max(4, h // 60), max(4, w // 60))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)) * soft * 0.6
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        part = cv2.remap(part, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        part = cv2.GaussianBlur(part, (0, 0), max(0.6, soft * 0.35))
        # A dried edge: pigment pulled to the rim of the wash.
        rim = np.clip(part - cv2.GaussianBlur(part, (0, 0), max(1.5, soft * 0.5)), 0, 1)
        part = part * (1.0 + 0.6 * float(edges) * (0.5 + low)) + 1.6 * float(edges) * rim
        if bloom > 0 and k < n - 1:
            spots = (cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), soft * 1.5) > 0.5 + 0.02 * (1 - bloom)) * part
            part = part - 0.35 * float(bloom) * cv2.GaussianBlur(spots.astype(np.float32), (0, 0), soft * 0.4)
        part = np.clip(part, 0, None) * (1.0 + float(granulation) * 0.12 * fine * (0.4 + 0.6 * low))
        acc += (hi - lo) * np.clip(part, 0, None)[..., None] * hue
        if record:
            _frame(_wash_out(pap, acc, m), f"{record}: wash {k + 1}")
    out = _wash_out(pap, acc, m)
    print(f"wash: {n} washes; darkest absorbance {top:.2f}; the paper shows through wherever the study is paper-light")
    return out


def _wash_out(pap, acc, m):
    lin = to_linear(pap) * np.exp(-acc)
    return np.clip(pap * (1 - m[..., None]) + to_srgb(lin) * m[..., None], 0, 1).astype(np.float32)


def _frame(picture, name):
    from PIL import Image
    os.makedirs("stages", exist_ok=True)
    frames = _state.setdefault("stages", [])
    path = os.path.join("stages", f"{len(frames) + 1:02d}-{''.join(c if c.isalnum() else '-' for c in name.lower())[:30]}.png")
    Image.fromarray(_to8(picture)).save(path)
    frames.append({"name": name, "note": "", "file": path})
    with open(os.path.join("stages", "index.json"), "w") as f:
        json.dump(frames, f)


def _smooth(points, per=6):
    """Catmull-Rom curve through the points, as a dense list."""
    p = np.asarray(points, np.float32).reshape(-1, 2)
    if len(p) < 3:
        return p
    q = np.r_[p[:1], p, p[-1:]]
    t = np.linspace(0, 1, per, endpoint=False)[:, None]
    out = []
    for i in range(1, len(q) - 2):
        p0, p1, p2, p3 = q[i - 1], q[i], q[i + 1], q[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.r_[np.concatenate(out), p[-1:]]


def ink_lines(like, paths, width=2.5, taper=0.6, smooth=True, color="#151515", opacity=1.0):
    """Line art: draw each path [(x, y), ...] as one pen line, and return them as a layer (H, W, 4).

    width   the line's thickness at its fullest, in pixels.
    taper   0: the same thickness from end to end; 1: swelling from a point to full width and back to a point, as a
            brush pen or a pressed pencil does.
    smooth  True runs a curve through the points; False joins them straight.
    Each call draws its paths at one width; separate calls give separate widths."""
    a = np.asarray(like)
    h, w = a.shape[:2]
    S = 2
    m = np.zeros((h * S, w * S), np.uint8)
    for path in paths:
        p = _smooth(path) if smooth else np.asarray(path, np.float32).reshape(-1, 2)
        p = p[np.isfinite(p).all(1)]
        if len(p) < 2:
            continue
        seg = np.hypot(*np.diff(p, axis=0).T)
        d = np.r_[0, np.cumsum(seg)]
        if d[-1] < 1e-3:
            continue
        n = int(d[-1] * S) + 2
        t = np.linspace(0, d[-1], n)
        xs, ys = np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])
        press = (1 - taper) + taper * np.clip(np.sin(np.pi * t / d[-1]), 0, 1) ** 0.6
        for x, y, r in zip(xs, ys, np.maximum(0.35, width * 0.5 * press) * S):
            cv2.circle(m, (int(round(x * S * 16)), int(round(y * S * 16))), int(round(r * 16)), 255, -1, cv2.LINE_AA, 4)
    alpha = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 * float(opacity)
    return np.dstack([np.broadcast_to(as_color(color), (h, w, 3)), alpha]).astype(np.float32)


def hatch(tone, mask=None, spacing=5.0, length=16.0, width=1.1, levels=4, direction="contour", angle=40.0,
          color="#151515", jitter=0.3, seed=1):
    """Pen-and-ink shading: short strokes, denser where the tone is darker. Returns a layer (H, W, 4).

    tone       a picture (its lightness is used) or an array (H, W) 0..1, light to dark being 1 to 0: your value
               plan for the drawing. White stays paper; each darker step gets another layer of strokes.
    spacing    distance between strokes in pixels (smaller is darker and finer); length: stroke length in pixels.
    levels     how many layers of strokes the darkest tone gets; each layer crosses the one before.
    direction  'contour': strokes follow the form (they run along the tone's contours, as an engraver's do);
               'straight': all strokes of a layer are parallel, starting at `angle` degrees.
    jitter     0..1 unevenness of position, length and angle (0: perfectly regular).
    A mask limits it to one area, so areas can be hatched with different settings."""
    t = np.asarray(tone, np.float32)
    L = rgb2oklab(t)[..., 0] if t.ndim == 3 else np.clip(t, 0, 1)
    h, w = L.shape
    dark = 1.0 - cv2.GaussianBlur(L, (0, 0), 1.0)
    sel = np.ones((h, w), np.float32) if mask is None else np.clip(np.asarray(mask, np.float32), 0, 1)
    rng = np.random.default_rng(seed)
    Lb = cv2.GaussianBlur(L, (0, 0), max(2.0, spacing))
    gx, gy = cv2.Sobel(Lb, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(Lb, cv2.CV_32F, 0, 1, ksize=3)
    base = np.arctan2(gx, -gy)  # along the contour
    flat = np.hypot(gx, gy) < 1e-3
    S = 2
    m = np.zeros((h * S, w * S), np.uint8)
    count = 0
    for k in range(int(levels)):
        thr = (k + 0.6) / (levels + 0.6)
        ys, xs = np.mgrid[0:h:spacing, 0:w:spacing]
        ys = (ys + rng.uniform(-0.5, 0.5, ys.shape) * spacing * (0.4 + jitter)).ravel()
        xs = (xs + rng.uniform(-0.5, 0.5, xs.shape) * spacing * (0.4 + jitter)).ravel()
        iy, ix = np.clip(ys.astype(int), 0, h - 1), np.clip(xs.astype(int), 0, w - 1)
        # A stroke is laid where the tone is darker than this layer's step; near the step, only some are.
        keep = (dark[iy, ix] - thr) * 6 + 0.5 > rng.random(len(ys))
        keep &= sel[iy, ix] > 0.5
        ys, xs, iy, ix = ys[keep], xs[keep], iy[keep], ix[keep]
        if not len(ys):
            continue
        turn = np.deg2rad(angle + 55.0 * k)
        if direction == "contour":
            ang = np.where(flat[iy, ix], turn, base[iy, ix] + (k % 2) * np.deg2rad(60) * (1 if k % 4 < 2 else -1))
        else:
            ang = np.full(len(ys), turn, np.float32)
        ang = ang + rng.normal(0, 0.12 * jitter + 0.02, len(ys))
        half = 0.5 * length * (1 + rng.uniform(-0.5, 0.5, len(ys)) * jitter) * (0.6 + 0.6 * np.clip(dark[iy, ix], 0, 1))
        bend = rng.normal(0, 0.08 * jitter, len(ys))
        pts = []
        for s in (-1.0, 0.0, 1.0):
            a_ = ang + bend * s
            pts.append(np.stack([xs + np.cos(a_) * half * s, ys + np.sin(a_) * half * s], -1))
        poly = np.round(np.stack(pts, 1) * S * 16).astype(np.int32)
        cv2.polylines(m, list(poly), False, 255, max(1, int(round(width * S))), cv2.LINE_AA, 4)
        count += len(ys)
    alpha = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    print(f"hatch: {count} strokes in {levels} layers")
    return np.dstack([np.broadcast_to(as_color(color), (h, w, 3)), alpha]).astype(np.float32)
