"""Putting one thing into another so the join cannot be seen: compositing, seamless blending, colour and grain matching."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .core import oklab2rgb, rgb2oklab, to_linear, to_srgb

__all__ = ["blend", "behind", "seamless", "match_color", "grain_level", "add_grain"]


def _same(a, b, what="the two pictures"):
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(f"{what} differ in size: {a.shape[1]}x{a.shape[0]} and {b.shape[1]}x{b.shape[0]}")


def blend(base, top, mask=None, opacity=1.0):
    """Lay `top` over `base` through a mask, mixing light correctly (in linear light). The basic way to apply any edit.

    top is a picture, or a layer whose alpha is used (times the mask, if one is given). Where the mask is 0 the
    result is exactly `base`, so an edit made this way cannot leak outside its mask."""
    b = np.asarray(base, np.float32)[..., :3]
    t = np.asarray(top, np.float32)
    _same(b, t)
    alpha = t[..., 3] if t.ndim == 3 and t.shape[2] == 4 else np.ones(b.shape[:2], np.float32)
    if mask is not None:
        m = np.asarray(mask, np.float32)
        _same(b, m, "the picture and the mask")
        alpha = alpha * m
    alpha = np.clip(alpha * float(opacity), 0, 1)[..., None]
    out = to_srgb(to_linear(b) * (1 - alpha) + to_linear(t[..., :3]) * alpha)
    return np.where(alpha > 0, out, b).astype(np.float32)


def behind(img, layer, front):
    """Put a layer (H, W, 4) behind something in the picture that stays in front of it: hair, fur, a hill or dune
    crest, foliage, a person, glass. Returns the picture.

    front  a soft mask of what stays in front (select_object, refine_edge for hair and fur). At its soft edge a
           pixel is a mixture of the front thing and the old background; blending there would leave a fringe of the
           old background round it. Instead the front thing's own colour is estimated at every edge pixel (alpha
           matting) and mixed with the new layer, so the edge comes out as if photographed against the new content.
    Pixels the layer does not reach are left exactly as they were."""
    a = np.asarray(img, np.float32)[..., :3]
    L = np.asarray(layer, np.float32)
    _same(a, L[..., :3], "the picture and the layer")
    la = np.clip(L[..., 3], 0, 1) if L.shape[-1] == 4 else np.ones(a.shape[:2], np.float32)
    f = np.clip(np.asarray(front, np.float32), 0, 1)
    ys, xs = np.nonzero(la > 1e-3)
    if not len(ys):
        return a.copy()
    h, w = la.shape
    pad = 16
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    A, Fm, La, Lc = a[y0:y1, x0:x1], f[y0:y1, x0:x1], la[y0:y1, x0:x1, None], L[y0:y1, x0:x1, :3]
    try:
        from pymatting import estimate_foreground_ml
        Fg = estimate_foreground_ml(A.astype(np.float64), Fm.astype(np.float64)).astype(np.float32)
    except Exception:  # without matting: the front thing's colour taken as it is
        Fg = A
    back = A * (1 - La) + Lc * La          # the old background with the new layer laid over it
    region = Fm[..., None] * Fg + (1 - Fm[..., None]) * back
    out = a.copy()
    out[y0:y1, x0:x1] = np.where(La > 1e-3, region, A)
    return np.clip(out, 0, 1)


def _harmonic(region, values, known, max_unknowns=60000):
    """Smoothly continue `values` (given on the `known` pixels around a region) through the region: the solution of
    Laplace's equation, which is the smoothest field that agrees with its boundary. Solved on a reduced grid, since
    the answer has no fine detail. Returns an array of values' shape, zero outside the region."""
    out = np.zeros_like(values, dtype=np.float32)
    if not region.any():
        return out
    ys, xs = np.where(region.any(1))[0], np.where(region.any(0))[0]
    H, W = region.shape
    y0, y1, x0, x1 = max(0, ys[0] - 2), min(H, ys[-1] + 3), max(0, xs[0] - 2), min(W, xs[-1] + 3)
    reg, kn, val = region[y0:y1, x0:x1], known[y0:y1, x0:x1] & ~region[y0:y1, x0:x1], values[y0:y1, x0:x1].astype(np.float32)
    h, w = reg.shape
    f = max(1, int(np.ceil(np.sqrt(reg.sum() / max_unknowns))))
    if f > 1:
        hs, ws = int(np.ceil(h / f)), int(np.ceil(w / f))
        size = (ws, hs)
        regs = cv2.resize(reg.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.5
        kw = cv2.resize(kn.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        kv = cv2.resize(val * kn[..., None], size, interpolation=cv2.INTER_AREA)
        if kv.ndim == 2:
            kv = kv[..., None]
        kv = kv / np.maximum(kw, 1e-6)[..., None]
        kns = (kw > 0) & ~regs
    else:
        hs, ws, regs, kns, kv = h, w, reg, kn, val
    n = int(regs.sum())
    if n == 0:
        return out
    idx = -np.ones((hs, ws), np.int64)
    idx[regs] = np.arange(n)
    ry, rx = np.nonzero(regs)
    diag = np.full(n, 1e-6, np.float64)
    rows, cols = [], []
    rhs = np.zeros((n, kv.shape[2]), np.float64)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ny, nx = ry + dy, rx + dx
        ok = (ny >= 0) & (ny < hs) & (nx >= 0) & (nx < ws)
        nyc, nxc = np.clip(ny, 0, hs - 1), np.clip(nx, 0, ws - 1)
        inr = ok & regs[nyc, nxc]
        isk = ok & kns[nyc, nxc]
        diag += inr + isk
        rows.append(np.nonzero(inr)[0])
        cols.append(idx[nyc[inr], nxc[inr]])
        np.add.at(rhs, np.nonzero(isk)[0], kv[nyc[isk], nxc[isk]])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sparse.coo_matrix((np.concatenate([diag, -np.ones(len(rows))]), (np.concatenate([np.arange(n), rows]), np.concatenate([np.arange(n), cols]))), shape=(n, n)).tocsc()
    u = splu(A).solve(rhs)
    field = np.zeros((hs, ws, kv.shape[2]), np.float32)
    field[regs] = u
    field[kns] = kv[kns]
    # Outside the region and its rim nothing is defined; take the nearest defined value so enlarging stays clean.
    undefined = ~(regs | kns)
    if undefined.any():
        _, lab = cv2.distanceTransformWithLabels(undefined.astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        dy_, dx_ = np.nonzero(~undefined)
        lut = np.zeros(lab.max() + 1, np.int64)
        lut[lab[dy_, dx_]] = np.arange(len(dy_))
        near = lut[lab]
        field = field[dy_[near], dx_[near]]
    if f > 1:
        field = cv2.resize(field, (w, h), interpolation=cv2.INTER_LINEAR)
        if field.ndim == 2:
            field = field[..., None]
    out[y0:y1, x0:x1][reg] = field[reg]
    return out


def seamless(base, top, mask, strength=1.0):
    """Lay `top` over `base` through a mask and shift its tones so they meet the surroundings (Poisson blending).

    `top` keeps its detail and texture; its overall colour and brightness are bent, smoothly across the masked area,
    until its border agrees with `base`. Use it for a patch copied from elsewhere, a moved object, a filled region.
    The mask's border must run through similar material on both sides (sky into sky, wall into wall): where it
    crosses an edge of the object itself, colour bleeds in. Use blend() when the tones are already right."""
    b = np.asarray(base, np.float32)[..., :3]
    t = np.asarray(top, np.float32)[..., :3]
    m = np.clip(np.asarray(mask, np.float32), 0, 1)
    _same(b, t)
    _same(b, m, "the picture and the mask")
    region = m > 0.02
    if not region.any():
        return b.copy()
    ring = (cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~region
    if not ring.any():
        return blend(b, t, m)
    err = cv2.GaussianBlur(np.where(ring[..., None], b - t, 0), (0, 0), 2.5) / np.maximum(cv2.GaussianBlur(ring.astype(np.float32), (0, 0), 2.5), 1e-6)[..., None]
    fixed = np.clip(t + float(strength) * _harmonic(region, err, ring), 0, 1)
    return blend(b, fixed, m)


def match_color(img, like, mask=None, like_mask=None, strength=1.0):
    """Give `img` the colour balance and contrast of `like` (mean and spread in Oklab).

    The statistics are measured inside `mask` on img and inside `like_mask` on like, and the change is applied to
    the whole of img: blend the result back through your mask. Use it so that something brought in from elsewhere
    sits in the scene's light: measure `like` on the surroundings it must match."""
    a, r = rgb2oklab(img), rgb2oklab(like)
    wa = np.asarray(mask, np.float32) > 0.5 if mask is not None else np.ones(a.shape[:2], bool)
    wr = np.asarray(like_mask, np.float32) > 0.5 if like_mask is not None else np.ones(r.shape[:2], bool)
    if not wa.any() or not wr.any():
        raise ValueError("match_color: a mask selects nothing")
    ma, sa = a[wa].mean(0), a[wa].std(0) + 1e-5
    mr, sr = r[wr].mean(0), r[wr].std(0) + 1e-5
    gain = np.clip(sr / sa, 0.5, 2.0)
    gain[1:] = np.clip(np.hypot(sr[1], sr[2]) / np.hypot(sa[1], sa[2]), 0.5, 2.0)
    out = (a - ma) * gain + mr
    return oklab2rgb(a + float(strength) * (out - a))


# ---------------------------------------------------------------------------------------------------------------
# Grain

_LAP2 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def _noise_map(img):
    """Per-pixel noise estimate (Immerkaer 1996): the difference of two Laplacians cancels image structure."""
    g = np.asarray(img, np.float32)[..., :3].mean(-1)
    return np.abs(cv2.filter2D(g, -1, _LAP2, borderType=cv2.BORDER_REFLECT)) * (np.sqrt(np.pi / 2) / 6.0)


#: The noise that rounding to 8 bits alone puts in a picture (1/255 / sqrt(12)): grain below this cannot be told
#: apart, so comparisons of grain are floored here.
GRAIN_FLOOR = 0.0012


def grain_level(img, mask=None):
    """How much fine noise (film grain, sensor noise) a picture has in the masked area: its standard deviation, in
    0..1 units. A clean render is near 0; a typical photo 0.004 to 0.02. Compare an edited area with its surroundings."""
    nm = _noise_map(img)
    g = np.asarray(img, np.float32)[..., :3].mean(-1)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    w = np.asarray(mask, np.float32) > 0.5 if mask is not None else np.ones(g.shape, bool)
    if not w.any():
        raise ValueError("grain_level: the mask selects nothing")
    flat = w & (grad <= np.percentile(grad[w], 50))  # the smoother half: texture and edges would count as noise
    return float(nm[flat].mean()) if flat.any() else float(nm[w].mean())


def add_grain(img, mask, like=None, like_mask=None, amount=None, size=1.0, seed=7):
    """Give the masked area the same grain as the rest of the photo. Anything drawn, rendered, blurred or heavily
    recoloured is cleaner than a photograph and shows as a smooth patch until this is done.

    By default the grain is measured on `img` outside the mask; give like= (a picture) and like_mask= to measure it
    somewhere specific, or amount= to set the standard deviation yourself. Only what is missing is added.
    size is the grain's coarseness in pixels (1 = per-pixel noise; 1.5 to 2.5 for film or an enlarged photo)."""
    a = np.asarray(img, np.float32)[..., :3]
    m = np.clip(np.asarray(mask, np.float32), 0, 1)
    _same(a, m, "the picture and the mask")
    if not (m > 0.5).any():
        return a.copy()
    if amount is None:
        ref = a if like is None else np.asarray(like, np.float32)[..., :3]
        rm = (1 - m if like is None else np.ones(ref.shape[:2], np.float32)) if like_mask is None else np.asarray(like_mask, np.float32)
        if like is None and like_mask is None:
            # Measure near the edit, not across the whole picture: grain varies with brightness and focus.
            near = cv2.dilate((m > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=max(8, int(0.15 * np.sqrt((m > 0.5).sum())))) > 0
            if (near & (m < 0.02)).sum() > 500:
                rm = (near & (m < 0.02)).astype(np.float32)
        target = grain_level(ref, rm)
    else:
        target = float(amount)
    have = grain_level(a, m)
    sigma = float(np.sqrt(max(target ** 2 - have ** 2, 0.0)))
    if sigma < 2e-4:
        print(f"add_grain: nothing to add (area {have:.4f}, wanted {target:.4f})")
        return a.copy()
    rng = np.random.default_rng(seed)
    h, w = a.shape[:2]
    n = rng.standard_normal((h, w, 3)).astype(np.float32)
    lum = rng.standard_normal((h, w, 1)).astype(np.float32)
    n = 0.85 * lum + 0.35 * n  # mostly brightness noise, a little colour noise
    if size > 1.01:
        n = cv2.GaussianBlur(n, (0, 0), (size - 1) * 0.7 + 0.3)
    n = n / (n.std() + 1e-8) * sigma
    out = np.clip(a + n * m[..., None], 0, 1)
    print(f"add_grain: area had {have:.4f}, surroundings {target:.4f}; added {sigma:.4f}")
    return out.astype(np.float32)
