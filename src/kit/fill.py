"""Filling a hole with the picture's own surroundings (content-aware fill), and the clone stamp.

The fill is PatchMatch image completion (Barnes et al. 2009, after Wexler et al. 2007): every patch that overlaps the
hole looks for the most similar patch among the allowed source pixels, the overlapping patches vote on each hole
pixel, and this repeats from a coarse copy of the picture up to full size. Nothing is generated: every filled pixel is
copied from somewhere else in the same picture, which is why the fill keeps the picture's own texture and grain.
"""
from __future__ import annotations

import cv2
import numpy as np
from numba import njit

from .blend import _harmonic, blend

__all__ = ["fill", "fill_options", "clone"]

#: Weight of the texture-energy channel against the three colour channels in patch matching.
_ENERGY = 3.0


@njit(cache=False, fastmath=True)
def _dist(tgt, src, ty, tx, sy, sx, r, best):
    d = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            for c in range(tgt.shape[2]):
                e = tgt[ty + dy, tx + dx, c] - src[sy + dy, sx + dx, c]
                d += e * e
        if d >= best:
            return d
    return d


@njit(cache=False)
def _rand(state):
    x = state[0]
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    state[0] = x & 0xFFFFFFFF
    return state[0]


@njit(cache=False, fastmath=True)
def _search(tgt, src, valid, is_tgt, ty, tx, nny, nnx, nnd, r, iters, seed, coherence):
    """PatchMatch: propagation from the two already-visited neighbours, then a random search around the best match."""
    H, W = valid.shape
    n = len(ty)
    state = np.array([seed * 2654435761 % 4294967296 + 1], np.int64)
    for i in range(n):
        y, x = ty[i], tx[i]
        nnd[y, x] = _dist(tgt, src, y, x, nny[y, x], nnx[y, x], r, 1e30)
    for it in range(iters):
        fwd = it % 2 == 0
        step = -1 if fwd else 1
        for k in range(n):
            i = k if fwd else n - 1 - k
            y, x = ty[i], tx[i]
            by, bx, bd = nny[y, x], nnx[y, x], nnd[y, x]
            # A neighbour's match, shifted by one, continues the same piece of source: prefer it a little.
            for a in range(2):
                yy = y + (step if a == 1 else 0)
                xx = x + (step if a == 0 else 0)
                if yy < 0 or xx < 0 or yy >= H or xx >= W or not is_tgt[yy, xx]:
                    continue
                cy = nny[yy, xx] - (step if a == 1 else 0)
                cx = nnx[yy, xx] - (step if a == 0 else 0)
                if cy < 0 or cx < 0 or cy >= H or cx >= W or not valid[cy, cx] or (cy == by and cx == bx):
                    continue
                d = _dist(tgt, src, y, x, cy, cx, r, bd / coherence)
                if d * coherence < bd:
                    by, bx, bd = cy, cx, d * coherence
            rad = max(H, W)
            while rad >= 1:
                cy = by + _rand(state) % (2 * rad + 1) - rad
                cx = bx + _rand(state) % (2 * rad + 1) - rad
                if cy >= 0 and cx >= 0 and cy < H and cx < W and valid[cy, cx]:
                    d = _dist(tgt, src, y, x, cy, cx, r, bd)
                    if d < bd:
                        by, bx, bd = cy, cx, d
                rad //= 2
            nny[y, x], nnx[y, x], nnd[y, x] = by, bx, bd


@njit(cache=False, fastmath=True)
def _vote(src, ty, tx, nny, nnx, wgt, r):
    H, W, C = src.shape
    acc = np.zeros((H, W, C), np.float64)
    ws = np.zeros((H, W), np.float64)
    for i in range(len(ty)):
        y, x = ty[i], tx[i]
        sy, sx = nny[y, x], nnx[y, x]
        w = wgt[i]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                for c in range(C):
                    acc[y + dy, x + dx, c] += w * src[sy + dy, sx + dx, c]
                ws[y + dy, x + dx] += w
    return acc, ws


@njit(cache=False, fastmath=True)
def _vote_best(src, ty, tx, nny, nnx, nnd, hole_p, r):
    """The sharp vote: each hole pixel is copied from the one best-matching patch that covers it (ties go to the
    patch centred nearest), so the fill is made of real, unblurred pieces of the picture."""
    H, W = hole_p.shape
    C = src.shape[2]
    best = np.full((H, W), 1e30, np.float32)
    out = np.zeros((H, W, C), np.float32)
    for i in range(len(ty)):
        y, x = ty[i], tx[i]
        sy, sx = nny[y, x], nnx[y, x]
        d = nnd[y, x]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy, xx = y + dy, x + dx
                if not hole_p[yy, xx]:
                    continue
                off = abs(dy) + abs(dx)
                sc = d * (1.0 + 0.2 * off) + 1e-7 * off
                if sc < best[yy, xx]:
                    best[yy, xx] = sc
                    for c in range(C):
                        out[yy, xx, c] = src[sy + dy, sx + dx, c]
    return out, best


def _weights(nnd, ty, tx, conf):
    """A patch's say in the vote: more for a close match, and more the nearer it sits to the known picture."""
    d = nnd[ty, tx]
    sigma2 = max(float(np.percentile(d, 75)), 1e-6)
    return (conf[ty, tx] * np.maximum(np.exp(-d / (2 * sigma2)), 1e-4)).astype(np.float32)


def _settle(tgt, acc, ws, hole_p):
    """Write the vote's result into the hole."""
    got = hole_p & (ws > 0)
    tgt[got] = acc[got] / ws[got][:, None]


def _with_energy(img, hole, weight):
    """The picture plus one more channel: how busy the texture is around each pixel (smoothed gradient size).

    Matching on colour alone drifts toward the smoothest material in reach, because a flat patch is on average
    closer to any texture than another sample of that texture is. The extra channel is voted and copied like a
    colour, so a place that should be textured keeps asking for textured patches."""
    g = img.mean(-1)
    e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    near = cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    e[near] = 0  # gradients that touch the hole say nothing about the material
    e = cv2.GaussianBlur(e, (0, 0), 1.2)
    return np.dstack([img, weight * e]).astype(np.float32)


def _level(img, hole, allowed, r):
    """Pad by r and work out, for one pyramid level, which pixels are targets and which may be copied from."""
    p = r
    src = np.pad(_with_energy(img, hole, _ENERGY), ((p, p), (p, p), (0, 0)), mode="edge").astype(np.float32)
    hole_p = np.pad(hole, p, mode="constant", constant_values=False)
    ok = np.pad(allowed & ~hole, p, mode="constant", constant_values=False)
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    valid = cv2.erode(ok.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    is_tgt = cv2.dilate(hole_p.astype(np.uint8), k) > 0
    is_tgt[:p, :] = is_tgt[-p:, :] = False
    is_tgt[:, :p] = is_tgt[:, -p:] = False
    return src, hole_p, valid, is_tgt


def _complete(img, hole, allowed, r, seed, coherence, sharp_levels=2):
    """Coarse-to-fine completion. Returns the filled picture, the mean vote (for seams) and where the vote reached.

    Coarse levels average the overlapping patches, which settles the broad layout; the finest `sharp_levels` levels
    copy whole patches instead, which keeps texture crisp."""
    # Pyramid: halve until the hole is only a few patches thick, so the coarsest level sees its whole surroundings.
    pyr = [(img, hole, allowed)]
    while True:
        im, ho, al = pyr[-1]
        h, w = ho.shape
        thick = 2 * float(cv2.distanceTransform(ho.astype(np.uint8), cv2.DIST_L2, 3).max())
        if thick <= 2 * r + 1 or min(h, w) < 12 * (2 * r + 1) or len(pyr) >= 8:
            break
        nh, nw = (h + 1) // 2, (w + 1) // 2
        pyr.append((
            cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA),
            cv2.resize(ho.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) > 0.0,
            cv2.resize(al.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) > 0.999,
        ))

    nny = nnx = None
    rng = np.random.default_rng(seed)
    for li in range(len(pyr) - 1, -1, -1):
        im, ho, al = pyr[li]
        src, hole_p, valid, is_tgt = _level(im, ho, al, r)
        if not valid.any():
            if li > 0:
                nny = nnx = None
                continue
            raise ValueError("fill: there is nowhere to copy from. The source area is empty or thinner than a patch; widen `source` or use a smaller `patch`.")
        H, W = valid.shape
        ty, tx = np.nonzero(is_tgt)
        ty, tx = ty.astype(np.int32), tx.astype(np.int32)
        vy, vx = np.nonzero(valid)
        dt = cv2.distanceTransform(hole_p.astype(np.uint8), cv2.DIST_L2, 3)
        conf = (1.3 ** (-np.minimum(dt, 40.0))).astype(np.float32)  # capped: deep inside, all patches count alike
        tgt = src.copy()
        nnd = np.zeros((H, W), np.float32)
        sharp = li < sharp_levels

        def vote(first=False):
            """Rebuild the hole from the current matches; returns the mean vote's sums (for the seam correction)."""
            w = conf[ty, tx].astype(np.float32) if first else _weights(nnd, ty, tx, conf)
            acc, ws = _vote(src, ty, tx, nny, nnx, w, r)
            if sharp and not first:
                out, best = _vote_best(src, ty, tx, nny, nnx, nnd, hole_p, r)
                got = hole_p & (best < 1e29)
                tgt[got] = out[got]
            else:
                _settle(tgt, acc, ws, hole_p)
            return acc, ws

        if nny is None:
            # Coarsest level: a smooth guess for the hole, and random matches to start from.
            guess = cv2.inpaint((np.clip(src[..., :3], 0, 1) * 255).astype(np.uint8), hole_p.astype(np.uint8), 3, cv2.INPAINT_TELEA)
            pick = rng.integers(0, len(vy), size=(H, W))
            nny, nnx = vy[pick].astype(np.int32), vx[pick].astype(np.int32)
            vote(first=True)  # gives the hole the surroundings' average texture energy
            tgt[hole_p, :3] = guess[hole_p].astype(np.float32) / 255
            loops, iters = 8, 4
        else:
            # Carry the matches up from the coarser level: same offsets, doubled.
            ph, pw = nny.shape
            yy, xx = np.mgrid[0:H, 0:W]
            cy, cx = np.clip((yy - r) // 2 + r, 0, ph - 1), np.clip((xx - r) // 2 + r, 0, pw - 1)
            uy = (nny[cy, cx] - r) * 2 + (yy - r) % 2 + r
            ux = (nnx[cy, cx] - r) * 2 + (xx - r) % 2 + r
            bad = (uy < 0) | (ux < 0) | (uy >= H) | (ux >= W)
            uy, ux = np.clip(uy, 0, H - 1), np.clip(ux, 0, W - 1)
            bad |= ~valid[uy, ux]
            pick = rng.integers(0, len(vy), size=int(bad.sum()))
            uy[bad], ux[bad] = vy[pick], vx[pick]
            nny, nnx = uy.astype(np.int32), ux.astype(np.int32)
            vote(first=True)
            loops, iters = (3, 3) if li == 0 else (4, 3)

        for lp in range(loops):
            _search(tgt, src, valid, is_tgt, ty, tx, nny, nnx, nnd, r, iters, seed * 131 + li * 17 + lp, coherence)
            acc, ws = vote()

    result = tgt[r:-r, r:-r, :3]
    # What the mean vote makes of the known pixels next to the hole, for the seam correction.
    rim = (acc / np.maximum(ws, 1e-30)[..., None])[r:-r, r:-r, :3].astype(np.float32)
    return result, rim, (ws > 0)[r:-r, r:-r]


def fill(img, mask, source=None, patch=7, seed=1, coherence=0.8, _sharp=2):
    """Fill the masked area with matching material copied from the rest of the picture (content-aware fill).

    Use it to remove things. The mask must cover ALL of the thing, with its shadow, reflection and soft edge:
    grow the selection by 4 to 12 px first. Anything left uncovered stays, and is copied back into the hole.
    Every pixel the mask selects at all is replaced, however faintly it is selected.
    source  a mask (or a box) of where it may copy from. By default the surroundings of the hole, out to about its
            own width. Give it whenever that pulls in the wrong material or there is better material further away:
            say exactly which stretch of wall, grass, water or sky to use. This is the main control over the result.
    patch   patch width in pixels (odd). 7 for fine texture; 9 to 15 keep larger structure (bricks, planks, waves).
    seed    another seed gives another fill; coherence (0.5..1) below 1 favours copying larger unbroken pieces.
    Lines and edges that should run on through the hole are not guaranteed to: check, and continue them with clone()."""
    a = np.ascontiguousarray(np.asarray(img, np.float32)[..., :3])
    m = np.clip(np.asarray(mask, np.float32), 0, 1)
    h, w = a.shape[:2]
    if m.shape != (h, w):
        raise ValueError(f"mask is {m.shape[1]}x{m.shape[0]} but the picture is {w}x{h}")
    hole = m > 0.02
    if not hole.any():
        return a.copy()
    if hole.mean() > 0.6:
        raise ValueError("fill: the mask covers most of the picture; there is too little left to copy from")
    r = max(1, int(patch) // 2)
    if source is None:
        allowed = np.ones((h, w), bool)
    elif np.ndim(source) == 1:
        x0, y0, x1, y1 = [int(round(v)) for v in source]
        allowed = np.zeros((h, w), bool)
        allowed[max(0, y0):y1, max(0, x0):x1] = True
    else:
        allowed = np.asarray(source, np.float32) > 0.5
    # Pixels right next to the hole are often the fringe of the thing being removed: never copy from them.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 3, 2 * r + 3))
    allowed &= ~(cv2.dilate(hole.astype(np.uint8), k) > 0)

    # Work in a window around the hole and its sources, not the whole picture.
    pad = 6 * (2 * r + 1)
    if source is None:
        # With no source given, copy only from near the hole: what is next to a thing is likeliest to be behind it.
        hy, hx = np.where(hole.any(1))[0], np.where(hole.any(0))[0]
        span = max(hy[-1] - hy[0], hx[-1] - hx[0]) + 1
        reach = max(48, int(0.75 * span))
        allowed &= cv2.distanceTransform((~hole).astype(np.uint8), cv2.DIST_L2, 3) <= reach
    need = hole | allowed
    ys, xs = np.where(need.any(1))[0], np.where(need.any(0))[0]
    y0, y1 = max(0, int(ys[0]) - pad), min(h, int(ys[-1]) + 1 + pad)
    x0, x1 = max(0, int(xs[0]) - pad), min(w, int(xs[-1]) + 1 + pad)
    sub, sh, sa = a[y0:y1, x0:x1], hole[y0:y1, x0:x1], allowed[y0:y1, x0:x1]

    filled, rim, seen = _complete(sub, sh, sa, r, int(seed), float(np.clip(coherence, 0.3, 1.0)), int(_sharp))

    # Seam: where the fill meets the untouched picture, shift its tones so the two agree, fading into the hole.
    ring = (cv2.dilate(sh.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~sh & seen
    if ring.any():
        err = cv2.GaussianBlur(np.where(ring[..., None], sub - rim, 0), (0, 0), 3) / np.maximum(cv2.GaussianBlur(ring.astype(np.float32), (0, 0), 3), 1e-6)[..., None]
        filled = filled + _harmonic(sh, err, ring)

    # Everything the mask selects at all is replaced; only the outermost pixel is mixed, so the edge is not a hard cut.
    out = a.copy()
    soft = np.clip(cv2.distanceTransform(sh.astype(np.uint8), cv2.DIST_L2, 3) / 1.5, 0, 1)[..., None]
    out[y0:y1, x0:x1] = sub * (1 - soft) + np.clip(filled, 0, 1) * soft
    return out.astype(np.float32)


def fill_options(img, mask, sources=(None,), patches=(7, 11), seeds=(1, 2)):
    """Several fills of the same hole at once, shown side by side for you to choose from. Returns the list of
    pictures in the order shown (the sheet numbers them from 0).

    Every combination of the given sources, patch sizes and seeds is tried (each takes about a second), at most 6.
    A fill is partly chance: take the best of several instead of accepting the first."""
    from .core import _emit, _fit, _to8
    from .masks import mask_box
    outs, labels = [], []
    for so in sources:
        for pa in patches:
            for se in seeds:
                if len(outs) < 6:
                    outs.append(fill(img, mask, source=so, patch=pa, seed=se))
                    labels.append(f"{len(outs) - 1}: patch {pa}, seed {se}" + ("" if so is None else ", given source"))
    b = mask_box(mask, pad=int(0.5 * max(8, np.sqrt(float((np.asarray(mask) > 0.02).sum())))))
    tiles = []
    for o, lab in zip(outs, labels):
        t, _ = _fit(_to8(o[b[1]:b[3], b[0]:b[2]]), 640)
        t = np.ascontiguousarray(t)
        cv2.rectangle(t, (0, 0), (min(t.shape[1], 12 * len(lab) + 12), 26), (0, 0, 0), -1)
        cv2.putText(t, lab, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    cols = 3 if len(tiles) > 2 else len(tiles)
    hh, ww = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * hh, cols * ww, 3), 24, np.uint8)
    for i, t in enumerate(tiles):
        sheet[(i // cols) * hh:(i // cols) * hh + hh, (i % cols) * ww:(i % cols) * ww + ww] = t
    _emit(sheet, "fill options", b, mappable=False)
    print("fill_options: " + "; ".join(labels))
    return outs


def clone(img, mask, offset, seamless=True):
    """Clone stamp: paint the masked area with the pixels found `offset` = (dx, dy) away from it.

    The source is the same shape as the mask, shifted by the offset: clone(img, m, (-120, 0)) copies from 120 px to
    the left. Use it to continue a line, an edge or a regular pattern through a hole, where fill() cannot know the
    structure; choose the offset so the pattern lines up (a multiple of the brick, plank or tile spacing).
    seamless=True adjusts the copy's tones to its new surroundings; the mask's soft edge is used for the join."""
    a = np.asarray(img, np.float32)[..., :3]
    m = np.clip(np.asarray(mask, np.float32), 0, 1)
    h, w = a.shape[:2]
    dx, dy = float(offset[0]), float(offset[1])
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    moved = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    inside = cv2.warpAffine(np.ones((h, w), np.float32), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if float((m * (1 - inside)).sum()) > 0.02 * float(m.sum()) + 1:
        print("clone: part of the source lies outside the picture; that part is mirrored from the edge")
    if seamless:
        from .blend import seamless as _seamless
        return _seamless(a, moved, m)
    return blend(a, moved, m)
