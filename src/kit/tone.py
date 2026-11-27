"""Colour and tone: recolouring a thing while keeping its shading, and the standard darkroom adjustments."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from .core import as_color, oklab2rgb, rgb2oklab, to_linear, to_srgb

__all__ = ["recolor", "adjust", "curves"]


def recolor(img, color, mask=None, lightness=1.0, keep_highlights=True):
    """Repaint a surface in a new colour, keeping its shading, folds, texture and highlights.

    color      the colour the surface should have in normal light: '#rrggbb' or (r, g, b).
    mask       the surface. It also tells the function what the surface looks like now, so select only the surface.
               With a mask the result is the whole picture with the masked part recoloured; without one, the whole
               picture is treated as the surface.
    lightness  0 keeps the surface exactly as light or dark as it was and changes only hue and saturation;
               1 also brings it to the new colour's lightness. Going from very dark to light (black to white)
               flattens shading; values below 1 keep more of it.
    Shading is carried over multiplicatively: a fold that was 30% darker than the surface's typical tone stays 30%
    darker in the new colour. Bright colourless highlights are kept as highlights."""
    a = np.asarray(img, np.float32)[..., :3]
    lab = rgb2oklab(a)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1], lab[..., 2])
    sel = (np.asarray(mask, np.float32) > 0.5) if mask is not None else np.ones(L.shape, bool)
    if not sel.any():
        return a.copy()
    T = rgb2oklab(as_color(color).reshape(1, 1, 3))[0, 0]
    Lm = float(np.median(L[sel]))
    Cm = float(np.median(C[sel]))
    Lt = Lm + float(lightness) * (float(T[0]) - Lm)
    Tn = T * (Lt / max(float(T[0]), 1e-3))  # the target colour, at the lightness the surface will have
    rel = L / max(Lm, 1e-3)
    new = Tn[None, None, :] * rel[..., None]
    if keep_highlights:
        # A highlight is brighter than the surface and less coloured than it: light reflected, not surface colour.
        hi = np.clip((L - Lm) / max(1.0 - Lm, 1e-3), 0, 1)
        if Cm > 0.03:
            pale = np.clip(1.0 - (C / np.maximum(L, 1e-3)) / (Cm / max(Lm, 1e-3)), 0, 1)
        else:
            pale = np.ones_like(L)
        s = (hi ** 1.5 * pale)[..., None]
        neutral = np.stack([np.maximum(L, new[..., 0]), lab[..., 1] * 0.3, lab[..., 2] * 0.3], -1)
        new = new * (1 - s) + neutral * s
    new[..., 0] = np.clip(new[..., 0], 0, 1)
    out = oklab2rgb(new)
    if mask is None:
        return out
    from .blend import blend
    return blend(a, out, mask)


def adjust(img, exposure=0.0, contrast=0.0, saturation=0.0, temperature=0.0, tint=0.0, highlights=0.0, shadows=0.0, vibrance=0.0):
    """Darkroom adjustments on the whole picture; blend the result through a mask to adjust one area.

    exposure     in stops (+1 doubles the light).           contrast    -1..1, around middle grey.
    saturation   -1 (grey)..1.                               vibrance    -1..1, saturation for the dull colours only.
    temperature  -1 (cooler, bluer)..1 (warmer, more amber). tint        -1 (greener)..1 (more magenta).
    highlights   -1 (pull back the bright parts)..1.         shadows     -1..1 (+ opens the dark parts).
    0 everywhere changes nothing."""
    a = np.asarray(img, np.float32)[..., :3]
    lin = to_linear(a)
    if exposure or temperature or tint:
        gain = np.array([2.0 ** (0.30 * temperature + 0.10 * tint), 2.0 ** (-0.20 * tint), 2.0 ** (-0.30 * temperature + 0.10 * tint)], np.float32)
        gain = gain / float(gain @ np.array([0.2126, 0.7152, 0.0722], np.float32))  # keep brightness
        lin = lin * gain * (2.0 ** float(exposure))
    lab = rgb2oklab(to_srgb(np.clip(lin, 0, None)))
    L = lab[..., 0]
    if highlights or shadows:
        Lb = cv2.GaussianBlur(L, (0, 0), max(2.0, 0.01 * max(L.shape)))  # by region, so local contrast survives
        L = L + 0.30 * float(shadows) * np.clip(1 - Lb / 0.5, 0, 1) ** 1.5 * (1 - L) + 0.30 * float(highlights) * np.clip((Lb - 0.5) / 0.5, 0, 1) ** 1.5 * L
    if contrast:
        k = 1.0 + float(contrast)
        x = np.clip(L, 0, 1)
        if k >= 1:  # S-curve through middle grey that never clips
            L = np.where(x < 0.5, 0.5 * (2 * x) ** k, 1 - 0.5 * (2 * (1 - x)) ** k)
        else:
            L = 0.5 + (x - 0.5) * k
    lab = np.stack([L, lab[..., 1], lab[..., 2]], -1)
    if saturation or vibrance:
        C = np.hypot(lab[..., 1], lab[..., 2])
        s = (1.0 + float(saturation)) * (1.0 + float(vibrance) * np.clip(1 - C / 0.25, 0, 1))
        lab[..., 1] *= s
        lab[..., 2] *= s
    return oklab2rgb(lab)


def curves(img, points, channel="rgb"):
    """A tone curve through the points [(input, output), ...], values 0..1, e.g. [(0, 0), (0.25, 0.18), (1, 1)].

    channel: 'rgb' (all three, like a darkroom curve), 'r', 'g', 'b', or 'l' (lightness only, colours untouched)."""
    pts = sorted((float(x), float(y)) for x, y in points)
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1] if pts[0][0] < 1e-6 else 0.0))
    if pts[-1][0] < 1:
        pts.append((1.0, 1.0))
    xs, ys = zip(*pts)
    if len(set(xs)) != len(xs):
        raise ValueError("curves: two points share the same input value")
    f = PchipInterpolator(xs, ys)
    lut = np.clip(f(np.linspace(0, 1, 4096)), 0, 1).astype(np.float32)
    ap = lambda v: lut[np.clip(np.round(np.clip(v, 0, 1) * 4095), 0, 4095).astype(np.int32)]
    a = np.asarray(img, np.float32)[..., :3].copy()
    ch = channel.lower()
    if ch == "l":
        lab = rgb2oklab(a)
        lab[..., 0] = ap(lab[..., 0])
        return oklab2rgb(lab)
    for i, c in enumerate("rgb"):
        if ch == "rgb" or ch == c:
            a[..., i] = ap(a[..., i])
    return a
