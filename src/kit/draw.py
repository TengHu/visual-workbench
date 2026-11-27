"""Making new picture content: a vector canvas (Skia), a page renderer (Chrome) and shadows."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from .core import as_color, load, oklab2rgb, rgb2oklab

__all__ = ["canvas", "from_canvas", "render_html", "form", "shadow"]

_CHROME = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
]


def canvas(width, height):
    """A transparent Skia drawing surface of the given size: returns (surface, canvas). Draw on the canvas with
    skia (antialiased paths, gradients, blurs, blend modes, text), then turn it into a layer with from_canvas(surface).

        surface, c = canvas(W, H)
        p = skia.Paint(AntiAlias=True, Color=skia.ColorSetARGB(255, 200, 30, 30))
        c.drawCircle(400, 300, 80, p)
        layer = from_canvas(surface)"""
    import skia

    surface = skia.Surface(int(width), int(height))
    c = surface.getCanvas()
    c.clear(skia.ColorTRANSPARENT)
    return surface, c


def from_canvas(surface):
    """A Skia surface as a layer (H, W, 4) with straight alpha, ready for place() or blend()."""
    import skia

    img = surface.makeImageSnapshot()
    arr = img.toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)
    return np.ascontiguousarray(arr, np.float32) / 255.0


def render_html(html, width, height, wait_ms=0):
    """Render an HTML page (inline SVG, 2D canvas, WebGL, CSS) in headless Chrome and return it as a layer (H, W, 4).

    The page is `width` x `height` CSS pixels with a transparent background: whatever the page does not paint stays
    transparent, so the result drops straight into place() or blend(). Everything must be inline: there is no
    network and no external file. Draw synchronously, or set wait_ms to let scripts and timers finish.
    Anything a browser can draw is available: SVG filters and gradients, canvas pixel work, WebGL shaders."""
    exe = next((p for p in _CHROME if os.path.exists(p)), None) or shutil.which("chromium") or shutil.which("google-chrome")
    if not exe:
        raise RuntimeError("render_html needs Google Chrome, which is not installed here; draw with canvas() instead")
    work = tempfile.mkdtemp(prefix="render-", dir=".")
    try:
        page = os.path.join(work, "page.html")
        with open(page, "w") as f:
            f.write(html if "<html" in html.lower() else f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>html,body{{margin:0;padding:0;overflow:hidden;background:transparent}}</style></head><body>{html}</body></html>")
        shot = os.path.abspath(os.path.join(work, "shot.png"))
        cmd = [
            exe, "--headless=new", "--no-sandbox", "--disable-gpu-sandbox", "--hide-scrollbars", "--no-first-run",
            "--disable-extensions", "--disable-background-networking", "--disable-component-update", "--disable-sync",
            "--force-device-scale-factor=1", "--default-background-color=00000000",
            f"--user-data-dir={os.path.abspath(os.path.join(work, 'profile'))}",
            f"--window-size={int(width)},{int(height)}", f"--screenshot={shot}",
        ]
        if wait_ms:
            cmd.append(f"--virtual-time-budget={int(wait_ms)}")
        cmd.append("file://" + os.path.abspath(page))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if not os.path.exists(shot):
            raise RuntimeError("render_html: Chrome made no picture. " + (r.stderr or "")[-600:])
        lay = load(shot, alpha=True)
        if lay.shape[0] != int(height) or lay.shape[1] != int(width):
            lay = lay[: int(height), : int(width)]
        return lay
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _inflate(mask):
    """A rounded height for a flat shape, in pixels: 0 at its outline, highest along its middle, as if the shape
    were blown up like a cushion. (The solution of a Poisson equation inside the shape: a band becomes a half
    cylinder, a disc a dome.)"""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    m = np.asarray(mask, np.float32) > 0.5
    h, w = m.shape
    ys, xs = np.nonzero(m)
    if not len(ys):
        return np.zeros((h, w), np.float32)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    f = max(1.0, max(x1 - x0, y1 - y0) / 150.0)
    sw, sh = max(3, int(round((x1 - x0) / f))), max(3, int(round((y1 - y0) / f)))
    small = np.pad(cv2.resize(m[y0:y1, x0:x1].astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.5, 1)
    idx = -np.ones(small.shape, np.int64)
    n = int(small.sum())
    if n < 4:
        return np.zeros((h, w), np.float32)
    idx[small] = np.arange(n)
    yy, xx = np.nonzero(small)
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [np.full(n, 4.0)]
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        j = idx[yy + dy, xx + dx]
        ok = j >= 0
        rows.append(np.arange(n)[ok]); cols.append(j[ok]); vals.append(np.full(int(ok.sum()), -1.0))
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    u = np.zeros(small.shape, np.float32)
    u[small] = spsolve(A.tocsc(), np.ones(n)).astype(np.float32)
    u = cv2.resize(u[1:-1, 1:-1], (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    if f > 1:  # the solve was done small: smooth away its grid before the height is taken
        u = cv2.GaussianBlur(u, (0, 0), 0.8 * f)
    out = np.zeros((h, w), np.float32)
    out[y0:y1, x0:x1] = f * np.sqrt(2.0 * np.clip(u, 0, None))
    return out * m


def _light_and_shade(base, tone, contrast=0.5, shade=None, lit=None):
    """Oklab colours between a thing's colour in the light (tone 1) and in its shadow (tone 0). The shadow is the
    colour darker and a little more saturated; or, given a shadow colour (and the lit colour it goes with), the
    same shift as from that lit colour to that shadow colour. Shared by form() and Sculpture.render()."""
    lab_lit = rgb2oklab(np.asarray(base, np.float32))
    lab_shade = lab_lit.copy()
    if shade is None:
        lab_shade[..., 0] *= 1.0 - 0.6 * float(contrast)
        lab_shade[..., 1:] *= 1.12
    else:
        ref_lit = rgb2oklab(as_color(lit if lit is not None else "#ffffff").reshape(1, 1, 3))[0, 0]
        ref_shade = rgb2oklab(as_color(shade).reshape(1, 1, 3))[0, 0]
        lab_shade[..., 0] *= float(ref_shade[0] / max(ref_lit[0], 1e-3))
        lab_shade[..., 1:] += ref_shade[1:] - ref_lit[1:]
    return lab_shade + (lab_lit - lab_shade) * np.clip(tone, 0, 1)[..., None]


def _shine(lab, spec):
    """A specular highlight (spec 0..1 per pixel) on Oklab colours."""
    lab = lab.copy()
    lab[..., 0] += spec * (1.0 - lab[..., 0])
    lab[..., 1:] *= (1.0 - 0.7 * spec)[..., None]
    return lab


def form(mask, color, light=(-0.6, -0.7), contrast=0.5, roundness=1.0, shade=None, highlight=0.0, reflected=0.15):
    """Give a flat shape volume: the shape filled with `color` and shaded as a rounded form under one light.
    Returns a layer (H, W, 4).

    mask       the shape. Its outline is where the form turns away; its middle is nearest to you. A long shape
               becomes a cylinder, a round one a dome. Several forms, each on its own layer, can be stacked
               nearer over farther.
    color      the colour in the light: '#rrggbb', or a picture (H, W, 3) to shade a pattern or a drawn surface.
    light      the direction toward the light, (dx, dy) (dy negative = from above; the same convention as
               Sculpture.render, which adds a z), or the dict that light_on() measured on the photograph or the reference, which
               also sets contrast and the colours of the lit and shadow sides.
    contrast   0..1: how dark the shadow side is (soft overcast light under 0.25, hard sun over 0.6).
    roundness  how strongly the form turns: 0 flat, 1 a full round, above 1 a sharper turn.
    shade      the colour of the shadow side; default: the lit colour darker and a little more saturated.
               highlight (0..1) adds a shine facing the light; reflected (0..1) lifts the far rim of the
               shadow side, as light bounced back from the surroundings does.
    The form turns at the mask's outline even when the mask is feathered; a colour blended through a feathered
    mask gives a soft patch with no turn. The layer goes on the stack (stack.put("arm", form(m, "#c98f6b"))) or
    into blend()."""
    m = np.clip(np.asarray(mask, np.float32), 0, 1)
    if m.ndim != 2:
        raise ValueError("form: the mask must be (H, W)")
    h, w = m.shape
    lit_c, shade_c = None, shade
    if isinstance(light, dict):
        contrast = float(light.get("contrast", contrast))
        lit_c, shade_c = light.get("lit"), (shade if shade is not None else light.get("shade"))
        light = light["direction"]
    base = np.asarray(color, np.float32)[..., :3] if (not isinstance(color, str) and np.ndim(color) == 3) else np.broadcast_to(as_color(color), (h, w, 3))
    if base.shape[:2] != (h, w):
        raise ValueError("form: the colour picture and the mask differ in size")
    hgt = cv2.GaussianBlur(_inflate(m), (0, 0), 1.0)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * float(roundness)
    gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * float(roundness)
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny = -gx * nz, -gy * nz
    L = np.array([float(light[0]), float(light[1]), 0.75], np.float32)
    L /= np.linalg.norm(L)
    d = nx * L[0] + ny * L[1] + nz * L[2]
    tone = np.clip(d * 1.15 + 0.12, 0, 1) + float(reflected) * np.clip(-d - 0.15, 0, 1)
    ref_lit = None if shade_c is None else (lit_c if lit_c is not None else (color if isinstance(color, str) else base.reshape(-1, 3).mean(0)))
    lab = _light_and_shade(base, tone, contrast, shade_c, ref_lit)
    if highlight > 0:
        H_ = L + np.array([0, 0, 1], np.float32)
        H_ /= np.linalg.norm(H_)
        spec = np.clip(nx * H_[0] + ny * H_[1] + nz * H_[2], 0, 1) ** 40 * float(highlight)
        lab = _shine(lab, spec)
    return np.dstack([oklab2rgb(lab), m]).astype(np.float32)


def shadow(mask, offset=(0, 0), blur=10.0, opacity=0.5, squash=1.0):
    """A shadow shape made from a mask or a layer's alpha: returns a mask to darken the picture with.

    offset   (dx, dy) the shadow is cast, in pixels, away from the light.
    blur     softness in pixels: small right under an object (contact shadow), larger the further it falls.
    squash   under 1 flattens the shadow vertically about the mask's bottom edge, as on a ground plane.
    Apply it with:  img = blend(img, adjust(img, exposure=-1.5), shadow_mask)   or multiply by a shadow colour.
    Two calls with different blur and offset give a tight shadow and a wider, softer one."""
    m = np.asarray(mask, np.float32)
    if m.ndim == 3:
        m = m[..., 3]
    h, w = m.shape
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    if squash != 1.0:
        ys = np.where((m > 0.5).any(1))[0]
        foot = float(ys[-1]) if len(ys) else h / 2.0
        M = np.float32([[1, 0, offset[0]], [0, squash, foot * (1 - squash) + offset[1]]])
    s = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if blur > 0:
        s = cv2.GaussianBlur(s, (0, 0), float(blur))
    return np.clip(s * float(opacity), 0, 1).astype(np.float32)
