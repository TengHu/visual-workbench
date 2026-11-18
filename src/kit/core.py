"""Pictures, files and looking: the part of the kit everything else rests on.

Conventions, used by every function in the kit:
  picture  float32 array (H, W, 3), sRGB values 0..1
  layer    float32 array (H, W, 4), a picture with an alpha channel (straight, not premultiplied)
  mask     float32 array (H, W), 0..1; soft edges are allowed everywhere
  box      (x0, y0, x1, y1) in pixels of the full picture; point (x, y)
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps

__all__ = [
    "load", "save", "show", "compare", "overlay", "View",
    "to_linear", "to_srgb", "rgb2oklab", "oklab2rgb", "luma", "as_color",
]

# ---------------------------------------------------------------------------------------------------------------
# Session state, shared with the worker that runs the model's code.

#: How the model writes positions it reads off a view: "yx1000" ([y, x], 0..1000 of the view; Gemini's convention)
#: or "xy" ([x, y] in pixels of the view image).
COORDS = os.environ.get("KIT_COORDS", "yx1000")
#: Longest side, in pixels, of a view sent to the model.
VIEW_SIDE = int(os.environ.get("KIT_VIEW_SIDE", "1536"))
#: Views made by show()/compare() since the worker last collected them.
OUTBOX: list[dict] = []
VIEWS: dict[str, "View"] = {}
_state = {"n": 0, "icc": None, "dir": "views", "saved": [], "calls": {}, "private": False, "brief": ""}


def _count(name):
    """Note that a tool was used (by the one doing the work, not by the review sheet): the run's record of method."""
    if not _state.get("private"):
        _state["calls"][name] = _state["calls"].get(name, 0) + 1

_FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/SFNS.ttf", "/Library/Fonts/Arial.ttf"]


def _font(size: int):
    for f in _FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------------------------------------------
# Colour


def to_linear(img):
    """sRGB-encoded values (0..1) to linear light. Mix, blur and scale light in linear; then come back with to_srgb."""
    img = np.asarray(img, np.float32)
    return np.where(img <= 0.04045, img / 12.92, ((np.maximum(img, 0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(lin):
    """Linear light back to sRGB-encoded values (0..1)."""
    lin = np.clip(np.asarray(lin, np.float32), 0, None)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099], [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)


def rgb2oklab(img):
    """Picture to Oklab (L 0..1, a and b about -0.4..0.4): the space to change colour in. Hue is atan2(b, a), chroma hypot(a, b)."""
    lms = np.cbrt(np.clip(to_linear(img)[..., :3] @ _M1.T, 0, None))
    return (lms @ _M2.T).astype(np.float32)


def oklab2rgb(lab):
    """Oklab back to a picture (values clipped to 0..1)."""
    lms = (np.asarray(lab, np.float32) @ np.linalg.inv(_M2).T) ** 3
    return np.clip(to_srgb(lms @ np.linalg.inv(_M1).T), 0, 1)


def luma(img):
    """Perceived lightness of each pixel, 0..1 (Oklab L)."""
    return rgb2oklab(img)[..., 0]


def as_color(c):
    """A colour as three sRGB floats 0..1, from '#rrggbb', (r, g, b) in 0..1 or in 0..255."""
    if isinstance(c, str):
        s = c.lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        return np.array([int(s[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)
    a = np.asarray(c, np.float32).reshape(-1)[:3]
    return a / 255 if a.max() > 1.0001 else a


# ---------------------------------------------------------------------------------------------------------------
# Files


def load(path, alpha=False):
    """Open a picture file as a float picture (H, W, 3). With alpha=True returns a layer (H, W, 4)."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if _state["icc"] is None and im.info.get("icc_profile"):
        _state["icc"] = im.info["icc_profile"]
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im, np.float32) / 65535.0
        a = np.stack([a] * 3, -1)
        return np.dstack([a, np.ones(a.shape[:2], np.float32)]) if alpha else a
    a = np.asarray(im.convert("RGBA" if alpha else "RGB"), np.float32) / 255.0
    return np.ascontiguousarray(a)


def _to8(img):
    return (np.clip(np.asarray(img, np.float32), 0, 1) * 255 + 0.5).astype(np.uint8)


def save(img, path="result.png"):
    """Write a picture (or layer, or mask) to a PNG file in the workspace. `result.png` is the file you deliver.

    Every save of a picture is also kept as a numbered snapshot, so an earlier version can be delivered instead."""
    a = _to8(img)
    im = Image.fromarray(a)
    kw = {"icc_profile": _state["icc"]} if _state["icc"] and a.ndim == 3 else {}
    im.save(path, **kw)
    if a.ndim == 3 and a.shape[2] == 3 and os.path.basename(path) == "result.png":
        os.makedirs("snapshots", exist_ok=True)
        n = len(_state["saved"]) + 1
        snap = os.path.join("snapshots", f"result-{n:02d}.png")
        im.save(snap, **kw)
        _state["saved"].append(snap)
        print(f"saved {path} ({a.shape[1]}x{a.shape[0]}); snapshot {snap}")
    else:
        print(f"saved {path} ({a.shape[1]}x{a.shape[0]})")
    return path


# ---------------------------------------------------------------------------------------------------------------
# Looking


class View:
    """One picture you were shown. It knows which part of the full picture it covers, and turns a position you read
    off it into full-picture pixels: v.pt(...), v.box(...), v.poly(...)."""

    def __init__(self, id, box, size, caption, mappable=True):
        self.id, self.region, self.size, self.caption, self.mappable = id, tuple(float(b) for b in box), size, caption, mappable

    def _xy(self, a, b):
        if not self.mappable:
            raise ValueError(f"{self.id} is a comparison sheet, not a plain view: read positions off a view made with show().")
        x0, y0, x1, y1 = self.region
        if COORDS == "yx1000":
            y, x = float(a), float(b)
            if max(abs(x), abs(y)) > 1100:
                raise ValueError("positions on a view are 0..1000 (1000 = the view's right or bottom edge), not pixels")
            return x0 + x / 1000.0 * (x1 - x0), y0 + y / 1000.0 * (y1 - y0)
        x, y = float(a), float(b)
        return x0 + x / self.size[0] * (x1 - x0), y0 + y / self.size[1] * (y1 - y0)

    def pt(self, p):
        """A position read off this view -> (x, y) in full-picture pixels."""
        return self._xy(p[0], p[1])

    def pts(self, ps):
        """Several positions read off this view -> [(x, y), ...] in full-picture pixels."""
        return [self.pt(p) for p in ps]

    poly = pts

    def box(self, b):
        """A box read off this view -> (x0, y0, x1, y1) in full-picture pixels."""
        xa, ya = self._xy(b[0], b[1])
        xb, yb = self._xy(b[2], b[3])
        return (int(round(min(xa, xb))), int(round(min(ya, yb))), int(round(max(xa, xb))), int(round(max(ya, yb))))

    def __repr__(self):
        x0, y0, x1, y1 = self.region
        return f"<View {self.id}: x {x0:.0f}..{x1:.0f}, y {y0:.0f}..{y1:.0f} of the picture, shown at {self.size[0]}x{self.size[1]}>"


def _as_rgb8(img):
    """Anything showable -> uint8 RGB. A mask becomes grey; a layer is laid over a checkerboard."""
    if isinstance(img, (str, os.PathLike)):
        img = load(img, alpha=True)
        if float(img[..., 3].min()) > 0.999:
            img = img[..., :3]
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255
    elif a.dtype == bool:
        a = a.astype(np.float32)
    a = a.astype(np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    elif a.shape[2] == 4:
        h, w = a.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        s = max(8, max(h, w) // 64)
        board = np.where(((yy // s + xx // s) % 2)[..., None] == 0, 0.62, 0.42).astype(np.float32)
        a = a[..., :3] * a[..., 3:4] + board * (1 - a[..., 3:4])
    return _to8(a[..., :3])


def _clip_box(box, w, h):
    x0, y0, x1, y1 = box
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError(f"box {tuple(box)} covers no pixels of a {w}x{h} picture")
    return x0, y0, x1, y1


def _fit(crop8, side):
    """Resize for viewing: shrink to `side`, or enlarge a small crop so its pixels can be seen."""
    h, w = crop8.shape[:2]
    scale = side / max(h, w)
    if scale >= 1:
        scale = min(scale, 8.0, max(1.0, 1024 / max(h, w)))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if (nw, nh) == (w, h):
        return crop8, 1.0
    interp = cv2.INTER_AREA if scale < 1 else (cv2.INTER_NEAREST if scale >= 3 else cv2.INTER_CUBIC)
    return cv2.resize(crop8, (nw, nh), interpolation=interp), scale


def _draw_grid(im8):
    """Lines every tenth of the view, labelled in the units positions are written in."""
    pil = Image.fromarray(im8).convert("RGBA")
    w, h = pil.size
    lay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    f = _font(max(11, min(w, h) // 60))
    for i in range(1, 10):
        x, y = round(w * i / 10), round(h * i / 10)
        for off, col in ((0, (255, 255, 255, 150)), (1, (0, 0, 0, 150))):
            d.line([(x + off, 0), (x + off, h)], fill=col, width=1)
            d.line([(0, y + off), (w, y + off)], fill=col, width=1)
        lx = str(i * 100) if COORDS == "yx1000" else str(x)
        ly = str(i * 100) if COORDS == "yx1000" else str(y)
        for (tx, ty, t) in ((x + 3, 2, lx), (x + 3, h - f.size - 4, lx), (3, y + 2, ly), (w - f.getlength(ly) - 4, y + 2, ly)):
            d.rectangle([tx - 2, ty - 1, tx + f.getlength(t) + 2, ty + f.size + 2], fill=(0, 0, 0, 170))
            d.text((tx, ty), t, fill=(255, 255, 255, 255), font=f)
    return np.asarray(Image.alpha_composite(pil, lay).convert("RGB"))


def _emit(im8, caption, region, mappable=True):
    _state["n"] += 1
    vid = f"v{_state['n']}"
    os.makedirs(_state["dir"], exist_ok=True)
    path = os.path.join(_state["dir"], f"{vid}.jpg")
    Image.fromarray(im8).save(path, "JPEG", quality=92, subsampling=0)
    h, w = im8.shape[:2]
    v = View(vid, region, (w, h), caption, mappable)
    VIEWS[vid] = v
    x0, y0, x1, y1 = region
    where = f"x {x0:.0f}..{x1:.0f}, y {y0:.0f}..{y1:.0f} of the picture" if mappable else "comparison sheet"
    OUTBOX.append({"id": vid, "file": path, "caption": caption, "note": f"{where}; shown at {w}x{h}"})
    return v


def show(img, caption="", box=None, grid=False):
    """Look at a picture, mask or layer (array or file name). Returns a View; you see it after this python call ends.

    box   (x0, y0, x1, y1) in full-picture pixels: look at just that part, enlarged. Use it to zoom.
    grid  draw coordinate lines every tenth of the view, labelled in the units you write positions in.
    A whole picture is shown reduced; to judge edges, texture and grain, zoom until one picture pixel is at least
    one view pixel (a box about 1500 px wide or less)."""
    a = _as_rgb8(img)
    h, w = a.shape[:2]
    region = _clip_box(box, w, h) if box is not None else (0, 0, w, h)
    x0, y0, x1, y1 = region
    if box is not None:
        _count("zoom")
    view, scale = _fit(a[y0:y1, x0:x1], VIEW_SIDE)
    if grid:
        view = _draw_grid(view)
    v = _emit(view, caption, region)
    if box is None and not grid and a.shape[:2] == _state.get("size") and np.ndim(img) == 3 and np.shape(img)[2] == 3:
        # The latest whole picture looked at: delivered if the job ends before anything was saved.
        Image.fromarray(a).save("last_shown.png")
    print(f"{v.id}: {caption or 'view'} [x {x0}..{x1}, y {y0}..{y1}; {scale:.2f}x]")
    return v


def compare(a, b, caption="", box=None, labels=("before", "after")):
    """Look at two pictures side by side (the same box of each), to judge a change. Positions cannot be read off it."""
    A, B = _as_rgb8(a), _as_rgb8(b)
    if A.shape != B.shape:
        raise ValueError(f"the two pictures differ in size: {A.shape[1]}x{A.shape[0]} and {B.shape[1]}x{B.shape[0]}")
    h, w = A.shape[:2]
    x0, y0, x1, y1 = _clip_box(box, w, h) if box is not None else (0, 0, w, h)
    if box is not None:
        _count("zoom")
    wide = (x1 - x0) >= (y1 - y0) * 1.25
    side = VIEW_SIDE if wide else VIEW_SIDE // 2
    pa, s = _fit(A[y0:y1, x0:x1], side)
    pb, _ = _fit(B[y0:y1, x0:x1], side)
    ph, pw = pa.shape[:2]
    bar, gap = 26, 6
    if wide:  # stack vertically
        sheet = np.full((2 * (ph + bar) + gap, pw, 3), 24, np.uint8)
        spots = [(0, 0), (0, ph + bar + gap)]
    else:
        sheet = np.full((ph + bar, 2 * pw + gap, 3), 24, np.uint8)
        spots = [(0, 0), (pw + gap, 0)]
    pil = Image.fromarray(sheet)
    d = ImageDraw.Draw(pil)
    for (px, py), tile, lab in zip(spots, (pa, pb), labels):
        pil.paste(Image.fromarray(tile), (px, py + bar))
        d.text((px + 6, py + 4), str(lab), fill=(255, 255, 255), font=_font(16))
    v = _emit(np.asarray(pil), caption or f"{labels[0]} | {labels[1]}", (x0, y0, x1, y1), mappable=False)
    print(f"{v.id}: {caption or 'comparison'} [{labels[0]} {'above' if wide else 'left of'} {labels[1]}; x {x0}..{x1}, y {y0}..{y1}; {s:.2f}x]")
    return v


def overlay(img, mask, color=(1.0, 0.1, 0.5), opacity=0.55, outline=True):
    """A picture with a mask tinted over it, to check a selection: show(overlay(img, m), box=...). Edge drawn in white."""
    a = np.asarray(img, np.float32)[..., :3]
    m = np.clip(np.asarray(mask, np.float32), 0, 1)
    if m.shape != a.shape[:2]:
        raise ValueError(f"mask is {m.shape[1]}x{m.shape[0]} but the picture is {a.shape[1]}x{a.shape[0]}")
    c = as_color(color)
    out = a * (1 - opacity * m[..., None]) + c * (opacity * m[..., None])
    if outline:
        hard = (m > 0.5).astype(np.uint8)
        k = max(1, round(max(a.shape[:2]) / 900))
        edge = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        out[edge] = 1.0
    return out.astype(np.float32)
