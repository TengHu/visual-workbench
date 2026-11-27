"""Measuring: exact outlines, body and face landmarks, palettes and shape comparisons, so that what you draw can be
checked in pixels instead of by impression. And references: pictures that came with the request, to work from.

These tools measure and compare. None of them makes pixels.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .core import _as_rgb8, _state, as_color, oklab2rgb, rgb2oklab, show
from .masks import fill_holes, keep_largest, mask_box, select_object

__all__ = ["outline", "landmarks", "palette", "light_on", "shape_diff", "landmark_diff", "edge_overlay", "Ref"]

POSE = ["nose", "eye_in_l", "eye_l", "eye_out_l", "eye_in_r", "eye_r", "eye_out_r", "ear_l", "ear_r", "mouth_l", "mouth_r", "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r", "pinky_l", "pinky_r", "index_l", "index_r", "thumb_l", "thumb_r", "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r", "heel_l", "heel_r", "toe_l", "toe_r"]
# Face-mesh points worth naming. "a" is the side on the picture's left, "b" on its right.
FACE = {"forehead": 10, "chin": 152, "nose_tip": 1, "nose_bridge": 6, "eye_a_outer": 33, "eye_a_inner": 133, "eye_a_top": 159, "eye_a_bottom": 145, "eye_b_inner": 362, "eye_b_outer": 263, "eye_b_top": 386, "eye_b_bottom": 374, "brow_a": 105, "brow_b": 334, "mouth_a": 61, "mouth_b": 291, "lip_top": 13, "lip_bottom": 14, "cheek_a": 234, "cheek_b": 454, "iris_a": 468, "iris_b": 473}
_detectors: dict = {}


def _models():
    return os.environ.get("KIT_MODELS") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".models")


def _contour(mask, points):
    cs, _ = cv2.findContours((np.asarray(mask) > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(np.r_[c, c[:1]], axis=0).T))]
    t = np.linspace(0, d[-1], int(points), endpoint=False)
    closed = np.r_[c, c[:1]]
    return [(float(np.interp(v, d, closed[:, 0])), float(np.interp(v, d, closed[:, 1]))) for v in t]


def outline(img, box, fg=(), bg=(), points=160):
    """The exact outline of the thing inside a box, as `points` points [(x, y), ...] going round it.

    Like select_object, but returns the contour instead of a mask: use it as a guide to draw to, to build a
    poly_mask, or to compare a drawn shape with a target shape."""
    return _contour(select_object(img, box, fg, bg, refine=False), points)


def landmarks(img, kind="pose"):
    """Find a person's landmarks in a picture: kind='pose' (33 body points: nose, shoulders, elbows, wrists, hips,
    knees, ankles ...), 'face' (named points: eyes, brows, nose, mouth, chin, cheeks) or 'hands'.

    Returns {name: (x, y)} in the picture's pixels ('hands': a list of 21-point lists), or {} when none is found.
    Works on photographs and on reasonably realistic drawings; a loose sketch may give nothing. landmark_diff()
    compares two sets of them."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
    except Exception as e:
        raise RuntimeError(f"landmarks needs MediaPipe, which is not installed here ({type(e).__name__})")
    spec = {"pose": ("pose_landmarker_heavy.task", vision.PoseLandmarker, vision.PoseLandmarkerOptions), "face": ("face_landmarker.task", vision.FaceLandmarker, vision.FaceLandmarkerOptions), "hands": ("hand_landmarker.task", vision.HandLandmarker, vision.HandLandmarkerOptions)}
    if kind not in spec:
        raise ValueError("kind must be 'pose', 'face' or 'hands'")
    file, cls, opt = spec[kind]
    path = os.path.join(_models(), file)
    if not os.path.exists(path):
        raise RuntimeError(f"landmarks: the model file {file} is missing; run setup.sh")
    if kind not in _detectors:
        extra = {"num_hands": 2} if kind == "hands" else {}
        _detectors[kind] = cls.create_from_options(opt(base_options=mpp.BaseOptions(model_asset_path=path), **extra))
    a = np.ascontiguousarray(_as_rgb8(img))
    h, w = a.shape[:2]
    r = _detectors[kind].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=a))
    if kind == "pose":
        if not r.pose_landmarks:
            return {}
        return {n: (p.x * w, p.y * h) for n, p in zip(POSE, r.pose_landmarks[0])}
    if kind == "face":
        if not r.face_landmarks:
            return {}
        pts = r.face_landmarks[0]
        return {n: (pts[i].x * w, pts[i].y * h) for n, i in FACE.items() if i < len(pts)}
    return [[(p.x * w, p.y * h) for p in hand] for hand in r.hand_landmarks]


def palette(img, mask=None, n=6):
    """The n main colours of a picture (or of its masked part): [('#rrggbb', share), ...], largest first. Printed too.
    """
    a = np.asarray(img, np.float32)[..., :3]
    px = a[np.asarray(mask) > 0.5] if mask is not None else a.reshape(-1, 3)
    if len(px) < n:
        raise ValueError("palette: the mask selects too few pixels")
    rng = np.random.default_rng(0)
    px = px[rng.choice(len(px), min(len(px), 40000), replace=False)]
    lab = rgb2oklab(px.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float32)
    _, lab_, centres = cv2.kmeans(lab, int(n), None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4), 3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lab_.ravel(), minlength=int(n)) / len(lab)
    rgb = oklab2rgb(centres.reshape(-1, 1, 3)).reshape(-1, 3)
    out = [("#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb[i]), float(share[i])) for i in np.argsort(share)[::-1]]
    print("palette: " + "  ".join(f"{c} {100 * s:.0f}%" for c, s in out))
    return out


def light_on(img, mask):
    """How a thing in the picture is lit, measured from its own shading. Prints and returns a dict:

    direction   (dx, dy), a unit vector in the picture pointing from the thing's shadow side to its lit side
                (dy negative = lit from above); `angle` is the same in degrees (0 = lit from the right, 90 = from below).
    contrast    how much darker the shadow side is than the lit side, 0..1 (soft overcast light is under 0.2,
                hard sun over 0.5).
    lit, shade  the typical colour of its lit and shadow parts ('#rrggbb'): their difference is the colour of the light.
    It is reliable on a simple rounded form of one plain colour (a face, an arm, a ball, a wall); on a patterned,
    shiny, nearly black or nearly white thing it misleads. The dict can be passed as `light=` to form()."""
    a = np.asarray(img, np.float32)[..., :3]
    m = np.asarray(mask, np.float32) > 0.5
    if m.sum() < 200:
        raise ValueError("light_on: the mask selects too little")
    L = rgb2oklab(a)[..., 0]
    Ls = cv2.GaussianBlur(L, (0, 0), max(2.0, 0.02 * np.sqrt(m.sum())))
    ys, xs = np.nonzero(m)
    A = np.c_[xs - xs.mean(), ys - ys.mean(), np.ones(len(xs))]
    coef, *_ = np.linalg.lstsq(A, Ls[ys, xs], rcond=None)
    g = np.array([coef[0], coef[1]], np.float64)
    n = float(np.hypot(*g))
    d = (g / n) if n > 1e-9 else np.array([0.0, -1.0])
    v = L[ys, xs]
    lo, hi = np.percentile(v, 15), np.percentile(v, 85)
    hexc = lambda sel: "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in np.clip(np.median(a[ys[sel], xs[sel]], 0), 0, 1))
    out = {"direction": (float(d[0]), float(d[1])), "angle": float(np.degrees(np.arctan2(d[1], d[0]))), "contrast": float(np.clip((hi - lo) / max(hi, 1e-3), 0, 1)), "lit": hexc(v >= hi), "shade": hexc(v <= lo)}
    horiz = "right" if d[0] > 0.38 else "left" if d[0] < -0.38 else ""
    vert = "" if abs(d[1]) < 0.38 else ("above" if d[1] < 0 else "below")
    side = {"above": "upper ", "below": "lower "}[vert] + horiz if horiz and vert else (horiz or vert or "front")
    print(f"light_on: lit from {'the ' if horiz else ''}{side} (direction ({d[0]:+.2f}, {d[1]:+.2f})); contrast {out['contrast']:.2f}; lit {out['lit']}, shade {out['shade']}")
    if float(np.median(v)) < 0.22 or float(np.median(v)) > 0.93:
        print("  (this thing is nearly " + ("black" if float(np.median(v)) < 0.5 else "white") + ": its shading says little about the light. Measure something of a middle tone near it, or judge the light by eye from the shadows it casts.)")
    return out


def shape_diff(mask, ref_mask, quiet=False):
    """Compare a shape with a target shape (two masks of the same picture size). Prints and returns:
    overlap (1 = identical), how far the centre is off (dx, dy in pixels, + is right/down), and how much too wide and
    too tall it is (1.10 = 10% larger than the target)."""
    a, b = np.asarray(mask) > 0.5, np.asarray(ref_mask) > 0.5
    if a.shape != b.shape:
        raise ValueError("shape_diff: the two masks differ in size")
    if not a.any() or not b.any():
        raise ValueError("shape_diff: a mask is empty")
    ba, bb = mask_box(a.astype(np.float32)), mask_box(b.astype(np.float32))
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    out = {
        "overlap": float((a & b).sum() / (a | b).sum()),
        "dx": float(xa.mean() - xb.mean()), "dy": float(ya.mean() - yb.mean()),
        "width": float((ba[2] - ba[0]) / max(1, bb[2] - bb[0])), "height": float((ba[3] - ba[1]) / max(1, bb[3] - bb[1])),
        "area": float(a.sum() / b.sum()),
    }
    if not quiet:
        print(f"shape_diff: overlap {out['overlap']:.2f}; centre off by ({out['dx']:+.0f}, {out['dy']:+.0f}) px; width x{out['width']:.2f}, height x{out['height']:.2f}, area x{out['area']:.2f}")
    return out


def landmark_diff(mine, theirs, tolerance=None):
    """Compare two sets of landmarks {name: (x, y)} in the same coordinates: yours and the ones to match. Prints each
    point that is off by more than `tolerance` pixels (default 1.5% of the figure's size), worst first, with the move
    that would fix it; returns {name: (dx, dy)} for all shared points."""
    keys = [k for k in mine if k in theirs]
    if not keys:
        print("landmark_diff: no landmark is in both sets")
        return {}
    ref = np.array([theirs[k] for k in keys], np.float32)
    size = float(max(np.ptp(ref[:, 0]), np.ptp(ref[:, 1]), 1))
    tol = tolerance if tolerance is not None else 0.015 * size
    d = {k: (float(mine[k][0] - theirs[k][0]), float(mine[k][1] - theirs[k][1])) for k in keys}
    off = sorted(((np.hypot(*v), k) for k, v in d.items() if np.hypot(*v) > tol), reverse=True)
    if not off:
        print(f"landmark_diff: all {len(keys)} points within {tol:.0f} px")
    else:
        print(f"landmark_diff: {len(off)} of {len(keys)} points off by more than {tol:.0f} px:")
        for dist, k in off[:14]:
            dx, dy = d[k]
            print(f"  {k}: {dist:.0f} px off; move it {abs(dx):.0f} {'left' if dx > 0 else 'right'}, {abs(dy):.0f} {'up' if dy > 0 else 'down'}")
    return d


def edge_overlay(drawing, target):
    """A picture that lays the edges of two same-size pictures over each other: the target's in blue, yours in
    orange, on grey. Where they run together the structure matches; where they part, yours is off by that much."""
    A, B = _as_rgb8(drawing), _as_rgb8(target)
    if A.shape != B.shape:
        raise ValueError("edge_overlay: the two pictures differ in size")
    k = max(1, round(max(A.shape[:2]) / 700))

    def edges(x):
        g = cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (0, 0), 1.2 * k)
        e = cv2.Canny(g, 30, 90)
        return cv2.dilate(e, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0

    out = np.full(A.shape, 0.78, np.float32) * (0.5 + 0.5 * cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)[..., None] / 255)
    eb, ea = edges(B), edges(A)
    out[eb] = (0.1, 0.35, 0.95)
    out[ea] = (0.95, 0.45, 0.05)
    out[ea & eb] = (0.1, 0.1, 0.1)
    return out


# ---------------------------------------------------------------------------------------------------------------
# References

_LEVELS = ("look", "measure", "sample")


class Ref:
    """A picture that came with the request, to work from. `refs` is the list of them (refs[0] is the first).

    Its own pixels are addressed in its own coordinates: ref.show() makes a view of it, and positions you read off
    that view convert with v.box / v.pt as usual, giving reference pixels. What a reference *returns* is in your
    picture's coordinates, mapped through ref.fit: by default the reference is fitted centred into your picture;
    ref.fit((x0, y0, x1, y1)) says instead which box of your picture the whole reference corresponds to.

    How much of it you may use is set for the run (ref.level):
      look     ref.show(...) only.
      measure  also ref.outline, ref.mask, ref.lines, ref.landmarks, ref.palette, ref.light, ref.tones and ref.check: shapes,
               colours as a palette, the light, and a coarse tone plan, but never its pixels.
      sample   also ref.picture (its pixels) and ref.on_canvas() (the picture fitted into your frame)."""

    def __init__(self, picture, level, name, canvas_size):
        self._px = np.asarray(picture, np.float32)[..., :3]
        self.level, self.name = level, name
        self.h, self.w = self._px.shape[:2]
        cw, ch = canvas_size
        s = min(cw / self.w, ch / self.h)
        self._default_fit = ((cw - self.w * s) / 2, (ch - self.h * s) / 2, (cw + self.w * s) / 2, (ch + self.h * s) / 2)
        self._canvas = (cw, ch)

    @property
    def _fit(self):
        # Kept with the session, not the object, so that the review sheet measures against the fit you set.
        return _state.setdefault("ref_fit", {}).get(self.name, self._default_fit)

    def _need(self, level, what):
        if _LEVELS.index(self.level) < _LEVELS.index(level):
            raise PermissionError(f"{what} is not available: this run lets you only {self.level} at references (it needs '{level}'). Work from ref.show().")

    def fit(self, box=None):
        """Set (or, with no argument, get) the box of your picture that the whole reference corresponds to."""
        if box is not None:
            _state.setdefault("ref_fit", {})[self.name] = tuple(float(v) for v in box)
        return self._fit

    def to_picture(self, pts):
        """Reference pixels -> your picture's pixels, for a list of (x, y)."""
        x0, y0, x1, y1 = self._fit
        return [(x0 + p[0] / self.w * (x1 - x0), y0 + p[1] / self.h * (y1 - y0)) for p in pts]

    def show(self, caption="", box=None, grid=False):
        """Look at the reference, or zoom into a box of it (in reference pixels). Allowed at every level."""
        return show(self._px, caption or f"reference {self.name}", box=box, grid=grid)

    def outline(self, box, fg=(), bg=(), points=160):
        """The outline of the thing inside a box of the reference, as points in YOUR picture's coordinates."""
        self._need("measure", "ref.outline")
        return self.to_picture(outline(self._px, box, fg, bg, points))

    def mask(self, box, fg=(), bg=()):
        """The shape of the thing inside a box of the reference, as a mask on YOUR picture (through ref.fit):
        the silhouette to draw a part into, exact where an outline of points would be coarse. See select_object."""
        self._need("measure", "ref.mask")
        # A clean shape and nothing else: one piece, no holes, a smooth edge. (A soft selection would carry the
        # reference's own detail along in its greys.)
        hard = fill_holes(keep_largest((select_object(self._px, box, fg, bg, refine=False) > 0.5).astype(np.float32))) > 0.5
        m = np.clip((cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.5) - 0.5) * 3 + 0.5, 0, 1)
        cw, ch = self._canvas
        x0, y0, x1, y1 = self._fit
        M = np.float32([[(x1 - x0) / self.w, 0, x0], [0, (y1 - y0) / self.h, y0]])
        return np.clip(cv2.warpAffine(np.asarray(m, np.float32), M, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, 1)

    def lines(self, strength=1.0):
        """For a reference that is a sketch or a line drawing: its drawn lines as a mask (H, W) on YOUR picture
        (through ref.fit), 1 on a dark line, 0 on paper. Put it on the stack as a coloured layer to draw over:
        stack.put("sketch", ref.lines(), color="#5a5248", opacity=0.4); fade it out at the end, or keep it faint
        as the ink of a line-and-wash picture. A sketch has no light to measure: decide the light in your brief."""
        self._need("measure", "ref.lines")
        L = rgb2oklab(self._px)[..., 0]
        paper, ink = float(np.percentile(L, 90)), float(np.percentile(L, 1))
        m = np.clip((paper - L) / max(paper - ink, 0.05) * 1.6 * float(strength) - 0.15, 0, 1)
        cw, ch = self._canvas
        x0, y0, x1, y1 = self._fit
        M = np.float32([[(x1 - x0) / self.w, 0, x0], [0, (y1 - y0) / self.h, y0]])
        return np.clip(cv2.warpAffine(m.astype(np.float32), M, (cw, ch), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, 1)

    def light(self, box, fg=(), bg=()):
        """How the thing inside a box of the reference is lit: direction, contrast, and the colours of its lit and
        shadow sides (see light_on). Give the result to form(..., light=...) to shade your own form the same way."""
        self._need("measure", "ref.light")
        return light_on(self._px, select_object(self._px, box, fg, bg, refine=False))

    def landmarks(self, kind="pose"):
        """Body, face or hand landmarks found in the reference, in YOUR picture's coordinates. See landmarks()."""
        self._need("measure", "ref.landmarks")
        r = landmarks(self._px, kind)
        if isinstance(r, list):
            return [self.to_picture(h) for h in r]
        return dict(zip(r.keys(), self.to_picture(list(r.values()))))

    def palette(self, box=None, n=6):
        """The n main colours of the reference, or of a box of it (reference pixels). See palette()."""
        self._need("measure", "ref.palette")
        if box is None:
            return palette(self._px, n=n)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        return palette(self._px[max(0, y0):y1, max(0, x0):x1], n=n)

    def tones(self, levels=4):
        """A coarse tone plan of the reference, the size of your picture: light and dark reduced to `levels` flat
        steps at low resolution (a notan). It says where the big lights and darks sit; it holds no detail.
        Returns a grey picture (H, W, 3): show() it, or take luma() of it for the tone at a place."""
        self._need("measure", "ref.tones")
        small = cv2.resize(rgb2oklab(self._px)[..., 0], (max(8, round(64 * self.w / max(self.w, self.h))), max(8, round(64 * self.h / max(self.w, self.h)))), interpolation=cv2.INTER_AREA)
        q = np.round(np.clip(small, 0, 1) * (levels - 1)) / (levels - 1)
        return self._place(np.dstack([q] * 3).astype(np.float32), cv2.INTER_LINEAR)

    def _place(self, pic, interp=cv2.INTER_AREA):
        cw, ch = self._canvas
        x0, y0, x1, y1 = self._fit
        M = np.float32([[(x1 - x0) / pic.shape[1], 0, x0], [0, (y1 - y0) / pic.shape[0], y0]])
        return cv2.warpAffine(pic, M, (cw, ch), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))

    def check(self, drawing, kinds=("pose", "face")):
        """Check your picture's structure against the reference: shows the edges of both laid over each other, and
        prints how far each body or face landmark of your figure is from the reference's, when both can be found."""
        self._need("measure", "ref.check")
        from .core import _count
        _count("ref.check")
        d = np.asarray(drawing, np.float32)[..., :3]
        v = show(edge_overlay(d, self._place(self._px)), f"structure against {self.name}: reference blue, yours orange")
        # The big arrangement of light and dark, both reduced to a few dozen patches: 1 = the same, 0 = unrelated.
        placed = self._place(self._px)
        x0, y0, x1, y1 = [int(round(c)) for c in self._fit]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(d.shape[1], x1), min(d.shape[0], y1)
        if x1 - x0 > 8 and y1 - y0 > 8:
            size = (max(4, round(24 * (x1 - x0) / max(x1 - x0, y1 - y0))), max(4, round(24 * (y1 - y0) / max(x1 - x0, y1 - y0))))
            a = cv2.resize(rgb2oklab(d[y0:y1, x0:x1])[..., 0], size, interpolation=cv2.INTER_AREA).ravel()
            b = cv2.resize(rgb2oklab(placed[y0:y1, x0:x1])[..., 0], size, interpolation=cv2.INTER_AREA).ravel()
            if a.std() > 1e-4 and b.std() > 1e-4:
                r = float(np.corrcoef(a, b)[0, 1])
                print(f"light and dark against {self.name}: agreement {r:.2f} (1 = the same arrangement; under 0.7 the big masses are off); yours runs {a.min():.2f}..{a.max():.2f}, the reference {b.min():.2f}..{b.max():.2f}")
        for kind in kinds:
            try:
                theirs = self.landmarks(kind)
            except RuntimeError:
                continue
            if not theirs:
                continue
            mine = landmarks(d, kind)
            if not mine:
                print(f"({kind}: found in the reference but not in your picture yet)")
                continue
            landmark_diff(mine, theirs)
        return v

    @property
    def picture(self):
        """The reference's pixels (reference size). Level 'sample' only."""
        self._need("sample", "ref.picture")
        return self._px.copy()

    def on_canvas(self):
        """The reference fitted into your picture's frame (through ref.fit). Level 'sample' only."""
        self._need("sample", "ref.on_canvas")
        return self._place(self._px)

    def __repr__(self):
        return f"<Ref {self.name}: {self.w}x{self.h}, level {self.level}>"
