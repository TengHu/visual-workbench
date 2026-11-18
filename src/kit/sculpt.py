"""Sculpting: build a solid thing in 3D from rounded or hard-edged parts, and render it with real light and shadow.

An animal, a figure, a fold of cloth or a cloud is hard to draw well from flat shapes: the outline, the turn of each
form, where one part shades another and where the light catches all have to be invented by hand. Modelled in 3D
from rounded parts that melt into each other, as a sculptor builds in clay, all of that is computed. The render is
a study with correct form and light, ready for paint() or wash().
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .core import as_color, oklab2rgb

__all__ = ["Sculpture"]


@njit(cache=False, fastmath=True, inline="always")
def _part(px, py, pz, P, i):
    """Signed distance from a point to part i: kind 0 is a tapered capsule from a (radius ra) to b (radius rb),
    kind 1 an ellipsoid at a with radii (ra, rb, rc), kind 2 a box at a with half-sizes (ra, rb, rc), turned by
    b[0] radians about the vertical and rounded by b[1], kind 3 a cylinder or cone with flat ends from a (radius
    ra) to b (radius rb)."""
    kind = P[i, 0]
    if kind == 2:
        qx, qy, qz = px - P[i, 1], py - P[i, 2], pz - P[i, 3]
        c, s = math.cos(P[i, 4]), math.sin(P[i, 4])
        qx, qz = c * qx - s * qz, s * qx + c * qz
        rr = P[i, 5]
        dx, dy, dz = abs(qx) - (P[i, 7] - rr), abs(qy) - (P[i, 8] - rr), abs(qz) - (P[i, 9] - rr)
        ox, oy, oz = max(dx, 0.0), max(dy, 0.0), max(dz, 0.0)
        return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(dx, max(dy, dz)), 0.0) - rr
    if kind == 3:
        ax, ay, az, bx, by, bz, ra, rb = P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6], P[i, 7], P[i, 8]
        ex, ey, ez = bx - ax, by - ay, bz - az
        qx, qy, qz = px - ax, py - ay, pz - az
        baba = ex * ex + ey * ey + ez * ez
        papa = qx * qx + qy * qy + qz * qz
        paba = (qx * ex + qy * ey + qz * ez) / max(baba, 1e-9)
        x = math.sqrt(max(papa - paba * paba * baba, 0.0))
        cax = max(0.0, x - (ra if paba < 0.5 else rb))
        cay = abs(paba - 0.5) - 0.5
        rba = rb - ra
        k = rba * rba + baba
        f = min(1.0, max(0.0, (rba * (x - ra) + paba * baba) / k))
        cbx = x - ra - f * rba
        cby = paba - f
        sg = -1.0 if (cbx < 0.0 and cay < 0.0) else 1.0
        return sg * math.sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba))
    if kind == 0:
        ax, ay, az, bx, by, bz, ra, rb = P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6], P[i, 7], P[i, 8]
        ex, ey, ez = bx - ax, by - ay, bz - az
        qx, qy, qz = px - ax, py - ay, pz - az
        ll = ex * ex + ey * ey + ez * ez
        t = 0.0 if ll < 1e-9 else min(1.0, max(0.0, (qx * ex + qy * ey + qz * ez) / ll))
        dx, dy, dz = qx - ex * t, qy - ey * t, qz - ez * t
        return math.sqrt(dx * dx + dy * dy + dz * dz) - (ra + (rb - ra) * t)
    qx, qy, qz = px - P[i, 1], py - P[i, 2], pz - P[i, 3]
    rx, ry, rz = P[i, 7], P[i, 8], P[i, 9]
    k0 = math.sqrt((qx / rx) ** 2 + (qy / ry) ** 2 + (qz / rz) ** 2)
    k1 = math.sqrt((qx / (rx * rx)) ** 2 + (qy / (ry * ry)) ** 2 + (qz / (rz * rz)) ** 2)
    return k0 * (k0 - 1.0) / max(k1, 1e-9)


@njit(cache=False, fastmath=True)
def _scene(px, py, pz, P, K):
    """Distance to the whole sculpture (parts melted together with a smooth minimum of width K) and the part
    nearest to the point."""
    d = 1e9
    best, bi = 1e9, 0
    for i in range(P.shape[0]):
        di = _part(px, py, pz, P, i)
        if di < best:
            best, bi = di, i
        k = K * P[i, 10]
        if k > 0:
            h = max(k - abs(d - di), 0.0) / k
            d = min(d, di) - h * h * k * 0.25
        else:
            d = min(d, di)
    return d, bi


@njit(cache=False, fastmath=True, parallel=True)
def _render(H, W, P, C, K, R, cx, cy, Lx, Ly, Lz, ground):
    """Orthographic ray-march. Rays go into the picture (-z), turned by the rotation R about (cx, cy).
    Returns per pixel: hit (0/1), part, normal (3), light (shadowed diffuse), occlusion, ground shadow, colour (3):
    the parts' colours mixed by how near each part's surface is, so colours meet as softly as the parts do."""
    out = np.zeros((H, W, 11), np.float32)
    span = 4.0 * max(H, W)
    for y in prange(H):
        for x in range(W):
            # Ray in camera space, then into the scene.
            ox, oy, oz = x + 0.5 - cx, y + 0.5 - cy, span
            sx = R[0, 0] * ox + R[0, 1] * oy + R[0, 2] * oz + cx
            sy = R[1, 0] * ox + R[1, 1] * oy + R[1, 2] * oz + cy
            sz = R[2, 0] * ox + R[2, 1] * oy + R[2, 2] * oz
            dx, dy, dz = -R[0, 2], -R[1, 2], -R[2, 2]
            t, hit, part = 0.0, False, 0
            tg = 1e9
            if ground > -1e8 and dy > 1e-6:
                tg = (ground - sy) / dy
            while t < 2.0 * span and t < tg:
                d, part = _scene(sx + dx * t, sy + dy * t, sz + dz * t, P, K)
                if d < 0.3:
                    hit = True
                    break
                t += max(d * 0.9, 0.2)
            if hit:
                hx, hy, hz = sx + dx * t, sy + dy * t, sz + dz * t
                e = 0.75
                nx = _scene(hx + e, hy, hz, P, K)[0] - _scene(hx - e, hy, hz, P, K)[0]
                ny = _scene(hx, hy + e, hz, P, K)[0] - _scene(hx, hy - e, hz, P, K)[0]
                nz = _scene(hx, hy, hz + e, P, K)[0] - _scene(hx, hy, hz - e, P, K)[0]
                nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
                nx, ny, nz = nx / nn, ny / nn, nz / nn
                diff = nx * Lx + ny * Ly + nz * Lz
                # Soft shadow: march toward the light.
                sh, s = 1.0, 2.0
                while s < span * 0.5 and sh > 0.02 and diff > 0:
                    d2 = _scene(hx + nx * 1.5 + Lx * s, hy + ny * 1.5 + Ly * s, hz + nz * 1.5 + Lz * s, P, K)[0]
                    sh = min(sh, 6.0 * d2 / s)
                    s += max(d2, 1.0)
                # Ambient occlusion: how open the surface is around the hit.
                ao = 0.0
                for k in range(1, 6):
                    hh = 3.0 * k
                    ao += (hh - _scene(hx + nx * hh, hy + ny * hh, hz + nz * hh, P, K)[0]) / (2.0 ** k)
                cr, cg, cb, wsum = 0.0, 0.0, 0.0, 0.0
                for i in range(P.shape[0]):
                    di = _part(hx, hy, hz, P, i)
                    kk = max(K * P[i, 10], 1.0)
                    wi = max(0.0, 1.0 - max(di, 0.0) / kk)
                    wi = wi * wi * wi
                    cr += C[i, 0] * wi
                    cg += C[i, 1] * wi
                    cb += C[i, 2] * wi
                    wsum += wi
                if wsum > 0:
                    out[y, x, 8], out[y, x, 9], out[y, x, 10] = cr / wsum, cg / wsum, cb / wsum
                else:
                    out[y, x, 8], out[y, x, 9], out[y, x, 10] = C[part, 0], C[part, 1], C[part, 2]
                out[y, x, 0] = 1.0
                out[y, x, 1] = part
                out[y, x, 2], out[y, x, 3], out[y, x, 4] = nx, ny, nz
                out[y, x, 5] = max(diff, 0.0) * max(sh, 0.0)
                out[y, x, 6] = min(1.0, max(0.0, 1.0 - 0.05 * ao))
            elif tg < 1e9:
                gx, gz = sx + dx * tg, sz + dz * tg
                sh, s = 1.0, 2.0
                while s < span * 0.5 and sh > 0.02:
                    d2 = _scene(gx + Lx * s, ground + Ly * s, gz + Lz * s, P, K)[0]
                    sh = min(sh, 8.0 * d2 / s)
                    s += max(d2, 1.0)
                out[y, x, 7] = 1.0 - max(sh, 0.0)
    return out


class Sculpture:
    """A solid thing modelled from rounded parts that melt into each other, rendered with its own light and shadow.

    Positions are in picture pixels: x to the right, y down, z toward you (0 is the picture plane; a part with
    larger z is nearer). Build the thing from its masses, largest first, each part on the real structure (a chest,
    a hip, upper and lower leg, a skull and a muzzle), and let neighbouring parts overlap so that they join:

        s = Sculpture(blend=0.6)
        s.ellipsoid("chest", (700, 520, 0), (150, 110, 95), "#c86a2c")
        s.ellipsoid("hip", (930, 540, 0), (120, 100, 85), "#c86a2c")
        s.capsule("neck", (620, 470, 0), (540, 400, 0), 60, 45, "#c86a2c")
        s.capsule("foreleg", (650, 600, 40), (640, 820, 40), 32, 18, "#5a3020")
        layer = s.render(W, H, light=(-0.6, -0.7, 0.4))
        stack.put("fox", layer)

    capsule(name, a, b, ra, rb=None, color)   a limb, a neck, a tail, a finger: radius ra at a, rb at b
    ellipsoid(name, center, (rx, ry, rz), color)   a body mass, a head, a muscle, a cloud puff
    box(name, center, (w, h, d), color, angle, round)   a wall, a block, a step, a slab (hard-edged)
    cylinder(name, a, b, ra, rb=None, color)  a column, a round tower, a cone or spire (rb=0), a trunk
    Build a building, a machine or furniture from boxes and cylinders; repeated details (battlements, windows,
    planks) are a loop in your code. A window or a door is a dark box set into the wall.
    Parts can be changed and rendered again: s.parts[name] is a dict you may edit ("a" and "b": the ends or the
    centre (x, y, z); "r": the radii; "color"); s.remove(name);
    s.turn(degrees) shows it from another side; s.recolor(colour, [names]).
    For an animal or a figure, start from quadruped() or figure(), which build one with sound proportions.
    blend: how softly parts melt together, as a share of the smaller radius (0 hard joins, 1 very soft).
    soft=False on a part keeps its join hard (a claw, a beak, a hat brim).

    render(W, H, light, ...) returns a layer (H, W, 4): each part in its colour, turned from light into shade by
    the real form, with the shadows parts cast on each other and the darkening in creases. After a render,
    s.mask(name) is where that part is seen, and s.ground_shadow the shadow cast on the ground (a mask). Look at the render from several sides (tilt, turn) before painting it."""

    def __init__(self, blend=0.5):
        self.blend = float(blend)
        self.parts: dict[str, dict] = {}
        self._last = None

    def capsule(self, name, a, b, ra, rb=None, color="#888888", soft=True):
        self.parts[str(name)] = {"kind": 0, "a": tuple(map(float, a)), "b": tuple(map(float, b)), "r": (float(ra), float(ra if rb is None else rb), 0.0), "color": color, "soft": bool(soft)}
        return self

    def ellipsoid(self, name, center, radii, color="#888888", soft=True):
        r = tuple(map(float, radii)) if np.ndim(radii) else (float(radii),) * 3
        self.parts[str(name)] = {"kind": 1, "a": tuple(map(float, center)), "b": (0.0, 0.0, 0.0), "r": r, "color": color, "soft": bool(soft)}
        return self

    def turn(self, degrees, about=None):
        """Turn the whole sculpture about a vertical axis (through `about` (x, y, z), default its middle):
        positive turns its right side toward you. Use it to show a thing from three-quarters instead of the side."""
        pts = np.array([p["a"] for p in self.parts.values()] + [p["b"] for p in self.parts.values() if p["kind"] in (0, 3)])
        c = np.asarray(about if about is not None else pts.mean(0), np.float64)
        u = math.radians(float(degrees))
        cu, su = math.cos(u), math.sin(u)
        rot = lambda q: (c[0] + (q[0] - c[0]) * cu + (q[2] - c[2]) * su, q[1], c[2] - (q[0] - c[0]) * su + (q[2] - c[2]) * cu)
        for p in self.parts.values():
            p["a"] = rot(p["a"])
            if p["kind"] in (0, 3):
                p["b"] = rot(p["b"])
            elif p["kind"] == 2:
                p["b"] = (p["b"][0] + u, p["b"][1], p["b"][2])
            else:  # an ellipsoid keeps its axes; its width and depth trade places as it turns
                rx, ry, rz = p["r"]
                p["r"] = (math.hypot(rx * cu, rz * su), ry, math.hypot(rx * su, rz * cu))
        return self

    def recolor(self, color, names=None):
        """Set the colour of the parts named (default: all)."""
        for n in (names or list(self.parts)):
            self.parts[n]["color"] = color
        return self

    def box(self, name, center, size, color="#888888", angle=0.0, round=0.0, soft=False):
        """A block: a wall, a tower, a step, a table top, a roof slab. size (w, h, d) in pixels; angle: degrees
        it is turned about the vertical; round: how rounded its edges are, in pixels. Hard-edged by default."""
        w, h, d = (float(v) for v in size)
        self.parts[str(name)] = {"kind": 2, "a": tuple(map(float, center)), "b": (math.radians(float(angle)), float(round), 0.0), "r": (w / 2, h / 2, d / 2), "color": color, "soft": bool(soft)}
        return self

    def cylinder(self, name, a, b, ra, rb=None, color="#888888", soft=False):
        """A cylinder with flat ends from a to b (radius ra), or a cone or frustum when rb differs (rb=0 a sharp
        point): a column, a round tower, a spire, a turret roof, a trunk, a cup. Hard-edged by default."""
        self.parts[str(name)] = {"kind": 3, "a": tuple(map(float, a)), "b": tuple(map(float, b)), "r": (float(ra), float(ra if rb is None else rb), 0.0), "color": color, "soft": bool(soft)}
        return self

    def remove(self, name):
        self.parts.pop(str(name), None)
        return self

    def _table(self):
        rows = []
        for p in self.parts.values():
            rmin = min((v for v in p["r"] if v > 0), default=1.0)
            rows.append([p["kind"], *p["a"], *p["b"], *p["r"], rmin if p["soft"] else 0.0])
        return np.asarray(rows, np.float64).reshape(-1, 11)

    def render(self, width, height, light=(-0.6, -0.7, 0.45), tilt=None, ground=None, contrast=0.6,
               ambient=0.25, highlight=0.0, shade=None):
        """Render the sculpture. Returns a layer (H, W, 4).

        light      (x, y, z): the direction toward the light (x negative = from the left, y negative = from above,
                   z positive = from the front; the same convention as form()), or the dict light_on()/ref.light() measured, which also sets contrast and shadow colour.
        tilt       degrees the view looks down on the thing (0 straight on; 20 from a little above; default 0,
                   or 12 when there is a ground, which seen straight on would be edge-on). To see it from
                   another side, turn() the sculpture first.
        ground     the y of a level ground under it. Only with it is the shadow on the ground computed
                   (s.ground_shadow; all zero without). Giving it also tilts the view by default (see tilt), which
                   moves far parts up a little: decide it before you place things against the render.
        contrast   how dark the shadow side is, 0..1; ambient: the light that reaches the shadows (sky, bounce).
        highlight  0..1: a shine facing the light (wet nose, eye, glossy fur); shade: the shadow colour."""
        if not self.parts:
            raise ValueError("Sculpture: no parts yet")
        W, H = int(width), int(height)
        lit_c = None
        if isinstance(light, dict):
            contrast = float(light.get("contrast", contrast))
            lit_c, shade = light.get("lit"), (shade if shade is not None else light.get("shade"))
            dx, dy = light["direction"]
            light = (dx, dy, 0.45)
        L = np.asarray(light, np.float64)
        L = L / (np.linalg.norm(L) + 1e-12)
        t = math.radians(float((12.0 if ground is not None else 0.0) if tilt is None else tilt))
        R = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]], np.float64)
        P = self._table()
        g = -1e9 if ground is None else float(ground)
        cols = np.array([as_color(p["color"]) for p in self.parts.values()], np.float64)
        o = _render(H, W, P, cols, self.blend, R, W / 2.0, H / 2.0, L[0], L[1], L[2], g)
        hit = o[..., 0] > 0.5
        part = o[..., 1].astype(np.int32)
        base = np.ascontiguousarray(o[..., 8:11])
        tone = np.clip(ambient * o[..., 6] + (1.0 - ambient) * o[..., 5] * (0.6 + 0.4 * o[..., 6]), 0, 1)
        from .draw import _light_and_shade, _shine
        lab = _light_and_shade(base, tone, contrast, shade, lit_c)
        if highlight > 0:
            Hv = L + np.array([0.0, 0.0, 1.0])
            Hv /= np.linalg.norm(Hv)
            spec = np.clip(o[..., 2] * Hv[0] + o[..., 3] * Hv[1] + o[..., 4] * Hv[2], 0, 1) ** 40 * float(highlight) * o[..., 5]
            lab = _shine(lab, spec.astype(np.float32))
        rgb = oklab2rgb(lab)
        names = list(self.parts)
        self._last = {"part": np.where(hit, part, -1), "names": names}
        self.ground_shadow = np.clip(o[..., 7], 0, 1).astype(np.float32) * (~hit)
        a = hit.astype(np.float32)
        try:
            import cv2
            a = np.clip(cv2.GaussianBlur(a, (0, 0), 0.6) * 1.15, 0, 1)  # an antialiased outline
        except Exception:
            pass
        print(f"sculpture: {len(names)} parts rendered at {W}x{H}" + (f", ground shadow at y {g:.0f}" if ground is not None else ""))
        return np.dstack([rgb, a]).astype(np.float32)

    def mask(self, name):
        """Where the named part is seen in the last render (a mask)."""
        if self._last is None:
            raise ValueError("render() first")
        if name not in self._last["names"]:
            raise KeyError(f"no part {name!r}; there are {self._last['names']}")
        return (self._last["part"] == self._last["names"].index(name)).astype(np.float32)

    def __repr__(self):
        return f"<Sculpture: {len(self.parts)} parts: {', '.join(self.parts)}>"
