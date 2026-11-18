#!/usr/bin/env python3
"""Checks of the kit that need no model and cost nothing: run after changing src/kit.

    vw eval kit            (or: .venv/bin/python eval/kit_test.py)

Each check states a property the pictures depend on, on a small made-up study:
  - paint() keeps the study's edges: no halo of mixed colour round a shape
  - no pass of paint() takes the canvas further from the study
  - paint() gives a flat area brushwork without changing its colour
  - form() turns a flat shape into a lit one, light side toward the light
  - drawing_report() finds flat, unshaded areas and passes shaded ones
"""
import os
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
VENV_PY = APP / ".venv" / "bin" / "python"
if Path(sys.prefix).resolve() != (APP / ".venv").resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), "-I", str(Path(__file__).resolve()), *sys.argv[1:]])
sys.path.insert(0, str(APP / "src"))
os.chdir(tempfile.mkdtemp(prefix="kit-test-"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import kit  # noqa: E402

H, W = 420, 560
like = np.zeros((H, W, 3), np.float32)
disc = kit.ellipse_mask(like, (160, 90, 400, 330))
bg, fg = kit.as_color("#7f8791"), kit.as_color("#d8a23a")
flat = (bg * (1 - disc[..., None]) + fg * disc[..., None]).astype(np.float32)
failed = []


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        failed.append(name)


# 1. Edges: away from the outline (more than 4 px), each side keeps its own colour.
p = kit.paint(flat, brushes=(28, 14, 7, 4, 2), brush="bristle", relief=0, variation=0)
edge = cv2.morphologyEx((disc > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((9, 9), np.uint8)) > 0
off = np.abs(p - flat).max(-1)
worst = float(np.percentile(off[~edge], 99.9))
check("no halo round a shape", worst < 0.06, f"difference away from the outline {worst:.3f} at its 99.9th percentile (limit 0.06)")

# 2. Each further pass brings the canvas closer to the study, never further away.
a = kit.paint(flat, brushes=(28,), brush="bristle", relief=0, variation=0)
b = kit.paint(flat, onto=a, brushes=(14, 7, 4), brush="bristle", relief=0, variation=0)
ea, eb = float(np.abs(a - flat).mean()), float(np.abs(b - flat).mean())
check("later passes only improve", eb <= ea + 1e-4, f"distance to the study {ea:.4f} after the first brush, {eb:.4f} after the rest")

# 3. Brushwork on a flat area: texture is there, the colour is not shifted.
q = kit.paint(flat, brushes=(28, 14, 7), brush="bristle")
inside = (disc > 0.99) & ~edge
tex, shift = float(q[inside].std(0).mean()), float(np.abs(q[inside].mean(0) - fg).max())
check("a flat area looks painted", tex > 0.004, f"variation inside the flat shape {tex:.4f} (at least 0.004)")
check("... without a change of colour", shift < 0.03, f"mean colour off by {shift:.3f} (limit 0.03)")

# 4. form(): lit from the upper left, the upper left of the shape is lighter than the lower right.
lay = kit.form(disc, "#d8a23a", light=(-0.7, -0.7), contrast=0.6)
L = kit.rgb2oklab(lay[..., :3])[..., 0]
lit, shade = float(L[130:170, 200:240].mean()), float(L[250:290, 320:360].mean())
check("form() lights a shape", lit - shade > 0.08, f"lightness {lit:.2f} toward the light, {shade:.2f} away from it")

# 5. drawing_report(): the flat study is flat, the shaded one is not.
shaded = kit.blend(np.broadcast_to(bg, like.shape) * (0.8 + 0.3 * np.linspace(1, 0, W, dtype=np.float32))[None, :, None], lay[..., :3], lay[..., 3])
f1, f2 = kit.drawing_report(flat, quiet=True)["flat"], kit.drawing_report(shaded, quiet=True)["flat"]
check("drawing_report() tells flat from shaded", f1 > 0.8 and f2 < 0.2, f"flat share {f1:.2f} for the flat study, {f2:.2f} for the shaded one")

# 6. A sculpture: lit from the upper left, its upper-left side is lighter, and a part in front shades the ground.
s = kit.Sculpture().ellipsoid("ball", (280, 200, 0), (90, 90, 90), "#3a7bd5")
lay = s.render(W, H, light=(-0.7, -0.7, 0.4), ground=300)
L = kit.rgb2oklab(lay[..., :3])[..., 0]
lit, shade = float(L[150:170, 220:240].mean()), float(L[230:250, 320:340].mean())
check("a sculpture is lit by its form", lit - shade > 0.1, f"lightness {lit:.2f} toward the light, {shade:.2f} away from it")
check("... and casts a shadow on the ground", float(s.ground_shadow.max()) > 0.5, f"darkest ground shadow {s.ground_shadow.max():.2f}")
q = kit.quadruped(280, 380, 120, kind="dog")
q.render(W, H)
check("quadruped() makes a whole animal", all(float(q.mask(n).sum()) > 0 for n in ("skull", "chest", "hip", "tail")), "head, body, hip and tail all seen")

# 7. Hard parts: a box and a cone are seen whole, and a turned box still renders.
hp = kit.Sculpture().box("block", (200, 220, 0), (120, 120, 120), "#a0a0a0", angle=30).cylinder("spire", (380, 280, 0), (380, 120, 0), 60, 0, "#a05050")
hp.render(W, H)
check("box() and cylinder() render as solids", float(hp.mask("block").sum()) > 8000 and float(hp.mask("spire").sum()) > 4000, f"block {hp.mask('block').sum():.0f} px, spire {hp.mask('spire').sum():.0f} px")

# 8. behind(): new content behind a soft-edged front thing keeps no fringe of the old background.
yy = np.mgrid[0:H, 0:W][0].astype(np.float32)
front = np.clip((yy - 200) / 6.0, 0, 1)                      # a soft horizon: the front thing below, sky above
sky, ground = kit.as_color("#f0d0e0"), kit.as_color("#3050a0")
photo = (sky * (1 - front[..., None]) + ground * front[..., None]).astype(np.float32)
lay = np.zeros((H, W, 4), np.float32); lay[..., :3] = kit.as_color("#202020"); lay[120:260, 200:360, 3] = 1.0   # a dark thing behind
out = kit.behind(photo, lay, front)
edge = (front > 0.2) & (front < 0.8) & (lay[..., 3] > 0)
expect = kit.as_color("#202020") * (1 - front[edge][:, None]) + ground * front[edge][:, None]
err = float(np.abs(out[edge] - expect).max())
check("behind() rebuilds a soft edge against the new content", err < 0.05, f"largest error on the edge {err:.3f} (limit 0.05)")
check("... and leaves the rest untouched", float(np.abs(out - photo)[lay[..., 3] == 0].max()) == 0.0, "")

# 9. The grain ratio of an edit does not swing when the picture is saved to 8 bits.
clean = np.broadcast_to(kit.as_color("#8090a0"), (H, W, 3)).copy()
edited = clean.copy(); edited[100:300, 150:400] = kit.as_color("#806040")
q = np.round(edited * 255) / 255
r1, r2 = kit.edit_report(clean, edited, quiet=True).get("grain_ratio", 1), kit.edit_report(clean, q.astype(np.float32), quiet=True).get("grain_ratio", 1)
check("edit_report's grain ratio is steady on a clean picture", abs(r1 - r2) < 0.3, f"x{r1:.2f} before saving, x{r2:.2f} after")

print("\n" + ("all checks passed" if not failed else f"{len(failed)} failed: " + ", ".join(failed)))
sys.exit(1 if failed else 0)
