"""Figures: animals and people built as sculptures with sound proportions, from a few settings.

Proportion is the first thing a drawing of a living thing gets wrong, and the hardest to see once wrong. These
start you from a correct one: the masses and limbs in the right sizes and places, as named parts of a Sculpture.
Change any part afterwards (s.parts[name]), add what the subject needs (a collar, a mane, a hat), then render.
"""
from __future__ import annotations

import math

from .sculpt import Sculpture

__all__ = ["quadruped", "figure"]

# Four-legged animals, in shares of the body length L (front of the chest to the back of the hip). The shoulder
# height is about leg + body. Values are typical of the adult animal standing.
ANIMALS = {
    "fox":   dict(body=0.42, leg=0.62, leg_r=0.065, neck=0.30, head=0.20, muzzle=0.30, muzzle_r=0.075, ear=0.24, ear_r=0.08, tail=0.95, tail_r=0.14, tail_up=-20),
    "dog":   dict(body=0.48, leg=0.72, leg_r=0.08, neck=0.34, head=0.22, muzzle=0.24, muzzle_r=0.10, ear=0.16, ear_r=0.08, tail=0.50, tail_r=0.05, tail_up=25),
    "wolf":  dict(body=0.48, leg=0.80, leg_r=0.075, neck=0.34, head=0.22, muzzle=0.30, muzzle_r=0.09, ear=0.18, ear_r=0.075, tail=0.70, tail_r=0.09, tail_up=-25),
    "cat":   dict(body=0.44, leg=0.58, leg_r=0.06, neck=0.20, head=0.21, muzzle=0.08, muzzle_r=0.09, ear=0.14, ear_r=0.07, tail=0.90, tail_r=0.045, tail_up=40),
    "horse": dict(body=0.55, leg=1.05, leg_r=0.07, neck=0.55, head=0.20, muzzle=0.40, muzzle_r=0.11, ear=0.12, ear_r=0.04, tail=0.65, tail_r=0.07, tail_up=-55),
    "deer":  dict(body=0.48, leg=1.05, leg_r=0.05, neck=0.45, head=0.17, muzzle=0.26, muzzle_r=0.07, ear=0.20, ear_r=0.07, tail=0.15, tail_r=0.05, tail_up=30),
}


def quadruped(x, ground, length, kind="fox", facing="left", head="level", walk=0.0, color="#b0702c",
              under="#e8dcc8", feet=None, blend=0.6):
    """A four-legged animal standing on the ground, side on. Returns a Sculpture with named parts.

    x, ground   the middle of the body, and the y where the feet stand.
    length      the body's length in pixels, front of the chest to the back of the hip (about a third of the
                whole animal with head and tail).
    kind        'fox', 'dog', 'wolf', 'cat', 'horse' or 'deer': proportions of that animal (ANIMALS[kind];
                pass a dict of the same keys for another).
    facing      'left' or 'right'. head: 'level', 'up' or 'down'. walk: 0 standing, 1 a full stride.
    color       the coat; under: the throat, chest and belly (None for the same as the coat); feet: the lower
                legs (None for the coat).
    Parts: chest, belly, hip, neck, skull, muzzle, nose, ear_near, ear_far, leg_{fore,hind}_{near,far}_{upper,lower},
    tail, tail_tip. Show it from three-quarters with s.turn(25)."""
    P = ANIMALS[kind] if isinstance(kind, str) else dict(kind)
    L, d = float(length), (-1.0 if facing == "left" else 1.0)
    f = lambda v: x + d * v * L  # along the body: positive toward the head
    bh = P["body"] * L
    top = ground - (P["leg"] + P["body"]) * L  # the line of the back
    cy = top + bh * 0.5
    s = Sculpture(blend=blend)
    under = under or color
    s.ellipsoid("chest", (f(0.32), cy, 0), (0.30 * L, bh * 0.55, bh * 0.45), color)
    s.ellipsoid("belly", (f(0.0), cy + bh * 0.05, 0), (0.32 * L, bh * 0.42, bh * 0.40), color)
    s.ellipsoid("hip", (f(-0.34), cy - bh * 0.02, 0), (0.26 * L, bh * 0.50, bh * 0.42), color)
    s.ellipsoid("underside", (f(0.22), cy + bh * 0.32, 0), (0.28 * L, bh * 0.22, bh * 0.30), under)
    # Neck and head.
    lift = {"level": 0.0, "up": 0.25, "down": -0.45}[head]
    n0 = (f(0.48), cy - bh * 0.15, 0)
    n1 = (f(0.48 + P["neck"] * (0.75 - 0.3 * lift)), cy - bh * 0.15 - P["neck"] * L * (0.75 + lift), 0)
    s.capsule("neck", n0, n1, bh * 0.36, P["head"] * L * 0.75, color)
    hr = P["head"] * L
    sk = (n1[0] + d * hr * 0.35, n1[1] - hr * 0.15, 0)
    s.ellipsoid("skull", sk, (hr * 1.0, hr * 0.85, hr * 0.85), color)
    m1 = (sk[0] + d * (hr * 0.6 + P["muzzle"] * L), sk[1] + hr * 0.35, 0)
    s.capsule("muzzle", (sk[0] + d * hr * 0.5, sk[1] + hr * 0.2, 0), m1, P["muzzle_r"] * L * 1.2, P["muzzle_r"] * L * 0.55, color)
    s.capsule("jaw", (sk[0], sk[1] + hr * 0.55, hr * 0.2), (m1[0] - d * P["muzzle"] * L * 0.3, m1[1] + hr * 0.15, 0), hr * 0.35, P["muzzle_r"] * L * 0.4, under)
    s.ellipsoid("nose", (m1[0] + d * P["muzzle_r"] * L * 0.3, m1[1] - P["muzzle_r"] * L * 0.1, 0), (P["muzzle_r"] * L * 0.55,) * 3, "#1c1614")
    for side, z in (("near", hr * 0.45), ("far", -hr * 0.45)):
        e0 = (sk[0] - d * hr * 0.25, sk[1] - hr * 0.6, z)
        s.capsule(f"ear_{side}", e0, (e0[0] - d * hr * 0.15, e0[1] - P["ear"] * L, z * 1.1), P["ear_r"] * L, P["ear_r"] * L * 0.15, color)
    # Legs: upper and lower, near (toward you) and far; the far pair a little behind and out of step.
    lr = P["leg_r"] * L
    for end, at, thick in (("fore", 0.30, 1.0), ("hind", -0.36, 1.35)):
        for side, z, step in (("near", bh * 0.32, 1.0), ("far", -bh * 0.32, -1.0)):
            swing = d * walk * step * (0.12 if end == "fore" else -0.12) * L
            top_ = (f(at) + (d * 0.04 * L if side == "far" else 0), cy + bh * 0.1, z)
            foot = (top_[0] + swing, ground - lr * 0.6, z)
            # The joint between: forward at the elbow of a foreleg, back at the hock of a hind leg.
            knee = ((top_[0] + foot[0]) / 2 + (d * 0.06 * L if end == "hind" else -d * 0.02 * L), top_[1] + (foot[1] - top_[1]) * 0.5, z)
            s.capsule(f"leg_{end}_{side}_upper", top_, knee, lr * 1.6 * thick, lr * 0.9, color)
            s.capsule(f"leg_{end}_{side}_lower", knee, foot, lr * 0.9, lr * 0.75, feet or color)
    # Tail: two segments, the second bending further down (or up), thickest where the animal's tail is bushiest.
    a = math.radians(P["tail_up"])
    t0 = (f(-0.58), cy - bh * 0.15, 0)
    seg = P["tail"] * L / 2
    t1 = (t0[0] - d * seg * math.cos(a), t0[1] - seg * math.sin(a), 0)
    a2 = a - math.radians(25) if P["tail_up"] <= 0 else a + math.radians(30)
    t2 = (t1[0] - d * seg * math.cos(a2), t1[1] - seg * math.sin(a2), 0)
    s.capsule("tail", t0, t1, P["tail_r"] * L * 0.6, P["tail_r"] * L, color)
    s.capsule("tail_tip", t1, t2, P["tail_r"] * L, P["tail_r"] * L * 0.35, color)
    return s


def figure(x, ground, height, facing=0.0, arms="down", stride=0.0, skin="#d9a487", top="#3d5a80", bottom="#2b2d42",
           hair="#3b2a20", shoes="#1d1d1d", blend=0.45):
    """A standing person, front on, built on the classical proportions (7.5 heads tall). Returns a Sculpture.

    x, ground   where the person stands; height: their height in pixels.
    facing      degrees turned: 0 front on, 45 three-quarters, 90 side on (positive turns them to their left).
    arms        'down', 'akimbo' (hands on hips), 'raised' or 'out'. stride: 0 feet together, 1 a walking step.
    skin, top, bottom, hair, shoes: colours. Parts: head, hair, neck, chest, waist, pelvis, arm_{l,r}_{upper,lower},
    hand_{l,r}, leg_{l,r}_{upper,lower}, foot_{l,r}. A face is drawn on afterwards, small and soft (see the manual)."""
    h = float(height) / 7.5  # one head
    s = Sculpture(blend=blend)
    y = lambda heads: ground - heads * h  # heights in heads from the ground
    s.ellipsoid("head", (x, y(7.0), 0), (h * 0.40, h * 0.52, h * 0.45), skin)
    s.ellipsoid("hair", (x, y(7.18), -h * 0.06), (h * 0.44, h * 0.42, h * 0.46), hair)
    s.capsule("neck", (x, y(6.55), 0), (x, y(6.15), 0), h * 0.20, h * 0.22, skin)
    s.ellipsoid("chest", (x, y(5.45), 0), (h * 0.80, h * 0.85, h * 0.48), top)
    s.ellipsoid("waist", (x, y(4.55), 0), (h * 0.62, h * 0.60, h * 0.40), top)
    s.ellipsoid("pelvis", (x, y(3.95), 0), (h * 0.72, h * 0.55, h * 0.45), bottom)
    for side, sx in (("l", -1.0), ("r", 1.0)):
        sh = (x + sx * h * 0.85, y(5.95), 0)
        if arms == "down":
            el, wr = (sh[0] + sx * h * 0.12, y(4.6), 0), (sh[0] + sx * h * 0.18, y(3.4), h * 0.1)
        elif arms == "akimbo":
            el, wr = (sh[0] + sx * h * 0.85, y(4.9), -h * 0.1), (x + sx * h * 0.72, y(4.15), h * 0.15)
        elif arms == "raised":
            el, wr = (sh[0] + sx * h * 0.4, y(7.2), 0), (sh[0] + sx * h * 0.55, y(8.4), 0)
        else:  # out
            el, wr = (sh[0] + sx * h * 1.4, y(5.85), 0), (sh[0] + sx * h * 2.7, y(5.75), 0)
        s.capsule(f"arm_{side}_upper", sh, el, h * 0.22, h * 0.17, top)
        s.capsule(f"arm_{side}_lower", el, wr, h * 0.16, h * 0.12, skin)
        s.ellipsoid(f"hand_{side}", (wr[0] + (wr[0] - el[0]) * 0.18, wr[1] + (wr[1] - el[1]) * 0.18, wr[2]), (h * 0.13, h * 0.2, h * 0.08), skin)
        step = stride * h * 0.6 * sx
        hp = (x + sx * h * 0.42, y(3.75), 0)
        kn = (x + sx * h * 0.38, y(2.0), step * 0.5)
        an = (x + sx * h * 0.34, y(0.25), step)
        s.capsule(f"leg_{side}_upper", hp, kn, h * 0.34, h * 0.22, bottom)
        s.capsule(f"leg_{side}_lower", kn, an, h * 0.22, h * 0.13, bottom)
        s.capsule(f"foot_{side}", (an[0], ground - h * 0.12, an[2] - h * 0.05), (an[0] + sx * h * 0.05, ground - h * 0.1, an[2] + h * 0.6), h * 0.13, h * 0.11, shoes)
    if facing:
        s.turn(-float(facing), about=(x, ground, 0))
    return s
