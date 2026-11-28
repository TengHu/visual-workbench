"""Layers and stages: build a picture in named layers, one stage of work at a time, and keep a record of each stage.

A drawing made in stages (construction, forms, line, flat colour, shading, detail, background) can be corrected where
the mistake is, instead of being redrawn. An edit kept in its own layer can be redone without touching the others.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .core import _state, _to8, show, to_linear, to_srgb

__all__ = ["Stack", "brief", "plan", "stage"]

_MODES = ("normal", "multiply", "screen")


class Stack:
    """A pile of named layers over a base picture. `stack` is one, ready in the session, with the picture
    (or an empty plate) as its base.

        stack.put("line", layer)                 add a layer on top, or replace the one of that name where it is
        stack.put("shade", layer, mode="multiply", opacity=0.8)     darkens what is under it (shadows)
        stack.put("light", layer, mode="screen")                    lightens (highlights, glow)
        stack.put("flats", layer, below="line")  slip it in under another layer
        stack.get("line")                        the layer (h, w, 4), to change and put back
        stack.opacity("construction", 0.15)      fade a layer to a guide; 0 hides it; stack.remove(name) deletes it
        stack.flatten()                          the picture as it now stands
        stack.flatten(without=["construction"])  ... leaving layers out; only=[...] takes just those
        stack.names                              the layers, bottom to top

    A layer is (H, W, 4) with straight alpha; a picture (H, W, 3) is taken as fully opaque, and a mask can be
    given with color= to make a flat-coloured layer."""

    def __init__(self, base):
        self.base = np.array(base, np.float32)[..., :3]
        self._layers: dict[str, dict] = {}
        self._order: list[str] = []

    @property
    def names(self):
        return list(self._order)

    def put(self, name, layer, mode="normal", opacity=1.0, below=None, color=None):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}")
        a = np.asarray(layer, np.float32)
        h, w = self.base.shape[:2]
        if a.shape[:2] != (h, w):
            raise ValueError(f"layer is {a.shape[1]}x{a.shape[0]} but the picture is {w}x{h}; place() it first")
        if a.ndim == 2:
            from .core import as_color
            c = as_color(color if color is not None else "#000000")
            a = np.dstack([np.broadcast_to(c, (h, w, 3)), np.clip(a, 0, 1)])
        elif a.shape[2] == 3:
            a = np.dstack([a, np.ones((h, w), np.float32)])
        self._layers[name] = {"px": np.clip(a, 0, 1).astype(np.float32), "mode": mode, "opacity": float(opacity)}
        if name in self._order and below is None:
            return self  # replaced where it was
        if name in self._order:
            self._order.remove(name)
        if below is not None and below in self._order:
            self._order.insert(self._order.index(below), name)
        else:
            self._order.append(name)
        return self

    def get(self, name):
        if name not in self._layers:
            raise KeyError(f"no layer {name!r}; there are {self._order}")
        return self._layers[name]["px"].copy()

    def remove(self, name):
        self._layers.pop(name, None)
        if name in self._order:
            self._order.remove(name)
        return self

    def opacity(self, name, value):
        if name not in self._layers:
            raise KeyError(f"no layer {name!r}; there are {self._order}")
        self._layers[name]["opacity"] = float(value)
        return self

    def flatten(self, only=None, without=None):
        lin = to_linear(self.base)
        for name in self._order:
            if (only is not None and name not in only) or (without is not None and name in without):
                continue
            L = self._layers[name]
            a = (L["px"][..., 3] * L["opacity"])[..., None]
            if not float(a.max()) > 0:
                continue
            top = to_linear(L["px"][..., :3])
            if L["mode"] == "multiply":
                top = lin * top
            elif L["mode"] == "screen":
                top = 1 - (1 - lin) * (1 - top)
            lin = lin * (1 - a) + top * a
        return to_srgb(lin)

    def __repr__(self):
        return "<Stack " + ", ".join(f"{n}({self._layers[n]['mode']}, {self._layers[n]['opacity']:.2f})" for n in self._order) + ">"


def brief(text):
    """Record a description of the intended result, in your own words. Optional. It is kept with the run
    (brief.md) and shown in its replay; calling it again replaces it."""
    text = str(text).strip()
    _state["brief"] = text
    with open("brief.md", "w") as f:
        f.write(text + "\n")
    print("brief recorded")


def plan(stages):
    """Record a list of steps, in order, e.g. plan(["step one", "step two"]). Optional. It is kept with the run and
    shown in its replay; calling it again replaces it."""
    stages = [str(s) for s in stages]
    _state.setdefault("plan", [])
    _state["plan"] = stages
    with open("plan.json", "w") as f:
        json.dump(stages, f)
    print("plan: " + " -> ".join(stages))


def stage(name, picture, note=""):
    """Record the picture as it stands at the end of step `name`, and show it: stage("step one", stack.flatten()).

    The frames make the run's step-by-step replay. Optional. Recording a stage of the same name again replaces its
    frame."""
    a = _to8(np.asarray(picture, np.float32)[..., :3])
    os.makedirs("stages", exist_ok=True)
    frames = _state.setdefault("stages", [])
    from PIL import Image
    again = next((f for f in frames if f["name"] == str(name)), None)
    if again:  # the stage done again: its frame is replaced, in its place
        Image.fromarray(a).save(again["file"])
        again["note"] = str(note)
    else:
        path = os.path.join("stages", f"{len(frames) + 1:02d}-{''.join(c if c.isalnum() else '-' for c in name.lower())[:30]}.png")
        Image.fromarray(a).save(path)
        frames.append({"name": str(name), "note": str(note), "file": path})
    with open(os.path.join("stages", "index.json"), "w") as f:
        json.dump(frames, f)
    return show(picture, f"end of stage: {name}")
