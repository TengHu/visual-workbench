"""The session: one long-lived Python process that runs the code the model writes.

It is started by session.py inside the macOS sandbox (sandbox.sb), in the run's own work folder, and keeps its
variables from one piece of code to the next, like a notebook. It speaks one JSON object per line: a request
{"code": ...} on stdin, a reply {"stdout", "error", "views", ...} on the pipe that was stdout when it started.
Everything else that would have gone to stdout or stderr (C libraries, Chrome) goes to session.log.
"""
import ast
import contextlib
import io
import json
import linecache
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MAX_VIEWS = int(os.environ.get("KIT_MAX_VIEWS", "6"))
MAX_TEXT = 6000


def clip_text(s, limit=MAX_TEXT):
    if len(s) <= limit:
        return s
    head, tail = limit * 2 // 3, limit // 3
    return f"{s[:head]}\n... ({len(s) - head - tail} characters left out) ...\n{s[-tail:]}"


def namespace():
    """What the model's code starts with: the libraries, the kit, and the picture."""
    import cv2
    import numpy as np
    import PIL
    import scipy
    import skimage
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    from scipy import ndimage as ndi

    import kit

    ns = {"__name__": "__main__", "np": np, "cv2": cv2, "PIL": PIL, "Image": Image, "ImageDraw": ImageDraw, "ImageFilter": ImageFilter, "ImageFont": ImageFont, "scipy": scipy, "ndi": ndi, "skimage": skimage, "kit": kit}
    try:
        import skia
        ns["skia"] = skia
    except Exception:
        pass
    for name in kit.__all__:
        ns[name] = getattr(kit, name)
    if os.path.exists("input.png"):
        import glob

        import json as _json
        import re as _re

        import kit.core as core
        # After a restart, carry on where the files stop, so that nothing recorded so far is overwritten.
        core._state["n"] = max([int(m.group(1)) for m in (_re.fullmatch(r"v(\d+)\.jpg", os.path.basename(f)) for f in glob.glob("views/v*.jpg")) if m] + [core._state["n"]])
        core._state["saved"] = sorted(glob.glob("snapshots/result-*.png")) or core._state["saved"]
        if os.path.exists("stages/index.json") and not core._state.get("stages"):
            core._state["stages"] = _json.load(open("stages/index.json"))
        if os.path.exists(".process.json") and not core._state["calls"]:
            _p = _json.load(open(".process.json"))
            core._state["calls"], core._state["brief"] = _p.get("calls", {}), _p.get("brief", "")
        if os.path.exists("plan.json") and not core._state.get("plan"):
            core._state["plan"] = _json.load(open("plan.json"))
        plate = kit.load("input.png")
        plate.setflags(write=False)
        ns["plate"] = plate
        ns["H"], ns["W"] = plate.shape[:2]
        core._state["size"] = plate.shape[:2]
        ns["stack"] = kit.Stack(plate)
        level = os.environ.get("KIT_REF_LEVEL", "sample")
        ns["refs"] = [kit.Ref(kit.load(f), level, f"refs[{i}]", (ns["W"], ns["H"])) for i, f in enumerate(sorted(glob.glob(".refs/*")))]
        if ns["refs"]:
            ns["ref"] = ns["refs"][0]
    return ns


def inventory(ns):
    """What the model has in its session, in one line each: for the handover note when the conversation is cut."""
    import types

    import numpy as np

    import kit
    import kit.core as core
    skip = set(kit.__all__) | {"np", "cv2", "PIL", "Image", "ImageDraw", "ImageFilter", "ImageFont", "scipy", "ndi", "skimage", "skia", "kit", "plate", "H", "W", "refs", "ref"}
    out = []
    for k, v in ns.items():
        if k.startswith("_") or k in skip or isinstance(v, (types.ModuleType, core.View)):
            continue
        if isinstance(v, np.ndarray):
            kind = {2: "mask", 3: {3: "picture", 4: "layer"}.get(v.shape[2] if v.ndim == 3 else 0, "array")}.get(v.ndim, "array")
            out.append(f"{k}: {kind} {v.shape[1]}x{v.shape[0]}" if v.ndim >= 2 else f"{k}: array {v.shape}")
        elif isinstance(v, kit.Stack):
            out.append(f"{k}: {v!r}")
        elif callable(v):
            out.append(f"{k}: function")
        else:
            out.append(f"{k}: {type(v).__name__} {repr(v)[:70]}")
    return "\n".join(out[:80]) or "(none)"


def run(code, ns, n, private=False):
    import kit.core as core

    core._state["private"] = private

    # Every view the model has been shown is a variable of its id (v1, v2, ...), including views the harness made,
    # unless the model has used that name for something else.
    for vid, view in core.VIEWS.items():
        if vid not in ns or isinstance(ns[vid], core.View):
            ns[vid] = view
    name = f"<code {n}>"
    linecache.cache[name] = (len(code), None, code.splitlines(True), name)
    buf = io.StringIO()
    error = None
    t0 = time.time()
    try:
        tree = ast.parse(code, name, "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(tree, name, "exec"), ns)
            if last is not None:
                value = eval(compile(last, name, "eval"), ns)
                if value is not None and not isinstance(value, core.View):
                    print(clip_text(repr(value), 2000))
    except BaseException as e:  # the model's code may raise anything, SystemExit included
        if isinstance(e, KeyboardInterrupt):
            raise
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename != __file__]
        error = "".join(traceback.format_list(frames) + traceback.format_exception_only(type(e), e)).strip()
    views = core.OUTBOX[:]
    core.OUTBOX.clear()
    dropped = max(0, len(views) - MAX_VIEWS)
    return {
        "stdout": clip_text(buf.getvalue()),
        "error": clip_text(error, 4000) if error else None,
        "views": views[:MAX_VIEWS],
        "dropped_views": dropped,
        "saved": list(core._state["saved"]),
        "stages": list(core._state.get("stages", [])),
        "plan": list(core._state.get("plan", [])),
        "calls": dict(core._state.get("calls", {})),
        "brief": core._state.get("brief", ""),
        "seconds": round(time.time() - t0, 2),
    }


def main():
    import warnings
    warnings.filterwarnings("ignore")  # library deprecation chatter is noise in the model's output
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    log = open("session.log", "a", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    try:
        ns = namespace()
        proto.write(json.dumps({"ready": True, "size": [ns.get("W"), ns.get("H")]}) + "\n")
    except Exception:
        proto.write(json.dumps({"ready": False, "error": traceback.format_exc()}) + "\n")
        return
    private = None
    n = 0
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if req.get("op") == "inventory":
            proto.write(json.dumps({"inventory": inventory(ns)}) + "\n")
            continue
        n += 1
        if req.get("private"):
            # The harness's own questions (the first look, the review sheet) must not touch the model's variables.
            if private is None:
                private = namespace()
            reply = run(req["code"], private, n, private=True)
        else:
            reply = run(req["code"], ns, n)
            with open(".process.json", "w") as f:
                json.dump({"calls": reply["calls"], "brief": reply["brief"]}, f)
        proto.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    main()
