"""The manual the model is given as its system prompt: how the session works, how delivery works, and the kit's
reference. It describes mechanisms only: it prescribes no method, no amount of work and no style.

The reference part is generated from the kit's own signatures and docstrings, so it cannot drift from the code.
"""
from __future__ import annotations

import inspect
import textwrap

COORDS = {
    "yx1000": """You read a position off a view and write it in that view's own coordinates: a point as `[y, x]`, a box as `[ymin, xmin, ymax, xmax]`, every number from 0 to 1000, where `[0, 0]` is the view's top-left corner and `[1000, 1000]` its bottom-right corner, whatever the view's size or shape.""",
    "xy": """You read a position off a view and write it in pixels of that view image as it was shown to you: a point as `[x, y]`, a box as `[x0, y0, x1, y1]`, where `[0, 0]` is the view's top-left corner. Each view's size in pixels is stated with it.""",
}

EXAMPLE = {
    "yx1000": "v2.box([412, 230, 655, 480])   # ymin, xmin, ymax, xmax on view v2, each 0..1000",
    "xy": "v2.box([353, 422, 737, 671])   # x0, y0, x1, y1 in pixels of view v2",
}


def _signature(fn) -> str:
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if not p.name.startswith("_")]
    return f"{fn.__name__}{sig.replace(parameters=params)}"


def reference() -> str:
    """Every kit function, by section: its signature and its docstring."""
    import kit
    from kit.core import View

    out = []
    for heading, mod in kit.SECTIONS:
        out.append(f"### {heading}\n")
        intro = (mod.__doc__ or "").strip().split("\n\n", 1)
        if heading.startswith("Selecting") and len(intro) > 1:
            out.append(intro[1].strip() + "\n")
        for name in mod.__all__:
            obj = getattr(mod, name)
            if name == "View":
                out.append("`View`: what show() returns. " + inspect.cleandoc(View.__doc__))
                for meth in ("pt", "pts", "box"):
                    out.append(f"- `v.{meth}(...)`: " + inspect.cleandoc(getattr(View, meth).__doc__))
                out.append("")
                continue
            doc = inspect.cleandoc(obj.__doc__ or "")
            out.append(f"`{_signature(obj)}`\n{textwrap.indent(doc, '    ')}\n")
    return "\n".join(out)


REF_LEVEL = {
    "look": "You may look at a reference (`refs[0].show(...)`, zooming as you like). You may not measure it or take pixels from it.",
    "measure": "You may look at a reference and measure it: the shapes of its parts, its landmarks, its palette, its light and a coarse tone plan come back in your picture's coordinates (`ref.mask`, `ref.outline`, `ref.landmarks`, `ref.palette`, `ref.light`, `ref.tones`, `ref.check`). You may not take pixels from it.",
    "sample": "You may look at a reference, measure it, and use its pixels (`ref.picture`, `ref.on_canvas()`).",
}

INTRO = """You work on a plate: the picture that came with the request, or a blank sheet if none was given. You change the plate by writing Python that runs in your session, using the kit described below. There is no image-generation model: every changed pixel comes from your code."""


HOW = {
    "tools": """You have two tools.

`python(code)` runs code in your Python session. The session stays alive for the whole job, like a notebook: variables, imports and functions you define are still there in the next call. It answers with what the code printed, the value of its last expression, or the error. One call may run for up to {seconds} seconds.

`finish(file)` delivers a picture file and ends the job, once it has passed the checks described in "Delivering".

You have {rounds} rounds. Each of your replies is one round, and one reply may make several python calls.""",
    "cli": """You do this job yourself, through shell commands. The job has a folder, `{run}`, and a Python session of its own that stays alive between your commands, like a notebook: variables, imports and functions you define are still there in the next command.

    {vw} py {run} step.py       run a file of Python in the session
    {vw} py {run} - <<'EOF'     ... or code from standard input
    show(orig, "the picture")
    EOF
    {vw} finish {run}           deliver result.png (the first call brings a review instead; see "Delivering")
    {vw} finish {run} --summary "what was done, and what could not be done well"

Keep your step files in the job's folder (`{run}`); a file name given to `vw py` is looked for there too, and read by `vw` itself. The code that runs can read and write only the job's work folder (`{run}/work`, its current folder): a file your code opens, such as a module of helpers you wrote, must be saved there.

`vw py` answers with what the code printed, the value of its last expression, or the error, and then lists the views the code made, each as a picture file. A view is seen only by opening its file. One call may run for up to {seconds} seconds.

Write only inside the job's folder, and make every pixel with the session: the picture is not to be changed with other programs.""",
}

SEEING = {
    "tools": "the views a python call made are shown to you when it returns, at most {max_views} per call",
    "cli": "the views a `vw py` call made are listed when it returns, as picture files for you to open, at most {max_views} per call",
}

DELIVERING = {
    "tools": "When you call `finish`, the result first goes through the checks above. Then, if the run has a reviewer, an independent reviewer who has seen none of your work is shown the request, the plate and your result (with close-ups of the changed areas and the measurements) and reports which parts of the request it judges not done and what faults it sees. When it reports any, the delivery is held back and the report is given to you; call `finish` again to have the result reviewed again. At most {reviews} reviews can hold a delivery back; after that, `finish` delivers what you saved, and that version is reviewed once more for the record. If the rounds or the budget run out first, the last saved `result.png` is delivered as it is.",
    "cli": "When you run `vw finish` (with `--accept \"reason\"` for a check the request calls for an exception to, and `--summary \"...\"`), the result first goes through the checks above and is shown in check views (listed as files). Then, if the job has a reviewer, an independent reviewer who has seen none of your work is shown the request, the plate and your result, and reports which parts of the request it judges not done and what faults it sees. When it reports any, the delivery is held back and the report is printed; run `vw finish` again to have the result reviewed again. At most {reviews} reviews can hold a delivery back; after that, `vw finish` delivers what you saved, and that version is reviewed once more for the record. Report the faults of that last review with the picture. Only what `vw finish` delivers counts as the result of the job.",
}


def build(*, coords: str, rounds: int, sam: bool, seconds: int, max_views: int, mode: str = "edit", ref_level: str = "sample", refs: int = 0, interface: str = "tools", run: str = "RUN", vw: str = "vw", reviews: int = 2) -> str:
    select_note = (
        "`select_object` uses a segmentation model (SAM 2.1) that follows real outlines closely, including through clutter and partial overlap."
        if sam
        else "`select_object` uses a colour graph cut: it works when the thing differs in colour or brightness from what is around it, and needs your fg/bg points when it does not."
    )
    sheet = "`orig` is the plate, read-only; `W` and `H` are its width and height. It is also the file `input.png`. It is blank if no picture came with the request."
    ref_text = "" if not refs else f"\n\n`refs` holds the {refs} reference picture{'s' if refs > 1 else ''} that came with the request; `ref` is the first (`refs[0]`). {REF_LEVEL[ref_level]} `ref.fit()` is the box of your plate that the whole reference corresponds to; by default the reference is fitted whole and centred, and a blank sheet whose size was not given has the reference's shape."
    how = HOW[interface].format(seconds=seconds, rounds=rounds, run=run, vw=vw)
    seeing = SEEING[interface].format(max_views=max_views)
    return f"""{INTRO}

# How you work

{how}

This manual is complete: everything the session offers is described here.

## The session

Already loaded: `np`, `cv2` (OpenCV with contrib), `PIL` (`Image`, `ImageDraw`, `ImageFilter`, `ImageFont`), `scipy` (`ndi` is scipy.ndimage), `skimage`, `skia`, and every function of the kit below. `numba` is installed.

{sheet} `stack` is a layer stack with it as the base.{ref_text}

{select_note}

The session can read and write only its own folder, and has no network. Nothing can be downloaded or installed.

## Seeing

You see nothing unless you look. `show(x, caption)` makes a view of a picture, a mask or a layer; {seeing}. Each view has an id (v1, v2, ...) and is also a variable of that name.

A whole picture is shown reduced, so single pixels are not visible in it. `show(img, box=...)` shows a region enlarged; one picture pixel is at least one view pixel when the box is about 1500 px across or less. Old views are dropped from the conversation after a while; showing a thing again brings it back.

## Positions

{COORDS[coords]}

A view knows which part of the full picture it shows, and converts for you:

    {EXAMPLE[coords]}
    v2.pt(...)    one point  -> (x, y) in full-picture pixels
    v2.pts([...]) several    -> [(x, y), ...]
    v2.box(...)   a box      -> (x0, y0, x1, y1) in full-picture pixels

Every kit function takes full-picture pixels: points `(x, y)`, boxes `(x0, y0, x1, y1)`. Arrays are indexed `[y, x]`.

A position is only as exact as the view it is read from: on a reduced view, one view pixel spans several picture pixels. `show(..., grid=True)` draws labelled coordinate lines; showing a mask or box (`overlay`) shows where it actually lies.

## Data

- picture: float32 array `(H, W, 3)`, sRGB values 0..1.
- layer: float32 `(h, w, 4)`, a picture with straight alpha.
- mask: float32 `(H, W)`, 1 selected, 0 not, soft in between.

`orig` is read-only; `img = orig.copy()` gives a picture to change. Variables keep their values between calls.

## Layers, stages and records

`paint`, `wash`, `hatch`, `form` and `ink_lines` return a picture or a layer; they do not put it on the stack. `stack.put(name, x)` puts it there; `stack.flatten()` holds only what was put on the stack.

`brief(text)`, `plan(stages)` and `stage(name, picture)` record a description of the intended result, a list of steps, and the picture at the end of a step. They are optional. What they record is kept with the run and shown in its replay; `stage` also shows you the picture.

## Delivering

`save(img, "result.png")` writes the picture to deliver (with layers: `save(stack.flatten(without=[...]), "result.png")`). Every save is also kept as a snapshot, so an earlier one can be delivered by name.

Before any review, the delivery goes through checks that need no judgement and are the same for every request. They hold it back, saying what they found, when the result is the same file a review already sent back, when it is not the size of the plate, or when no pixel differs from the plate. When the request calls for one of these (for example a request to change the size), give the reason with the delivery (`accept`); it is kept with the job.

{DELIVERING[interface].format(reviews=reviews)}

What the reviewer says of the delivered version is recorded as the job's verdict: **passed** when it reports no major fault and every part of the request done, **not passed** otherwise. A picture that is not passed is still delivered when the work ends, marked so, with its faults.

# The kit

All of these are already in the session's namespace. Anything they do not cover can be written with numpy, OpenCV, scipy and scikit-image.

{reference()}"""


TOOLS = [
    {
        "name": "python",
        "description": "Run Python code in your session. Variables persist between calls. Returns what the code printed and the value of its last expression, or the error; pictures passed to show() are shown to you afterwards.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The Python code to run."}}, "required": ["code"]},
    },
    {
        "name": "finish",
        "description": "Deliver a picture file and end the job. The result first goes through the checks and the review described in \"Delivering\"; when they hold it back, you get their report instead, and calling finish again tries again.",
        "parameters": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "The picture file to deliver, e.g. result.png or snapshots/result-02.png. Default: result.png."},
                "summary": {"type": "string", "description": "One or two sentences on what was done, and any part of the request that was not."},
                "accept": {"type": "string", "description": "Only when the request calls for what a check held the result back for (for example a request to change the size): the reason, in a sentence."},
            },
        },
    },
]
