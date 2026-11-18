"""The one command, `vw`. What it does follows from what it is given:

    vw photo.jpg "make the car dark green"            a picture given: a model edits it
    vw "a heron in reeds at dawn, pen and ink"         no picture: a model draws on a blank sheet
    vw "paint a study of this in oils" --ref photo.jpg draw, working from a reference
    vw open [photo.jpg] "REQUEST"                      hands-on: you (an agent) do the work; then vw py, vw finish
    vw runs                                            the latest runs
    vw learn RUN...                                    a page that explains how runs were done
    vw eval run base | vw eval compare base no-critic  the test set
    vw setup                                           once (handled by the vw script itself)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
PICTURE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic"}

USAGE = """vw [PICTURE] "REQUEST" [options]            a model does the work
       vw open [PICTURE] "REQUEST" [options]       you do the work: then vw py RUN, vw finish RUN
       vw runs | vw learn RUN... | vw eval ... | vw setup"""

EXAMPLES = """examples:
  vw photo.jpg "make the car dark green"                  edit (a picture is given)
  vw photo.jpg "replace the mug with a cactus" --rounds 30 --max-usd 4
  vw room.jpg "put the lamp from the other picture on the table" --ref lamp.png
  vw "a heron in reeds at dawn, pen and ink"              draw (no picture given)
  vw "paint a study of this picture in oils" --ref photo.jpg --reference sample
  vw photo.jpg "make the car dark green" --reviewer off   no independent review
  vw photo.jpg "make the car dark green" --model anthropic/claude-opus-5.5

hands-on (an agent writes the code itself; same kit, manual, session and review):
  vw open photo.jpg "make the car dark green"             sets the job up, prints its folder RUN and its manual
  vw py RUN step.py                                       run code in the job's session; lists the views to open
  vw finish RUN                                           deliver; the first call brings a review instead

other commands:
  vw runs              the latest runs, with cost and where to open them
  vw learn RUN...      one HTML page that teaches how those runs were done
  vw eval run CONFIG   run the test set (vw eval --help)
  vw setup             set up the environment, once

SKILL.md says how to get the best picture out of it."""


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    sys.stdout.reconfigure(line_buffering=True)  # progress shows at once, also when written to a file
    if argv[:1] == ["runs"]:
        return runs(argv[1:])
    if argv[:1] == ["learn"]:
        from learn import main as learn
        return learn(argv[1:])
    if argv[:1] in (["open"], ["py"], ["finish"]):
        import hands
        return {"open": hands.open_, "py": hands.py, "finish": hands.finish}[argv[0]](argv[1:])
    if argv[:1] == ["eval"]:
        sys.argv = [str(APP / "eval" / "run.py"), *argv[1:]]
        import runpy
        runpy.run_path(sys.argv[0], run_name="__main__")
        return

    p = argparse.ArgumentParser(prog="vw", usage=USAGE, epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="A language model edits a picture, or makes a new one, by writing the code itself. Give a picture to edit it; give none to draw.")
    p.add_argument("inputs", nargs="+", metavar="[PICTURE] REQUEST", help="the picture to edit (optional) and what you want, in your own words (sent to the model word for word)")
    p.add_argument("--ref", action="append", default=[], help="a reference picture the request refers to; repeatable")
    p.add_argument("--reference", choices=["look", "measure", "sample"], help="what the model may do with references: look at them only; also measure them (outlines, landmarks, palette, coarse tones); or also use their pixels (default sample when editing, measure when drawing)")
    p.add_argument("--size", help="the blank sheet when drawing, WIDTHxHEIGHT (default: the shape of the first reference at about 1.9 megapixels; 1600x1200 without one)")
    p.add_argument("--model", default="argon", help="'argon' (TARGET_MODEL from the repo's .env) or any OpenRouter model id")
    p.add_argument("--reviewer", choices=["on", "off"], default="on", help="on: every finish is checked by an independent reviewer before delivery; off: no reviewer model (the free delivery checks still run) (default on)")
    p.add_argument("--critic", default="same", help="who reviews when --reviewer is on: 'same' (the same model, fresh conversation), an OpenRouter model id, or 'none' (the model reviews itself, once)")
    p.add_argument("--critic-passes", type=int, default=2, help="how many times a finish may be held back for review (default 2)")
    p.add_argument("--provider", help="pin OpenRouter providers, comma-separated, e.g. google-vertex")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high"], help="thinking effort (default high)")
    p.add_argument("--rounds", type=int, help="how many replies the model gets (default 24 editing, 36 drawing)")
    p.add_argument("--max-usd", type=float, default=5.0, help="budget: once spent, the model gets one last reply to save and finish (default 5)")
    p.add_argument("--max-tokens", type=int, help="output cap per reply, thinking included (default 32000 editing, 60000 drawing)")
    p.add_argument("--coords", choices=["yx1000", "xy"], help="how the model writes positions: yx1000 (Gemini's native [y, x] 0..1000, default for Gemini) or xy (pixels of the view)")
    p.add_argument("--detail", choices=["low", "high", "original"], help="image detail sent to the model; 'original' costs twice the tokens per view")
    p.add_argument("--max-side", type=int, default=4096, help="pictures larger than this are reduced first (default 4096)")
    p.add_argument("--no-sam", action="store_true", help="select objects with the colour graph cut only, not SAM 2.1")
    p.add_argument("--no-review", action="store_true", help=argparse.SUPPRESS)  # older spelling of --reviewer off
    p.add_argument("--timeout", type=int, default=300, help="seconds one python call may run (default 300)")
    p.add_argument("--out", help="the run's folder (default runs/<time>-<request>)")
    a = p.parse_args(argv)

    image, request = sort_inputs(p, a.inputs, a.ref)
    mode = "edit" if image else "draw"
    edit = mode == "edit"

    size = sheet_size(p, a.size, a.ref) if not edit else (0, 0)

    from agent import Run
    from openrouter import create

    review = a.reviewer == "on" and not a.no_review
    providers = a.provider.split(",") if a.provider else None
    model = create(a.model, APP, providers, a.detail)
    critic_model = None if a.critic == "none" or not review else create(a.model if a.critic == "same" else a.critic, APP, providers if a.critic == "same" else None, a.detail)
    coords = a.coords or ("yx1000" if "gemini" in model.id.lower() else "xy")
    slug = re.sub(r"[^a-z0-9]+", "-", request.lower()).strip("-")[:40] or mode
    out = Path(a.out) if a.out else APP / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "REQUEST.md").write_text(request)
    sam = not a.no_sam and (APP / ".models" / "sam2.1_hiera_large.pt").exists()
    run = Run(mode=mode, model=model, critic_model=critic_model, image=image, size=size, request=request, refs=a.ref,
              ref_level=a.reference or ("sample" if edit else "measure"), out=out,
              rounds=a.rounds or (24 if edit else 36), effort=a.effort, max_tokens=a.max_tokens or (32000 if edit else 60000),
              max_usd=a.max_usd, coords=coords, sam=sam, max_side=a.max_side, seconds=a.timeout, review=review, critic_passes=a.critic_passes)
    print(f"{'editing ' + image if edit else f'drawing on a blank {size[0]}x{size[1]} sheet' + (' (the shape of the reference)' if a.ref and not a.size else '')}")
    print(f"reviewer: {('on, ' + critic_model.id) if review and critic_model else 'on, self-review only' if review else 'off'}")
    print(f"watch it: open {out / 'index.html'}")
    run.go()


def sort_inputs(p, inputs: list[str], refs: list[str]) -> tuple[str | None, str]:
    """The positional words: an existing picture file (the picture to edit, optional) and the request."""
    def is_picture(x: str) -> bool:
        try:
            return Path(x).is_file() and Path(x).suffix.lower() in PICTURE
        except OSError:  # e.g. a long request word that cannot be a filename
            return False
    pictures, words = [], []
    for x in inputs:
        (pictures if is_picture(x) else words).append(x)
    for w in words:
        if Path(w).suffix.lower() in PICTURE and " " not in w:
            p.error(f"no such picture: {w}")
    if not words:
        p.error("say what you want, in quotes: vw photo.jpg \"make the car dark green\"")
    if len(words) > 1:
        p.error(f"put the request in one pair of quotes (got {len(words)} separate words: {' '.join(words)[:80]})")
    if len(pictures) > 1:
        p.error("give one picture to edit; give the others with --ref")
    for f in refs:
        if not Path(f).is_file():
            p.error(f"no such picture: {f}")
    return (pictures[0] if pictures else None), words[0]


def sheet_size(p, size: str | None, refs: list[str]) -> tuple[int, int]:
    """The blank sheet: as asked for; else the shape of the first reference (a study of an upright picture on a
    wide sheet leaves empty bands at the sides); else 1600x1200."""
    if size:
        m = re.fullmatch(r"(\d+)x(\d+)", size)
        if not m:
            p.error("--size must look like 1600x1200")
        return int(m.group(1)), int(m.group(2))
    if refs:
        from PIL import Image, ImageOps
        w, h = ImageOps.exif_transpose(Image.open(refs[0])).size
        k = (1600 * 1200 / (w * h)) ** 0.5
        return max(64, int(round(w * k / 8)) * 8), max(64, int(round(h * k / 8)) * 8)
    return 1600, 1200


def runs(argv: list[str]) -> None:
    """The latest runs under runs/, newest first."""
    n = int(argv[0]) if argv and argv[0].isdigit() else 10
    folders = sorted((d for d in (APP / "runs").glob("*") if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)[:n]
    if not folders:
        print("no runs yet: vw photo.jpg \"what to change\"  or  vw \"what to draw\"")
        return
    for d in folders:
        info = {}
        try:
            info = json.loads((d / "work" / "run.json").read_text())
        except (OSError, ValueError):
            pass
        request = (d / "REQUEST.md").read_text().strip() if (d / "REQUEST.md").exists() else ""
        state = "done" if (d / "result.png").exists() else "running or stopped"
        print(f"{time.strftime('%m-%d %H:%M', time.localtime(d.stat().st_mtime))}  {info.get('mode', '?'):4}  ${info.get('spent_usd') or 0:5.2f}  {state:18}  {request[:60]}")
        print(f"      {d / 'index.html'}")
