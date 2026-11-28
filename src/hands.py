"""Hands-on jobs: an agent that writes the code itself (Claude Code, Antigravity, Codex, anything with a shell and
eyes for a picture file) works through three commands, and gets what the built-in loop gives a model: the manual,
a persistent sandboxed session, views to look at, and a review that every delivery goes through.

    vw open [PICTURE] "REQUEST"    set a job up: its folder, its manual (MANUAL.md) and its first views
    vw py RUN [FILE | -]           run code in the job's session; prints the output and the views it made
    vw finish RUN [FILE]           deliver: the first call brings the check views and a review instead
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import gate
import pages

SRC = Path(__file__).resolve().parent
APP = SRC.parent
SOCK = ".vw.sock"

SELF_REVIEW = """No independent reviewer is set for this job: above are the result's measurements and views. Open each view.

Compare the result with the request, part by part: is each part of it done? Is anything changed that the request did not ask for?

If something is not done, change it, save, and run finish again. Otherwise run finish again now to deliver."""


def _plate(cfg: dict) -> bool:
    """Whether the job's plate is a picture that came with the request (older jobs recorded mode edit/draw)."""
    return bool(cfg.get("has_plate", cfg.get("mode") == "edit"))


def resolve(run: str) -> Path:
    for d in (Path(run), APP / "runs" / run):
        if (d / "work" / "hands.json").exists():
            return d.resolve()
    raise SystemExit(f"no hands-on job at {run!r} (looked for {run}/work/hands.json, also under {APP / 'runs'}). Start one with: vw open [PICTURE] \"REQUEST\"")


def _config(work: Path) -> dict:
    return json.loads((work / "hands.json").read_text())


def _save(work: Path, cfg: dict) -> None:
    (work / "hands.json").write_text(json.dumps(cfg, indent=1))


def _ask(work: Path, req: dict, wait: float) -> dict:
    cwd = os.getcwd()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        os.chdir(work)  # socket paths are limited to about 100 bytes: connect by the short relative name
        try:
            s.connect(SOCK)
        finally:
            os.chdir(cwd)
        s.settimeout(wait)
        f = s.makefile("rw", encoding="utf-8")
        f.write(json.dumps(req) + "\n")
        f.flush()
        line = f.readline()
        if not line:
            raise ConnectionError("the session server closed the connection")
        return json.loads(line)
    finally:
        s.close()


def _server(work: Path) -> None:
    """Make sure the job's session server is running."""
    try:
        _ask(work, {"op": "ping"}, 10)
        return
    except (OSError, ConnectionError, ValueError):
        pass
    for f in (SOCK, ".vw.error"):
        if (work / f).exists():
            (work / f).unlink()
    log = open(work / "server.log", "a")
    subprocess.Popen([sys.executable, "-I", str(SRC / "server.py"), str(work)], stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, cwd=str(work))
    t0 = time.time()
    while time.time() - t0 < 180:
        time.sleep(0.4)
        if (work / ".vw.error").exists():
            why = (work / ".vw.error").read_text()
            hint = "\nIf this command is itself running inside a sandbox, open the job again with --no-sandbox." if "sandbox" in why.lower() or "not permitted" in why.lower() else ""
            raise SystemExit(f"The job's Python session could not start: {why}{hint}")
        try:
            _ask(work, {"op": "ping"}, 10)
            return
        except (OSError, ConnectionError, ValueError):
            continue
    raise SystemExit(f"The job's Python session did not answer within 3 minutes; see {work / 'server.log'}")


def call(work: Path, code: str, private: bool = False) -> dict:
    """Run code in the job's session (starting the session if need be). Returns the session's reply."""
    cfg = _config(work)
    _server(work)
    reply = _ask(work, {"code": code, "private": private}, cfg["timeout"] + 240)
    if cfg.get("boot") and reply.get("boot") != cfg["boot"] and not private:
        reply["stdout"] = "(The session was not running any more and has been started again: variables from earlier steps are gone; `plate`, W, H, `refs` and the kit are back, and every file you saved is still there. Re-create what you need.)\n" + (reply.get("stdout") or "")
    if reply.get("boot") != cfg.get("boot") or (reply.get("stages") and not private):
        cfg = _config(work)
        cfg["boot"] = reply.get("boot")
        if reply.get("stages") and not private:
            cfg["stages"] = reply["stages"]
        _save(work, cfg)
    return reply


def _text(r: dict) -> str:
    out = []
    if (r.get("stdout") or "").strip():
        out.append(r["stdout"].rstrip())
    if r.get("error"):
        out.append("Error:\n" + r["error"])
    if r.get("dropped_views"):
        out.append(f"({r['dropped_views']} more views were made but not kept: at most 6 per call.)")
    if not out:
        out.append("(no output)")
    out.append(f"[{r.get('seconds', 0):.1f} s]")
    return "\n".join(out)


def _views(work: Path, views: list[dict]) -> str:
    if not views:
        return ""
    lines = ["", "Views to open and look at:"]
    for v in views:
        lines.append(f"  {v['id']}: {v['caption'] or 'view'} ({v['note']})\n      {work / v['file']}")
    return "\n".join(lines)


def _rows(work: Path) -> list[dict]:
    f = work / "steps.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []


def _add_row(work: Path, steps: list[dict], seconds: float) -> None:
    n = len(_rows(work)) + 1
    with open(work / "steps.jsonl", "a") as f:
        f.write(json.dumps({"round": n, "text": "", "steps": steps, "usage": {}, "seconds": seconds}) + "\n")


def _pages(out: Path, done: bool = False, final: Path | None = None) -> None:
    cfg = _config(out / "work")
    who = "hands-on" + (f": {cfg['agent']}" if cfg.get("agent") else "")
    pages.write(pages.Record(dir=out, mode="edit" if _plate(cfg) else "draw", request=(out / "REQUEST.md").read_text(), model_id=who, w=cfg["w"], h=cfg["h"], rows=_rows(out / "work"), stages=cfg.get("stages") or [], reviews=cfg.get("review_reports") or [], summary=cfg.get("summary") or "", delivered=cfg.get("delivered"), hands_on=True, passed=cfg.get("passed"), accepted=cfg.get("accepted") or ""), done, final)


def command() -> str:
    """How to call vw from here: by name if it is on the PATH, else by its full path."""
    found = shutil.which("vw")
    return "vw" if found and Path(found).resolve() == (APP / "vw").resolve() else str(APP / "vw")


def open_(argv: list[str]) -> None:
    import manual
    from agent import set_up
    from cli import sheet_size, sort_inputs

    p = argparse.ArgumentParser(prog="vw open", description="Set up a hands-on job: you (an agent) write the code; vw gives you the manual, the session, the views and the review. Give a picture to edit it; give none to draw.")
    p.add_argument("inputs", nargs="+", metavar="[PICTURE] REQUEST", help="the picture to edit (optional) and the request, in quotes")
    p.add_argument("--ref", action="append", default=[], help="a reference picture the request refers to; repeatable")
    p.add_argument("--reference", choices=["look", "measure", "sample"], help="what may be taken from references (default sample when editing, measure when drawing)")
    p.add_argument("--size", help="the plate when there is no picture, WIDTHxHEIGHT (default: the shape of the first reference; 1600x1200 without one)")
    p.add_argument("--coords", choices=["xy", "yx1000"], default="xy", help="how you write positions read off a view: xy (pixels of the view, the default) or yx1000 ([y, x] 0..1000, native to Gemini models)")
    p.add_argument("--critic", default="argon", help="who reviews before delivery: 'argon' or any OpenRouter model id (the request, the plate and the result are sent to it), or 'none' (you compare the result with the request yourself)")
    p.add_argument("--critic-passes", type=int, default=2, help="how many times a delivery may be held back for review (default 2)")
    p.add_argument("--agent", default="", help="a name for whoever does the work, for the record (e.g. claude-code, antigravity)")
    p.add_argument("--max-side", type=int, default=4096, help="pictures larger than this are reduced first (default 4096)")
    p.add_argument("--no-sam", action="store_true", help="select objects with the colour graph cut only, not SAM 2.1")
    p.add_argument("--no-sandbox", action="store_true", help="run the session without the macOS sandbox: only when this command already runs inside one")
    p.add_argument("--timeout", type=int, default=300, help="seconds one `vw py` call may run (default 300)")
    p.add_argument("--out", help="the job's folder (default runs/<time>-<request>)")
    a = p.parse_args(argv)
    image, request = sort_inputs(p, a.inputs, a.ref)
    mode = "make"
    has_plate = image is not None
    size = (0, 0) if has_plate else sheet_size(p, a.size, a.ref)
    import re
    slug = re.sub(r"[^a-z0-9]+", "-", request.lower()).strip("-")[:40] or mode
    out = (Path(a.out) if a.out else APP / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}").resolve()
    if (out / "work" / "hands.json").exists():
        p.error(f"{out} already holds a job; give another --out")
    out.mkdir(parents=True, exist_ok=True)
    (out / "REQUEST.md").write_text(request)
    ref_level = a.reference or ("sample" if has_plate else "measure")
    w, h, notes, refs = set_up(out, mode, image, size, a.ref, ref_level, a.max_side)
    work = out / "work"
    sam = not a.no_sam and (APP / ".models" / "sam2.1_hiera_large.pt").exists()
    _save(work, {"mode": mode, "has_plate": has_plate, "coords": a.coords, "sam": sam, "ref_level": ref_level, "timeout": a.timeout, "sandbox": not a.no_sandbox, "critic": a.critic, "critic_passes": a.critic_passes, "agent": a.agent, "w": w, "h": h, "refs": refs, "reviews": 0, "boot": None, "delivered": None})
    vw = command()
    text = manual.build(coords=a.coords, rounds=0, sam=sam, seconds=a.timeout, max_views=6, mode=mode, ref_level=ref_level, refs=len(refs), interface="cli", run=str(out), vw=vw, reviews=a.critic_passes if a.critic != "none" else 1)
    (out / "MANUAL.md").write_text(text)
    (work / "system.md").write_text(text)

    first = []
    first += call(work, "show(plate, 'the plate')", private=True)["views"]
    for i in range(len(refs)):
        first += call(work, f"refs[{i}].show('reference refs[{i}]')", private=True)["views"]
    cfg = _config(work)
    _pages(out)
    print(f"Job opened: {out}\n")
    print(f"The request: {request}\n")
    print(f"Your plate is {w}x{h}; it is `plate` and `input.png` in the session" + (" — the picture that came with the request." if has_plate else " — empty." + (" It has the shape of the reference." if refs and not a.size else "")) + (" " + " ".join(notes) if notes else ""))
    if refs:
        print(f"{len(refs)} reference picture{'s' if len(refs) > 1 else ''}: `refs` in the session; you may {ref_level} " + ("at them only." if ref_level == "look" else "them." if ref_level == "measure" else "them (use their pixels)."))
    print("Reviewer before delivery: " + ("you (--critic none)." if cfg["critic"] == "none" else f"{cfg['critic']}, independent (the request, the plate and the result are sent to it)."))
    print(_views(work, first).lstrip("\n"))
    print(f"""
What to do now, in this order:
1. Read the manual, all of it, before writing any code: {out / 'MANUAL.md'}
2. Open the views listed above and look at them.
3. Run code:  {vw} py {out} step.py   (or  {vw} py {out} - <<'EOF' ... EOF);  open every view each call lists.
4. Save with save(img, "result.png") in the session, then:  {vw} finish {out}
   The first finish brings check views and a review; run it again to deliver. Only that delivers.""")


def py(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="vw py", description="Run Python in a hands-on job's session. Variables stay between calls. Prints what the code printed and lists the views it made, as picture files to open.")
    p.add_argument("run", help="the job's folder (or its name under runs/)")
    p.add_argument("file", nargs="?", default="-", help="a file of Python (looked for in the current folder, then in the job's folder), or - (the default) to read the code from standard input")
    a = p.parse_args(argv)
    out = resolve(a.run)
    work = out / "work"
    if a.file == "-":
        code = sys.stdin.read()
    else:
        found = next((f for f in (Path(a.file), out / a.file, work / a.file) if f.is_file()), None)
        if not found:
            raise SystemExit(f"no such file: {a.file} (looked in the current folder, in {out} and in {work})")
        code = found.read_text()
    if not code.strip():
        raise SystemExit("no code given: pass a file, or pipe the code in")
    r = call(work, code)
    text = _text(r)
    _add_row(work, [{"kind": "python", "code": code, "out": text, "views": [v["file"] for v in r["views"]], "error": bool(r.get("error"))}], r.get("seconds", 0))
    _pages(out)
    print(text + _views(work, r["views"]))
    if r.get("error"):
        sys.exit(1)


def finish(argv: list[str]) -> None:
    from agent import REVIEW_CODE

    p = argparse.ArgumentParser(prog="vw finish", description="Deliver a hands-on job's result. The first call measures the result, lists check views to open and brings a review; fix what is found and call it again.")
    p.add_argument("run", help="the job's folder (or its name under runs/)")
    p.add_argument("file", nargs="?", default="result.png", help="the picture to deliver, in the job's work folder (default result.png)")
    p.add_argument("--summary", default="", help="one or two sentences on what was done and what could not be done well")
    p.add_argument("--accept", default="", help="only when the request truly calls for an exception to a check that held the result back: the reason, in a sentence (kept with the job)")
    a = p.parse_args(argv)
    out = resolve(a.run)
    work = out / "work"
    cfg = _config(work)
    path = (work / a.file).resolve()
    if not str(path).startswith(str(work.resolve())) or not path.exists():
        raise SystemExit(f"There is no file {a.file!r} in the job's work folder. Save it first, in the session: save(img, 'result.png').")
    passes = cfg["critic_passes"] if cfg["critic"] != "none" else 1
    reviewed_now = False
    request = (out / "REQUEST.md").read_text()
    if a.accept:
        cfg["accepted"] = a.accept
        _save(work, cfg)
    sheet = None
    if not a.accept:
        # The free checks come first: how the work was done, then what a ruler can tell. They hold back as often as
        # they find something; only `--accept "reason"` lets a result past them.
        reports = cfg.get("review_reports") or []
        resub = bool(reports) and reports[-1].get("verdict") == "fix" and gate.digest(path) == cfg.get("last_reviewed")
        faults, kind = gate.process({**_process(work), "resubmitted": resub}), "process"
        if not faults:
            sheet = call(work, REVIEW_CODE.format(file=a.file, plate=_plate(cfg)), private=True)
            try:
                faults, kind = gate.measured(json.loads((work / "gate.json").read_text()), _plate(cfg)), "measured"
            except (OSError, ValueError):
                faults = []
        if faults:
            note = gate.text(faults, kind).replace('finish with accept="the reason"', f'run {command()} finish {out} --accept "the reason"')
            _add_row(work, [{"kind": "check", "views": [v["file"] for v in (sheet or {}).get("views", [])], "note": note}], 0)
            _pages(out)
            print(note + (_views(work, sheet["views"]) if sheet else "") + f"\n\nNOT DELIVERED YET. Run again when it is done:  {command()} finish {out}")
            return
    if cfg["reviews"] < passes:
        r = sheet or call(work, REVIEW_CODE.format(file=a.file, plate=_plate(cfg)), private=True)
        numbers = (r.get("stdout") or "").strip()
        text, ok, report = _review(out, cfg, numbers, r["views"])
        cfg = _config(work)
        cfg["reviews"] += 1
        cfg["last_reviewed"] = gate.digest(path)
        if report:
            cfg.setdefault("review_reports", []).append(report)
        _save(work, cfg)
        steps = [{"kind": "review", "views": [v["file"] for v in r["views"]], "note": text}]
        print("Review of " + a.file + _views(work, r["views"]) + "\n\n" + text)
        if not ok:
            _add_row(work, steps, r.get("seconds", 0))
            _pages(out)
            print(f"\nNOT DELIVERED YET. Run again when it is fixed:  {command()} finish {out}")
            return
        _add_row(work, steps, r.get("seconds", 0))
        reviewed_now = True
    final = out / "result.png"
    shutil.copy(path, final)
    cfg = _config(work)
    note = ""
    if not reviewed_now and cfg["critic"] != "none" and cfg.get("last_reviewed") != gate.digest(path):
        # The reviews that could hold the delivery back are used. This version still gets a review of its own, for
        # the record: whoever is handed the picture should know what the reviewer made of exactly this version.
        r = call(work, REVIEW_CODE.format(file=a.file, plate=_plate(cfg)), private=True)
        numbers = (r.get("stdout") or "").strip()
        text, ok, report = _review(out, cfg, numbers, r["views"], final=True)
        if report is not None:
            cfg = _config(work)
            cfg["last_reviewed"] = gate.digest(path)
            _save(work, cfg)
        if report is None:
            text = (numbers + "\n\n" if numbers else "") + "(No reviewer could be reached, so this version was only measured.)"
        cfg = _config(work)
        if report:
            cfg.setdefault("review_reports", []).append(report)
            note = f"The delivered version was reviewed for the record: {report.get('verdict')}, {len(report.get('faults', []))} faults."
        _save(work, cfg)
        _add_row(work, [{"kind": "review", "views": [v["file"] for v in r["views"]], "note": text}], r.get("seconds", 0))
        print(f"The job's {passes} reviews that hold a delivery back are used, so this version is delivered. A review of it, for the record (report its faults with the picture):\n\n" + text)
    cfg = _config(work)
    reports = cfg.get("review_reports") or []
    if reports and cfg.get("last_reviewed") == gate.digest(path):
        cfg["passed"] = reports[-1].get("verdict") == "deliver"
    else:
        cfg["passed"] = None
    cfg["delivered"], cfg["summary"] = str(path), (a.summary + (" " if a.summary and note else "") + (f"[{note}]" if note else ""))
    _save(work, cfg)
    _add_row(work, [{"kind": "finish", "note": a.summary}], 0)
    _pages(out, done=True, final=final)
    try:
        _ask(work, {"op": "stop"}, 10)
    except (OSError, ConnectionError, ValueError):
        pass
    verdict = {True: "PASSED the review.", False: "NOT PASSED: the review of this version still lists faults. Report them with the picture.", None: "Not independently reviewed (--critic none)." if cfg["critic"] == "none" else "Not independently reviewed: the reviewer could not be reached."}[cfg["passed"]]
    print(f"\nDELIVERED: {final}\n{verdict}\nThe whole job, step by step: {out / 'index.html'}")


def _process(work: Path) -> dict:
    """The session's record of method: tools used, brief, plan, stages."""
    try:
        p = json.loads((work / ".process.json").read_text())
    except (OSError, ValueError):
        p = {}
    for name, key in (("plan.json", "plan"), ("stages/index.json", "stages")):
        try:
            p[key] = json.loads((work / name).read_text())
        except (OSError, ValueError):
            p[key] = []
    return p


def _review(out: Path, cfg: dict, numbers: str, views: list[dict], final: bool = False) -> tuple[str, bool, dict | None]:
    """The review text for a held-back delivery, whether the result may go out, and the reviewer's report."""
    import critic
    from openrouter import ModelError, create

    work = out / "work"
    head = numbers + "\n\n" if numbers else ""
    if cfg["critic"] == "none":
        return head + SELF_REVIEW, False, None
    try:
        model = create(cfg["critic"], APP)
    except SystemExit as e:
        return head + f"(No independent reviewer: {e}.)\n\n" + SELF_REVIEW, False, None
    shown = []
    if _plate(cfg):
        shown.append(("The plate (the picture that came with the request)", str(out / "original.png")))
    shown += [(f"Reference {i + 1}", str(work / ".refs" / n)) for i, n in enumerate(cfg.get("refs") or [])]
    shown += [(v["caption"], str(work / v["file"])) for v in views]
    try:
        rev, usage = critic.review(model, request=(out / "REQUEST.md").read_text(), pictures=shown, numbers=numbers)
    except ModelError as e:
        return head + f"(The reviewer could not be reached: {e})\n\n" + SELF_REVIEW, False, None
    ok = rev["verdict"] == "deliver"
    text = head + f"An independent reviewer ({model.id}), who saw only the request, the starting point and this result, reports:\n\n" + critic.as_text(rev)
    text += "" if ok or final else "\n\nLook at each reported place yourself, close up; fix what is really there (say so in --summary if a report is mistaken), save, and run finish again."
    return text, ok, rev
