"""The checks a result must pass before it is reviewed, and what counts as passed. They cost nothing, they are the
same for every model and every agent, and they are about how the work was done and what can be measured, not taste.

  process(...)   were the steps taken that good work needs (a brief, stages, looking close, checking)?
  measured(...)  does the result fail something a ruler can tell (wrong size, nothing changed, all flat, no tones)?

A fault here holds the delivery back with a plain sentence saying what to do. `accept` (a reason, in words) lets a
result through when the request really calls for the exception; the reason is kept with the run.
"""
from __future__ import annotations

CHECKS = ("squint", "tones", "mirror", "drawing_report", "ref.check", "landmarks", "shape_diff", "edge_overlay")


def digest(path) -> str:
    """A fingerprint of a picture file, to tell whether it changed."""
    import hashlib
    from pathlib import Path
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def process(mode: str, st: dict) -> list[str]:
    """What the work skipped. st: the session's record: calls {name: times}, plan [..], stages [..], brief str,
    and, when a review sent the result back, `resubmitted`: whether the very same picture is offered again."""
    calls = st.get("calls") or {}
    out = []
    if st.get("resubmitted"):
        out.append("This is the same picture the reviewer sent back: nothing was changed. Look at each reported fault close up, fix what is really there, save, and finish again.")
    if mode == "draw":
        if not (st.get("brief") or "").strip():
            out.append("There is no brief. Before drawing, decide the picture and say it: brief(\"...\") with the subject, the viewpoint and composition, where the light comes from, the palette and the medium. A short request needs this most.")
        if not st.get("plan"):
            out.append("There is no plan. Say the stages first: plan([...]).")
        if len(st.get("stages") or []) < 3:
            out.append(f"Only {len(st.get('stages') or [])} stage(s) were closed with stage(). A picture is built and judged in stages (at least: big shapes, light and dark, rendering); a picture made in one go is not accepted.")
        if not any(calls.get(c) for c in CHECKS):
            out.append("None of the checks was used (squint, tones, mirror, drawing_report, ref.check). Use them on the picture and act on what they show.")
    else:
        if not calls.get("zoom"):
            out.append("The picture was never looked at close up. Zoom into the edited area and along its border (show(img, box=...) or compare(orig, img, box=...)) and fix what shows.")
        if not (calls.get("compare") or calls.get("changes") or calls.get("edit_report")):
            out.append("The edit was never compared with the original. Use compare(orig, img, box=...) and changes(orig, img, mask) to see what changed and that nothing else did.")
    return out


def measured(mode: str, g: dict, request: str = "") -> list[str]:
    """What a ruler can tell is wrong with the result. g: the numbers the review sheet wrote (gate.json)."""
    out = []
    words = request.lower()
    if g.get("size_ok") is False and not any(w in words for w in ("crop", "resize", "size", "frame", "extend", "wider", "taller", "square", "aspect", "ratio", "rotate")):
        out.append("The result is not the size of the " + ("original" if mode == "edit" else "sheet") + ". Deliver it at that size.")
    if mode == "edit":
        if g.get("changed") is not None and g["changed"] <= 0:
            out.append("The result is identical to the original: nothing was changed.")
    else:
        flat_ok = any(w in words for w in ("flat", "poster", "cut-out", "cutout", "silhouette", "pictogram", "icon", "logo", "diagram", "line", "ink", "pencil", "sketch", "drawing", "woodcut", "linocut", "stencil"))
        if (g.get("flat") or 0) > 0.45 and not flat_ok:
            out.append(f"{100 * g['flat']:.0f}% of the picture lies in flat areas with no light and shade. Give those things volume (form()), a gradient or a cast shadow; drawing_report() says where they are.")
        if g.get("range") is not None and g["range"] < 0.3 and not flat_ok:
            out.append(f"The tones span only {g['range']:.2f} of the scale from black to white: there is no real dark or no real light. Set the darkest darks and the lightest lights (tones(img, 3) must show three clear steps).")
        paper_ok = any(w in words for w in ("ink", "pencil", "sketch", "line", "drawing", "hatch", "etching", "charcoal", "watercolour", "watercolor", "wash"))
        if (g.get("bare") or 0) > 0.25 and not paper_ok:
            out.append(f"{100 * g['bare']:.0f}% of the sheet is still bare paper. A painting covers its sheet.")
    return out


def text(faults: list[str], kind: str) -> str:
    head = {"process": "Not accepted yet: the work skipped steps that good work needs.", "measured": "Not accepted yet: the result fails checks that need no reviewer."}[kind]
    return head + "\n" + "\n".join(f"- {f}" for f in faults) + "\nDo these, save, and finish again. (If the request truly calls for an exception, finish with accept=\"the reason\".)"
