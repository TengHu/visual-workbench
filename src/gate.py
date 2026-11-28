"""The checks a result must pass before it is reviewed. They cost nothing, they are the same for every model, every
agent and every request, and they are only about facts that need no judgement: is this the picture a review already
sent back, is it the size of the plate, is it different from the plate.

  process(...)   facts about the work: is the same picture offered again after a review sent it back?
  measured(...)  facts about the result: its size, and whether it differs from the plate at all.

A fault here holds the delivery back with a plain sentence saying what was found. `accept` (a reason, in words) lets a
result through when the request calls for it (for example a request to change the size); the reason is kept with
the run.
"""
from __future__ import annotations


def digest(path) -> str:
    """A fingerprint of a picture file, to tell whether it changed."""
    import hashlib
    from pathlib import Path
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def process(st: dict) -> list[str]:
    """Facts about the work. st: the session's record, and, when a review sent the result back, `resubmitted`:
    whether the very same picture file is offered again."""
    out = []
    if st.get("resubmitted"):
        out.append("This is the same picture file the reviewer sent back: it has not changed since that review.")
    return out


def measured(g: dict, has_plate: bool = True) -> list[str]:
    """Facts about the result. g: the numbers the review sheet wrote (gate.json): size_ok, changed."""
    out = []
    if g.get("size_ok") is False:
        out.append("The result is not the size of the plate (" + ("the picture that came with the request" if has_plate else "empty") + ").")
    if g.get("changed") is not None and g["changed"] <= 0:
        out.append("The result is identical to the plate: no pixel differs.")
    return out


def text(faults: list[str], kind: str) -> str:
    head = {"process": "Not accepted yet:", "measured": "Not accepted yet:"}[kind]
    return head + "\n" + "\n".join(f"- {f}" for f in faults) + "\nChange the result, save, and finish again; or, if the request calls for this, finish with accept=\"the reason\"."
