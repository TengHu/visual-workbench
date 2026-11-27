"""The independent reviewer: a model call with a fresh conversation that sees only the request, the starting point,
the result and measurements, never the work that led to it. It reports faults; it does not fix them.

The same function serves as the judge of the test set (eval/run.py), which asks it to choose between two results.
"""
from __future__ import annotations

import json
import re

SYSTEM = """You review a picture against a request. You are shown the request, the plate (the picture that came with the request, when there was one), any references that came with the request, and the result: whole, in close-up views of the areas that changed, and with some measurements. You did not make it and have not seen how it was made.

Judge the result only against the request. Check:
1. The request: list its parts as it states them. For each, is it done in the result? Where the request leaves something open, do not supply a preference of your own.
2. Changes the request did not ask for: anything different from the plate that no part of the request accounts for.
3. Visible defects in carrying out the request: places where the result shows something the request did not ask for, such as an unintended seam, fringe, gap, smear, leftover of something removed, or a part that is missing or broken off.

Report only what you can see in the pictures. Do not report matters of style or taste that the request does not address. Say where each fault is in plain words (for example "left edge of the bonnet, just under the mirror")."""

FORMAT = """Answer with one JSON object and nothing else:
{"request": [{"part": "<one part of the request>", "done": true or false, "note": "<why not, if not>"}],
 "faults": [{"where": "<place in the picture>", "what": "<what is there, and which part of the request it concerns>", "severity": "major" or "minor"}],
 "verdict": "deliver" or "fix"}
"deliver" means you would let it go out as it is: every part of the request done and no major fault. A fault is "major" when it leaves a part of the request undone or contradicted. List at most 8 faults, major ones first."""


def _json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def review(model, *, request: str, pictures: list[tuple[str, str]], numbers: str = "", effort: str = "medium") -> tuple[dict, dict]:
    """Ask for a review. pictures: (caption, file) in the order to show them. Returns (review, usage)."""
    parts = [{"type": "text", "text": f"# The request\n\n{request}"}]
    for caption, file in pictures:
        parts += [{"type": "text", "text": caption + ":"}, {"type": "image", "file": file}]
    if numbers.strip():
        parts.append({"type": "text", "text": "Measurements:\n" + numbers.strip()})
    parts.append({"type": "text", "text": FORMAT})
    reply = model.send([{"role": "system", "content": SYSTEM}, {"role": "user", "content": parts}], [], effort, 8000)
    out = _json(reply.text) or {"request": [], "faults": [{"where": "", "what": "the reviewer's answer could not be read: " + reply.text[:300], "severity": "minor"}], "verdict": "fix"}
    out.setdefault("request", [])
    out.setdefault("faults", [])
    if out.get("verdict") not in ("deliver", "fix"):
        out["verdict"] = "fix" if out["faults"] or any(not r.get("done", True) for r in out["request"]) else "deliver"
    return out, reply.usage


def as_text(r: dict) -> str:
    """A review as the plain text handed to the model that did the work."""
    lines = []
    undone = [x for x in r.get("request", []) if not x.get("done", True)]
    if undone:
        lines.append("Not done:")
        lines += [f"- {x.get('part', '?')}" + (f" ({x['note']})" if x.get("note") else "") for x in undone]
    elif r.get("request"):
        lines.append("Every part of the request is judged done.")
    if r.get("faults"):
        lines.append("Faults, worst first:")
        lines += [f"- [{x.get('severity', 'minor')}] {x.get('where', '')}: {x.get('what', '')}" for x in r["faults"]]
    else:
        lines.append("No faults reported.")
    lines.append(f"Verdict: {r.get('verdict')}.")
    return "\n".join(lines)


def choose(model, *, request: str, original: str | None, a: str, b: str, effort: str = "medium") -> tuple[str, str, dict]:
    """Which of two results is better? Returns ('A' | 'B' | 'tie', reason, usage). The caller shuffles A and B."""
    parts = [{"type": "text", "text": f"# The request\n\n{request}"}]
    if original:
        parts += [{"type": "text", "text": "The plate:"}, {"type": "image", "file": original}]
    parts += [{"type": "text", "text": "Result A:"}, {"type": "image", "file": a}, {"type": "text", "text": "Result B:"}, {"type": "image", "file": b}]
    parts.append({"type": "text", "text": 'Which result carries out the request better, judged as described? Answer with one JSON object and nothing else: {"better": "A" or "B" or "tie", "reason": "<one sentence>"}'})
    reply = model.send([{"role": "system", "content": SYSTEM}, {"role": "user", "content": parts}], [], effort, 4000)
    j = _json(reply.text) or {}
    return (j.get("better") if j.get("better") in ("A", "B", "tie") else "tie"), str(j.get("reason", "")), reply.usage
