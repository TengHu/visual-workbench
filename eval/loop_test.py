#!/usr/bin/env python3
"""Checks of the built-in loop with a scripted model: no API call, no cost. Run after changing src/agent.py,
src/gate.py or src/hands.py.

    vw eval loop

The scripted model does a drawing job badly on purpose (no brief, no stages, a flat picture), then properly, and
the checks must hold the bad deliveries back and let the good one through. The same is done through the hands-on
commands (vw open / py / finish).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
VENV_PY = APP / ".venv" / "bin" / "python"
if Path(sys.prefix).resolve() != (APP / ".venv").resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), "-I", str(Path(__file__).resolve()), *sys.argv[1:]])
sys.path.insert(0, str(APP / "src"))

from agent import Run  # noqa: E402
from openrouter import Reply  # noqa: E402

FLAT = 'img = orig.copy(); img[:] = as_color("#7f8791"); img = blend(img, np.broadcast_to(as_color("#d8a23a"), img.shape), ellipse_mask(img, (150, 100, 350, 300))); save(img, "result.png")'
GOOD = '''brief("A single orange ball on a grey table, seen from slightly above, large and centred, lit from the upper left by a soft window light, in grey, orange and a warm dark, painted in oils, loosely.")
plan(["big shapes", "light and dark", "rendering"])
m = ellipse_mask(orig, (150, 100, 350, 300))
table = np.broadcast_to(as_color("#8a8f98"), orig.shape) * np.linspace(1.15, 0.75, H, dtype=np.float32)[:, None, None]
stack.put("table", table)
stack.put("ball", m, color="#d8a23a")
stage("big shapes", stack.flatten())
stack.put("ball", form(m, "#d8a23a", light=(-0.7, -0.7), contrast=0.7, highlight=0.3))
stack.put("shadow", shadow(m, offset=(40, 60), blur=12, opacity=0.7, squash=0.4), below="ball", color="#2a2622")
stage("light and dark", stack.flatten())
show(tones(stack.flatten(), 3), "tones")
img = paint(stack.flatten(), brushes=(24, 12, 6, 3), brush="bristle")
stage("rendering", img)
save(img, "result.png")'''


class Scripted:
    """Stands in for a model: each reply is the next scripted list of tool calls."""
    id = "scripted/test"

    def __init__(self, turns):
        self.turns, self.n = list(turns), 0

    def body(self, *a, **k):
        return {}

    def send(self, messages, tools, effort, max_tokens, on_progress=None):
        calls = self.turns.pop(0) if self.turns else [("finish", {})]
        self.n += 1
        tc = [{"id": f"c{self.n}-{i}", "name": n, "args": a, "error": None} for i, (n, a) in enumerate(calls)]
        return Reply(text="", reasoning="", tool_calls=tc, assistant={"role": "assistant", "raw": {"role": "assistant", "content": "", "tool_calls": []}}, finish="tool_calls", provider="script", usage={"usd": 0.0})


failed = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    if not ok:
        failed.append(name)


tmp = Path(tempfile.mkdtemp(prefix="loop-test-"))
# 1. The built-in loop, with no reviewer model (self-review), so it costs nothing.
model = Scripted([
    [("python", {"code": FLAT}), ("finish", {})],     # flat, no brief, no stages: held back by the process checks
    [("python", {"code": GOOD}), ("finish", {})],     # the work done properly: passes the free checks, then self-review
    [("finish", {})],                                  # delivers after the self-review
])
run = Run(mode="draw", model=model, critic_model=None, image=None, size=(500, 400), request="An orange ball on a grey table, painted in oils.", refs=[], ref_level="measure", out=tmp / "loop", rounds=6, effort="low", max_tokens=1000, max_usd=1, coords="xy", sam=False, max_side=4096, seconds=120, review=True, critic_passes=2)
run.session.sandbox = run.session.sandbox  # the session runs sandboxed, as in a real run
final = run.go()
kinds = [s["kind"] for r in run.rows for s in r["steps"]]
check("loop: a flat picture made without a brief or stages is held back", kinds.count("check") >= 1 and "There is no brief" in next(s["note"] for r in run.rows for s in r["steps"] if s["kind"] == "check"), str(kinds))
check("loop: the properly made picture is reviewed, then delivered", kinds[-2:] == ["review", "finish"] and final is not None, str(kinds))

# 2. The hands-on commands.
vw = str(APP / "vw")
job = tmp / "hands"
def sh(*args, code=None):
    return subprocess.run([vw, *args], input=code, capture_output=True, text=True, timeout=600)
o = sh("open", "An orange ball on a grey table, painted in oils.", "--size", "500x400", "--critic", "none", "--no-sam", "--out", str(job))
check("hands: open", o.returncode == 0 and (job / "MANUAL.md").exists(), o.stderr[-300:])
sh("py", str(job), "-", code=FLAT)
f1 = sh("finish", str(job))
check("hands: a flat, unstaged picture is held back", "NOT DELIVERED" in f1.stdout and "brief" in f1.stdout, f1.stdout[-300:])
p = sh("py", str(job), "-", code=GOOD)
check("hands: the proper steps run", p.returncode == 0, (p.stdout + p.stderr)[-400:])
f2 = sh("finish", str(job))
check("hands: then the self-review comes", "NOT DELIVERED" in f2.stdout and "review it yourself" in f2.stdout, f2.stdout[-400:])
f3 = sh("finish", str(job), "--summary", "test")
check("hands: then it is delivered", "DELIVERED:" in f3.stdout and (job / "result.png").exists(), f3.stdout[-300:])
info = json.loads((job / "work" / "run.json").read_text())
check("hands: the run's record says it was not independently reviewed", info.get("passed") is None, str(info.get("passed")))

import gate  # noqa: E402
check("a picture sent back and offered again unchanged is held back", any("nothing was changed" in f for f in gate.process("draw", {"resubmitted": True, "brief": "x", "plan": [1], "stages": [1, 2, 3], "calls": {"tones": 1}})))

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + ("all checks passed" if not failed else f"{len(failed)} failed: " + ", ".join(failed)))
sys.exit(1 if failed else 0)
