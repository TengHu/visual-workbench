#!/usr/bin/env python3
"""Run the test set under a named configuration, and compare two configurations with a blind judge.

    vw eval run base                      every task with the default settings
    vw eval run no-critic --tasks recolour-car,remove-tents --max-usd 2
    vw eval compare base no-critic        a judge model picks the better result per task; writes eval/report.html
    vw eval kit                           checks of the kit alone (edges, brushwork, form, the drawing report): free
    vw eval loop                          checks of the delivery gate, built-in loop and hands-on, with a scripted model: free

A configuration is a list of extra command-line flags (eval/configs.json). Results go to eval/results/<config>/<task>/,
each a normal run folder. A task that already has a result is skipped unless --redo is given.
"""
import argparse
import html
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent
VENV_PY = APP / ".venv" / "bin" / "python"
if Path(sys.prefix).resolve() != (APP / ".venv").resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), "-I", str(Path(__file__).resolve()), *sys.argv[1:]])
sys.path.insert(0, str(APP / "src"))

TASKS = json.loads((HERE / "tasks.json").read_text())
CONFIGS = json.loads((HERE / "configs.json").read_text())


def run(a):
    flags = CONFIGS[a.config]
    for t in TASKS:
        if a.tasks and t["id"] not in a.tasks.split(","):
            continue
        out = HERE / "results" / a.config / t["id"]
        if (out / "result.png").exists() and not a.redo:
            print(f"{t['id']}: already done")
            continue
        cmd = [str(APP / "vw")]
        if t["mode"] == "edit":
            cmd.append(str(HERE / "images" / t["image"]))
        cmd.append(t["request"])
        if t.get("size"):
            cmd += ["--size", t["size"]]
        if t.get("ref"):
            cmd += ["--ref", str(HERE / "images" / t["ref"])]
        cmd += ["--model", a.model, "--max-usd", str(a.max_usd), "--out", str(out), *flags]
        print(f"=== {a.config} / {t['id']}")
        if a.dry:
            print("   " + " ".join(cmd))
            continue
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "console.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        info = json.loads((out / "work" / "run.json").read_text()) if (out / "work" / "run.json").exists() else {}
        print(f"   {'result' if (out / 'result.png').exists() else 'NO RESULT'} · {len(info.get('rounds', []))} rounds · ${info.get('spent_usd', 0):.2f} · {info.get('stopped') or 'finished'}")


def compare(a):
    import critic
    from openrouter import create
    judge = create(a.judge, APP)
    rows, score, spent = [], {a.a: 0, a.b: 0, "tie": 0}, 0.0
    rnd = random.Random(7)
    for t in TASKS:
        ra, rb = (HERE / "results" / c / t["id"] / "result.png" for c in (a.a, a.b))
        if not (ra.exists() and rb.exists()):
            continue
        original = HERE / "results" / a.a / t["id"] / "original.png" if t["mode"] == "edit" else None
        flip = rnd.random() < 0.5  # the judge never knows which configuration is A
        better, reason, usage = critic.choose(judge, mode=t["mode"], request=t["request"], original=str(original) if original else None, a=str(rb if flip else ra), b=str(ra if flip else rb))
        spent += usage.get("usd") or 0
        winner = "tie" if better == "tie" else (a.b if (better == "A") == flip else a.a)
        score[winner] += 1
        cost = lambda c: json.loads((HERE / "results" / c / t["id"] / "work" / "run.json").read_text()).get("spent_usd", 0)
        rows.append((t, winner, reason, cost(a.a), cost(a.b)))
        print(f"{t['id']}: {winner} ({reason})")
    rel = lambda c, t, f: html.escape(f"results/{c}/{t['id']}/{f}")
    body = "".join(f"<tr><td><b>{html.escape(t['id'])}</b><br>{html.escape(t['request'])}<br><i>better: {html.escape(w)}</i><br>{html.escape(r)}</td><td>{'<img src=' + chr(34) + rel(a.a, t, 'original.png') + chr(34) + '>' if t['mode'] == 'edit' else ''}</td><td><a href='{rel(a.a, t, 'index.html')}'><img src='{rel(a.a, t, 'result.png')}'></a><br>${ca:.2f}</td><td><a href='{rel(a.b, t, 'index.html')}'><img src='{rel(a.b, t, 'result.png')}'></a><br>${cb:.2f}</td></tr>" for t, w, r, ca, cb in rows)
    (HERE / "report.html").write_text(f"<!DOCTYPE html><meta charset='utf-8'><title>{a.a} vs {a.b}</title><style>body{{font:14px system-ui;margin:20px}}td{{vertical-align:top;padding:8px;border-top:1px solid #8884;max-width:420px}}img{{max-width:400px}}</style><h1>{html.escape(a.a)} {score[a.a]} : {score[a.b]} {html.escape(a.b)} ({score['tie']} ties)</h1><p>Judge: {html.escape(judge.id)}, blind to which is which.</p><table><tr><th>task</th><th>original</th><th>{html.escape(a.a)}</th><th>{html.escape(a.b)}</th></tr>{body}</table>")
    print(f"\n{a.a} {score[a.a]} : {score[a.b]} {a.b}, {score['tie']} ties · judge cost ${spent:.2f}\n{HERE / 'report.html'}")


if sys.argv[1:2] in (["kit"], ["loop"]):  # checks of the kit, and of the loop with a scripted model: no cost
    sys.exit(subprocess.run([str(VENV_PY), "-I", str(HERE / f"{sys.argv[1]}_test.py"), *sys.argv[2:]]).returncode)

p = argparse.ArgumentParser(prog="vw eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("run")
r.add_argument("config", choices=sorted(CONFIGS))
r.add_argument("--tasks")
r.add_argument("--model", default="argon")
r.add_argument("--max-usd", type=float, default=3.0)
r.add_argument("--redo", action="store_true")
r.add_argument("--dry", action="store_true", help="print the commands without running them")
c = sub.add_parser("compare")
c.add_argument("a")
c.add_argument("b")
c.add_argument("--judge", default="argon")
args = p.parse_args()
run(args) if args.cmd == "run" else compare(args)
