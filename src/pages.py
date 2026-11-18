"""The pages a run leaves: index.html (every step: code, output, views), replay.html (the stages) and work/run.json.

Used by the built-in loop (agent.py) and by hands-on jobs (hands.py), so both kinds of run read the same.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Record:
    dir: Path
    mode: str
    request: str
    model_id: str
    w: int
    h: int
    rounds: int = 0
    effort: str = ""
    spent: float = 0.0
    rows: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    reviews: list = field(default_factory=list)
    stopped: str = ""
    summary: str = ""
    delivered: str | None = None
    unsaved: bool = False
    hands_on: bool = False
    passed: bool | None = None  # the review of the delivered picture: True, False, or None when there was none
    accepted: str = ""  # the reason given for letting a result past the free checks

    @property
    def work(self) -> Path:
        return self.dir / "work"


def write(r: Record, done: bool = False, final: Path | None = None) -> None:
    rel = lambda f: html.escape(str(Path("work") / f))
    cards = []
    for row in r.rows:
        items = []
        if row["text"]:
            items.append(f"<p class='say'>{html.escape(row['text'])}</p>")
        for s in row["steps"]:
            if s["kind"] == "python":
                items.append(f"<details{' open' if s['error'] else ''}><summary>python{' (error)' if s['error'] else ''}</summary><pre>{html.escape(s['code'])}</pre><pre class='out'>{html.escape(s['out'])}</pre></details>")
            elif s["kind"] == "check":
                items.append(f"<details open><summary>delivery checks</summary><pre class='out'>{html.escape(s.get('note', ''))}</pre></details>")
            elif s["kind"] == "review":
                items.append(f"<details open><summary>review</summary><pre class='out'>{html.escape(s.get('note', ''))}</pre></details>")
            elif s["kind"] == "finish":
                items.append(f"<p class='say'><b>finish</b> {html.escape(s.get('note', ''))}</p>")
            items += [f"<a href='{rel(v)}'><img src='{rel(v)}' loading='lazy'></a>" for v in s.get("views", [])]
        u = row["usage"]
        facts = f"{row['seconds']:.0f} s" + (f" · {u.get('output')} output tokens ({u.get('reasoning')} thinking) · {u.get('input')} input ({u.get('cached')} cached) · ${u.get('usd') or 0:.3f}" if u else "")
        cards.append(f"<section><h2>{'Step' if r.hands_on else 'Round'} {row['round']}</h2><p class='facts'>{facts}</p>{''.join(items)}</section>")
    state = (r.stopped or "finished") if done else (f"working: {len(r.rows)} steps so far (reload to update)" if r.hands_on else f"working: {len(r.rows)} of {r.rounds} rounds (reload to update)")
    if done and final:
        state += " · " + ("PASSED the review" if r.passed else "NOT PASSED: the reviewer still lists faults (see the last review below)" if r.passed is False else "not independently reviewed")
        if r.accepted:
            state += f" · accepted past the checks because: {r.accepted}"
    if final and r.unsaved:
        state += " · the model never saved a result; this is the last whole picture it looked at, unreviewed"
    before = "<figure><img src='original.png'><figcaption>original</figcaption></figure>" if r.mode == "edit" else ""
    top = f"<div class='{'pair' if before else ''}'>{before}<figure><img src='result.png'><figcaption>result</figcaption></figure></div>" if final else ""
    replay = "<a href='replay.html'>step-by-step replay</a> · " if r.stages else ""
    css = ":root{color-scheme:light dark}body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:19px}h2{font-size:16px;margin:28px 0 4px}img{max-width:100%;display:block;margin:6px 0;border:1px solid #8884}.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:700px){.pair{grid-template-columns:1fr}}figure{margin:0}figcaption,.facts{opacity:.7;font-size:13px}pre{white-space:pre-wrap;font-size:12px;background:#8881;padding:8px}pre.out{opacity:.85}.say{white-space:pre-wrap}"
    (r.dir / "index.html").write_text(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{'Photo edit' if r.mode == 'edit' else 'Drawing'}</title>
<style>{css}</style></head><body>
<h1>{html.escape(r.request)}</h1><p class="facts">{html.escape(r.model_id)} {(' · effort ' + r.effort) if r.effort else ''} · {r.w}×{r.h} · ${r.spent:.2f} · {html.escape(state)}{(' · ' + html.escape(r.summary)) if r.summary else ''}</p>
{top}<p>{replay}<a href="work/system.md">the manual the model was given</a></p>{''.join(cards)}</body></html>""")
    if r.stages:
        frames = [{"src": str(Path("work") / s["file"]), "name": s["name"]} for s in r.stages]
        if final:
            frames.append({"src": "result.png", "name": "result"})
        (r.dir / "replay.html").write_text(f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Replay</title>
<style>:root{{color-scheme:dark}}body{{margin:0;background:#151413;color:#e9e4da;font:14px/1.4 system-ui,sans-serif;padding:16px}}.bar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}}button{{font:inherit;color:inherit;background:#2a2724;border:1px solid #4a4540;border-radius:6px;padding:6px 12px;cursor:pointer}}input[type=range]{{flex:1;min-width:160px}}img{{max-width:100%;max-height:82vh;display:block}}#label{{min-width:240px}}</style></head><body>
<div class="bar"><button id="prev">◀</button><button id="play">Play</button><button id="next">▶</button><input id="slider" type="range" min="0" max="{len(frames) - 1}" value="0"><span id="label"></span></div>
<img id="pic" alt="The picture at this step">
<script>
const frames = {json.dumps(frames).replace("<", "\\u003c")};
const $ = (id) => document.getElementById(id); let i = 0, timer = null;
frames.forEach((f) => {{ new Image().src = f.src; }});
function show(n) {{ i = (n + frames.length) % frames.length; $("pic").src = frames[i].src; $("slider").value = i; $("label").textContent = (i + 1) + " / " + frames.length + ": " + frames[i].name; }}
function stop() {{ clearInterval(timer); timer = null; $("play").textContent = "Play"; }}
$("prev").onclick = () => {{ stop(); show(i - 1); }}; $("next").onclick = () => {{ stop(); show(i + 1); }};
$("slider").oninput = (e) => {{ stop(); show(Number(e.target.value)); }};
$("play").onclick = () => {{ if (timer) return stop(); $("play").textContent = "Pause"; if (i === frames.length - 1) show(0); timer = setInterval(() => (i === frames.length - 1 ? stop() : show(i + 1)), 1100); }};
show(0);
</script></body></html>""")
    (r.work / "run.json").write_text(json.dumps({"mode": r.mode, "model": r.model_id, "hands_on": r.hands_on, "passed": r.passed, "accepted": r.accepted or None, "request": r.request, "spent_usd": r.spent, "stopped": r.stopped or None, "delivered": r.delivered, "unsaved": r.unsaved, "reviews": r.reviews, "stages": r.stages, "rounds": r.rows}, indent=1, default=str))
