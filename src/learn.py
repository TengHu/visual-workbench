"""`vw learn RUN [RUN ...]`: one self-contained HTML page that teaches how runs were done.

For each run folder it gathers the request, the before and after, the stages, the reviewer's reports, and every round:
the model's own reasoning summary, the code it wrote, what the code printed, and the views it looked at. A `NOTES.md`
in a run folder (written afterwards, by whoever studied the run) is shown at the top of that run as the guided reading; "round 12" there links to
the round. Pictures are embedded, so the page is one file that can be sent as it is.
"""
from __future__ import annotations

import ast
import base64
import html
import io
import json
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image

APP = Path(__file__).resolve().parent.parent


def kit_names() -> set[str]:
    """Public functions of the kit, to mark them in the model's code."""
    names = set()
    for f in (APP / "src" / "kit").glob("*.py"):
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                names.add(node.name)
    return names - {"available", "segment"}


KIT = kit_names()
KIT_RE = re.compile(r"\b(" + "|".join(sorted(KIT, key=len, reverse=True)) + r")(?=\()")


def picture(path: Path, width: int = 760, quality: int = 80) -> str:
    """A JPEG data URL, at most `width` wide."""
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def code_html(code: str) -> str:
    out = html.escape(code)
    out = KIT_RE.sub(r"<b class='kit'>\1</b>", out)
    return re.sub(r"(?m)^(\s*)(#.*)$", r"\1<i class='cm'>\2</i>", out)


def notes_html(text: str, rid: str) -> str:
    """A small Markdown subset: headings, bullets, paragraphs, `code`, **bold**, and round links."""
    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        return re.sub(r"\b([Rr]ounds?) (\d+)(?:(–|-| to )(\d+))?", lambda m: f"<a href='#{rid}-r{m.group(2)}'>{m.group(0)}</a>", s)
    out, para, items = [], [], []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            flush()
        elif s.startswith("#"):
            flush()
            out.append(f"<h4>{inline(s.lstrip('#').strip())}</h4>")
        elif s.startswith(("- ", "* ")):
            if para:
                flush()
            items.append(s[2:])
        elif items and line.startswith("  "):
            items[-1] += " " + s
        else:
            para.append(s)
    flush()
    return "".join(out)


def view_captions(out: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in re.finditer(r"(?m)^(v\d+): (.*?) \[", out or "")}


def run_section(d: Path, k: int) -> tuple[str, str, str]:
    """(nav label, section html, system prompt) for one run folder."""
    rid = f"run{k}"
    info = json.loads((d / "work" / "run.json").read_text())
    request = (d / "REQUEST.md").read_text().strip() if (d / "REQUEST.md").exists() else info.get("request", "")
    console = (d / "console.log").read_text() if (d / "console.log").exists() else ""
    head = re.search(r"up to (\d+) rounds · cap \$([\d.]+)", console)
    rows = info["rounds"]
    llm = d / "work" / "llm"

    # numbers
    usage = [r.get("usage") or {} for r in rows]
    tin, tcached = sum(u.get("input") or 0 for u in usage), sum(u.get("cached") or 0 for u in usage)
    seconds = sum(r.get("seconds") or 0 for r in rows)
    steps = [s for r in rows for s in r["steps"]]
    lines = sum(len(s["code"].splitlines()) for s in steps if s["kind"] == "python")
    nviews = sum(len(s.get("views") or []) for s in steps)
    calls = Counter(m for s in steps if s["kind"] == "python" for m in KIT_RE.findall(s["code"]))
    verdicts = [r.get("verdict") for r in info.get("reviews") or []]

    # pictures: before, the version first sent to review, delivered
    before = d / "original.png"
    after = d / "result.png"
    reviewed = None
    last_snap = None
    for r in rows:
        for s in r["steps"]:
            snaps = re.findall(r"snapshots/result-\d+\.png", s.get("out") or "")
            if snaps:
                last_snap = snaps[-1]
            if s["kind"] == "review" and reviewed is None and last_snap:
                reviewed = d / "work" / last_snap
    figs = []
    if before.exists():
        figs.append(("before", before))
    if reviewed and reviewed.exists():
        figs.append(("first finish, sent back by the reviewer", reviewed))
    if after.exists():
        figs.append(("delivered", after))
    pics = "".join(f"<figure><img src='{picture(p, 900)}'><figcaption>{html.escape(c)}</figcaption></figure>" for c, p in figs)

    stages = info.get("stages") or []
    strip = "".join(f"<figure><img src='{picture(d / 'work' / s['file'], 420)}'><figcaption><b>{i + 1}. {html.escape(s['name'])}</b><br>{html.escape(s.get('note') or '')}</figcaption></figure>"
                    for i, s in enumerate(stages) if (d / "work" / s["file"]).exists())

    notes = (d / "NOTES.md").read_text() if (d / "NOTES.md").exists() else ""
    reviews = info.get("reviews") or []

    # timeline
    items, nrev = [], 0
    for r in rows:
        n = r["round"]
        u = r.get("usage") or {}
        reasoning = ""
        f = llm / f"response-{n:02d}.json"
        if f.exists():
            reasoning = json.loads(f.read_text()).get("reasoning") or ""
        body, tags = [], []
        if reasoning.strip():
            body.append(f"<div class='think'><div class='lbl'>The model's reasoning (its own summary)</div>{notes_html(reasoning, rid)}</div>")
        if r.get("text"):
            body.append(f"<div class='say'><div class='lbl'>The model said</div>{html.escape(r['text'])}</div>")
        for s in r["steps"]:
            if s["kind"] == "python":
                used = sorted(set(KIT_RE.findall(s["code"])))
                tags += used
                body.append(f"<div class='lbl'>Code it ran ({len(s['code'].splitlines())} lines)</div><pre class='code'>{code_html(s['code'])}</pre>")
                if (s.get("out") or "").strip():
                    body.append(f"<div class='lbl'>What it printed{' (error)' if s.get('error') else ''}</div><pre class='out{' err' if s.get('error') else ''}'>{html.escape(s['out'].strip()[:6000])}</pre>")
                caps = view_captions(s.get("out") or "")
                vs = [v for v in s.get("views") or [] if (d / "work" / v).exists()]
                if vs:
                    body.append("<div class='lbl'>What it looked at</div><div class='views'>" + "".join(
                        f"<figure><img loading='lazy' src='{picture(d / 'work' / v, 520, 72)}'><figcaption>{html.escape(Path(v).stem)}: {html.escape(caps.get(Path(v).stem, ''))}</figcaption></figure>" for v in vs) + "</div>")
            elif s["kind"] == "review":
                rv = reviews[nrev] if nrev < len(reviews) else {}
                nrev += 1
                tags.append("REVIEW")
                faults = "".join(f"<li><span class='sev {html.escape(x.get('severity', ''))}'>{html.escape(x.get('severity', ''))}</span> <b>{html.escape(x.get('where', ''))}</b>: {html.escape(x.get('what', ''))}</li>" for x in rv.get("faults", []))
                parts = "".join(f"<li>{'✓' if x.get('done') else '✗'} {html.escape(x.get('part', ''))}{(' — ' + html.escape(x['note'])) if x.get('note') else ''}</li>" for x in rv.get("request", []))
                vs = [v for v in s.get("views") or [] if (d / "work" / v).exists()]
                rcaps = view_captions(s.get("note") or "")
                body.append(f"<div class='review'><div class='lbl'>Independent reviewer (fresh conversation, saw only the request, the original and this result) — verdict: <b>{html.escape(rv.get('verdict', '?'))}</b></div>"
                            f"<p>Request:</p><ul>{parts}</ul><p>Faults:</p><ul>{faults}</ul>"
                            + ("<div class='lbl'>What the reviewer was shown</div><div class='views'>" + "".join(f"<figure><img loading='lazy' src='{picture(d / 'work' / v, 520, 72)}'><figcaption>{html.escape(rcaps.get(Path(v).stem, ''))}</figcaption></figure>" for v in vs) + "</div>" if vs else "") + "</div>")
            elif s["kind"] == "finish":
                tags.append("FINISH")
                body.append(f"<div class='say'><div class='lbl'>finish</div>{html.escape(s.get('note') or '')}</div>")
        tagtxt = " ".join(f"<span class='tag{' hot' if t in ('REVIEW', 'FINISH') else ''}'>{html.escape(t)}</span>" for t in dict.fromkeys(tags))
        items.append(f"<details id='{rid}-r{n}'><summary><span class='rn'>Round {n}</span> <span class='meta'>{(r.get('seconds') or 0):.0f} s · ${u.get('usd') or 0:.3f} · {u.get('output') or 0} tokens out</span> {tagtxt}</summary>{''.join(body)}</details>")

    flags = f" --rounds {head.group(1)} --max-usd {head.group(2)}" if head else ""
    cmd = f"vw {'original.png ' if info['mode'] == 'edit' else ''}{json.dumps(request)}{flags}"
    stats = [("model", info["model"]), ("rounds", str(len(rows))), ("cost", f"${info.get('spent_usd') or 0:.2f}"), ("model time", f"{seconds / 60:.0f} min"),
             ("code written", f"{lines} lines"), ("views looked at", str(nviews)), ("input cached", f"{100 * tcached / max(tin, 1):.0f}%"),
             ("reviews", ", ".join(verdicts) or "none")]
    section = f"""
<section id='{rid}'>
<h2>{k}. {html.escape(request[:90])}</h2>
<p class='request'><b>The request, word for word:</b> {html.escape(request)}</p>
<dl class='stats'>{''.join(f'<div><dt>{a}</dt><dd>{html.escape(b)}</dd></div>' for a, b in stats)}</dl>
<div class='pics'>{pics}</div>
{f"<div class='notes'><div class='lbl'>Reading guide (written afterwards from the trace, not by the model that did the work)</div>{notes_html(notes, rid)}</div>" if notes else ''}
{f"<h3>Stages, as the model recorded them</h3><div class='strip'>{strip}</div>" if strip else ''}
<h3>Kit functions it called</h3>
<p class='calls'>{' '.join(f"<span class='tag'>{html.escape(c)} ×{n}</span>" for c, n in calls.most_common())}</p>
<h3>Every round</h3>
<p class='hint'>Click a round to open it. <a href='#' onclick="document.querySelectorAll('#{rid} details').forEach(e=>e.open=true);return false">Open all</a> · <a href='#' onclick="document.querySelectorAll('#{rid} details').forEach(e=>e.open=false);return false">close all</a></p>
{''.join(items)}
<h3>Run it again</h3>
<pre class='out'>{html.escape(cmd)}</pre>
<p class='hint'>Results differ from run to run; the method should not. The raw trace is in <code>{html.escape(d.name)}/</code>: <code>index.html</code>, <code>replay.html</code>, <code>work/run.json</code>, <code>work/llm/</code>.</p>
</section>"""
    system = (d / "work" / "system.md").read_text() if (d / "work" / "system.md").exists() else ""
    return request[:60], section, system


OVERVIEW = """
<h2>How the workbench works</h2>
<p>A language model that can only write text edits pictures by <b>writing Python</b>. It never outputs pixels and no image-generation model is involved: every pixel comes from code the model wrote, or from a tool in the kit that the model called with arguments it chose.</p>
<div class='arch'>
  <div class='box'><b>You</b><br>a picture and a request, e.g. <code>vw photo.jpg "make the dress green"</code></div>
  <div class='arrow'>→</div>
  <div class='box main'><b>The loop</b> <code>src/agent.py</code><br>sends the conversation to the model; runs each tool call; returns what the code printed and the pictures it showed; tracks rounds and budget</div>
  <div class='arrow'>⇄</div>
  <div class='col'>
    <div class='box'><b>The model</b> <code>src/openrouter.py</code><br>Gemini 4 Argon (or any OpenRouter model). Two tools: <code>python(code)</code> and <code>finish(file)</code></div>
    <div class='box'><b>The sandboxed session</b> <code>src/session.py</code>, <code>src/worker.py</code><br>a persistent Python, like a notebook, with no network and no access outside the run folder. Holds <code>plate</code> and the kit</div>
    <div class='box'><b>The kit</b> <code>src/kit/</code><br>SAM 2.1 selection, PatchMatch fill, Poisson blending, recolour, a stroke engine, layers and stages, landmarks, edit checks</div>
    <div class='box'><b>The reviewer</b> <code>src/critic.py</code><br>a fresh conversation that sees only the request, the original and the result; answers deliver or fix</div>
  </div>
</div>
<h3>One round</h3>
<ol>
<li>The model replies with a <code>python</code> call. Its code calls kit functions (marked <b class='kit'>like this</b> in the code below) and plain numpy, OpenCV and Skia.</li>
<li>The session runs it. What it printed goes back as the tool result; every <code>show(img, box=...)</code> becomes a picture the model sees next. <b>The model only sees what it asks to see</b>, so zooming in is how it judges detail (each image costs the same tokens whatever its size).</li>
<li>A status line tells it the rounds and money left; near the end it is told to wrap up.</li>
<li>When it calls <code>finish</code>, the harness builds close-ups and measurements and asks the independent reviewer. On "fix" the report goes back to the model and it keeps working (at most twice).</li>
</ol>
<h3>What to read in each run</h3>
<ul>
<li><b>The model's reasoning</b>: the summary of its thinking for that round, why it did what it did.</li>
<li><b>The code</b>: exactly what ran. Kit calls are highlighted; everything else is the model's own maths.</li>
<li><b>What it looked at</b>: the views it requested, including the close-ups it used to find faults.</li>
<li><b>The review</b>: what an outside eye caught, and the rounds after it, where the model fixes those faults.</li>
</ul>
"""

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mute:#6b6a65;--line:#e4e1d8;--card:#fff;--code:#f4f2ec;--kit:#0b6b4f;--cm:#8a877d;--accent:#9a3b1b;--think:#f1f5fb;--rev:#fff6e8;--err:#fdeeee}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151514;--fg:#ecebe6;--mute:#a19f97;--line:#33322f;--card:#1d1d1b;--code:#232220;--kit:#5fd3a6;--cm:#8f8c82;--accent:#f0a07c;--think:#1b2230;--rev:#2a2215;--err:#341c1c}}
:root[data-theme=dark]{--bg:#151514;--fg:#ecebe6;--mute:#a19f97;--line:#33322f;--card:#1d1d1b;--code:#232220;--kit:#5fd3a6;--cm:#8f8c82;--accent:#f0a07c;--think:#1b2230;--rev:#2a2215;--err:#341c1c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 80px}h1{font-size:28px;margin:8px 0 4px}h2{font-size:22px;margin:48px 0 8px;padding-top:16px;border-top:2px solid var(--line)}h3{font-size:16px;margin:28px 0 8px}h4{margin:14px 0 4px}
a{color:var(--accent)}code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.9em}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;z-index:5;font-size:14px;overflow-x:auto;white-space:nowrap}nav a{margin-right:16px}
.lead{color:var(--mute);max-width:760px}
.arch{display:grid;grid-template-columns:1fr auto 1.2fr auto 1.6fr;gap:10px;align-items:center;margin:16px 0}.arch .col{display:grid;gap:8px}.box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:13.5px}.box.main{border-color:var(--accent)}.arrow{font-size:22px;color:var(--mute);text-align:center}
@media (max-width:820px){.arch{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}}
.request{background:var(--card);border-left:3px solid var(--accent);padding:10px 14px;border-radius:6px}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.stats div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px}.stats dt{font-size:11px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}.stats dd{margin:0;font-weight:600}
.pics{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}.pics img{max-height:460px;width:auto;max-width:100%}figure{margin:0}figure img{width:100%;border-radius:6px;border:1px solid var(--line);display:block}figcaption{font-size:12.5px;color:var(--mute);margin-top:4px}
.notes{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:4px 18px 12px;margin:18px 0}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.tag{display:inline-block;font:12px ui-monospace,Menlo,monospace;background:var(--code);border-radius:4px;padding:1px 6px;margin:2px 2px 0 0}.tag.hot{background:var(--accent);color:var(--bg)}
details{border:1px solid var(--line);border-radius:8px;margin:6px 0;background:var(--card)}summary{cursor:pointer;padding:8px 12px;list-style:none}summary::-webkit-details-marker{display:none}summary .rn{font-weight:700;margin-right:6px}summary .meta{color:var(--mute);font-size:13px;margin-right:6px}
details[open]{padding-bottom:12px}details>*:not(summary){margin-left:12px;margin-right:12px}
.lbl{font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute);margin:12px 0 4px}
pre{white-space:pre-wrap;word-break:break-word;background:var(--code);border-radius:6px;padding:10px 12px;font:12.5px/1.45 ui-monospace,Menlo,monospace;margin:0;max-height:560px;overflow:auto}pre.err{background:var(--err)}
b.kit{color:var(--kit)}i.cm{color:var(--cm)}
.think{background:var(--think);border-radius:6px;padding:2px 12px 6px;margin-top:10px;font-size:14px}.think h4{font-size:14px;margin:8px 0 2px}.think p{margin:2px 0 6px}
.say{margin-top:8px}.review{background:var(--rev);border-radius:6px;padding:4px 12px 10px;margin-top:10px}.review ul{margin:4px 0}
.sev{font:11px ui-monospace,Menlo,monospace;padding:0 5px;border-radius:3px;background:var(--code)}.sev.major{background:var(--accent);color:var(--bg)}
.views{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px}
.hint{color:var(--mute);font-size:13px}
details.manual pre{max-height:none}
"""


def build(folders: list[Path], out: Path, title: str) -> Path:
    navs, sections, system = [], [], ""
    for k, d in enumerate(folders, 1):
        label, section, sysmd = run_section(d, k)
        navs.append(f"<a href='#run{k}'>{k}. {html.escape(label[:40])}</a>")
        sections.append(section)
        system = system or sysmd
    manual = f"<h2 id='manual'>What the model was told</h2><p class='hint'>The full system prompt, generated by <code>src/manual.py</code> from the method text and the kit's own docstrings. This is all the model knows about the kit.</p><details class='manual'><summary>Open the system prompt</summary><pre>{html.escape(system)}</pre></details>" if system else ""
    page = f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<nav><a href='#overview'>How it works</a>{''.join(navs)}{"<a href='#manual'>The system prompt</a>" if manual else ''}</nav>
<main>
<h1>{html.escape(title)}</h1>
<p class='lead'>How a language model edited these pictures by writing code, traced round by round from the logs. Made {time.strftime('%Y-%m-%d')} with <code>vw learn</code>.</p>
<div id='overview'>{OVERVIEW}</div>
{''.join(sections)}
{manual}
</main></body></html>"""
    out.write_text(page)
    return out


def main(argv: list[str]) -> None:
    import argparse
    p = argparse.ArgumentParser(prog="vw learn", description="Make one self-contained HTML page that teaches how runs were done. Put a NOTES.md in a run folder to add guided reading.")
    p.add_argument("runs", nargs="+", help="run folders (or names under runs/)")
    p.add_argument("--out", help="the page to write (default runs/learn-<time>.html)")
    p.add_argument("--title", default="How these pictures were made")
    a = p.parse_args(argv)
    folders = []
    for r in a.runs:
        d = Path(r) if Path(r).is_dir() else APP / "runs" / r
        if not (d / "work" / "run.json").exists():
            p.error(f"not a run folder: {r}")
        folders.append(d.resolve())
    out = Path(a.out) if a.out else APP / "runs" / f"learn-{time.strftime('%Y%m%d-%H%M%S')}.html"
    build(folders, out, a.title)
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")
