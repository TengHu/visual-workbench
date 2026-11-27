"""The loop. The request (words and pictures) goes to the model with the manual; the model acts through `python` and
`finish`; each result goes back; this repeats until it finishes, or the rounds or the money run out.

The harness knows nothing about any particular picture. It runs code, shows pictures, keeps the conversation cheap
to resend, never ends a run empty-handed, and holds `finish` back for an independent review of the result.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps

import critic
import gate
import manual
import pages
from openrouter import ModelError, OpenRouter
from session import Session

# The conversation is resent every round, and a provider caches it only while its beginning stays the same. So
# earlier messages are left alone, and trimmed rarely and in one go.
MAX_PICTURES = 20  # when more views than this are in the conversation ...
KEEP_PICTURES = 8  # ... the views of all but the newest messages holding this many are dropped, at once
COMPACT_AT = 160_000  # input tokens above which the oldest rounds are replaced by a handover note
KEEP_ROUNDS = 2
PAPER = (246, 242, 233)

REVIEW_CODE = {
    "edit": r"""
_r = load({file!r})
if _r.shape != orig.shape:
    print(f"The file is {{_r.shape[1]}}x{{_r.shape[0]}}; the original is {{W}}x{{H}}.")
    show(_r, "the result, whole")
else:
    _c = changes(orig, _r)
    edit_report(orig, _r)
    show(_r, "the result, whole")
    for _i, (_x0, _y0, _x1, _y1) in enumerate([a for a in _c["areas"] if a][:3]):
        _pad = max(24, int(0.25 * max(_x1 - _x0, _y1 - _y0)))
        compare(orig, _r, f"changed area {{_i + 1}}: original | result", box=(max(0, _x0 - _pad), max(0, _y0 - _pad), min(W, _x1 + _pad), min(H, _y1 + _pad)))
    if _c["box"]:
        show(change_map(orig, _r), "where pixels changed (black = untouched)")
import json as _json
_json.dump({{"size_ok": bool(_r.shape == orig.shape), "changed": float(changes(orig, _r, quiet=True)["changed"]) if _r.shape == orig.shape else None}}, open("gate.json", "w"))
""",
    "draw": r"""
_r = load({file!r})
_d = drawing_report(_r)
import json as _json
_json.dump({{"size_ok": bool(_r.shape == orig.shape), "flat": _d["flat"], "bare": _d["bare"], "range": _d["lightest"] - _d["darkest"]}}, open("gate.json", "w"))
show(_r, "the picture, whole")
show(tones(_r, 3), "tones only (three steps of light and dark)")
show(mirror(_r), "mirrored")
_h, _w = _r.shape[:2]
_e = np.abs(cv2.Laplacian(cv2.GaussianBlur(_r.mean(-1), (0, 0), 2), cv2.CV_32F))
_s = cv2.blur(_e, (max(8, _w // 3), max(8, _h // 3)))
_cy, _cx = np.unravel_index(int(_s.argmax()), _s.shape)
show(_r, "the busiest part, close up", box=(max(0, _cx - _w // 5), max(0, _cy - _h // 5), min(_w, _cx + _w // 5), min(_h, _cy + _h // 5)))
_qy, _qx = np.unravel_index(int(_s[_h // 6:_h - _h // 6, _w // 6:_w - _w // 6].argmin()), (_h - 2 * (_h // 6), _w - 2 * (_w // 6)))
_qy, _qx = _qy + _h // 6, _qx + _w // 6
show(_r, "the quietest part, close up", box=(max(0, _qx - _w // 6), max(0, _qy - _h // 6), min(_w, _qx + _w // 6), min(_h, _qy + _h // 6)))
if refs and refs[0].level != "look":
    refs[0].check(_r)
""",
}

SELF_REVIEW = """Before this is delivered: above is the result with check views and measurements.

Look at it as if someone else made it and you have to find what is wrong with it. Go through the manual's lists. Then check the request once more, word by word: is every part of it done?

If you find anything, fix it, save, and call finish again. If it truly holds up, call finish again now to deliver."""


def prepare(src: str, dest: Path, max_side: int) -> tuple[int, int, bool]:
    im = ImageOps.exif_transpose(Image.open(src))
    icc = im.info.get("icc_profile")
    im = im.convert("RGB")
    shrunk = False
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        shrunk = True
    im.save(dest, **({"icc_profile": icc} if icc else {}))
    return im.size[0], im.size[1], shrunk


def set_up(out: Path, mode: str, image: str | None, size: tuple[int, int], refs: list[str], ref_level: str, max_side: int) -> tuple[int, int, list[str], list[str]]:
    """Lay out a run's folder: the picture (or the blank sheet) as work/input.png, and the references.
    Returns (width, height, notes for whoever does the work, the references' file names)."""
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    notes = []
    if image:
        w, h, shrunk = prepare(image, work / "input.png", max_side)
        if shrunk:
            notes.append(f"The picture was reduced to {w}x{h} to fit --max-side.")
    else:
        w, h = size
        Image.new("RGB", size, PAPER).save(work / "input.png")
    shutil.copy(work / "input.png", out / "original.png")
    # References live in a folder of their own; whether their pixels may be used is the run's reference level.
    (work / ".refs").mkdir(exist_ok=True)
    names = []
    for i, r in enumerate(refs, 1):
        name = f"ref-{i}.png"
        prepare(r, work / ".refs" / name, max_side)
        if ref_level == "sample":
            shutil.copy(work / ".refs" / name, work / name)
        names.append(name)
    return w, h, notes, names


class Run:
    def __init__(self, *, mode: str, model: OpenRouter, critic_model: OpenRouter | None, image: str | None, size: tuple[int, int], request: str, refs: list[str], ref_level: str, out: Path, rounds: int, effort: str, max_tokens: int, max_usd: float, coords: str, sam: bool, max_side: int, seconds: int, review: bool, critic_passes: int):
        self.mode, self.model, self.critic_model, self.request = mode, model, critic_model, request
        self.has_plate = image is not None
        self.kind = "edit" if self.has_plate else "draw"   # mechanical label: is there a plate to preserve?
        self.rounds, self.effort, self.max_tokens, self.max_usd = rounds, effort, max_tokens, max_usd
        self.review, self.critic_passes, self.ref_level = review, critic_passes, ref_level
        self.dir = out
        self.work = out / "work"
        (self.work / "llm").mkdir(parents=True, exist_ok=True)
        self.w, self.h, self.notes, self.refs = set_up(out, mode, image, size, refs, ref_level, max_side)
        self.session = Session(self.work, coords=coords, sam=sam, timeout=seconds, ref_level=ref_level)
        self.system = manual.build(coords=coords, rounds=rounds, sam=sam, seconds=seconds, max_views=6, mode=mode, ref_level=ref_level, refs=len(refs), reviews=critic_passes if critic_model else 1)
        (self.work / "system.md").write_text(self.system)
        self.messages: list[dict] = [{"role": "system", "content": self.system}]
        self.round_starts: list[int] = []
        self.rows: list[dict] = []
        self.stages: list[dict] = []
        self.spent = 0.0
        self.last_input = 0
        self.reviews: list[dict] = []
        self.last_reviewed = ""  # digest of the file the latest review looked at
        self.process: dict = {}  # the session's record of method: tools used, brief, plan, stages
        self.free_holds = 0
        self.accepted = ""
        self.passed: bool | None = None
        self.last_call = False
        self.delivered: str | None = None
        self.unsaved = False
        self.summary = ""
        self.stopped = ""

    # -- messages ----------------------------------------------------------------------------------------------

    def _views_parts(self, views: list[dict]) -> list[dict]:
        parts = []
        for v in views:
            parts.append({"type": "text", "text": f"{v['id']}: {v['caption'] or 'view'} ({v['note']})"})
            parts.append({"type": "image", "file": str(self.work / v["file"])})
        return parts

    def _trim_pictures(self) -> None:
        """Rarely, and all at once: drop the views of older messages, keeping the request's and the newest."""
        holders = [i for i, m in enumerate(self.messages) if m["role"] == "user" and isinstance(m["content"], list) and any(p["type"] == "image" for p in m["content"])]
        count = lambda i: sum(p["type"] == "image" for p in self.messages[i]["content"])
        if sum(count(i) for i in holders[1:]) <= MAX_PICTURES:
            return
        kept = 0
        for i in reversed(holders[1:]):
            if kept < KEEP_PICTURES:
                kept += count(i)
                continue
            m = self.messages[i]
            m["content"] = [{"type": "text", "text": "(The pictures of this step were removed to save space; show them again if you need them.)"}] + [p for p in m["content"] if p["type"] == "text"]

    def _compact(self) -> None:
        """Replace the oldest rounds with a handover note the model writes for itself."""
        if self.last_input < COMPACT_AT or len(self.round_starts) <= KEEP_ROUNDS + 2:
            return
        ask = {"role": "user", "content": [{"type": "text", "text": "The conversation is getting long and its older part is about to be removed. Write a handover note for yourself, in plain text, with no tool call: the request as you understand it, your plan and which stages are finished, what each of your important variables and layers holds, what you found does and does not work on this picture, and exactly what remains to be done."}]}
        try:
            reply = self.model.send(self.messages + [ask], [], "low", 6000)
        except ModelError:
            return
        self.spent += reply.usage.get("usd") or 0
        cut = self.round_starts[-KEEP_ROUNDS]
        note = {"role": "user", "content": [{"type": "text", "text": "The earlier rounds of this job were removed to save space. Your own handover note:\n\n" + reply.text.strip() + "\n\nVariables now in your session:\n" + self.session.inventory() + "\n\nThe latest rounds follow."}]}
        dropped = len(self.messages) - cut
        self.messages = self.messages[:2] + [note] + self.messages[cut:]
        self.round_starts = [3 + (i - cut) for i in self.round_starts[-KEEP_ROUNDS:]]
        print(f"    the conversation was compacted ({self.last_input} input tokens; kept the last {KEEP_ROUNDS} rounds, {dropped} messages)")
        self.last_input = 0

    def _result_text(self, r: dict) -> str:
        out = []
        if r.get("stdout", "").strip():
            out.append(r["stdout"].rstrip())
        if r.get("error"):
            out.append("Error:\n" + r["error"])
        if r.get("dropped_views"):
            out.append(f"({r['dropped_views']} more views were made but not shown: at most 6 per call.)")
        if not out:
            out.append("(no output)")
        out.append(f"[{r.get('seconds', 0):.1f} s]")
        return "\n".join(out)

    # -- the loop ----------------------------------------------------------------------------------------------

    def go(self) -> Path | None:
        print(f"{self.mode} · {self.model.id} · effort {self.effort} · {self.w}x{self.h} · up to {self.rounds} rounds · cap ${self.max_usd}\n{self.dir}\n")
        intro = [{"type": "text", "text": f"# The request\n\n{self.request}"}]
        first = self.session.run("show(orig, 'the plate')", private=True)
        intro += self._views_parts(first["views"])
        intro.append({"type": "text", "text": f"Your plate is {self.w}x{self.h}; it is `orig` and `input.png` in your session" + (" — the picture that came with the request." if self.has_plate else " — a blank sheet.") + (" " + " ".join(self.notes) if self.notes else "")})
        for i, name in enumerate(self.refs):
            r = self.session.run(f"refs[{i}].show('reference refs[{i}]')", private=True)
            intro.append({"type": "text", "text": f"A picture that came with the request, `refs[{i}]` in your session" + (f" (also the file `{name}`)" if self.ref_level == "sample" else "") + ":"})
            intro += self._views_parts(r["views"])
        self.messages.append({"role": "user", "content": intro})
        self.write_pages()

        try:
            for rnd in range(1, self.rounds + 1):
                if self.spent >= self.max_usd:
                    if self.last_call:
                        self.stopped = f"the budget ran out: ${self.spent:.2f} spent, cap ${self.max_usd}"
                        break
                    self.last_call = True  # one more round, told to save and finish, so the money buys a result
                if self._round(rnd):
                    break
            else:
                self.stopped = self.stopped or f"the {self.rounds} rounds ran out"
            self._review_for_the_record()
        except ModelError as e:
            self.stopped = f"stopped by a model error: {e}"
        except KeyboardInterrupt:
            self.stopped = "stopped by you (Ctrl+C)"
        finally:
            self.session.close()

        final = None
        src = self.delivered or next((str(p) for p in (self.work / "result.png", self.work / "last_shown.png") if p.exists()), None)
        if src and Path(src).exists():
            self.unsaved = Path(src).name == "last_shown.png"
            final = self.dir / "result.png"
            shutil.copy(src, final)
        if self.reviews and final and self.last_reviewed == self._digest(final):
            self.passed = self.reviews[-1].get("verdict") == "deliver"
        self.write_pages(done=True, final=final)
        if final and self.passed is not None:
            majors = [f for f in self.reviews[-1].get("faults", []) if f.get("severity") == "major"]
            print("\nPASSED the review." if self.passed else f"\nNOT PASSED: the reviewer of the delivered picture still lists {len(self.reviews[-1].get('faults', []))} faults ({len(majors)} major):\n" + "\n".join(f"  - {f.get('where', '')}: {f.get('what', '')}" for f in majors[:6]))
        elif final:
            print("\nNot reviewed: there is no independent verdict on this picture.")
        if self.stopped:
            print(f"\n{self.stopped}")
        print(f"\n{len(self.rows)} rounds, ${self.spent:.2f}")
        if final:
            print(f"\nResult: {final}" + ("  (never saved by the model: this is the last whole picture it looked at)" if self.unsaved else ""))
        else:
            print("\nNo picture was made; see " + str(self.dir / "index.html"))
        return final

    @staticmethod
    def _digest(path) -> str:
        return gate.digest(path) if path else ""

    def _review_for_the_record(self) -> None:
        """Whatever is about to be delivered gets a review of exactly that version. It holds nothing back: it is
        there so that whoever reads the run knows what the reviewer made of the picture they are given."""
        if not (self.review and self.critic_model):
            return
        src = self.delivered or next((str(p) for p in (self.work / "result.png", self.work / "last_shown.png") if p.exists()), None)
        if not src or self._digest(src) == self.last_reviewed:
            return
        views, text, _ = self._review(str(Path(src).resolve().relative_to(self.work.resolve())), final=True)
        self.rows.append({"round": len(self.rows) + 1, "text": "", "steps": [{"kind": "review", "views": [v["file"] for v in views], "note": text}], "usage": {}, "seconds": 0})

    def _review(self, file: str, final: bool = False, sheet: dict | None = None) -> tuple[list[dict], str, bool]:
        """Views and text for a held-back finish, and whether the reviewer would let the result go."""
        self.last_reviewed = self._digest(self.work / file)
        r = sheet or self.session.run(REVIEW_CODE[self.kind].format(file=file), private=True)
        numbers = r["stdout"].strip()
        if self.critic_model is None:
            return r["views"], (numbers + "\n\n" if numbers else "") + SELF_REVIEW, False
        pictures = []
        if self.kind == "edit":
            pictures.append(("The original", str(self.dir / "original.png")))
        pictures += [(f"Reference {i + 1}", str(self.work / ".refs" / n)) for i, n in enumerate(self.refs)]
        pictures += [(v["caption"], str(self.work / v["file"])) for v in r["views"]]
        try:
            rev, usage = critic.review(self.critic_model, mode=self.kind, request=self.request, pictures=pictures, numbers=numbers)
        except ModelError as e:
            return r["views"], (numbers + "\n\n" if numbers else "") + f"(The reviewer could not be reached: {e})\n\n" + SELF_REVIEW, False
        self.spent += usage.get("usd") or 0
        self.reviews.append(rev)
        ok = rev["verdict"] == "deliver"
        print(f"    review: {rev['verdict']}, {len(rev['faults'])} faults · ${usage.get('usd') or 0:.3f}")
        text = (numbers + "\n\n" if numbers else "") + "An independent reviewer, who saw only the request, the starting point and this result, reports:\n\n" + critic.as_text(rev)
        if final:
            text = "For the record, a review of the picture as it was delivered (made after the work ended; nothing was changed in answer to it).\n\n" + text
        else:
            text += "\n\nDelivered." if ok else "\n\nNot delivered yet. Look at each reported place yourself, close up; fix what is really there (say so if a report is mistaken), save, and call finish again."
        return r["views"], text, ok

    def _round(self, rnd: int) -> bool:
        n = f"{rnd:02d}"
        tools = manual.TOOLS
        self._compact()
        self.round_starts.append(len(self.messages))
        (self.work / "llm" / f"request-{n}.json").write_text(json.dumps(self.model.body(self.messages, tools, self.effort, self.max_tokens, with_images=False), indent=1))
        print(f"round {rnd}: asking the model")
        reply = self.model.send(self.messages, tools, self.effort, self.max_tokens, lambda s, r, t, a: print(f"    {s:.0f}s  thinking {r} chars, text {t}, code {a}"))
        (self.work / "llm" / f"response-{n}.json").write_text(json.dumps({"text": reply.text, "reasoning": reply.reasoning, "tool_calls": reply.tool_calls, "finish": reply.finish, "provider": reply.provider, "usage": reply.usage, "seconds": reply.seconds}, indent=1))
        self.spent += reply.usage.get("usd") or 0
        u = reply.usage
        self.last_input = u.get("input") or 0
        print(f"    {reply.seconds:.0f}s · {u.get('output')} out ({u.get('reasoning')} thinking) · {u.get('input')} in ({u.get('cached')} cached) · ${u.get('usd') or 0:.3f} · {reply.provider}")
        self.messages.append(reply.assistant)

        left = self.rounds - rnd
        final_round = left == 0 or self.last_call
        views, steps, extra, done = [], [], [], False
        for call in reply.tool_calls:
            if call["args"] is None:
                self.messages.append({"role": "tool", "call_id": call["id"], "content": f"Error: {call['error']}. Nothing was run; send less in one call."})
                steps.append({"kind": "broken"})
                continue
            if call["name"] == "python":
                code = str(call["args"].get("code", ""))
                r = self.session.run(code)
                if r.get("calls") is not None:
                    self.process = {k: r.get(k) for k in ("calls", "brief", "plan", "stages")}
                self.messages.append({"role": "tool", "call_id": call["id"], "content": self._result_text(r)})
                views += r["views"]
                if r.get("stages") is not None and len(r["stages"]) >= len(self.stages):
                    self.stages = r["stages"]
                steps.append({"kind": "python", "code": code, "out": self._result_text(r), "views": [v["file"] for v in r["views"]], "error": bool(r.get("error"))})
                print(f"    python: {len(code.splitlines())} lines, {len(r['views'])} views" + (", ERROR" if r.get("error") else "") + f", {r.get('seconds', 0):.1f}s")
            elif call["name"] == "finish":
                file = str(call["args"].get("file") or "result.png")
                path = (self.work / file).resolve()
                if not str(path).startswith(str(self.work.resolve())) or not path.exists():
                    self.messages.append({"role": "tool", "call_id": call["id"], "content": f"There is no file {file!r} in your folder. Save it first: save(img, 'result.png')."})
                    steps.append({"kind": "finish", "note": "no such file"})
                    continue
                passes = self.critic_passes if self.critic_model else 1
                accept = str(call["args"].get("accept") or "").strip()
                sheet = None
                if not final_round and not accept and self.free_holds < 4:  # the free checks run even with the reviewer off
                    # The free checks come first: how the work was done, then what a ruler can tell.
                    resub = bool(self.reviews) and self.reviews[-1].get("verdict") == "fix" and gate.digest(path) == self.last_reviewed
                    faults, kind = gate.process(self.kind, {**self.process, "resubmitted": resub}), "process"
                    if not faults:
                        sheet = self.session.run(REVIEW_CODE[self.kind].format(file=file), private=True)
                        try:
                            faults, kind = gate.measured(self.kind, json.loads((self.work / "gate.json").read_text()), self.request), "measured"
                        except (OSError, ValueError):
                            faults = []
                    if faults:
                        self.free_holds += 1
                        note = gate.text(faults, kind)
                        self.messages.append({"role": "tool", "call_id": call["id"], "content": note})
                        if sheet:
                            views += sheet["views"]
                        steps.append({"kind": "check", "views": [v["file"] for v in (sheet or {}).get("views", [])], "note": note})
                        print(f"    finish: not accepted ({kind}: {len(faults)})")
                        continue
                if accept:
                    self.accepted = accept
                if self.review and len([s for row in self.rows for s in row["steps"] if s["kind"] == "review"]) + sum(s["kind"] == "review" for s in steps) < passes and not final_round:
                    rviews, text, ok = self._review(file, sheet=sheet)
                    if not ok:
                        self.messages.append({"role": "tool", "call_id": call["id"], "content": "Held back for review: see the pictures and the report that follow."})
                        views += rviews
                        extra.append(text)
                        steps.append({"kind": "review", "views": [v["file"] for v in rviews], "note": text})
                        print("    finish: held back for review")
                        continue
                    steps.append({"kind": "review", "views": [v["file"] for v in rviews], "note": text})
                self.delivered = str(path)
                self.summary = str(call["args"].get("summary") or "")
                self.messages.append({"role": "tool", "call_id": call["id"], "content": "Delivered."})
                steps.append({"kind": "finish", "note": self.summary})
                print(f"    finish: {file}")
                done = True
            else:
                self.messages.append({"role": "tool", "call_id": call["id"], "content": f"There is no tool {call['name']!r}; use python or finish."})

        if not done:
            status = f"Round {rnd} of {self.rounds} done; {left} left. ${self.spent:.2f} of ${self.max_usd:.2f} spent."
            if self.spent >= self.max_usd and not final_round:
                status += " The budget is spent: your next reply is the last. Save the best version as result.png and call finish in that reply."
            elif left <= 2 or self.spent >= 0.85 * self.max_usd:
                status += " Wrap up: save your best version as result.png now and call finish; whatever is saved last is what gets delivered."
            if not reply.tool_calls:
                status = "You did not call a tool. Use python to work, or finish to deliver. " + status
            parts = self._views_parts(views) + [{"type": "text", "text": t} for t in extra]
            self.messages.append({"role": "user", "content": parts + [{"type": "text", "text": status}]})
            self._trim_pictures()
        self.rows.append({"round": rnd, "text": reply.text.strip(), "steps": steps, "usage": reply.usage, "seconds": reply.seconds})
        self.write_pages()
        return done

    # -- the run's pages ---------------------------------------------------------------------------------------

    def write_pages(self, done: bool = False, final: Path | None = None) -> None:
        pages.write(pages.Record(dir=self.dir, mode=self.kind, request=self.request, model_id=self.model.id, w=self.w, h=self.h, rounds=self.rounds, effort=self.effort, spent=self.spent, rows=self.rows, stages=self.stages, reviews=self.reviews, stopped=self.stopped, summary=self.summary, delivered=self.delivered, unsaved=self.unsaved, passed=self.passed, accepted=self.accepted), done, final)
