---
name: visual-workbench
description: Use whenever someone wants a picture edited (recolour, remove, add, replace, retouch, change the light) or drawn or painted (from a few words, a description, a sketch or a reference photo), and no image-generation model is to be used. You make the picture by writing code in the `vw` workbench, which gives you the tools, a sandboxed session, views to look at, and checks and a review before delivery. Also covers directing another model to do it.
---

# Making pictures with `vw`

`vw` is a workbench for making pictures by writing code. It gives you a kit of picture tools, a Python session that keeps its state, views to look at, measurements, and checks and an independent review before anything is delivered. No image-generation model is involved: every pixel comes from code run in the workbench.

Follow this file from top to bottom. Each job also gets its own `MANUAL.md` with how the session and delivery work and every function of the kit. This file tells you how to run a job.

## 1. Find `vw` and check it is ready

`vw` is the command in the same folder as this file. Call it by its full path, written `vw` below. If you are reading a copy of this file, ask the person where `visual-workbench` is.

```
VW=/path/to/visual-workbench/vw                    # the folder that holds this SKILL.md
$VW --help >/dev/null && echo ready                # "not set up": run  $VW setup  once (about 1.6 GB, needs uv)
```

The reviewer is a model reached through OpenRouter, using the key in the repo's `.env` (`GATEWAY_API_KEY` or `OPENROUTER_API_KEY`). Check that it exists without printing it:

```
grep -qE '^(GATEWAY|OPENROUTER)_API_KEY=' "$(git -C "$(dirname "$VW")" rev-parse --show-toplevel)/.env" && echo key found
```

Without a key, work with `--critic none`: you then compare the result with the request yourself. Tell the person the result was not reviewed independently.

## 2. Choose how to work

- **You do it (hands-on): the normal way.** Use this when you can run shell commands and open picture files to look at them. It costs your own time plus about one cent per review.
- **Direct a model.** `vw PICTURE "REQUEST"` has a model called by `vw` do the work (Gemini 4 Argon by default, or `--model ID`). Use it when you cannot look at pictures yourself, or the person asks for it. It takes 10 to 30 minutes and $0.50 to $2.50; see section 8.

**Never import the kit into a script of your own outside `vw`, and never run the job without `vw open` / `vw py` / `vw finish`.** That skips the manual, the views and the review. Code of your own *inside* the session is fine: your own helpers, geometry, renderers or loops, run through `vw py`.

## 3. Pass on the request as it was given

Put the person's words in the request exactly as they gave them. A few words stay a few words. Pass the picture to work on as the first argument, and any other pictures with `--ref`. The reviewer checks the result against the request part by part, so the request is the only standard the result is held to.

If you write the request yourself, from a longer conversation, use the person's own words and add nothing they did not ask for. Naming things by what is visible ("the brown car on the right") lets the reviewer find them.

## 4. Open the job

```
$VW open photo.jpg "Make the brown car on the right a deep racing green."  # the plate is photo.jpg
$VW open "a fox in snow"                                                    # the plate is a blank sheet
$VW open "Paint a study of this picture in oils." --ref photo.jpg           # a blank sheet and a reference
```

Options:
- `--reference look|measure|sample` sets what the session may do with references: only look at them; also measure them (shapes, landmarks, palette, light, coarse tones); or also use their pixels. The default is `sample` with a plate picture and `measure` on a blank sheet.
- `--size 1200x1600` sets the blank sheet's size. The default is the reference's shape, or 1600x1200 without one.
- `--coords yx1000` if you are a Gemini model.
- `--critic ID` picks the reviewer (default `argon`, the repo's `TARGET_MODEL`); `--critic none` when there is no key.
- `--agent NAME` records who did the work.

`vw open` prints the job's folder (`RUN` below), the first views, and `RUN/MANUAL.md`.

**Read `RUN/MANUAL.md` from beginning to end before you write any code.**

## 5. Do the work

```
$VW py RUN step1.py            # a file of Python, run in the job's session (keep step files in RUN)
$VW py RUN - <<'EOF'           # or the code inline
show(orig, "the picture")
EOF
```

The session keeps its variables between steps, like a notebook. Each step prints its output and lists the views it made as picture files. You see a view only by opening its file.

## 6. Deliver through `finish`

```
$VW finish RUN --summary "what was done, and any part of the request that was not"
```

`finish` first runs free checks that need no judgement. They hold the delivery back when:
- the result is the same file a review already sent back, unchanged;
- it is not the size of the plate;
- no pixel differs from the plate.

When the request calls for one of these (for example a request to change the size), pass `--accept "reason"`; the reason is recorded.

Then the reviewer is shown the request, the plate and the result, and reports which parts of the request it judges not done and what faults it sees. When it reports any, the delivery is held back and the report is printed with check views to open; run `finish` again to have the result reviewed again. Two reviews can hold a delivery back. After that the picture is delivered, reviewed once more, and marked **PASSED** or **NOT PASSED**.

## 7. Report honestly

Give the person:
- the path of `RUN/result.png`;
- the verdict, PASSED or NOT PASSED;
- the faults the last review listed;
- anything else you can see wrong yourself.

Never call a NOT PASSED picture finished or perfect. `RUN/index.html` shows every step, view and review. `$VW learn RUN --out page.html` writes one page explaining how it was made, to send to someone.

**If the result does not do what was asked:**
1. Do it again, which costs nothing but time.
2. Do a narrow second pass on the result, naming the fault: `$VW open RUN/result.png "Remove the thin yellow strip by her arm."`
3. Say what still fails instead of spending more.

## 8. Directing a model instead

```
$VW photo.jpg "REQUEST" --out RUN > RUN.log 2>&1 &     # the plate is photo.jpg; without a picture, a blank sheet
$VW "REQUEST" --ref photo.jpg --out RUN > RUN.log 2>&1 &
```

The same manual, checks and review apply. It is finished when the log's last line starts with `Result:`, and the log says PASSED or NOT PASSED. Then judge and report as in section 7.

- `--rounds`, `--max-usd` and `--max-tokens` set how many replies the model gets, the spending cap and the output cap per reply.
- `--max-usd` is a hard cap. Tell the person the cap before you start.
- `--model ID` uses any OpenRouter model with tool calling and image input.
- `--critic ID` has another model review.
- `$VW runs` lists the latest runs.

## 9. Limits

- The session has no network and can touch only the job's folder. Pictures go only to the reviewer's provider, and with `--critic none` nowhere.
- Make no sexualised or undressed pictures of real people. Make nothing meant to pass as a genuine photo of a real person doing something they did not do. Label edited pictures of real people as edited.
- Do not change the code under `src/` to get one picture through. Report a fault in the kit with the run that shows it.

## Installing this skill (for the person)

- **Claude Code:** `ln -s /path/to/visual-workbench ~/.claude/skills/visual-workbench`. It is then used whenever a picture is asked for.
- **Any other coding agent** (Codex, Gemini CLI, Antigravity, Cursor): tell it "Read /path/to/visual-workbench/SKILL.md and follow it", or put that line in its `AGENTS.md` / `GEMINI.md`.
