---
name: visual-workbench
description: Use whenever someone wants a picture edited (recolour, remove, add, replace, retouch, change the light) or drawn or painted (from a few words, a description, a sketch or a reference photo), and no image-generation model is to be used. You make the picture by writing code in the `vw` workbench, which gives you the tools, a sandboxed session, views to look at, and checks and a review before delivery. Also covers directing another model to do it.
---

# Making pictures with `vw`

`vw` is a workbench for making pictures by writing code. It gives you a kit of picture tools, a Python session that keeps its state, views to look at, measurements, and checks and an independent review before anything is delivered. No image-generation model is involved: every pixel comes from code run in the workbench.

Follow this file from top to bottom. Each job also gets its own `MANUAL.md` with the craft and every function of the kit. This file tells you how to run the job and what decides its quality.

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

Without a key, work with `--critic none`: you then review your own work against a checklist. Tell the person the result was not reviewed independently.

## 2. Choose how to work

- **You do it (hands-on): the normal way.** Use this when you can run shell commands and open picture files to look at them. It costs your own time plus about one cent per review.
- **Direct a model.** `vw PICTURE "REQUEST"` has a model called by `vw` do the work (Gemini 4 Argon by default, or `--model ID`). Use it when you cannot look at pictures yourself, or the person asks for it. It takes 10 to 30 minutes and $0.50 to $2.50; see section 8.

**Never import the kit into a script of your own outside `vw`, and never run the job without `vw open` / `vw py` / `vw finish`.** That skips the manual, the views and the review. Code of your own *inside* the session is fine and often needed: your own helpers, geometry, renderers or loops, run through `vw py`. It is how the worst picture this workbench has produced was made: flat tube arms, speckle, empty bands filled with a pattern.

## 3. Pass on the request as it was given

Put the person's words in the request exactly as they gave them. A few words stay a few words: inside the job you decide what they leave open, in a `brief`. Pass any picture they gave with `--ref`. The reviewer checks the result against the request part by part.

If you write the request yourself, from a longer conversation:
- Name things by what is visible: "the brown car on the right".
- Say what must stay: "Keep everything else as it is."
- Describe new content concretely: material, colour, style, where it stands.
- Name the medium for a drawing: pen and ink, gouache, oils, watercolour.
- Give one job per run. Do two changes as two runs, the second on the first run's `result.png`.

## 4. Open the job

```
$VW open photo.jpg "Make the brown car on the right a deep racing green. Keep everything else as it is."   # an edit
$VW open "a fox in snow"                                                    # a drawing from a few words
$VW open "Paint a study of this picture in oils." --ref photo.jpg           # from a reference photo
$VW open "Finish this sketch as a watercolour." --ref sketch.png            # from a sketch
```

Options:
- `--reference sample` lets you use a reference's pixels. That gives the most faithful study of a photo, so use it when the person wants a good painting of their photo. The default for a drawing is `measure`: shapes, landmarks, palette and light only, never pixels.
- `--size 1200x1600` sets the sheet for a drawing. The default is the reference's shape, or 1600x1200 (landscape) without one: choose an upright sheet for an upright subject.
- `--coords yx1000` if you are a Gemini model.
- `--critic ID` picks the reviewer (default `argon`, the repo's `TARGET_MODEL`); `--critic none` when there is no key.
- `--agent NAME` records who did the work.

`vw open` prints the job's folder (`RUN` below), the first views, and `RUN/MANUAL.md`.

**Read `RUN/MANUAL.md` from beginning to end before you write any code.**

## 5. Do the work in steps, looking after each

```
$VW py RUN step1.py            # a file of Python, run in the job's session (keep step files in RUN)
$VW py RUN - <<'EOF'           # or the code inline
show(orig, "the picture")
EOF
```

The session keeps its variables between steps, like a notebook. Each step prints its output and lists the views it made as picture files. **Open every listed view and look at it before writing the next step.** A job takes 15 to 30 steps, about half of them looking.

These rules decide quality. `finish` enforces the ones that can be measured:

0. **Decide the picture first.** For a drawing, start with `brief("...")`: subject, composition, light, palette and medium, in a few sentences. For a few words, choose as a careful illustrator would: a clear view of the subject, large on the sheet, one light, a quiet background. For a long request, list everything it asks for. For a sketch, keep its composition and drawing (`ref.lines()`) and add form, light and colour. For an edit, write a plan of the edit.
1. **Work in stages.** `plan([...])`, then one stage at a time, each closed with `stage(name, picture)`. Judge each stage before building on it.
2. **Measure instead of guessing.**
   - Shapes: `select_object`, `ref.mask`.
   - Positions: `landmarks`.
   - Light: `light_on`, `ref.light`.
   - Colour: `palette`.
   - An edit: `changes`, `edit_report`.
   - A drawing: `drawing_report`, `ref.check`, `tones`, `squint`, `mirror`.
3. **Form before brushwork.**
   - Anything living or rounded (an animal, a person, cloth, a cloud) is built in 3D first, then rendered in the picture's light. Start from `quadruped()` or `figure()`, which have correct proportions, or build a `Sculpture` of rounded parts.
   - Simpler things get their real outline (`curve_mask`) and volume (`form`).
   - Only then paint: `paint()` for oil or gouache, `wash()` for watercolour, `hatch()` for ink.
4. **Keep the study clean.** Never add noise or random dabs to look painted. The stroke engine supplies the brushwork.
5. **Fill the sheet.** Never leave empty bands, and never fill them with a border or pattern.
6. **In an edit, change only what was asked.** Work through a mask. Zoom into the edited area and along its border, compare with the original, and check that nothing outside moved.

## 6. Deliver through `finish`

```
$VW finish RUN --summary "what was done, and what could not be done well"
```

Before you call it, run the painter's checks on the final picture yourself (`drawing_report`, `tones`, `squint`, `mirror`, and `ref.check` with a reference) and act on what they show.

`finish` first runs free checks.
- For a drawing: a brief, a plan, three closed stages, and the painter's checks were used.
- For an edit: it was zoomed into and compared with the original.
- For any picture, measurable faults hold it back:
  - the wrong size;
  - an edit that changed nothing;
  - a picture mostly flat;
  - no real darks or lights;
  - a bare sheet.

If something is missing, it says what to do. Do it and finish again. If the request itself rules a check out (a flat poster was asked for), pass `--accept "reason"`; the reason is recorded.

Then the reviewer looks at the result.
- Open the check views it lists.
- Look at each reported fault yourself, close up.
- Fix what is really there, save with `save(img, "result.png")`, and finish again.

A picture sent back is not accepted again unchanged: change it before the next `finish` (that refusal does not use up a review). Two reviews can hold a delivery back. After that the picture is delivered, reviewed once more, and marked **PASSED** or **NOT PASSED**.

## 7. Report honestly

Give the person:
- the path of `RUN/result.png`;
- the verdict, PASSED or NOT PASSED;
- the faults the last review listed;
- anything else you can see wrong yourself.

Never call a NOT PASSED picture finished or perfect. `RUN/index.html` shows every step, view and review. `$VW learn RUN --out page.html` writes one page explaining how it was made, to send to someone.

**If it is not good enough, in this order:**
1. Do it again, which costs nothing but time.
2. Do a narrow second pass on the result, naming the fault: `$VW open RUN/result.png "Remove the thin yellow strip by her arm. Keep everything else as it is."`
3. For a study of a photo that came out badly drawn, use `--reference sample`.
4. Say what still fails instead of spending more.

**What to expect:**
- Recolouring, removing, retouching and changing the light can be hard to tell from a photograph.
- Structure, proportion, light and form are right when you follow the rules.
- Fine organic detail (fur, faces, foliage) and photoreal new people are the weakest part. Say so before starting such a job.

## 8. Directing a model instead

```
$VW photo.jpg "REQUEST" --out RUN > RUN.log 2>&1 &     # an edit; without a picture it draws
$VW "REQUEST" --ref photo.jpg --out RUN > RUN.log 2>&1 &
```

The same manual, checks and review apply. It is finished when the log's last line starts with `Result:`, and the log says PASSED or NOT PASSED. Then judge and report as in section 7.

- Replacing or adding something needs `--rounds 30 --max-usd 4 --max-tokens 60000`.
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
