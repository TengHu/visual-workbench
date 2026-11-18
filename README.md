# Visual workbench

A workbench for editing a picture, or drawing one on a blank sheet, by writing Python that runs in a sandboxed session: a kit of picture tools, views to look at, measurements, stages, and a review before delivery. No image-generation model is involved: every pixel comes from code and from the engines it directs.

There are two ways to use it, with the same kit, manual, session and review:

- **Direct a model**: `vw photo.jpg "REQUEST"`. A model called by `vw` (Gemini 4 Argon by default, or any OpenRouter model such as Claude Opus) does the work in `vw`'s own loop.
- **Hands-on**: `vw open photo.jpg "REQUEST"`, then `vw py` and `vw finish`. An agent with a shell (Claude Code, Antigravity, Codex, ...) writes the code itself.

## Quick start: give your coding agent `SKILL.md`

[`SKILL.md`](SKILL.md) is everything an agent needs, in one file: how to find and check `vw`, which way to work, how to pass on the request, the steps and rules that decide quality, how to deliver, and how to report. Each job adds its own `MANUAL.md` with the craft and the kit.

- **Claude Code:** install it once as a skill, and it is used whenever a picture is asked for:
  ```
  ln -s "$PWD" ~/.claude/skills/visual-workbench      # run inside this folder
  ```
- **Any other coding agent** (Codex, Gemini CLI, Antigravity, Cursor): tell it, or put in its `AGENTS.md` / `GEMINI.md`:
  > Read /full/path/to/visual-workbench/SKILL.md and follow it.

Then ask for a picture in plain words: "a fox in snow", "make the car on the right dark green", "finish this sketch as a watercolour" (with the file).

Once, before the first job: `./vw setup` (about 1.6 GB, needs [uv](https://docs.astral.sh/uv/)), and an OpenRouter key in the repo's `.env` for the reviewer.

## One command: `vw`

Everything goes through `vw`. Give it a picture and it edits that picture; give it none and it draws on a blank sheet.

```
vw photo.jpg "make the car dark green"           # edit
vw "a heron in reeds at dawn, pen and ink"        # draw
vw open photo.jpg "make the car dark green"      # hands-on: you write the code; then vw py RUN, vw finish RUN
vw runs                                           # the latest runs and where to open them
vw learn RUN...                                   # one HTML page that teaches how those runs were done
vw --help                                         # every option, with examples
```

Once, to set up (needs [uv](https://docs.astral.sh/uv/): `brew install uv`; about 1.6 GB with SAM and the landmark models):

```
git clone https://github.com/TengHu/visual-workbench && cd visual-workbench
./vw setup
ln -s "$PWD/vw" ~/.local/bin/vw      # optional: run `vw` from any folder (any folder on your PATH will do)
```

The OpenRouter key is read from `GATEWAY_API_KEY` (or `OPENROUTER_API_KEY`) in a `.env` in this folder (or any folder above it), and `TARGET_MODEL=google/gemini-4-argon` there is what `argon` means.

More examples:

```
# edit: the picture can come before or after the request
vw photo.jpg "remove the man on the left and his shadow"
vw photo.jpg "replace the mug with a small cactus in a terracotta pot" --rounds 30 --max-usd 4
vw room.jpg "put the lamp from the other picture on the table" --ref lamp.png

# draw; a --ref is something to work from, not the picture to change
vw "paint a study of this picture in oils" --ref photo.jpg                      # drawn from measurements
vw "paint a study of this picture in oils" --ref photo.jpg --reference sample   # painted from its pixels

# reviewer on (default) or off; another model; another reviewer
vw photo.jpg "make the car dark green" --reviewer off
vw photo.jpg "make the car dark green" --model anthropic/claude-opus-5.5
vw photo.jpg "make the car dark green" --critic anthropic/claude-opus-5.5

# the test set
vw eval run base
vw eval compare base no-critic
```

The run prints where its folder is; open `index.html` there to watch it round by round. The picture ends up in `result.png`. A typical edit takes 10 to 30 minutes and $0.50 to $2.50 with Argon.

## What a run leaves

Your request goes to the model word for word. Each run makes a folder under `runs/` (or `--out`) with:

- `result.png` (next to `original.png` for an edit)
- `index.html`: the run round by round: what the model said, the code it ran, every view it looked at, the reviews. Rewritten after every round; reload it while the run goes.
- `replay.html`: the picture stage by stage, with play and a slider, when the model worked in stages.
- `work/`: the session's folder (files the model saved, `snapshots/`, `stages/`, `views/`), `system.md` (the manual it was given), `llm/` (each request and reply).

| Option | What it does | Default |
|---|---|---|
| `--ref FILE` | a reference picture (repeatable) | none |
| `--reference` | what may be taken from references: `look` (see them only), `measure` (also shapes, landmarks, palette, light, coarse tones; never pixels), `sample` (also their pixels) | `sample` for edit, `measure` for draw |
| `--size WxH` | the blank sheet, when drawing | the first reference's shape at about 1.9 megapixels; 1600x1200 without one |
| `--model ID` | `argon` (`TARGET_MODEL` in the repo's `.env`) or any OpenRouter id | `argon` |
| `--reviewer on\|off` | `on`: every `finish` is checked by an independent reviewer before delivery; `off`: no reviewer model (the free delivery checks still run) | `on` |
| `--critic` | who reviews when the reviewer is on: `same` model in a fresh conversation, another model id, or `none` (the model checks itself once) | `same` |
| `--critic-passes N` | how many times a finish may be held back | 2 |
| `--effort` | `low`, `medium`, `high` thinking | `high` |
| `--rounds N` | replies the model gets | 24 edit, 36 draw |
| `--max-usd N` | budget; once spent, the model gets one last reply to save and finish | 5 |
| `--coords` | how the model writes positions: `yx1000` (Gemini's own `[y, x]`, 0..1000 of a view) or `xy` (pixels of a view) | `yx1000` for Gemini, `xy` otherwise |
| `--no-sam` | select objects with a colour graph cut instead of SAM 2.1 | SAM if installed |

The reference levels are a working rule the model is told and the kit enforces in its own functions; they are not a security barrier (the reference is in the session's memory).

## How it works

**Two tools.** `python(code)` runs code in a Python session that lives for the whole job, like a notebook. `finish(file)` delivers. The system prompt (`src/manual.py`, saved per run as `work/system.md`) explains the session, the method, the checks, and documents the kit; its reference part is generated from the kit's own docstrings.

**The model sees only what it asks to see.** `show(img, box=...)` sends a view; a zoomed box shows real pixels. Every view knows which part of the picture it covers, so a position read off any view (for Gemini in its native `[y, x]` on 0..1000) converts exactly to picture pixels.

**The kit (`src/kit/`)** does the precise parts a model cannot write well on the spot:

| Job | Functions |
|---|---|
| Selecting | `select_object` (SAM 2.1 from a box and points, or a colour graph cut), `select_color`, shape masks, `grow`, `feather`, `refine_edge` |
| Applying | `blend` (linear light; cannot leak outside its mask), `behind` (new content behind a soft-edged thing, edge rebuilt by alpha matting), `seamless` (Poisson), `match_color`, `grain_level`, `add_grain` |
| Removing | `fill` (our own PatchMatch with a chosen `source`), `fill_options` (several fills side by side to choose from), `clone` |
| Colour and tone | `recolor` (keeps shading and highlights), `adjust`, `curves` |
| Moving | `cut`, `place`, `warp` |
| Sculpting | `Sculpture` (a solid built from soft rounded parts and hard boxes, cylinders and cones, ray-marched with real light, cast shadows, creases and a ground shadow: the study for anything living or rounded), `quadruped` and `figure` (animals and people with sound proportions, as sculptures to adjust) |
| New content | `form` (turns a flat shape into a lit, rounded volume under a given or measured light), `curve_mask` (an organic outline through points), `render_html` (SVG, canvas, WebGL in headless Chrome), `canvas` (Skia), `shadow` |
| Strokes | `wash` (transparent watercolour: the paper is the light), `paint` (coarse-to-fine brushwork from a study: strokes wrap round the forms, stop at the study's edges, and are laid only where they bring the canvas closer to it; brush-load variation, bristle streaks and paint relief come from the engine; round, bristle, flat, pencil, dab; own flow fields; wet blending), `hatch` (pen and ink, denser where darker, following contours), `ink_lines` (tapered line art) |
| Layers and stages | `stack` (named layers, normal / multiply / screen), `plan`, `stage` (records and shows each stage) |
| Measuring | `outline` (exact contour), `landmarks` (body, face, hands; MediaPipe), `palette`, `shape_diff`, `landmark_diff`, `edge_overlay`; `refs[i]` with `show`, `mask`, `outline`, `landmarks`, `palette`, `light`, `tones`, `check` |
| Checking | `changes`, `change_map`, `edit_report` (grain, sharpness and border of the edit against its surroundings), `drawing_report` (bare paper, flat areas with no light and shade, range of tones), `tones`, `squint`, `mirror`, `compare` |

**Stages.** For anything drawn, the manual teaches the taught order (construction, placement, big shapes, light and dark, line or edge, colour, rendering, finish), each in its own layer, each closed with `stage()`, measured before the next. This is the process of the Ko_da reference video; the measuring tools are what keep proportions from drifting, and the stroke engine is the method of the Sol and raven references, pointed at the model's own study instead of a photo.

**Two ways in, one workbench.** The built-in loop (`src/agent.py`) drives a model through an API with the two tools. A hands-on job (`src/hands.py`) gives an outside agent the same things through the shell: `vw open` writes the same manual to `MANUAL.md`, `vw py` runs code in the same sandboxed session (kept alive between commands by `src/server.py`) and lists the views as files, and `vw finish` holds the result back for the same measurements and review. Both leave the same `index.html`, `replay.html` and `run.json`.

**Good practice is enforced, not only described.** The kit keeps a record of how the work was done (which tools were used, the brief, the plan, the stages). Before any review, `src/gate.py` checks that record and the measurable faults of the result, the same way for every model and agent, and holds the delivery back with a sentence saying what to do. An exception the request calls for is let through with a stated reason, which is kept. The delivered version is then reviewed and marked passed or not passed; `vw eval loop` tests this with a scripted model, for free.

**Delivery is reviewed by someone else.** On `finish` the harness builds close-ups and measurements and asks an independent reviewer (a fresh conversation that has seen none of the work) for faults and for whether each part of the request is done. The report goes back to the model; up to two passes.

**The run never ends empty.** The last whole picture the model looked at is kept and delivered if it never saved; the last rounds and the last 15% of the budget are announced as wrap-up; when the budget is spent the model gets one final reply.

**The conversation stays cheap.** Earlier messages are never rewritten round by round, so the provider's prompt cache holds (measured: about 80% of input cached from round 2). Old views are dropped rarely and in one go; a very long job is compacted once into a handover note the model writes for itself.

**The session is locked in.** It runs under the macOS sandbox (`src/sandbox.sb`): it can write only to the run's `work/` folder, cannot read anything else in your home folder, has **no network**, and starts with an empty environment, so no API key reaches it. Tested: listing the home folder, reading the `.env`, writing into `src/`, HTTPS and raw sockets all fail; SAM, MediaPipe, the fill and headless Chrome work.

## Testing changes

```
vw eval run base --dry                 # show the commands
vw eval run base                       # 16 tasks: recolour, remove, add, tone, restyle, draw from a few words or a description, study, finish a sketch
vw eval run no-critic
vw eval compare base no-critic         # a blind judge picks per task; writes eval/report.html
vw eval kit                            # free: the kit's key properties (edges, brushwork, form, drawing report)
vw eval loop                           # free: the delivery checks, built-in loop and hands-on, with a scripted model
```

The task pictures are not in the repo. Put your own in `eval/images/` under the names `eval/tasks.json` gives (`camp.jpg`, `cars.jpg`, `fjord.jpg`, `jellyfish.jpg`, `mug.jpg`, `puppy.jpg`, `sketch-mug.png`); `vw eval kit` and `vw eval loop` need none.

`eval/tasks.json` holds the tasks, `eval/configs.json` the configurations (each a list of flags). A full configuration costs roughly $15 to $30 at current Argon prices; use `--tasks` to run a few.

## Research behind it (2026-11-16)

- **Argon cannot output images**, so editing and drawing are code. Agent-editing papers (CV-Arena, IMAGAgent, RetouchLLM) agree that failures come from missing planning and independent checking, hence stages and the reviewer.
- **Models point well and outline badly.** Argon's boxes and points are near the top of detection benchmarks; SAM 2.1 turns them into exact masks. MediaPipe does the same for figures.
- **Views.** Measured on Argon through OpenRouter: every image costs 1,129 tokens whatever its size, and 11 px digits are legible on a 2048 px image. So zooming is cheap and is how detail is judged.
- **No existing skill or MCP server does this.** The ones found wrap image-generation APIs or remote-control an editor. Anthropic's `canvas-design` skill is a prompt for abstract posters; its two useful ideas (decide the look first, refine instead of adding) are in the manual.
- **Left out on purpose:** learned inpainting (LaMa, IOPaint) and image models, which generate pixels. Not done yet: a 3D route through Blender for lit objects and correctly proportioned figures, real pigment mixing (`paint`'s `wet` blending is an approximation), shape fitting with a differentiable rasteriser.

## Files

- `SKILL.md`: best practice for getting a good picture: the two ways to work, the rules that decide quality, wording the request, options per kind of job, judging the result, what to do when it falls short. Written as an agent skill, readable by a person.
- `vw`: the one command (edit, draw, `open` / `py` / `finish`, `runs`, `learn`, `eval`, `setup`). `setup.sh`, `requirements.txt`: the environment.
- `src/cli.py`: the options, and how `vw` tells an edit from a drawing. `src/agent.py`: the built-in loop. `src/gate.py`: the free delivery checks. `src/hands.py`, `src/server.py`: hands-on jobs and the server that keeps a job's session alive. `src/critic.py`: the reviewer and the judge. `src/manual.py`: the manual (the system prompt, and `MANUAL.md` of a hands-on job). `src/pages.py`: the pages a run leaves. `src/openrouter.py`: the model client.
- `src/learn.py`: `vw learn`, the page that explains finished runs (reasoning, code, views and reviews per round; a `NOTES.md` in a run folder becomes its reading guide).
- `src/session.py`, `src/worker.py`, `src/sandbox.sb`: the sandboxed Python session.
- `src/kit/`: the toolkit. A docstring there is what the model is told. `sculpt.py` and `figures.py` are the 3D route.
- `POSITIONING.md`: what a coding agent with `vw` is for next to diffusion and ComfyUI, and the research behind the design.
- `eval/`: the test set.
