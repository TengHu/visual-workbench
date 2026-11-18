# Coding agent + `vw`: what it is for, next to diffusion, text-to-image and ComfyUI

Written 2026-11-17, to revisit.

## The short answer

On generating new content from a prompt, diffusion models win, and by a wide margin. A coding agent with `vw` has a different strength: precise, checkable control over the pixels. The two complement each other.

## Where a coding agent with `vw` is the better tool

1. **Edits that leave everything else untouched.** A diffusion edit, even inpainting, regenerates the region and often shifts colour, texture or detail around it. With `vw` the edit runs through a mask, and `changes()` proves that every pixel outside it is identical. Retouching, product photos, documents and evidence-grade edits need that guarantee.
2. **Exactness.** It can hit an exact hex colour, exact geometry, correct text and spelling, correct counts ("five windows"), and charts or diagrams that are true to the data. Diffusion approximates all of these and often gets text, counts and geometry wrong.
3. **The work can be checked.** Every result comes with measurements (what changed, flatness, tonal range, how far it is from the reference), free delivery checks on how the work was done, and an independent review that marks it passed or not passed. It is checked before delivery.
4. **The process is the artifact.** The code, every stage and every view are kept. You can rerun it, change one step, apply the same treatment to 500 photos, or keep a series consistent across many pictures.
5. **Provenance.** No generative model's training data is in the picture, so every pixel can be traced to the original or to code. That matters for licensing, for newsrooms, and for honest labelling of edited photos.
6. **It fits where agents already work.** A coding agent with a shell can use it as part of a larger task (fix the screenshots in the docs, make the figures for a report) with no GPU, no node graph to build, and no service to send pictures to. The session is sandboxed and offline.
7. **It measures a model's visual intelligence.** This is what the workbench was first built for. Because nothing generates pixels, the result shows what the model itself understands about form, light, composition and editing. A diffusion model would hide that.

## Where it is not competitive

- Photorealistic new content: people, faces, rich natural textures, adding a convincing person to a photo. Our own runs show it: "add a person" came out mannequin-like.
- Speed and cost for one-off pretty pictures: diffusion takes seconds and costs cents, while a `vw` run takes 10 to 30 minutes and $1 to $3.
- Open-ended style and aesthetics at the level of a strong image model.

## The position

| Need | Better tool |
|---|---|
| "Make me a beautiful picture of X" | diffusion / ComfyUI |
| "Change exactly this, prove nothing else changed" | coding agent + `vw` |
| Diagrams, charts, text, precise layouts, series consistency, batch edits | coding agent + `vw` |
| Auditable or licensed or provenance-sensitive work | coding agent + `vw` |
| Testing what a multimodal model understands visually | coding agent + `vw` (diffusion can't) |

## An open option

Add a diffusion inpainting model as one more kit tool. The agent would still choose the mask, check the result and run the review, but the generator would draw what code draws badly, such as a person or fur. That would close the biggest quality gap.

It also breaks the current rule that nothing in the workbench generates pixels, so it would have to be a flag, off by default, and evaluations of models would leave it off. Undecided.

## Research: has this been done, and what does it teach? (2026-11-17)

Enabling a coding agent to make pictures is not new; making it *good* at organic subjects without a learned image prior is still open.

| Approach | Examples | What it shows |
|---|---|---|
| Agent paints stroke by stroke, looking after each stroke | [EASEL](https://arxiv.org/html/2608.25417v1), [Painter](https://arxiv.org/pdf/2308.08520), [agent-paint](https://github.com/Minebot17/agent-paint), [EaselAPI](https://github.com/Gemberkoekje/EaselAPI) | Of 25 models matching a reference, the best scores 0.535 (Gemini 3.1 Pro); a fitted data policy reaches 0.715. Models gain most of their similarity in the first 5% of steps and then stop improving: "the bottleneck is not initial perception but sustained, feedback-driven correction". |
| Agent sketches in strokes | [SketchAgent](https://openaccess.thecvf.com/content/CVPR2025/papers/Vinker_SketchAgent_Language-Driven_Sequential_Sketch_Generation_CVPR_2025_paper.pdf) | Recognisable, deliberately simple sketches. |
| LLM writes SVG | [StarVector](https://github.com/joanrod/star-vector), [SVGenius](https://github.com/ZJU-REAL/SVGenius) | Good for icons and diagrams; specialised models are strong in their niche and break outside it. |
| LLM lays out the structure, then the detail is optimised | [Chat2SVG](https://arxiv.org/html/2411.16602v2), [VectorFusion](https://openaccess.thecvf.com/content/CVPR2023/papers/Jain_VectorFusion_Text_to_SVG_by_Abstracting_Pixel_Based_Diffusion_Models_CVPR_2023_paper.pdf) | The best-looking code-based results. The LLM writes a template of basic shapes; a diffusion model gives a target picture; an optimiser fits the paths to it. Both stages are needed. |
| Fitting shapes to a target picture | [DiffVG, LIVE, CLIPDraw](https://www.github.com/ximinng/PyTorch-SVGRender) | Gradients make thousands of shape parameters converge on a target: detail is made by fitting, not placed by hand. |
| LLM writes Blender code | [SceneCraft](https://arxiv.org/abs/2403.01248), [BlenderGym](https://arxiv.org/html/2504.01786), [LL3M](https://www.alphaxiv.org/abs/2508.08228), [3DCodeBench](https://arxiv.org/html/2606.01057), [Procedura](https://arxiv.org/pdf/2608.26238) | Layout, light and perspective right, because the renderer computes them; the objects "toy-like, extremely low-poly and without fine detail". |
| Agent skill for code art | Anthropic's [algorithmic-art](https://claudemarketplaces.com/skills/anthropics/skills/algorithmic-art) | Generative abstract art, not depiction. |

**Why "it writes complex systems, so it can draw" does not carry over directly.** (1) The wrong level of control: a good picture is tens of thousands of continuous decisions, and code lets an agent place a dozen points at a time. Systems that reach detail fit it to a target instead of typing it in. (2) Correction from looking is weak (EASEL): agents look, improve once, then stall. (3) An artist does not draw from first principles either, but from thousands of hours of learned shapes.

**What `vw` took from this (no image generation):**
- *A 3D route:* `Sculpture`, a small ray-marched modeller in the kit (soft rounded parts, real light, cast and contact shadows, ground shadow), rendered as the study. Form, outline and light come out right by construction.
- *Shape knowledge as code:* `quadruped()` (fox, dog, wolf, cat, horse, deer) and `figure()` (7.5 heads, poses) build sculptures with sound proportions for the agent to adjust.
- *The optimiser:* `paint()` and `wash()` already fit strokes to a target; the 3D render is that target. A separate vector optimiser would duplicate it, so it was left out.
- *Correction enforced:* the free delivery checks and the review, and a picture sent back by the reviewer is not accepted again unchanged.

The remaining ceiling: photoreal organic detail (fur, faces, foliage) still needs either more of the above or the opt-in learned prior described in "An open option".
