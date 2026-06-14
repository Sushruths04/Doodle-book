---
title: DoodleBook
emoji: 📚
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
pinned: false
tags:
  - hackathon
  - build-small
  - adventure-in-thousand-token-wood
  - gradio
  - modal
  - flux
  - minicmp
  - voxcpm
  - storybook
  - coloring-book
---

# DoodleBook

Draw a character, upload it, and DoodleBook turns it into a narrated six-page picture book plus a matching printable coloring book.

The project was built for the Build Small Hackathon 2026. The core idea is to keep the reasoning stack small, use a strong image renderer only where it matters, and make the whole flow feel like a child-facing product instead of a model demo.

## What it does

- Takes a doodle photo from upload or webcam.
- Generates a six-page children's story with a consistent hero.
- Renders six full-color story pages with FLUX.
- Generates narration audio for the whole book.
- Exports a story PDF.
- Generates a matching black-and-white coloring book as a second output.

## Current architecture

There are two runtime modes in this repo.

### 1. Local Modal-backed app

Use [run_modal.py](run_modal.py) for the real end-to-end flow during development.

- UI: Gradio 5 custom Blocks layout
- Story: local generator by default, optional Modal MiniCPM route
- Images: Modal FLUX pipeline
- TTS: Modal VoxCPM pipeline
- PDFs: local export
- Coloring book: direct FLUX line-art render, with traced fallback

Start it with:

```bash
python run_modal.py
```

The default local URL is:

```text
http://127.0.0.1:7880
```

### 2. HF Spaces / ZeroGPU-oriented app

Use [app.py](app.py) or [app_zerogpu.py](app_zerogpu.py) depending on the target deployment mode.

- `app.py` contains the Gradio app logic and local GPU/ZeroGPU-style orchestration work.
- `app_zerogpu.py` is the simplified free-hosting path intended for Hugging Face ZeroGPU experiments.

## Stack used in the hackathon

This project deliberately mixes a small-model reasoning stack, a stronger dedicated image renderer, a custom Gradio presentation layer, and remote inference infrastructure that is cheap enough to demo but strong enough to feel like a real product.

The important distinction is:

- the app "brain" is small
- the renderer is specialized
- the UX is product-shaped, not notebook-shaped
- the deployment path is built around a Gradio Space front-end

### Full stack at a glance

| Layer | Stack | Role in the product |
|---|---|---|
| Product UI | Gradio 5 Blocks + custom CSS/HTML/JS | Child-facing scrapbook interface, status streaming, downloads |
| Story engine | MiniCPM5-1B + local structured fallback | Writes the six-page narrative and scene plan |
| Image engine | FLUX.2-klein-4B on Modal | Draws consistent full-color pages and dedicated coloring pages |
| Voice engine | VoxCPM2 on Modal | Narrates the full storybook |
| Coloring engine | Direct FLUX line-art pass + cleanup fallback | Produces printable black-and-white pages |
| Export layer | Pillow + FPDF | Builds printable story and coloring PDFs |
| Hosting target | Hugging Face Spaces | Gradio app shell and user entrypoint |
| Remote compute | Modal | GPU execution for heavy image and TTS work |
| Observability | Heartbeat streaming + stage timing in trace panel | Keeps long runs visible and debuggable |

### Frontend and product shell

- Gradio 5
- Custom scrapbook-style UI in [ui/layout.py](ui/layout.py)
- HTML-based book rendering in [book_builder.py](book_builder.py)
- Fixed-position PDF downloads under the status panel
- Streaming progress heartbeats to keep long jobs alive in the browser
- File-backed page rendering instead of giant inline base64 payloads

### Story generation stack

- `openbmb/MiniCPM5-1B`
- Local fast fallback story generator in [services/story.py](services/story.py)
- Optional Modal story worker in [modal_workers/modal_story_gen.py](modal_workers/modal_story_gen.py)

Why it matters:
- The story model is the small-model "brain" of the app.
- It keeps the narrative stack small and hackathon-aligned.
- The story system outputs both prose and scene prompts, so downstream image generation stays structured.
- The local structured fallback means the Space can still produce a valid book if the remote story path is unavailable.

### Image generation stack

- `black-forest-labs/FLUX.2-klein-4B`
- Modal deployment for image generation in [modal_workers/modal_image_gen.py](modal_workers/modal_image_gen.py)
- Parallel canonical-character plus per-page render flow in [services/images.py](services/images.py)
- One canonical character render from the child doodle, then scene-specific page renders
- Separate direct line-art render path for the coloring book

Why it matters:
- The app needs high visual quality and character consistency.
- FLUX is used as the renderer, not as the reasoning engine.
- The character consistency pipeline is what makes the book feel authored rather than randomly reimagined on every page.
- The line-art renderer is separate because tracing finished crayon pages produced bad coloring results.

### TTS stack

- `openbmb/VoxCPM2`
- Modal TTS worker in [modal_workers/modal_tts.py](modal_workers/modal_tts.py)
- Service wrapper in [services/tts.py](services/tts.py)
- Parallelized with image generation in the real Modal-backed app

Why it matters:
- Narration is part of the child-facing experience, not a side feature.
- TTS runs in parallel with image generation in the real local pipeline.
- Overlapping TTS with illustration time reduces total wait without degrading output quality.

### Coloring-book stack

- Direct FLUX line-art rendering for the same scenes
- Cleanup and fallback pipeline in [services/coloring.py](services/coloring.py)
- Modal `render_coloring_page` for dedicated line-art scene generation
- Local cleanup for thresholding, despeckling, and printable black-on-white output

Why it matters:
- The main bug fixed in this version was that the coloring book used to trace finished crayon-textured images.
- The improved pipeline renders dedicated line-art pages instead of trying to strip color out after the fact.
- This is the main quality improvement that separates the current version from the earlier broken coloring-book output.

### Infrastructure stack

- Modal for remote GPU inference
- Hugging Face Spaces as the target host
- Python 3.11 / 3.13 local development
- `diffusers`, `transformers`, `torch`, `accelerate`
- `Pillow`, `OpenCV`, `FPDF`
- Gradio client-compatible API surface for testing and debugging
- Hugging Face org deployment target: `build-small-hackathon`

### Sponsor and hackathon alignment

This app directly reflects the hackathon sponsor/tool stack:

- `OpenBMB`: MiniCPM5-1B and VoxCPM2
- `Black Forest Labs`: FLUX.2-klein-4B
- `Modal`: remote GPU inference
- `OpenAI Codex`: debugging, architecture fixes, deployment preparation, README/release work
- `Hugging Face Spaces`: final Gradio app surface

For hackathon judging, the main narrative is:

- Tiny Titan reasoning stack
- Off-brand custom UI
- Real multimodal product loop
- Remote GPU orchestration with a Gradio user experience
- Child-usable output artifacts: storybook PDF, audio, coloring book PDF

## Key engineering fixes in this version

- Added direct Modal coloring-page rendering with `render_coloring_page`.
- Fixed the live app to keep the Gradio stream alive during long coloring generation.
- Added stage timing so story, image, PDF, TTS, and coloring costs are visible.
- Reduced final-page payload size by replacing giant inline base64 book HTML with file-backed image URLs.
- Fixed download serving through Gradio temp-file paths.
- Removed port confusion between the local test app and the real Modal-backed app.

## Measured performance

Measured against the real local Modal-backed app flow:

- Story-only stage: about `0.3s`
- Full-color book, warm: about `75s to 80s`
- Full-color book + coloring book, warm: about `200s`
- Slowest stage: coloring-book generation

The current bottleneck is still the coloring-book path, even after the direct line-art fix.

## Repository layout

```text
app.py                  Main Gradio app variant
app_zerogpu.py          ZeroGPU-oriented app variant
run_modal.py            Real local Modal-backed app
book_builder.py         HTML and PDF assembly
services/               Orchestration and fallbacks
modal_workers/          Modal remote workers
ui/                     Custom Gradio layout
assets/                 Sample doodles and sample book pages
docs/                   Specs and notes
```

## Local setup

```bash
pip install -r requirements.txt
python run_modal.py
```

If you want the real Modal-backed app, use `run_modal.py`, not `app.py`.

## Hugging Face Space deployment target

The intended hosted version is a Gradio Space in the `build-small-hackathon` org.

Target format:

```text
build-small-hackathon/DoodleBook
```

Recommended runtime shape:

- Hugging Face Space SDK: `gradio`
- Space entrypoint: `app.py`
- Space frontend and API live on Hugging Face
- Story and image generation run in-Space
- TTS degrades gracefully if the voice stack is unavailable

Optional secrets only if you also keep the separate Modal-backed local/dev path:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- any Hugging Face token needed by Modal workers for model pulls

Why this deployment shape is preferred:

- keeps the user-facing app as a normal Gradio Space
- avoids overloading ZeroGPU with the entire image pipeline
- keeps the high-quality FLUX and VoxCPM paths intact
- makes the frontend easy to judge, share, and demo from the official org

## Hackathon fit

This project targets the hackathon stack in a deliberate way:

- Small-model reasoning for story generation
- Strong but scoped rendering model for visuals
- Distinct multimodal outputs: story, illustrations, narration, coloring book
- Real product UX instead of a bare prompt box
- Clear deployment story for Hugging Face Spaces plus Modal GPU workers

## Contributors

- Sushruth S.
- OpenAI Codex: debugging, architecture fixes, rendering pipeline fixes, README and release preparation

## License

Apache-2.0. See [LICENSE](LICENSE).
