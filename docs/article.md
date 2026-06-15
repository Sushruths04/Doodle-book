# DoodleBook: A Child's Crayon Drawing Becomes a Narrated Storybook

*Built for the Build Small Hackathon 2026 · Adventure in Thousand Token Wood*

---

Imagine a child scribbles a robot on a napkin. Wobbly arms, two big eyes, a lightning-bolt antenna. Forty-five seconds later they're holding a six-page illustrated storybook — their robot, in their style, starring in a real adventure — with a narrator's voice reading it aloud. That's DoodleBook.

> **Try it now →** [huggingface.co/spaces/build-small-hackathon/DoodleBook](https://huggingface.co/spaces/build-small-hackathon/DoodleBook)

---

## What DoodleBook Does

The flow is three taps:

1. **Draw something** — upload a photo of a drawing, or snap one with the webcam. Anything works: a cat, a dragon, a kid's self-portrait.
2. **Pick a theme** — brave adventure, making a new friend, overcoming a fear, and more.
3. **Tap "Make my book!"**

DoodleBook then:

- **Writes** a complete six-page story where the character in the drawing is the hero
- **Illustrates** every page in a crayon storybook style — keeping **the same character** across all six scenes
- **Narrates** the whole story aloud in a warm, child-friendly voice
- **Exports** a printable **Story PDF** and a matching **Coloring Book PDF** ready to print

The sample book loads the moment you open the Space, so you see the quality before waiting for GPU time.

---

## The "Tiny Titan" Philosophy

The hackathon theme is *Build Small* — and we took that seriously.

The *reasoning* behind DoodleBook — writing the story, planning the scenes, narrating the book — is done by a **~3 billion parameter model stack**:

| What it does | Model | Size |
|---|---|---|
| Writes the story + scene plan | MiniCPM5-1B | **1 B** |
| Reads the story aloud | VoxCPM2 | **2 B** |
| Illustrates the pages | FLUX.2-klein | 4 B |

FLUX is the *renderer*, not the brain. It prints what the small models decided. That's the Tiny Titan bet: small models drive the product experience; a strong image model just executes the instructions.

---

## The Hard Part: One Hero, Six Pages

The hardest engineering problem in AI storybooks is **character consistency**. Generate six pages independently and you get six different creatures — different colors, different faces, different proportions. The magic collapses.

We solved it without per-user fine-tuning (LoRA training is minutes-to-hours — impractical for a live demo). Instead, DoodleBook uses a **canonical-character pass**:

1. The child's doodle is run through FLUX **img2img once**, turning the rough drawing into a single clean "model-sheet" version of the character on a plain background.
2. Every one of the six story pages is then rendered with **that canonical image as the reference** — so the same creature appears in every scene.
3. A locked seed and a fixed character description provide a second identity anchor and make every book reproducible.

The child's drawing is the source of truth. A cat doodle becomes a cat hero. A robot becomes a robot. No training required.

For the full technical write-up: **[Field Notes: Cross-Page Character Consistency →](blog.md)**

---

## The Coloring Book: Redraw, Don't Trace

The matching coloring book was trickier than expected. Our first attempt traced the finished crayon pages using classical edge detection — but crayon texture shattered busy backgrounds into speckle. No child could color a page that looked like static.

The fix: hand each finished color page back to **FLUX as img2img** with a *"clean coloring-book line art"* prompt. The model **redraws** clean, colorable outlines that match the composition of the story page — instead of tracing texture, it understands the scene and produces shapes. A tiny local pass then thresholds the result to crisp black-on-white.

---

## Sponsors Who Made DoodleBook Possible

### OpenBMB — The Brain

**[OpenBMB](https://huggingface.co/openbmb)** builds the MiniCPM family of small language models and the VoxCPM voice models. Two of their models power the reasoning core of DoodleBook:

**[MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)** writes the story. Given the character name, a theme, and the child's hero description, MiniCPM5-1B produces a complete structured six-page story with per-page scene descriptions — in a single inference pass. At 1B parameters it is fast enough to run as the first stage in a live demo, and the quality of its structured JSON output is high enough that the story rarely needs a fallback.

**[VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)** narrates the book. It runs in parallel with the image generation stage, so the narration audio is typically ready before the illustrations finish — the "narration ready ▶" status appears mid-generation. VoxCPM2 supports expressive voice design, letting us give the narrator a warm, storytelling tone suited to young listeners.

Together, MiniCPM5-1B and VoxCPM2 are the **3B "brain"** of DoodleBook. They turn a drawing and a theme into a complete authored, performed story — before FLUX renders a single pixel.

---

### Black Forest Labs — The Illustrator

**[Black Forest Labs](https://huggingface.co/black-forest-labs)** built the FLUX family of image models. DoodleBook uses:

**[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** for every illustration in the book. Klein runs in as few as **6 inference steps at guidance_scale 1.0**, which gives us six story-page illustrations and a canonical-character pass in a GPU slot that fits inside ZeroGPU's per-request limit. Despite running fast, the crayon storybook quality is strong: warm textures, clear character shapes, and readable scene composition.

FLUX.2-klein is also the engine for the **coloring book pass** — the same model that illustrated the story redraws each page as clean line art, keeping the composition consistent between the color book and the coloring book.

Without FLUX.2-klein's speed and quality at 4B, DoodleBook's full pipeline — canonical hero pass + six story pages + optional six coloring pages — would be too slow for a real demo.

---

### Hugging Face — The Stage

**[Hugging Face Spaces](https://huggingface.co/spaces)** hosts DoodleBook and provides **ZeroGPU** — free, on-demand GPU time that scales to zero when the Space is idle. The ZeroGPU pattern requires loading models on CUDA at module scope (not inside the request handler), which meant some non-obvious initialization work — but it made the Space fast and cost-free to run publicly.

The **Gradio 6** framework provides the product shell. DoodleBook uses a fully custom Gradio Blocks layout with its own CSS, fonts (Gaegu, Caveat, Nunito), SVG filters for hand-drawn wobble borders, and a construction-paper scrapbook aesthetic — **zero Gradio defaults visible**.

---

## Open Trace

Every book DoodleBook produces is fully reproducible. Open **"Behind the magic → Trace"** in the app and you get:

- The locked **seed** (same seed + inputs → same book)
- The **per-page scene plan** and prompts
- The exact **model IDs** used
- **Stage timings**: story / illustrations / narration / PDF / coloring
- Any fallback reasons — nothing is silent

This is the hackathon's **Open Trace** badge in practice: full transparency into every generation.

---

## Try It

**Live Space:** [huggingface.co/spaces/build-small-hackathon/DoodleBook](https://huggingface.co/spaces/build-small-hackathon/DoodleBook)

Open the Space — a real sample book loads instantly. Then upload a drawing, pick a theme, and make your book.

**Source code:** [github.com/Sushruths04/Doodle-book](https://github.com/Sushruths04/Doodle-book)

**Technical Field Notes:** [Cross-page character consistency without per-user training →](blog.md)

---

## The Stack at a Glance

| Layer | Technology | Sponsor |
|---|---|---|
| Story writing | [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | OpenBMB |
| Narration | [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | OpenBMB |
| Illustration + line art | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Black Forest Labs |
| Hosting + GPU | [HF Spaces + ZeroGPU](https://huggingface.co/spaces) | Hugging Face |
| UI framework | [Gradio 6](https://www.gradio.app) | Hugging Face |

---

*DoodleBook was built for the **Build Small Hackathon 2026 · Adventure in Thousand Token Wood**. Apache-2.0 licensed.*
