---
title: DoodleBook
emoji: 📚
colorFrom: orange
colorTo: yellow
sdk: gradio
sdk_version: "6.18.0"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - hackathon
  - build-small
  - adventure-in-thousand-token-wood
  - gradio
  - flux
  - minicpm
  - voxcpm
  - storybook
  - coloring-book
  - voice-cloning
  - children
models:
  - black-forest-labs/FLUX.2-klein-4B
  - openbmb/MiniCPM5-1B
  - openbmb/VoxCPM2
---

# DoodleBook 📚🖍️

**A child draws a character → DoodleBook turns it into a narrated, illustrated crayon storybook *and* a matching printable coloring book — with their own voice if they want.**

Built for the **Build Small Hackathon 2026 · Adventure in Thousand Token Wood**. Every model is under 32B; the reasoning stack (story + voice) is just ~3B.

> Open the Space and a real sample book loads instantly. Then draw your own character, upload it, and watch your hero come to life across 6–10 pages.

---

## ✅ Pre-flight Checklist

| Requirement | Status | Notes |
|---|---|---|
| **Stay under 32B** | ✅ | MiniCPM5-1B (1B) + VoxCPM2 (2B) + FLUX.2-klein-4B (4B) = **7B total** |
| **Ship a Gradio app** | ✅ | Gradio 6 Space in the Build Small org |
| **Record a demo** | 📹 | *(link to be added)* |
| **Post on social media** | 🐦 | *(link to be added)* |
| **GPU limit (≤10 ZeroGPU)** | ✅ | 1 ZeroGPU Space |

---

## 🔗 Links

| | |
|---|---|
| 🚀 **Live Space** | [huggingface.co/spaces/build-small-hackathon/DoodleBook](https://huggingface.co/spaces/build-small-hackathon/DoodleBook) |
| 🔬 **Field Notes** (technical deep-dive) | [docs/blog.md](docs/blog.md) |
| 💻 **Source code** | [github.com/Sushruths04/Doodle-book](https://github.com/Sushruths04/Doodle-book) |
| 🤖 **MiniCPM5-1B** | [huggingface.co/openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) |
| 🔊 **VoxCPM2** | [huggingface.co/openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) |
| 🎨 **FLUX.2-klein-4B** | [huggingface.co/black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |

---

## ✨ What it does

1. **Draw & upload** a character (upload or webcam snap).
2. **Story** — MiniCPM5-1B writes a complete 6–10-page story with a consistent hero and a real emotional arc.
3. **Illustrations** — FLUX.2-klein renders each page where **your character stays consistent** across all pages (built from *your* drawing via img2img).
4. **Narration** — VoxCPM2 reads the whole book aloud in the child's choice of voice — kid, storyteller, grandpa, or **their own cloned voice** (record 5–60 s).
5. **Coloring book** — a matching black-and-white line-art version to print and color.
6. **Downloads** — one-tap **Story PDF** and **Coloring PDF** (works on mobile too).

### What makes a story good

DoodleBook uses a structured prompting system to guarantee quality:

- **Few-shot exemplar**: A full 6-page story is shown as a gold example so MiniCPM learns the exact format and richness required.
- **Story arc rules**: Pages 1–2 introduce the hero and challenge; middle pages build it; final pages resolve and teach a clear lesson.
- **2–3 rich sentences per page**: Every page uses sensory detail — colours, sounds, textures, feelings — not just plot.
- **10 diverse themes** covering kindness, imagination, friendship, courage, identity, and adventure.
- **Optional story spark**: Parents/kids can give a hint ("loves dinosaurs") to steer the story without breaking the structure.
- **Variable page count (6–10)**: More pages = longer story and longer narration.

---

## 🧠 Models & the "Tiny Titan" argument

| Role | Model | Params | Sponsor | What it does |
|---|---|---|---|---|
| 📖 Story writer | `openbmb/MiniCPM5-1B` | **1B** | **OpenBMB** | Writes the full story as structured JSON — title, character description, per-page text and scene descriptions |
| 🔊 Voice narrator | `openbmb/VoxCPM2` | **2B** | **OpenBMB** | Reads the story aloud with voice design prefixes; optionally clones the user's own voice via reference audio |
| 🎨 Illustrator | `black-forest-labs/FLUX.2-klein-4B` | **4B** | **Black Forest Labs** | Renders each page + the coloring line-art as FLUX img2img; keeps the child's drawn character consistent |

**Total parameter count: 7B.** Each individual model is well below the 32B cap.

**The product's reasoning is a ~3B small-model stack.** MiniCPM5-1B writes the narrative and scene plans; VoxCPM2 performs it. FLUX is not the "intelligence" — it's the renderer. That's the Tiny Titan story: the small models drive the experience.

---

## 🎙️ Voice Cloning — the "My Voice" option

When a parent selects **"🎙️ My Voice"**, they record 5–60 seconds of clear speech. VoxCPM2 uses this as a `reference_wav_path` to clone the voice and narrate the book in it — so children hear their parent's actual voice reading their story, even on demand.

The reference audio is processed locally (never stored). Sentences are capped at 15 for the ZeroGPU budget.

---

## 🎨 Cross-page character consistency

Generate pages independently and you get six different characters. DoodleBook keeps **one** hero — *the one the child drew* — with **no per-user training**:

1. **Canonical-character pass** — the doodle goes through FLUX **img2img once** to produce a clean "model-sheet" hero.
2. **Every page is conditioned on that canonical image**, so the same creature appears in every scene.
3. **Seed-locking** (deterministic per page) + a **fixed character description** anchor identity.

A cat doodle becomes a cat hero across all pages. Full write-up: **[Field Notes →](docs/blog.md)**.

---

## 🖍️ Coloring book: redraw, don't trace

Tracing finished crayon pages turned textures into speckle. Instead, DoodleBook passes each color page back to FLUX as img2img with a *"clean coloring-book line art"* prompt — it **redraws** clean outlines, then a local pass crisps them to pure black-on-white.

---

## 🔍 Open Trace — every book is reproducible

Open **"Behind the magic → Trace"** on any generated book:

- locked **seed** (same inputs → same book)
- per-page prompts and scene plan
- exact **model IDs** and stage timings
- fallback reasons (surfaced, never silent)

---

## 🏗️ Architecture

| Layer | What | File |
|---|---|---|
| Product UI | Custom scrapbook Gradio 6 Blocks | [`ui/layout.py`](ui/layout.py) |
| Book / PDF | HTML book + printable PDFs | [`book_builder.py`](book_builder.py) |
| Story | MiniCPM5-1B + deterministic arc fallback | [`app.py`](app.py) |
| Images | FLUX.2-klein canonical + per-page img2img | [`app.py`](app.py) |
| Voice | VoxCPM2 narration (parallel) + voice cloning | [`app.py`](app.py) |
| Coloring | FLUX line art + crisp/threshold cleanup | [`services/coloring.py`](services/coloring.py) |
| Config | Models, seeds, voices, palette | [`config.py`](config.py) |

### How it runs (ZeroGPU)

- **Gradio 6** Space, **ZeroGPU** hardware — free T4 GPU per request.
- All three models load **on CUDA at module scope** (ZeroGPU pattern); each stage is a `@spaces.GPU` call.
- Narration runs **in parallel** with illustration and surfaces the moment it's ready.
- A real, pre-generated **sample book loads instantly** on open — no GPU required.

---

## 🏅 Hackathon badges

| Badge | Status | Evidence |
|---|---|---|
| **Off-Brand** | ✅ | Fully custom scrapbook UI — Gaegu/Caveat fonts, paper textures, hand-drawn SVG frames, light-locked. Zero Gradio defaults. See [`ui/layout.py`](ui/layout.py). |
| **Open Trace** | ✅ | Every book exposes seed, per-page prompts, model IDs, and timings in the in-app Trace panel. |
| **Field Notes** | ✅ | Engineering write-up on cross-page consistency: **[docs/blog.md](docs/blog.md)**. |
| **Tiny Titan** | ✅ | MiniCPM5-1B + VoxCPM2 ≈ **3B reasoning stack**; FLUX is the renderer. |
| **Sponsor — OpenBMB** | ✅ | `MiniCPM5-1B` writes every story; `VoxCPM2` narrates and clones voices. |
| **Sponsor — Black Forest Labs** | ✅ | `FLUX.2-klein-4B` renders every page illustration and coloring line art. |

---

## ▶️ Run locally

```bash
pip install -r requirements.txt
python app.py
```

The example doodle (`assets/sample_doodle.jpg`) and the instant sample book are included.

---

## 🤝 Sponsor & tool stack

- **OpenBMB** — MiniCPM5-1B (story) + VoxCPM2 (voice + cloning)
- **Black Forest Labs** — FLUX.2-klein-4B (illustration + line art)
- **Hugging Face Spaces / ZeroGPU** — hosting + GPU
- **Gradio 6** — the product shell

---

## License

Apache-2.0. See [LICENSE](LICENSE).
