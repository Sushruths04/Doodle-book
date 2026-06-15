---
title: DoodleBook
emoji: 📚
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: "6.18.0"
app_file: app.py
pinned: false
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
models:
  - black-forest-labs/FLUX.2-klein-4B
  - openbmb/MiniCPM5-1B
  - openbmb/VoxCPM2
---

# DoodleBook 📚🖍️

**A child draws a character → DoodleBook turns it into a narrated, hand-drawn crayon storybook *and* a matching printable coloring book.**

Built for the **Build Small Hackathon 2026 · Adventure in Thousand Token Wood**. The bet: keep the *reasoning* brain tiny (a ~3B small-model stack writes and narrates the story), use a strong image model only as the *renderer*, and wrap it all in a UI that feels like a children's product — not a model demo.

> Open the Space and a real sample book loads instantly. Then draw your own character, upload it, and watch your hero come to life across six pages.

---

## ✨ What it does

1. **Draw & upload** a character (upload or webcam).
2. **Story** — a small model writes a complete six-page story with a consistent hero.
3. **Illustrations** — FLUX renders six crayon pages where **your character stays the same on every page** (it's built from *your* drawing).
4. **Narration** — the whole book is read aloud in a child-friendly voice.
5. **Coloring book** — a matching black-and-white line-art version to print and color.
6. **Downloads** — one-tap **Story PDF** and **Coloring PDF** (works on mobile too).

---

## 🧠 Models & the "Tiny Titan" argument

| Role | Model | Size | Sponsor |
|---|---|---|---|
| 📖 Story (the brain) | `openbmb/MiniCPM5-1B` | **1B** | OpenBMB |
| 🔊 Narration (the brain) | `openbmb/VoxCPM2` | **2B** | OpenBMB |
| 🎨 Illustration (the printer) | `black-forest-labs/FLUX.2-klein-4B` | 4B | Black Forest Labs |

**The product's *reasoning* is a ~3B small-model stack.** MiniCPM5-1B authors the narrative and per-page scene plan; VoxCPM2 performs it. FLUX is not the "intelligence" — it's the renderer that prints what the small models decided. That's the Tiny Titan story: the small models drive the experience.

---

## 🎨 The hard part: cross-page character consistency

Generate six pages independently and you get six different characters. DoodleBook keeps **one** hero — *the one the child drew* — with **no per-user training**:

1. **Canonical-character pass** — the doodle is sent through FLUX **img2img once** to become a single clean "model-sheet" hero.
2. **Every page is conditioned on that canonical image**, so the same creature appears in each scene.
3. **Seed-locking** (deterministic per page) + a **fixed character description** anchor identity and reproducibility.

Identity comes from the child's drawing, carried by image conditioning — a cat doodle becomes a cat hero, a robot becomes a robot. Full write-up in **[Field Notes →](docs/blog.md)**.

---

## 🖍️ Coloring book: redraw, don't trace

Tracing finished crayon pages turned textured backgrounds into speckle. Instead, DoodleBook hands each color page back to FLUX as img2img with a *"clean coloring-book line art"* prompt — it **redraws** clean, colorable outlines that match the story page, then a tiny local pass crisps them to pure black-on-white.

---

## 🔍 Open Trace — every book is reproducible

Open **"Behind the magic → Trace"** on any generated book and you get the full, reproducible record:

- the locked **seed**
- the **per-page prompts** and scene plan
- the exact **model IDs** used
- **stage timings** (story / images / narration / PDF / coloring)
- any fallback reasons (surfaced, never silent)

Same inputs → same book.

---

## 🏗️ How it runs (Hugging Face ZeroGPU)

- **Gradio 6** Space, **ZeroGPU** hardware. `app.py` is the entrypoint.
- All three models load **on CUDA at module scope** (the ZeroGPU-recommended pattern); each generation stage is a `@spaces.GPU` call.
- Narration runs **in parallel** with illustration and is surfaced the moment it's ready.
- A real, pre-generated **sample book loads instantly** on open (no GPU needed to be impressed).

---

## 🏅 Hackathon badges

We only claim what we can show. Every claim below points to a real, inspectable artifact.

| Badge | Status | Evidence |
|---|---|---|
| **Off-Brand** | ✅ Claimed | A fully custom scrapbook UI — crayon fonts (Gaegu/Caveat), paper textures, hand-drawn SVG frames, light-locked theme. **Zero Gradio defaults.** See [`ui/layout.py`](ui/layout.py). |
| **Open Trace** | ✅ Claimed | Every book exposes a full reproducible trace (seed, per-page prompts, model IDs, timings) in the in-app **Trace** panel. |
| **Field Notes** | ✅ Claimed | Engineering write-up on cross-page character consistency without per-user training: **[docs/blog.md](docs/blog.md)**. |
| **Tiny Titan** | ✅ Argued | The reasoning stack (MiniCPM5-1B + VoxCPM2 ≈ **3B**) is the product brain; FLUX is the renderer. |
| **Sponsor — OpenBMB** | ✅ | `MiniCPM5-1B` writes the story; `VoxCPM2` narrates it. |
| **Sponsor — Black Forest Labs** | ✅ | `FLUX.2-klein-4B` renders every page and the coloring line art. |
| **Well-Tuned** | 🔜 Roadmap (not claimed) | A crayon-style FLUX LoRA is the planned next step. **No LoRA is trained or published yet**, so we make no Well-Tuned claim — consistency today is achieved by the canonical-character + seed + description stack. |

---

## 🧩 Architecture (Space)

| Layer | What | Where |
|---|---|---|
| Product UI | Custom scrapbook Gradio 6 Blocks + CSS/JS | [`ui/layout.py`](ui/layout.py) |
| Book/PDF assembly | HTML book + printable PDFs | [`book_builder.py`](book_builder.py) |
| Story | MiniCPM5-1B + deterministic structured fallback | [`app.py`](app.py) |
| Images | FLUX.2-klein canonical + per-page render | [`app.py`](app.py) |
| Voice | VoxCPM2 narration (parallel) | [`app.py`](app.py) |
| Coloring | FLUX img2img line art + crisp/threshold cleanup | [`services/coloring.py`](services/coloring.py) |
| Config | Models, seeds, voices, palette | [`config.py`](config.py) |

> The source repo also includes a **Modal-backed** development path (story/image/voice as deployed Modal functions) used for local quality validation; the hosted Space runs the self-contained ZeroGPU path in `app.py`.

---

## ▶️ Run locally

```bash
pip install -r requirements.txt
python app.py
```

The example doodle (`assets/sample_doodle.jpg`) and the instant sample book are included.

---

## 🤝 Sponsor & tool stack

- **OpenBMB** — MiniCPM5-1B (story) + VoxCPM2 (voice)
- **Black Forest Labs** — FLUX.2-klein-4B (illustration + line art)
- **Hugging Face Spaces / ZeroGPU** — hosting + GPU
- **Gradio 6** — the product shell

---

## License

Apache-2.0. See [LICENSE](LICENSE).
