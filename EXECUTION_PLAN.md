# DoodleBook — Master Execution Plan
**Build Small Hackathon 2026 · "Adventure in Thousand Token Wood" Track**
_Senior architect / strategist / tech-lead review of `B2_doodlebook_prompt.md`_

> Source-of-truth concept: A child draws a crayon character → photograph it → the app produces a
> **consistent, illustrated 6-page storybook** (same character + same art style on every page),
> written by **MiniCPM5-1B**, narrated by **VoxCPM2**, illustrated by **FLUX.2-klein + a crayon-style LoRA**.

---

## 0. Critical engineering corrections (read first)

The original prompt is excellent on vision and badge strategy but has **5 load-bearing technical risks**. Fix these before writing code or you will fail the live demo.

| # | Risk in original | Reality | Fix (this plan adopts) |
|---|---|---|---|
| **C1** | Implies per-child LoRA so each kid's character is reproduced | LoRA training = minutes–hours on A100. You **cannot** train per user at demo time. | LoRA trains **ONE crayon art-style** offline. Per-character consistency comes from **(a) locked seed, (b) the child's doodle used as an image prompt** via IP-Adapter / FLUX Redux / img2img, and **(c) a fixed character description** reused on every page. Never claim live per-child training. |
| **C2** | 1B model must emit strict JSON | 1B models break JSON constantly (trailing commas, prose, truncation). | Constrained decoding + a **3-layer parser**: (1) regex extract, (2) `json5`/repair, (3) deterministic template fallback so the app NEVER crashes. Few-shot prompt with one full exemplar. |
| **C3** | "60–90s for 6 images" | True for warm GPU; **Modal cold start of a 12B diffusion model = 2–4 min**. | Modal `@app.cls` with `keep_warm=1` during demo window + model weights on a **Modal Volume** (no re-download). Pre-warm button. Always ship a pre-generated sample book. |
| **C4** | "Small Models" hackathon, but hero model is 12B | Judges may discount the "small" claim. | **Reframe narrative:** the *brain* (story + voice) is a **3B total small-model stack** (MiniCPM5-1B + VoxCPM2); FLUX is the *renderer/printer*. Add a real **"Tiny Mode"** (SDXL-Turbo or SD-Turbo + style LoRA, runnable on T4/edge) to make the small-model claim defensible and unlock the edge-device track. |
| **C5** | Assumes model IDs are correct | Prompt itself flags FLUX.2-klein / MiniCPM5-1B / VoxCPM2 as unverified. | **Phase 1, Task 0:** verify every model-card ID on HF Hub; wire fallbacks (`FLUX.1-schnell`, `MiniCPM3-4B`, `MeloTTS`/`Kokoro`) behind one config block. |

Everything below assumes these corrections.

---

## 1. Project Analysis

### Executive summary
DoodleBook converts a child's hand-drawn crayon character (captured by photo) into a complete, narrated, visually-consistent 6-page picture book in ~2 minutes. It fuses three sponsor models into a single emotional, demo-perfect artifact: a 1B story writer, a 2B voice, and a 12B image model steered by a custom crayon-style LoRA. The moat is the **combination** — nobody else pairs a child's real drawing with a fine-tuned FLUX LoRA and on-model narration.

### Problem statement
Kids create characters constantly but those drawings die on the fridge. Parents can't turn them into the stories children imagine. Existing "AI storybook" apps (a) ignore the child's actual art, (b) produce inconsistent characters page-to-page, and (c) feel like generic AI slop, not *the child's* creation. There is no tool that preserves the child's own visual style across a coherent narrated book.

### Target users
- **Primary:** Parents of children 3–8 (gift/keepsake, bedtime, screen-time-with-purpose).
- **Secondary:** Early-years teachers (creative writing prompts), pediatric/occupational therapists (expressive activities), grandparents (remote bonding).
- **Demo persona:** Judge watching a 60-sec video of a kid hearing their own drawing read aloud as a book.

### Market need
- AI-storytime apps are a growing category but commoditized and character-inconsistent.
- Differentiator the market lacks: **"your child's actual drawing becomes the book's art."** Emotional keepsake > generic generation. High shareability (parents post their kids).

### Competitive advantage / moat
1. **Underused sponsor field** — FLUX.2-klein + custom LoRA is near-empty in the competition (per original recon).
2. **Style-faithful character consistency** via seed-lock + image-prompt conditioning + LoRA (multi-signal, robust).
3. **Full small-model stack** end-to-end on sponsor models (story + voice + image).
4. **Off-brand storybook UI** — zero Gradio defaults.
5. **Emotional demo** — strongest 60-second-video category in the whole event.

### Innovation score (self-assessed)
| Axis | Score /10 | Note |
|---|---|---|
| Concept originality | 9 | Doodle→consistent book is genuinely novel |
| Technical depth | 8 | LoRA + multi-model orchestration + consistency engineering |
| Feasibility in hackathon window | 7 | Achievable IF corrections C1–C5 applied |
| Demo/emotional impact | 10 | Best-in-show potential |
| Track coverage | 9 | 5+ awards reachable |
| **Composite** | **8.6** | Strong winner profile |

### Hackathon-winning potential
**High.** Realistic path to: Thousand Token Wood podium + Black Forest Labs sponsor award + OpenBMB award + Off-Brand + Best Demo + Community Choice. Five+ simultaneous award surfaces is the strategy (see §7).

---

## 2. Product Vision

### Long-term vision
The default way a family turns a child's imagination into a keepsake — "Instagram for the things your kid invents." Drawing → narrated book → printed photo-book → series with recurring characters.

### Future roadmap
- **v1 (hackathon):** Doodle → 6-page narrated book, PDF export, Open-Trace share.
- **v2:** Character library (recurring heroes across books), multi-character scenes, child-voice cloning (with guardian consent), print-on-demand.
- **v3:** Collaborative books (siblings co-create), classroom mode, multilingual narration, animation (short clips per page).
- **v4:** On-device "Tiny Mode" mobile app for offline bedtime generation.

### Scalability opportunities
- Stateless generation workers (Modal autoscale) behind a thin Gradio/HF front door.
- Cache by (doodle-hash + theme + seed) to dedupe regenerations.
- Batch the 6 pages in one warm container (already in the original `generate_book_pages`).

### Edge-device deployment possibilities
- **Story:** MiniCPM5-1B quantized (GGUF/llama.cpp, int4) runs on a laptop/phone NPU.
- **Voice:** VoxCPM2 or a Kokoro/MeloTTS fallback runs on CPU.
- **Image (Tiny Mode):** SD-Turbo / SDXL-Turbo + tiny style LoRA at 1–4 steps on a single consumer GPU or Apple Silicon; sub-second-to-a-few-seconds pages.
- Ship a documented "edge profile" config to claim the edge/small-model narrative credibly.

### Small-model optimization strategy
- 4-bit (NF4/bitsandbytes) for MiniCPM; `torch.compile` + bf16 for FLUX; sequential CPU offload to fit smaller GPUs.
- FLUX turbo settings: 4–20 steps; Tiny Mode = 1–4 steps with turbo image model.
- Modal Volume model cache; `keep_warm` only during judging.
- KV-cache reuse + greedy decode for deterministic, fast story gen.

---

## 3. Technical Architecture

### System architecture (text diagram)
```
                         ┌──────────────────────────────────────────────┐
                         │   HF Space (Gradio 5.x, custom storybook UI)  │
                         │   app.py  ·  custom.css  ·  book_builder.py    │
                         └───────────────┬──────────────────────────────┘
                                         │ orchestration (sync calls)
        ┌────────────────────────────────┼─────────────────────────────────┐
        ▼                                ▼                                 ▼
┌───────────────┐              ┌───────────────────┐             ┌──────────────────┐
│ modal_story   │              │ modal_image_gen   │             │ modal_tts        │
│ MiniCPM5-1B   │              │ FLUX.2-klein +    │             │ VoxCPM2          │
│ (T4) JSON     │  char_desc   │ crayon LoRA (A100)│   pages.png  │ (T4/A10G) wav    │
│ story+scenes  │ ───────────► │ + IP-Adapter/img2 │ ──────────► │ narration        │
└───────┬───────┘   scenes     │ img from doodle   │             └────────┬─────────┘
        │                      └─────────┬─────────┘                      │
        │  pages[text,scene]             │ 6 page images                  │ audio
        └───────────────┬────────────────┴────────────────────────────────┘
                        ▼
              ┌────────────────────┐
              │ book_builder.py    │  → storybook HTML (gr.HTML) + PDF (fpdf2)
              │ + open_trace.py    │  → HF dataset trace (prompts/seeds/lora)
              └────────────────────┘
```

### Frontend architecture
- **Gradio 5.x `gr.Blocks`** single-page, two-column: input panel (left), live book viewer (right).
- Custom **storybook CSS** (yellowed paper, serif, drop shadows, page-flip feel) → Off-Brand badge.
- Progressive reveal: pages stream in as generated ("Illustrating page 3 of 6…").
- `gr.HTML` book canvas, `gr.Audio` narration, `gr.DownloadButton` PDF, `gr.Gallery` fallback.

### Backend architecture
- **Modal** for all heavy compute, 3 apps (story/image/tts), each a warm-able class.
- **Stateless** functions; book assembly + trace logging in the Space process.
- Config module (`config.py`) holds every model ID + fallback + generation params (single source of truth → fixes C5).

### AI model architecture
- **Story (MiniCPM5-1B, T4):** few-shot, greedy, `max_new_tokens≈800`, constrained JSON + 3-layer parser (C2). Outputs `title`, `character_description`, `pages[{page,text,scene}]`.
- **Image (FLUX.2-klein + LoRA, A100):**
  - Crayon **style LoRA** (offline-trained, rank 16) fused at scale ~0.8.
  - **Consistency stack:** locked base seed `S`; page `i` uses `S+i`; reuse identical `character_description` token block; **doodle image fed as image prompt** (IP-Adapter / Redux / img2img strength ~0.3–0.5) so output resembles the child's drawing (this realizes original TODO 1).
  - 20 steps (Standard) / 4 steps (Tiny Mode), guidance ~3.5, 768×512.
- **Voice (VoxCPM2, T4/A10G):** narrate `title + page texts`; return wav.
- **Doodle understanding (MiniCPM-V, optional):** caption the drawing → style tokens prepended to FLUX prompt (original TODO 1).

### Data flow
1. User uploads/photographs doodle + name + theme.
2. Story worker → JSON (title, char desc, 6×{text,scene}).
3. (Opt) Doodle captioner → style tokens.
4. Image worker → 6 PNGs (seed-locked, LoRA + doodle-conditioned).
5. TTS worker → narration wav.
6. `book_builder` → HTML book + PDF; `open_trace` → HF dataset row.

### API structure (internal contracts)
```
generate_story(hero_name:str, theme:str, age:int=5) -> {title, character_description, pages:[{page,text,scene}]}
generate_book_pages(character_desc:str, story_beats:list[str], doodle:bytes|None,
                    art_style:str, seed:int=42, tiny:bool=False) -> list[bytes]
speak_book(text:str, voice:str="warm") -> bytes(wav)
build_book_html(images:list[bytes], texts:list[str], title:str) -> str
export_pdf(images, texts, title) -> path
log_trace(payload) -> dataset_url
```

### Storage strategy
- Model weights → **Modal Volume** (cache, no re-download → fixes C3).
- Generated assets → ephemeral `/tmp` in Space; user downloads PDF.
- Traces → **HF Dataset** `build-small-hackathon/doodlebook-traces` (Open Trace badge).
- LoRA weights → **HF model repo** `build-small-hackathon/doodlebook-flux-lora` (Well-Tuned badge).

### Deployment strategy
- Front end on **HF Spaces** (Gradio SDK 5.0, `app.py`).
- Compute on **Modal** (secrets: `HF_TOKEN`, endpoint URLs via `.env`).
- `keep_warm=1` on image app only during judging window; scale to 0 after.

### Performance optimization plan
- One warm container generates all 6 pages (already designed).
- bf16 + optional `torch.compile`; turbo step counts; sequential CPU offload fallback.
- Stream page-by-page UI updates (perceived speed).
- Pre-generated sample book for instant judge view (non-negotiable).
- Tiny Mode for sub-10s full books on cheap GPU.

---

## 4. UI/UX Design Plan

### Design philosophy
"**A warm digital picture book, not a dashboard.**" Tactile, nostalgic, magical — looks hand-made, hides all ML. Every interaction should feel like turning a page, not running a model.

### User journeys
1. **First-time parent (happy path):** land → see sample book glowing → upload kid's drawing → name + theme → "✨ Make my book!" → progress storybook fills page-by-page → narration auto-ready → download PDF / share. <2 min, zero jargon.
2. **Judge (cold, impatient):** lands on a finished sample book immediately (no generation wait) → clicks "Hear it" → reads the story → optionally generates one live → sees Open-Trace link. Wow in <15s.
3. **Returning user (v2 vision):** pick a saved character → new adventure → consistent hero.

### Wireframe descriptions
- **Header:** centered title "📚 DoodleBook", subtitle "Draw a character. Get a storybook.", soft paper texture.
- **Left input card (scale 1):** webcam/upload doodle, character name, hero name, theme dropdown, big orange "Make my book!" CTA, Examples row (loads sample), status line, "⚡ Tiny Mode" toggle.
- **Right book viewer (scale 2):** large `gr.HTML` book — title page then 6 illustrated text pages with page numbers, yellowed background, serif body; narration audio bar pinned above; "⬇ Download PDF" + "🔗 Share trace" buttons below.
- **Progress state:** skeleton page slots fill one-by-one with "Illustrating page N of 6…".

### Dashboard / layout
- Single page, two columns desktop; stacked on mobile (input → book).
- Optional collapsible "🔬 Behind the magic" panel showing prompts/seeds/LoRA (judge candy + Open Trace).

### Color palette
| Token | Hex | Use |
|---|---|---|
| Paper | `#FEF9E7` | app background |
| Page | `#FFFDE7` | book pages |
| Ink | `#3E2723` | body text |
| Title brown | `#5D4037` | headings |
| Crayon orange | `#FF7043` | primary CTA |
| Sky accent | `#4FC3F7` | secondary/links |
| Muted | `#BCAAA4` | page numbers/meta |

### Typography
- Display/title: **Georgia / "Fredoka" / "Baloo 2"** (rounded, child-friendly).
- Body: **Georgia serif** 20–22px, line-height 1.9 (read-aloud comfortable).
- Avoid system sans defaults — they read as "Gradio".

### Accessibility
- WCAG AA contrast (ink on page passes); 18px+ body.
- Audio narration = built-in alt for non-readers; captions = page text.
- All controls keyboard reachable; alt text on every generated image (use page `text`).
- Respect `prefers-reduced-motion` (disable page-flip animation).

### Mobile responsiveness
- Columns collapse to stack; CTA full-width sticky; webcam capture works on phones (parents photograph the drawing in-app).

### Demo-friendly interactions
- Auto-load sample book on launch (no empty state).
- Page-by-page streaming reveal (visible progress = perceived magic).
- One-tap "Play narration".
- "Tiny Mode" toggle to show edge story live without long waits.

---

## 5. Gradio Implementation Plan

### App structure
```
app.py
├─ config.py            # model IDs + fallbacks + params (single source of truth)
├─ ui/
│  ├─ layout.py         # gr.Blocks layout
│  └─ custom.css        # storybook styling
├─ services/
│  ├─ story.py          # calls modal_story_gen
│  ├─ images.py         # calls modal_image_gen
│  ├─ tts.py            # calls modal_tts
│  ├─ book_builder.py   # HTML + PDF
│  └─ trace.py          # Open Trace dataset logging
└─ modal/
   ├─ modal_story_gen.py
   ├─ modal_image_gen.py
   └─ modal_tts.py
```

### Component hierarchy
```
gr.Blocks(css, theme)
├─ Header (gr.Markdown)
├─ gr.Row
│  ├─ gr.Column(scale=1)  # inputs
│  │  ├─ gr.Image(sources=[webcam,upload])
│  │  ├─ gr.Textbox char_name / hero_name
│  │  ├─ gr.Dropdown theme
│  │  ├─ gr.Checkbox tiny_mode
│  │  ├─ gr.Button "Make my book!" (primary)
│  │  ├─ gr.Examples (sample)
│  │  └─ gr.Textbox status (interactive=False)
│  └─ gr.Column(scale=2)  # output
│     ├─ gr.Audio narration
│     ├─ gr.HTML book_display
│     ├─ gr.DownloadButton PDF
│     └─ gr.Accordion "Behind the magic" (prompts/seeds)
```

### Pages & navigation
Single page (hackathon-optimal). "Pages" = sections of the book inside the HTML canvas. No router needed.

### User interaction flow
`make_btn.click(create_book, inputs=[...], outputs=[book_html, status, audio, pdf])` — use a **generator function** (`yield`) so status + pages stream in, not one blocking return.

### Model integration approach
- Space process is thin orchestrator; all GPU work via `modal.Function.remote()`.
- Defensive: every remote call wrapped in try/except → graceful UI error + fallback (base FLUX if no LoRA, template story if JSON fails).

### Performance considerations
- Single warm Modal container per book (6 images batched).
- `gr.Progress()` for the progress bar; `yield` partial books.
- Cache sample book in memory at startup.

### Deployment on HF Spaces
- `sdk: gradio`, `sdk_version: "5.0"`, `app_file: app.py` (frontmatter already specified).
- Secrets: `HF_TOKEN`, `MODAL_ENDPOINT_URL` (or Modal token) in Space settings.
- Keep Space CPU-only (compute offloaded to Modal) → cheap, always-on.

---

## 6. Development Roadmap

> Effort assumes a single builder + coding agent. Sequence is dependency-ordered.

### Phase 1 — Foundation
- **Tasks:** Verify all model IDs on HF Hub (Task 0, fixes C5); scaffold repo per directory structure; `config.py` with IDs + fallbacks + params; `requirements.txt`; `.env.example`; Modal account + secrets; bare Gradio shell that loads and shows static sample book.
- **Dependencies:** HF + Modal accounts, tokens.
- **Effort:** ~0.5 day.
- **Risks:** Model IDs differ from prompt → fallback wiring matters.
- **Success:** `app.py` launches locally, shows sample book, `config.py` resolves real model IDs.

### Phase 2 — Core Features
- **Tasks:** `modal_story_gen.py` with 3-layer JSON parser + template fallback (C2); `book_builder.py` HTML; PDF export (`fpdf2`); storybook CSS; wire story→book (text only, placeholder images).
- **Dependencies:** Phase 1.
- **Effort:** ~1 day.
- **Risks:** 1B JSON instability → mitigated by parser + fallback.
- **Success:** Enter name+theme → get a valid 6-page text book + PDF, no crashes even on bad model output.

### Phase 3 — AI Integration
- **Tasks:** `modal_image_gen.py` FLUX pipeline; Modal Volume model cache + `keep_warm` (C3); seed-lock + doodle image-prompt consistency stack (C1); graceful base-FLUX fallback if no LoRA; `modal_tts.py` VoxCPM2 (+ Kokoro/MeloTTS fallback); full pipeline story→images→audio.
- **Dependencies:** Phase 2; LoRA may still be training (degrade gracefully).
- **Effort:** ~1.5 days.
- **Risks:** Cold starts (C3), VRAM (use A100, CPU offload fallback), model-ID drift.
- **Success:** End-to-end live book in <2 min warm; consistent character across pages; narration plays.

### Phase 4 — UI/UX Enhancement
- **Tasks:** Streaming page-by-page reveal (`yield`); progress text; "Behind the magic" accordion; Tiny Mode toggle (C4); mobile responsive CSS; Examples auto-load; accessibility pass (alt text, contrast, reduced-motion).
- **Dependencies:** Phase 3.
- **Effort:** ~1 day.
- **Risks:** Gradio streaming quirks; CSS scope leaks.
- **Success:** Off-Brand-worthy UI; live progress; works on phone; Tiny Mode produces a book fast.

### Phase 5 — Optimization
- **Tasks:** Train + publish crayon style LoRA (Well-Tuned); quantization/turbo settings; Tiny Mode SD-Turbo path; trace logging to HF dataset (Open Trace); pre-generate + commit sample book (6 pages); error hardening.
- **Dependencies:** Phases 3–4.
- **Effort:** ~1 day (+ LoRA train time in background).
- **Risks:** LoRA quality/time → app must run on base model meanwhile (non-negotiable from original).
- **Success:** LoRA on HF, traces logged, sample book committed, Tiny Mode real, no unhandled errors.

### Phase 6 — Submission Preparation
- **Tasks:** README with exact frontmatter + Tiny Titan argument; record 60-sec demo video (child hearing book); blog post (Field Notes) on FLUX+LoRA consistency; screenshots/GIFs; deploy + smoke test on Spaces; final checklist (§9).
- **Dependencies:** All prior.
- **Effort:** ~0.5–1 day.
- **Risks:** Last-minute deploy breakage → smoke test early, keep sample-book path independent of live compute.
- **Success:** Public Space loads sample instantly, live gen works, all badges' artifacts published, video submitted.

**Total:** ~5–6 focused days. Critical path: Phase 1 (model IDs) → Phase 3 (FLUX+consistency) → Phase 6 (deploy/video).

---

## 7. Hackathon Strategy

### Tracks / awards to target (stack as many as possible)
| Award | Lever |
|---|---|
| Thousand Token Wood podium | Unique doodle→consistent-book concept |
| Black Forest Labs ($3k) | FLUX.2-klein + published custom LoRA (sparse field) |
| OpenBMB award | MiniCPM5-1B story + VoxCPM2 narration |
| Well-Tuned | Published LoRA on HF |
| Off-Brand ($1,500) | Storybook UI, zero Gradio defaults |
| Best Demo ($1,000) | Child hearing their drawing narrated |
| Community Choice | Shareable, emotional, parents repost |
| Tiny Titan (claimable) | Argue story generator (1B) is the primary AI + real Tiny Mode |

### How to maximize scoring
- One artifact, many badges: every badge needs a concrete published thing (LoRA repo, trace dataset, blog post, off-brand UI) — produce all four.
- Lead with emotion + sponsor-model usage in README's first 5 lines.
- Show, don't tell: pre-generated sample book + 60-sec video carry the score even if live gen is slow.

### What judges look for
Working demo, clear sponsor-model use, originality, polish, reproducibility (traces + LoRA), and a story that makes them feel something. DoodleBook is built to hit all six.

### Demo strategy
- Open on the finished sample book (instant wow, no wait).
- Play narration immediately.
- Then generate one live (or Tiny Mode) to prove it's real.
- End on the Open-Trace link + "made by a 1B + 2B small-model brain."

### Presentation / storytelling
Narrative arc: "Kids invent characters every day and we throw them away. Watch what happens when a 1B model gives one a story and FLUX gives it a book — in the child's own art." Personal, concrete, sponsor-forward.

### Key differentiators
Child's real art preserved · cross-page character consistency engineering · full small-model stack · off-brand keepsake UX · reproducible traces + LoRA.

---

## 8. README Plan
```
# 📚 DoodleBook  (+ exact HF frontmatter block from original prompt)
> Elevator pitch: Draw a character → get a narrated, illustrated 6-page storybook in your child's own art.
1. ✨ Features (consistency, narration, doodle-faithful art, PDF, Tiny Mode, traces)
2. 🧠 Models & why (table: FLUX.2-klein+LoRA / MiniCPM5-1B / VoxCPM2) + Tiny Titan argument
3. 🏗️ Architecture (diagram + data flow)
4. ⚙️ Installation (clone, requirements, Modal setup, HF token, .env)
5. ▶️ Usage (run app.py, upload doodle, make book)
6. 🖼️ Screenshots (sample book pages, UI)
7. 🎬 Demo (60-sec video link + live Space link)
8. 🔬 Reproducibility (LoRA repo, Open-Trace dataset, seeds)
9. 🛣️ Future work (character library, voice cloning, print-on-demand, edge app)
10. 🏅 Hackathon badges (Well-Tuned, Off-Brand, Field Notes, Open Trace)
11. 📄 License (Apache-2.0 / MIT) · 👥 Contributors
```

## 9. Submission Checklist
**Code** ☐ app launches clean ☐ all Modal fns callable ☐ graceful fallbacks (no-LoRA, bad-JSON, remote error) ☐ config has verified model IDs + fallbacks
**Docs** ☐ README + exact frontmatter ☐ install/usage ☐ LoRA reproduce README ☐ Field Notes blog published
**UI polish** ☐ storybook CSS, no Gradio defaults ☐ mobile ok ☐ accessibility (alt/contrast/reduced-motion) ☐ Examples auto-load
**Performance** ☐ warm gen <2 min ☐ keep_warm during judging ☐ Tiny Mode works ☐ sample book loads instantly
**Model optimization** ☐ LoRA trained + published ☐ bf16/turbo steps ☐ Tiny Mode SD-Turbo path
**HF deployment** ☐ Space live ☐ secrets set ☐ smoke test ☐ trace dataset public ☐ LoRA repo public
**Demo** ☐ 60-sec video (child hearing book) ☐ live Space link ☐ screenshots/GIFs
**Presentation** ☐ pitch deck/blurb ☐ storytelling script ☐ badge artifacts linked
**Final validation** ☐ fresh-clone run ☐ cold-open judge path tested ☐ all badge claims have a published URL

## 10. Agent Execution Prompt
See `AGENT_HANDOFF.md` — a self-contained master prompt for Codex / OpenCode / Cursor / Claude Code to build the whole project with the C1–C5 corrections baked in.
```
