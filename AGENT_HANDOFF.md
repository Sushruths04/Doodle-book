# DoodleBook — Coding-Agent Handoff Prompt
Paste everything in the fenced block below into Codex / OpenCode / Cursor Agent / Claude Code as the build task.
It is self-contained and bakes in the 5 critical corrections (C1–C5) from `EXECUTION_PLAN.md`.

---

```
ROLE
You are the lead engineer building "DoodleBook" for the Build Small Hackathon 2026
(Adventure in Thousand Token Wood track). Build a Gradio app deployed to Hugging Face
Spaces that turns a child's crayon drawing into a consistent, narrated, illustrated
6-page storybook. Work in: D:\Project\Hugging_face_app\doodlebook

CONTEXT FILES (read first, in this order)
1. B2_doodlebook_prompt.md   — original concept, exact code sketches, frontmatter, badges
2. EXECUTION_PLAN.md         — architecture, phases, and the 5 corrections you MUST honor

NON-NEGOTIABLE CORRECTIONS (override the original prompt where they conflict)
C1  Do NOT train a LoRA per child at runtime (infeasible). Train ONE crayon-style LoRA
    OFFLINE. Achieve per-character consistency at inference via: (a) locked seed S, page i
    uses seed S+i; (b) reuse the identical character_description on every page; (c) feed the
    uploaded doodle as an IMAGE PROMPT (IP-Adapter / FLUX Redux / img2img strength ~0.3-0.5).
    The app MUST run on base FLUX with NO LoRA (degrade gracefully, show "LoRA coming").
C2  MiniCPM5-1B is unreliable at JSON. Use a few-shot prompt with ONE full exemplar, greedy
    decode, and a 3-layer parser: (1) regex extract {...}; (2) json-repair/json5; (3) a
    deterministic TEMPLATE fallback that always yields a valid 6-page book. App must NEVER
    crash on bad model output.
C3  Modal cold-starts of a 12B diffusion model take minutes. Cache weights on a Modal Volume,
    expose a keep_warm option, generate all 6 pages in ONE warm container call, and ALWAYS
    ship a pre-generated sample book in assets/sample_book/ that loads instantly with zero compute.
C4  This is a SMALL-MODELS hackathon. Frame the 1B story + 2B voice as the "brain" and FLUX as
    the "renderer" in the README (Tiny Titan argument). Implement a real "Tiny Mode" toggle that
    swaps FLUX for an SD-Turbo/SDXL-Turbo + style-LoRA path (1-4 steps) runnable on a T4/edge GPU.
C5  Treat ALL model IDs as UNVERIFIED. FIRST TASK: verify each on the HF Hub and put the resolved
    IDs + fallbacks in config.py. Fallbacks: FLUX.2-klein -> FLUX.1-schnell; MiniCPM5-1B ->
    MiniCPM3-4B; VoxCPM2 -> Kokoro or MeloTTS. Nothing else imports a raw model string.

TECH STACK
Gradio 5.x (gr.Blocks, custom storybook CSS) on HF Spaces (CPU) · Modal for all GPU compute
(FLUX on A100, MiniCPM on T4, TTS on T4/A10G) · diffusers · peft · transformers · fpdf2 ·
Python 3.11. Space is a thin orchestrator; every heavy call is modal.Function.remote().

DELIVERABLES (build in this dependency order; commit after each phase)
PHASE 1 Foundation
  - Verify model IDs on HF Hub; create config.py = single source of truth (IDs, fallbacks,
    seeds, step counts, dimensions, lora repo, dataset repo).
  - Scaffold the directory structure exactly as in B2_doodlebook_prompt.md plus:
    config.py, services/, ui/, modal/.
  - requirements.txt, .env.example (HF_TOKEN, MODAL endpoint/token).
  - Bare Gradio shell that launches and displays the static sample book.
PHASE 2 Core text pipeline
  - modal_story_gen.py: MiniCPM story -> JSON {title, character_description, pages:[{page,text,scene}]}
    with the C2 3-layer parser + template fallback.
  - book_builder.py: pages -> storybook HTML; PDF export via fpdf2 (gr.DownloadButton).
  - assets/custom.css storybook styling. Wire text-only book end to end.
PHASE 3 AI integration
  - modal_image_gen.py: FLUX pipeline; generate_book_pages() makes all 6 pages in one warm
    container; C1 consistency stack (seed-lock + char_desc reuse + doodle image-prompt);
    optional LoRA fuse (~0.85) with graceful base-model fallback; Modal Volume cache; keep_warm.
  - modal_tts.py: VoxCPM2 narration of title+page texts -> wav, with fallback voice (C5).
  - Full pipeline: doodle -> story -> 6 images -> narration -> assembled book.
PHASE 4 UX
  - Convert create_book to a GENERATOR (yield) so status + pages stream in page-by-page
    ("Illustrating page N of 6..."). Add "Behind the magic" gr.Accordion (prompts/seeds/LoRA).
  - Tiny Mode toggle (C4). Mobile-responsive CSS. Accessibility: alt text = page text on every
    image, AA contrast, prefers-reduced-motion. Examples auto-load the sample on launch.
PHASE 5 Optimization & badges
  - lora_finetune/: train_lora.py (DreamBooth-style, FLUX, rank16 alpha16, crayon style,
    trigger [DOODLECHAR]), dataset_prep.py, README.md (reproduce steps). Publish LoRA to HF
    (Well-Tuned). bf16/turbo settings; CPU-offload fallback for smaller GPUs.
  - services/trace.py: log prompts/seeds/lora-version to HF dataset build-small-hackathon/
    doodlebook-traces (Open Trace).
  - Pre-generate and COMMIT the 6-page sample book to assets/sample_book/ (C3 non-negotiable).
PHASE 6 Submission
  - README.md with the EXACT frontmatter from B2_doodlebook_prompt.md + Tiny Titan argument +
    model table + architecture diagram + install/usage + screenshots + demo + reproducibility +
    badges + license (Apache-2.0).
  - Field Notes blog draft (docs/blog.md) on FLUX+LoRA character consistency.
  - Deploy to HF Spaces; smoke test the cold-open judge path (sample loads with no compute);
    verify live generation; run the §9 submission checklist from EXECUTION_PLAN.md.

INTERNAL CONTRACTS (keep stable)
generate_story(hero_name, theme, age=5) -> {title, character_description, pages:[{page,text,scene}]}
generate_book_pages(character_desc, story_beats, doodle=None, art_style, seed=42, tiny=False) -> list[bytes png]
speak_book(text, voice="warm") -> bytes wav
build_book_html(images, texts, title) -> html ; export_pdf(images, texts, title) -> path
log_trace(payload) -> dataset_url

ENGINEERING RULES
- Every remote/model call wrapped in try/except with a user-friendly fallback; the app must
  never show a stack trace to a judge.
- config.py is the ONLY place model IDs/params live.
- Keep the sample-book path 100% independent of live compute so the demo always works.
- Match the storybook visual identity (paper #FEF9E7, page #FFFDE7, ink #3E2723, CTA #FF7043,
  serif body). No default Gradio look (Off-Brand badge).
- Prefer small, readable modules over clever code. Comment the consistency logic and the parser.

DEFINITION OF DONE
Public HF Space loads the sample book instantly; a live "Make my book!" produces a consistent,
narrated 6-page book in <2 min warm; Tiny Mode works on a cheap GPU; LoRA repo + trace dataset
are public; README + frontmatter + blog published; submission checklist fully ticked.

START NOW with Phase 1, Task 0: verify the model IDs on the HF Hub and write config.py. Report
what you find before proceeding.
```
