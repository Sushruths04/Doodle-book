# B2 — "DoodleBook" | Claude Code Build Prompt  
## Build Small Hackathon 2026 | Thousand Token Wood Track

---

## Mission
A child draws a crayon character. You photograph it. The app turns it into a consistent illustrated 6-page storybook — same character, same art style, across every page — narrated by MiniCPM5-1B and illustrated by a fine-tuned FLUX.2-klein LoRA. Nobody else in this competition is using FLUX.2-klein with a custom LoRA. That's the entire competitive moat.

---

## Models (ONLY sponsor models)

| Role | Model ID | Params | Sponsor |
|---|---|---|---|
| Image generation | `black-forest-labs/FLUX.2-klein` | ~12B | Black Forest Labs |
| LoRA fine-tune (character consistency) | Custom LoRA on FLUX.2-klein | — | Black Forest Labs |
| Story generation | `openbmb/MiniCPM5-1B` | 1B | OpenBMB |
| Voice narration | `openbmb/VoxCPM2` | 2B | OpenBMB |

Total: ~15B — under 32B cap.

**FLUX.2-klein model ID note:** Verify the exact HF model card ID at `https://huggingface.co/black-forest-labs`. At time of writing, check for `black-forest-labs/FLUX.2-klein` or `black-forest-labs/FLUX.1-schnell` as fallback. The hackathon docs specifically name FLUX.2-klein.

**Tiny Titan note:** Story model (MiniCPM5-1B) is 1B. If you argue the "primary AI" is the story generator (not the image model), you could claim Tiny Titan. Make this argument in the README.

---

## Badge stack

| Badge | How |
|---|---|
| ✅ Well-Tuned | Fine-tune LoRA on FLUX.2-klein for character consistency; publish to HF |
| ✅ Off-Brand | Custom storybook UI — not remotely like default Gradio |
| ✅ Field Notes | Blog post about FLUX.2-klein + LoRA character consistency approach |
| ✅ Open Trace | Publish generation traces (prompts, seeds, LoRA weights used) |

---

## Tech stack
- **Gradio 5.x** with `gr.Server` for storybook-style custom UI
- **Modal** for FLUX.2-klein image generation (A100 recommended — diffusion is memory-heavy)
- **Modal** for MiniCPM5-1B story generation + VoxCPM2 TTS
- **diffusers** library for FLUX pipeline
- **peft** for LoRA loading
- **Python 3.11**

---

## Directory structure
```
doodlebook/
├── app.py                      # Gradio entry point
├── modal_image_gen.py          # FLUX.2-klein + LoRA generation on Modal
├── modal_story_gen.py          # MiniCPM5-1B story generation on Modal
├── modal_tts.py                # VoxCPM2 TTS on Modal
├── book_builder.py             # Assembles pages into storybook HTML
├── lora_finetune/
│   ├── train_lora.py           # FLUX LoRA training script (run locally)
│   ├── dataset_prep.py         # Prepare character images for training
│   └── README.md               # How to reproduce the fine-tune
├── requirements.txt
├── .env.example                # MODAL_ENDPOINT_URL, HF_TOKEN
├── README.md
└── assets/
    ├── custom.css              # Storybook CSS (yellowed pages, serif font)
    ├── page_template.html      # Single page HTML template
    ├── sample_doodle.jpg       # Example child's drawing
    └── sample_book/            # Pre-generated example book (6 pages)
        ├── page_1.png
        └── ...
```

---

## README.md — EXACT frontmatter
```yaml
---
title: DoodleBook
emoji: 📚
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: false
tags:
  - hackathon
  - build-small
  - adventure-in-thousand-token-wood
  - black-forest-labs/FLUX.2-klein
  - openbmb/MiniCPM5-1B
  - openbmb/VoxCPM2
  - fine-tuned
  - lora
  - character-consistency
  - storybook
  - off-brand
---
```

---

## LoRA fine-tuning plan (run BEFORE submission)

### Goal
Train a LoRA that makes FLUX.2-klein reproduce the visual style of a child's crayon drawing and maintain character consistency across 6 different scene prompts.

### Training data strategy
1. Take a child's crayon drawing (or generate 10-15 "crayon-style" reference images)
2. Create variations: same character in different poses/scenes, keeping style consistent
3. Use DreamBooth-style fine-tuning with a trigger token: `[DOODLECHAR]`

### Train script sketch (lora_finetune/train_lora.py)
```python
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model
# Use diffusers DreamBooth LoRA training
# Follow: https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
# Target: FLUX.2-klein with rank=16, alpha=16
# Training images: 10-15 images of the character
# Instance prompt: "photo of [DOODLECHAR] character, crayon drawing style"
# Epochs: 200-400 steps (fast with FLUX)
```

After training:
```bash
huggingface-cli upload build-small-hackathon/doodlebook-flux-lora ./lora-weights
```

---

## modal_image_gen.py
```python
import modal
app = modal.App("doodlebook-image-gen")

flux_env = modal.Image.debian_slim().pip_install(
    "diffusers>=0.28", "torch", "accelerate", "transformers",
    "peft", "pillow", "sentencepiece"
)

@app.function(gpu="A100", image=flux_env, timeout=300, memory=32768)
def generate_page(
    prompt: str,
    lora_repo: str = "build-small-hackathon/doodlebook-flux-lora",
    seed: int = 42,
    width: int = 768,
    height: int = 512
) -> bytes:
    from diffusers import FluxPipeline
    import torch, io
    from PIL import Image

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein",  # verify ID on HF Hub
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # Load character LoRA for consistency
    pipe.load_lora_weights(lora_repo)
    pipe.fuse_lora(lora_scale=0.85)

    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=20,   # FLUX.2-klein is fast
        guidance_scale=3.5,
        width=width,
        height=height,
        generator=generator
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

@app.function(gpu="A100", image=flux_env, timeout=300, memory=32768)
def generate_book_pages(
    character_desc: str,
    story_beats: list[str],
    art_style: str = "crayon drawing, children's book, colorful, simple shapes",
    seed: int = 42
) -> list[bytes]:
    """Generate all 6 pages in one function call to reuse the loaded model."""
    from diffusers import FluxPipeline
    import torch, io

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.load_lora_weights("build-small-hackathon/doodlebook-flux-lora")
    pipe.fuse_lora(lora_scale=0.85)

    pages = []
    for i, beat in enumerate(story_beats):
        prompt = (
            f"[DOODLECHAR] {character_desc}, {beat}, "
            f"{art_style}, page {i+1} of children's book, "
            f"white background, simple illustration"
        )
        gen = torch.Generator("cuda").manual_seed(seed + i)  # deterministic per page
        image = pipe(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=3.5,
            width=768, height=512,
            generator=gen
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        pages.append(buf.getvalue())

    return pages
```

---

## modal_story_gen.py
```python
import modal
app = modal.App("doodlebook-story")

story_env = modal.Image.debian_slim().pip_install(
    "transformers>=4.40", "torch", "accelerate", "sentencepiece"
)

@app.function(gpu="T4", image=story_env, timeout=120)
def generate_story(character_name: str, theme: str, age: int = 5) -> dict:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch, json

    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-1B")
    model = AutoModelForCausalLM.from_pretrained(
        "openbmb/MiniCPM5-1B", torch_dtype=torch.float16
    ).cuda().eval()

    prompt = f"""Write a 6-page children's storybook for age {age} about {character_name} with theme: {theme}.

Return ONLY valid JSON:
{{
  "title": "Book title",
  "character_description": "Visual description of {character_name} for illustration",
  "pages": [
    {{"page": 1, "text": "1-2 sentence page text (age {age})", "scene": "visual scene description for illustrator"}},
    {{"page": 2, ...}},
    {{"page": 3, ...}},
    {{"page": 4, ...}},
    {{"page": 5, ...}},
    {{"page": 6, "text": "Gentle ending. Goodnight.", "scene": "closing scene"}}
  ]
}}"""

    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800, do_sample=False)
    response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    import re
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"error": response}
```

---

## book_builder.py — storybook HTML assembler
```python
import base64

PAGE_HTML = """
<div class="book-page" style="page-break-after: always;">
  <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:8px;"/>
  <p class="page-text">{text}</p>
  <span class="page-num">{page_num}</span>
</div>
"""

def build_book_html(pages_images: list[bytes], pages_texts: list[str], title: str) -> str:
    pages_html = ""
    for i, (img_bytes, text) in enumerate(zip(pages_images, pages_texts)):
        b64 = base64.b64encode(img_bytes).decode()
        pages_html += PAGE_HTML.format(img_b64=b64, text=text, page_num=i+1)

    return f"""<div class="book-container">
    <h1 class="book-title">{title}</h1>
    {pages_html}
    </div>"""
```

---

## app.py — full Gradio storybook UI
```python
import gradio as gr
from modal_story_gen import generate_story
from modal_image_gen import generate_book_pages
from modal_tts import speak_book
from book_builder import build_book_html
import json

THEMES = ["brave adventure", "making a new friend", "overcoming a fear",
          "helping someone", "lost and found", "learning something new"]

CSS = """
body { background: #fef9e7; font-family: 'Georgia', serif; }
.book-container { max-width: 800px; margin: 0 auto; }
.book-title { font-size: 32px; text-align: center; color: #5d4037; }
.book-page { margin: 24px 0; padding: 20px; background: #fffde7; 
             border-radius: 12px; box-shadow: 3px 3px 12px rgba(0,0,0,0.15); }
.page-text { font-size: 22px; line-height: 1.9; color: #3e2723; text-align: center; }
.page-num { color: #bcaaa4; font-size: 14px; }
.gr-button-primary { background: #ff7043 !important; font-size: 20px; }
"""

def create_book(doodle_image, character_name, theme, hero_name):
    if not character_name.strip():
        character_name = "Little Hero"
    if not hero_name.strip():
        hero_name = character_name

    # Step 1: Generate story
    story = generate_story.remote(hero_name, theme, age=5)
    if "error" in story:
        return None, f"Story generation failed: {story['error']}", None

    pages = story["pages"]
    char_desc = story["character_description"]
    title = story["title"]

    scene_beats = [p["scene"] for p in pages]
    page_texts = [p["text"] for p in pages]

    # Step 2: Generate all 6 images (one Modal call, model loaded once)
    img_bytes_list = generate_book_pages.remote(char_desc, scene_beats)

    # Step 3: Assemble HTML book
    book_html = build_book_html(img_bytes_list, page_texts, title)

    # Step 4: TTS narration of full book
    full_text = f"{title}. " + " ".join(page_texts)
    audio_bytes = speak_book.remote(full_text)
    audio_path = save_wav(audio_bytes)

    return book_html, f"✅ '{title}' — 6 pages generated!", audio_path

with gr.Blocks(css=CSS, title="📚 DoodleBook") as demo:
    gr.Markdown("# 📚 DoodleBook\n*Draw a character. Get a storybook.*")

    with gr.Row():
        with gr.Column(scale=1):
            doodle = gr.Image(sources=["webcam","upload"], label="📸 Photo of your doodle", type="numpy")
            char_name = gr.Textbox(label="Character name", placeholder="Ziggy the robot")
            hero_name = gr.Textbox(label="Hero name in the story", placeholder="Ziggy")
            theme = gr.Dropdown(choices=THEMES, value=THEMES[0], label="Story theme")
            make_btn = gr.Button("✨ Make my book!", variant="primary")
            gr.Examples(
                examples=[["assets/sample_doodle.jpg", "Ziggy", "Ziggy", "brave adventure"]],
                inputs=[doodle, char_name, hero_name, theme]
            )
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            book_display = gr.HTML(label="Your storybook")
            audio_narration = gr.Audio(label="🎙️ Listen to your book", autoplay=False)

    make_btn.click(
        create_book,
        inputs=[doodle, char_name, theme, hero_name],
        outputs=[book_display, status, audio_narration]
    )

demo.launch()
```

---

## TODO 1 — Doodle style extraction for LoRA prompt conditioning
After core pipeline works: use MiniCPM-V to *describe* the uploaded doodle in visual terms ("thick black outlines, bright primary colors, stick figure proportions, sun in top corner"). Prepend this extracted style description to every FLUX prompt so the generated images actually *match* the child's drawing style, not just the character concept. This is what makes the output genuinely feel like "their" character.

## TODO 2 — PDF export + shareable link
Assemble the 6 PNG pages into a downloadable PDF using `fpdf2` or `reportlab`. Add a "Download your book as PDF" button (gr.DownloadButton). Also export the full book as a shareable HF dataset entry (with the prompts, seeds, and LoRA version used) — this earns the Open Trace badge and means families can re-generate the same book later.

---

## Sponsor + badge alignment

| Award | Why |
|---|---|
| Thousand Token Wood podium | Unique concept — nobody combines child doodle + FLUX LoRA + story |
| Black Forest Labs ($3k pool) | FLUX.2-klein + custom LoRA — near-empty sponsor field |
| OpenBMB award (Wood track) | MiniCPM5-1B (story) + VoxCPM2 (narration) |
| Well-Tuned ($badge) | Published LoRA on HF |
| Off-Brand ($1,500) | Storybook CSS with yellowed pages, serif font — zero Gradio defaults |
| Best Demo ($1,000) | Child hearing their drawing narrated as a book = perfect 60-sec video |
| Community Choice | Shareable, emotional — parents will post this |

---

## Non-negotiables
- Pre-generate and include a complete sample book (all 6 pages) in `assets/sample_book/` so judges can see what it looks like without waiting for generation
- FLUX on A100 (not A10G) — FLUX.2-klein may need 24GB+ VRAM; check Modal memory settings
- If LoRA not yet trained, the app must still run with base FLUX.2-klein (no LoRA) — degrade gracefully, note "LoRA coming" in UI
- Generation time: expect 60-90 seconds for 6 images. Show progress: "Illustrating page 1 of 6..."
- Verify exact FLUX.2-klein model ID on HF Hub before writing any import statements
