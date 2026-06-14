"""
DoodleBook — HF ZeroGPU Version

Free T4 GPU on Hugging Face Spaces!
No Modal needed.
"""

import gradio as gr
import os
import sys
import torch
try:
    import spaces
except ModuleNotFoundError:
    # `spaces` only exists on HF ZeroGPU. Off-HF (local/dev) provide a no-op so
    # the app still runs; generation then uses whatever local GPU/CPU exists.
    class _SpacesShim:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):      # bare @spaces.GPU
                return args[0]
            def deco(fn):                       # @spaces.GPU(duration=...)
                return fn
            return deco
    spaces = _SpacesShim()
import json
import time
import tempfile
import logging

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    FLUX_MODEL, STORY_MODEL, TTS_MODEL,
    GENERATION_PARAMS, SAMPLE_BOOK_PATH, BASE_SEED, page_seed,
    DEFAULT_VOICE, voice_design,
)
from book_builder import (
    build_book_html, export_pdf, magic_loader_html,
    build_coloring_html, export_coloring_pdf,
)
from ui.layout import create_layout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLOR_ART_STYLE = (
    "children's crayon storybook illustration, bold black outlines, "
    "flat bright colors, simple shapes"
)
COLOR_PAGE_SUFFIX = "full colorful background scene, the character clearly visible."
LINE_ART_STYLE = (
    "children's coloring book page, pure black ink outlines on pure white paper, "
    "clean contour lines, no color, no gray, no shading, no texture, "
    "no hatching, no pencil marks, open spaces to color"
)
LINE_ART_SUFFIX = (
    "simple clean background shapes, same composition, thick readable outlines, "
    "no filled black areas, no extra sketch marks."
)


# ============================================================================
# SAMPLE BOOK (loads instantly, no GPU needed)
# ============================================================================

SAMPLE_BOOK_HTML = None

def load_sample_book() -> str:
    """Load pre-generated sample book (C3: always ship sample)."""
    global SAMPLE_BOOK_HTML
    if SAMPLE_BOOK_HTML:
        return SAMPLE_BOOK_HTML
    
    sample_path = os.path.join(SAMPLE_BOOK_PATH, "sample.html")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            SAMPLE_BOOK_HTML = f.read()
        return SAMPLE_BOOK_HTML
    
    return "<div class='page-loading'>Loading sample book...</div>"


# ============================================================================
# ZEROGPU INFERENCE FUNCTIONS
# ============================================================================

@spaces.GPU(duration=60)
def generate_story_gpu(hero_name: str, theme: str, age: int = 5) -> dict:
    """Generate story using MiniCPM5-1B on ZeroGPU."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from modal_workers.modal_story_gen import parse_story_json, build_prompt
    
    model_id = STORY_MODEL.hub_id
    logger.info(f"Loading story model: {model_id}")
    
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda().eval()
    
    prompt = build_prompt(hero_name, theme, age)
    inputs = tok(prompt, return_tensors="pt").cuda()
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=800, do_sample=False)
    
    response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    story = parse_story_json(response)
    
    while len(story.get("pages", [])) < 6:
        story.setdefault("pages", []).append({
            "page": len(story.get("pages", [])) + 1,
            "text": "And the adventure continued happily.",
            "scene": "Continuing adventure"
        })
    
    return story


@spaces.GPU(duration=120)
def generate_images_gpu(
    character_desc: str,
    scenes: list,
    doodle_bytes: bytes = None,
    seed: int = 42,
    tiny: bool = False
) -> list:
    """Generate all 6 images using FLUX on ZeroGPU."""
    import io
    from PIL import Image
    
    if tiny:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16
        ).cuda()
        num_steps = 4
        guidance = 0.0
    else:
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL.hub_id,
            torch_dtype=torch.bfloat16
        ).cuda()
        pipe.enable_model_cpu_offload()
        num_steps = 6
        guidance = 1.0
    
    canonical = None
    if doodle_bytes:
        try:
            ref = Image.open(io.BytesIO(doodle_bytes)).convert("RGB")
            kw = dict(
                prompt=(f"Turn this child's drawing into a clean, friendly, full-body cartoon "
                        f"character for a children's storybook. Keep the EXACT same creature, "
                        f"face, and features as the drawing. {COLOR_ART_STYLE}, "
                        f"plain white background, full character visible, centered."),
                height=768, width=768, guidance_scale=guidance,
                num_inference_steps=num_steps,
                generator=torch.Generator("cuda").manual_seed(seed)
            )
            if tiny:
                kw["prompt"] = f"A friendly cartoon character, {COLOR_ART_STYLE}"
            else:
                kw["image"] = ref
            canonical = pipe(**kw).images[0]
            logger.info("Canonical character built from doodle")
        except Exception as e:
            logger.warning(f"Canonical build failed ({e}); text2img fallback")
            canonical = None
    
    images = []
    for i, scene in enumerate(scenes):
        if canonical is not None and not tiny:
            prompt = f"The same character. {scene}. {COLOR_ART_STYLE}, {COLOR_PAGE_SUFFIX}"
            kw = dict(image=canonical, prompt=prompt)
        else:
            prompt = (
                f"{character_desc}. Scene: {scene}. {COLOR_ART_STYLE}, "
                f"white background, centered, full character visible"
            )
            kw = dict(prompt=prompt)
        
        kw.update(dict(
            height=768, width=768, guidance_scale=guidance,
            num_inference_steps=num_steps,
            generator=torch.Generator("cuda").manual_seed(seed + i + 1)
        ))
        
        image = pipe(**kw).images[0]
        images.append(image)
        logger.info(f"Generated page {i+1}/6")
    
    return images


@spaces.GPU(duration=120)
def generate_coloring_images_gpu(
    character_desc: str,
    scenes: list,
    doodle_bytes: bytes = None,
    seed: int = 42,
    tiny: bool = False
) -> list:
    """Generate coloring pages directly with FLUX instead of tracing color pages."""
    import io
    from PIL import Image

    if tiny:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16
        ).cuda()
        num_steps = 4
        guidance = 0.0
    else:
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL.hub_id,
            torch_dtype=torch.bfloat16
        ).cuda()
        pipe.enable_model_cpu_offload()
        num_steps = 6
        guidance = 1.0

    canonical = None
    if doodle_bytes:
        try:
            ref = Image.open(io.BytesIO(doodle_bytes)).convert("RGB")
            kw = dict(
                prompt=(f"Turn this child's drawing into a clean, friendly, full-body cartoon "
                        f"character for a children's coloring book. Keep the EXACT same creature, "
                        f"face, and features as the drawing. {LINE_ART_STYLE}, "
                        f"plain white background, full character visible, centered."),
                height=768, width=768, guidance_scale=guidance,
                num_inference_steps=num_steps,
                generator=torch.Generator('cuda').manual_seed(seed)
            )
            if tiny:
                kw["prompt"] = f"A friendly cartoon character, {LINE_ART_STYLE}"
            else:
                kw["image"] = ref
            canonical = pipe(**kw).images[0]
            logger.info("Line-art canonical character built from doodle")
        except Exception as e:
            logger.warning(f"Line-art canonical build failed ({e}); text2img fallback")
            canonical = None

    images = []
    for i, scene in enumerate(scenes):
        if canonical is not None and not tiny:
            prompt = f"The same character. {scene}. {LINE_ART_STYLE}, {LINE_ART_SUFFIX}"
            kw = dict(image=canonical, prompt=prompt)
        else:
            prompt = (
                f"{character_desc}. Scene: {scene}. {LINE_ART_STYLE}, "
                f"white background, centered, full character visible"
            )
            kw = dict(prompt=prompt)

        kw.update(dict(
            height=768, width=768, guidance_scale=guidance,
            num_inference_steps=num_steps,
            generator=torch.Generator("cuda").manual_seed(seed + i + 101)
        ))

        image = pipe(**kw).images[0]
        images.append(image)
        logger.info(f"Generated coloring page {i+1}/6")

    return images


@spaces.GPU(duration=30)
def generate_tts_gpu(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Generate TTS using VoxCPM2 on ZeroGPU with the chosen voice preset."""
    import io
    import numpy as np

    try:
        from voxcpm import VoxCPM
        model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

        design = voice_design(voice)

        import re
        chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not chunks:
            chunks = [text.strip() or "The end."]
        
        sr = model.tts_model.sample_rate
        pause = np.zeros(int(sr * 0.35), dtype=np.float32)
        pieces = []
        
        for i, sentence in enumerate(chunks):
            wav = model.generate(
                text=f"{design} {sentence}",
                cfg_value=2.0,
                inference_timesteps=10,
            )
            pieces.append(np.asarray(wav, dtype=np.float32))
            if i < len(chunks) - 1:
                pieces.append(pause)
        
        audio = np.concatenate(pieces)
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()
    
    except Exception as e:
        logger.warning(f"VoxCPM2 TTS failed: {e}, using local fallback")
        from modal_workers.modal_tts import speak_book_local
        return speak_book_local(text, voice)


# ============================================================================
# MAIN BOOK CREATION (Generator for streaming)
# ============================================================================

def create_book(doodle_image, character_name, theme, hero_name, tiny_mode=False, voice=DEFAULT_VOICE, make_coloring=False):
    """Create book with streaming progress + magic loader + optional coloring book."""
    if not character_name or not character_name.strip():
        character_name = "Little Hero"
    if not hero_name or not hero_name.strip():
        hero_name = character_name
    
    trace_data = {
        "hero_name": hero_name,
        "theme": theme,
        "tiny_mode": tiny_mode,
        "voice": voice,
        "make_coloring": make_coloring,
        "seed": BASE_SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Stage 1: Story
    loader = magic_loader_html("story", hero_name)
    yield (
        loader,
        "Generating story with MiniCPM5-1B...",
        None, None, {}, "", json.dumps(trace_data, indent=2),
        gr.update(visible=False), gr.update(visible=False),
    )
    
    try:
        story = generate_story_gpu(hero_name, theme)
    except Exception as e:
        logger.error(f"Story generation failed: {e}")
        yield (f"<div class='page-loading'>Error: {e}</div>", f"Error: {e}",
               None, None, {}, "", "",
               gr.update(visible=False), gr.update(visible=False))
        return
    
    pages = story.get("pages", [])
    char_desc = story.get("character_description", "")
    title = story.get("title", "Untitled Story")
    page_texts = [p.get("text", "") for p in pages]
    scenes = [p.get("scene", "") for p in pages]
    
    trace_data["title"] = title
    trace_data["character_description"] = char_desc
    
    # Stage 2: Images
    loader = magic_loader_html("images", hero_name)
    yield (
        loader,
        f"Story: {title} — Illustrating...",
        None, None, story, "", json.dumps(trace_data, indent=2),
        gr.update(visible=False), gr.update(visible=False),
    )
    
    doodle_bytes = None
    if doodle_image is not None:
        import io
        from PIL import Image
        img = Image.fromarray(doodle_image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        doodle_bytes = buf.getvalue()
    
    try:
        images = generate_images_gpu(char_desc, scenes, doodle_bytes, BASE_SEED, tiny_mode)
        import io
        img_bytes = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes.append(buf.getvalue())
        engine = "flux"
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        from services.images import generate_placeholder_images
        img_bytes = generate_placeholder_images(char_desc, scenes, doodle_bytes)
        engine = "sketch"
    
    book_html = build_book_html(img_bytes, page_texts, title, engine)
    
    # Stage 3: TTS
    loader = magic_loader_html("tts", hero_name)
    yield (
        loader,
        f"Story: {title} — Recording narration...",
        None, None, story, "", json.dumps(trace_data, indent=2),
        gr.update(visible=False), gr.update(visible=False),
    )
    
    audio_path = None
    try:
        full_text = f"{title}. {' '.join(page_texts)}"
        audio_bytes = generate_tts_gpu(full_text, voice)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name
    except Exception as e:
        logger.warning(f"TTS failed: {e}")
    
    # Stage 4: PDFs
    pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = export_pdf(img_bytes, page_texts, title, tmp.name)
    except Exception as e:
        logger.warning(f"PDF failed: {e}")
    
    # Coloring book: render the SAME scenes directly as line art with FLUX.
    # Fall back to the older trace-from-color path only if that render fails.
    coloring_html = ""
    coloring_pdf_path = None
    if make_coloring:
        try:
            from services.coloring import _crispen
            coloring_images = generate_coloring_images_gpu(
                char_desc, scenes, doodle_bytes, BASE_SEED, tiny_mode
            )
            import io
            outlines = []
            for img in coloring_images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                outlines.append(_crispen(buf.getvalue()))
            coloring_html = build_coloring_html(outlines, page_texts, title)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                coloring_pdf_path = export_coloring_pdf(outlines, page_texts, title, tmp.name)
            trace_data["coloring_book"] = True
            trace_data["coloring_engine"] = "flux-direct-lineart"
        except Exception as e:
            logger.warning(f"Direct FLUX coloring book failed ({e}); using traced fallback")
            try:
                from services.coloring import derive_coloring_pages
                outlines = derive_coloring_pages(img_bytes)
                coloring_html = build_coloring_html(outlines, page_texts, title)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    coloring_pdf_path = export_coloring_pdf(outlines, page_texts, title, tmp.name)
                trace_data["coloring_book"] = True
                trace_data["coloring_engine"] = "trace-fallback"
            except Exception as e2:
                logger.warning(f"Coloring book fallback failed: {e2}")
    
    trace_data["completed"] = True
    trace_data["pages_generated"] = len(img_bytes)
    trace_data["engine"] = engine

    pdf_update = (gr.update(visible=True, value=pdf_path) if pdf_path
                  else gr.update(visible=False))
    coloring_pdf_update = (gr.update(visible=True, value=coloring_pdf_path) if coloring_pdf_path
                           else gr.update(visible=False))
    coloring_display_update = (gr.update(visible=True, value=coloring_html) if coloring_html
                               else gr.update(visible=False))

    yield (
        book_html,
        f"Complete: {title} — 6 pages illustrated!",
        audio_path,
        pdf_update,
        story,
        f"Pages: {len(img_bytes)} | Seed: {BASE_SEED} | Mode: {'Tiny' if tiny_mode else 'Standard'} | Engine: {engine}",
        json.dumps(trace_data, indent=2),
        coloring_display_update,
        coloring_pdf_update,
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_layout(
        load_sample_fn=load_sample_book,
        create_book_fn=create_book,
    )
    demo.launch(server_port=7870, share=False)
