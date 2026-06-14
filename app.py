"""
DoodleBook — HF ZeroGPU Version

Free T4 GPU on Hugging Face Spaces!
No Modal needed.

Ports all improvements from run_modal.py:
- Parallel TTS (voice starts concurrently with images via threading)
- Heartbeat streaming keeps the browser alive during long GPU calls
- Queue with concurrency limits
- allowed_paths so PDF downloads actually work
- Full per-stage timing in trace
"""

import gradio as gr
import io
import json
import os
import sys
import tempfile
import threading
import time
import logging

import torch
try:
    import spaces
except ModuleNotFoundError:
    class _SpacesShim:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def deco(fn):
                return fn
            return deco
    spaces = _SpacesShim()

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
# HEARTBEAT HELPER — keeps Gradio SSE alive during long GPU calls
# ============================================================================

def _with_heartbeat(blocking_fn, frame_fn, poll=4.0):
    """
    Run blocking_fn() in a thread while pumping frame_fn(elapsed) heartbeats
    into the Gradio stream every `poll` seconds.

    Yields ("hb", <frame tuple>) heartbeats, then ("done", <return value>).
    Re-raises whatever blocking_fn raised.
    """
    box = {}

    def _run():
        try:
            box["val"] = blocking_fn()
        except BaseException as e:
            box["err"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    t0 = time.time()
    while th.is_alive():
        th.join(timeout=poll)
        if th.is_alive():
            yield ("hb", frame_fn(int(time.time() - t0)))
    if "err" in box:
        raise box["err"]
    yield ("done", box["val"])


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


@spaces.GPU(duration=180)
def generate_images_gpu(
    character_desc: str,
    scenes: list,
    doodle_bytes: bytes = None,
    seed: int = 42,
    tiny: bool = False
) -> list:
    """Generate all 6 images using FLUX on ZeroGPU."""
    from PIL import Image

    if tiny:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo", torch_dtype=torch.float16
        ).cuda()
        num_steps = 4
        guidance = 0.0
    else:
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL.hub_id, torch_dtype=torch.bfloat16
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


@spaces.GPU(duration=180)
def generate_coloring_images_gpu(
    character_desc: str,
    scenes: list,
    doodle_bytes: bytes = None,
    seed: int = 42,
    tiny: bool = False
) -> list:
    """Generate coloring pages directly with FLUX line-art renders."""
    from PIL import Image

    if tiny:
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo", torch_dtype=torch.float16
        ).cuda()
        num_steps = 4
        guidance = 0.0
    else:
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL.hub_id, torch_dtype=torch.bfloat16
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


@spaces.GPU(duration=60)
def generate_tts_gpu(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Generate TTS using VoxCPM2 on ZeroGPU with the chosen voice preset."""
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
# MAIN BOOK CREATION — streaming generator with parallel TTS + heartbeat
# ============================================================================

def create_book(doodle_image, character_name, theme, hero_name,
                tiny_mode=False, voice=DEFAULT_VOICE, make_coloring=False):
    """
    Create book with streaming progress.

    Key improvements over the old app.py:
    - TTS starts in a thread CONCURRENTLY with image generation
    - Heartbeat frames keep the browser SSE alive during long GPU calls
    - Full per-stage timing in the trace panel
    """
    t_total = time.perf_counter()
    character_name = (character_name or "").strip() or "Little Hero"
    hero_name = (hero_name or "").strip() or character_name

    _no = gr.update(visible=False)
    _keep = gr.update()  # no-op: leave the element as-is

    trace = {
        "backend": "zerogpu",
        "hero_name": hero_name,
        "theme": theme,
        "voice": voice,
        "tiny_mode": tiny_mode,
        "make_coloring": make_coloring,
        "seed": BASE_SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ---- 1) STORY ----
    yield (magic_loader_html("story", hero_name),
           "Writing the story\u2026", None, _keep, {}, "",
           json.dumps(trace, indent=2), _no, _keep)

    t_story = time.perf_counter()
    try:
        story = generate_story_gpu(hero_name, theme)
    except Exception as e:
        logger.error(f"Story generation failed: {e}")
        yield (f"<div class='page-loading'>Error: {e}</div>", f"Error: {e}",
               None, _keep, {}, "", "", _no, _no)
        return
    trace["story_sec"] = round(time.perf_counter() - t_story, 2)

    pages = story.get("pages", [])
    char_desc = story.get("character_description", "")
    title = story.get("title", "Untitled Story")
    page_texts = [p.get("text", "") for p in pages]
    scenes = [p.get("scene", "") for p in pages]
    trace.update(title=title, character_description=char_desc)

    # ---- 2) TTS starts NOW, concurrently with images ----
    #    Voice only needs the text (which is ready), so its ~30-60s overlaps
    #    the image render for free.
    voice_box = {}
    full_text = f"{title}. {' '.join(page_texts)}"
    t_voice = time.perf_counter()

    def _do_voice():
        try:
            voice_box["bytes"] = generate_tts_gpu(full_text, voice)
        except Exception as e:
            voice_box["err"] = e

    voice_thread = threading.Thread(target=_do_voice, daemon=True)
    voice_thread.start()

    # ---- 3) IMAGES (with heartbeat so browser stays alive) ----
    doodle_bytes = None
    if doodle_image is not None:
        from PIL import Image
        img = Image.fromarray(doodle_image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        doodle_bytes = buf.getvalue()

    yield (magic_loader_html("images", hero_name),
           f"{title} \u2014 illustrating\u2026 (voice recording in parallel)",
           None, _keep, story, "", json.dumps(trace, indent=2), _no, _keep)

    t_images = time.perf_counter()
    img_bytes, engine = None, "sketch"
    for kind, payload in _with_heartbeat(
        lambda: _run_images(char_desc, scenes, doodle_bytes, tiny_mode),
        lambda s: (magic_loader_html("images", hero_name),
                   f"{title} \u2014 illustrating\u2026 {s}s  (voice recording in parallel)",
                   None, _keep, story, "", json.dumps(trace, indent=2), _no, _keep),
    ):
        if kind == "hb":
            yield payload
        else:
            img_bytes, engine = payload
    trace["images_sec"] = round(time.perf_counter() - t_images, 2)

    book_html = build_book_html(img_bytes, page_texts, title, engine)

    # ---- 4) Collect the parallel VOICE result ----
    while voice_thread.is_alive():
        voice_thread.join(timeout=4)
        if voice_thread.is_alive():
            yield (book_html, f"{title} \u2014 finishing narration\u2026",
                   None, _keep, story, "", json.dumps(trace, indent=2), _no, _keep)

    audio_path = None
    trace["tts_sec"] = round(time.perf_counter() - t_voice, 2)
    if voice_box.get("bytes"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(voice_box["bytes"])
                audio_path = tmp.name
        except Exception as e:
            logger.warning(f"writing audio failed: {e}")
    elif "err" in voice_box:
        logger.warning(f"TTS failed: {voice_box['err']}")

    # ---- 5) PDF ----
    pdf_path = None
    t_pdf = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = export_pdf(img_bytes, page_texts, title, tmp.name)
    except Exception as e:
        logger.warning(f"PDF failed: {e}")
    trace["pdf_sec"] = round(time.perf_counter() - t_pdf, 2)

    # ---- 6) COLORING BOOK ----
    coloring_html = ""
    coloring_pdf_path = None
    if make_coloring:
        t_coloring = time.perf_counter()
        for kind, payload in _with_heartbeat(
            lambda: _run_coloring(char_desc, scenes, doodle_bytes, tiny_mode, img_bytes),
            lambda s: (
                book_html,
                f"{title} \u2014 building coloring book\u2026 {s}s",
                audio_path, _keep, story, "",
                json.dumps(trace, indent=2), _no, _keep,
            ),
        ):
            if kind == "hb":
                yield payload
            else:
                outlines, coloring_engine = payload
        try:
            coloring_html = build_coloring_html(outlines, page_texts, title)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                coloring_pdf_path = export_coloring_pdf(outlines, page_texts, title, tmp.name)
            trace["coloring_book"] = True
            trace["coloring_engine"] = coloring_engine
        except Exception as e:
            logger.warning(f"Coloring book failed: {e}")
        trace["coloring_sec"] = round(time.perf_counter() - t_coloring, 2)

    # ---- FINAL YIELD ----
    trace["completed"] = True
    trace["pages_generated"] = len(img_bytes)
    trace["engine"] = engine
    trace["total_sec"] = round(time.perf_counter() - t_total, 2)

    engine_label = "FLUX (ZeroGPU)" if engine == "flux" else "local sketch fallback"
    pdf_update = gr.update(value=pdf_path) if pdf_path else _keep
    coloring_pdf_update = gr.update(value=coloring_pdf_path) if coloring_pdf_path else _keep
    coloring_display_update = (gr.update(visible=True, value=coloring_html) if coloring_html
                               else gr.update(visible=False))

    yield (
        book_html,
        f"Complete: {title} \u2014 {len(img_bytes)} pages \u00b7 {engine_label} \u00b7 voice: {voice} \u00b7 total {trace['total_sec']}s",
        audio_path,
        pdf_update,
        story,
        f"Pages: {len(img_bytes)} | Seed: {BASE_SEED} | "
        f"Mode: {'Tiny' if tiny_mode else 'Standard'} | Engine: {engine} | "
        f"Story {trace.get('story_sec', 0)}s | Images {trace.get('images_sec', 0)}s | "
        f"PDF {trace.get('pdf_sec', 0)}s | Coloring {trace.get('coloring_sec', 0)}s",
        json.dumps(trace, indent=2),
        coloring_display_update,
        coloring_pdf_update,
    )


def _run_images(char_desc, scenes, doodle_bytes, tiny_mode):
    """Wrapper around image GPU call — returns (img_bytes, engine)."""
    try:
        images = generate_images_gpu(char_desc, scenes, doodle_bytes, BASE_SEED, tiny_mode)
        img_bytes = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes.append(buf.getvalue())
        return img_bytes, "flux"
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        from services.images import generate_placeholder_images
        return generate_placeholder_images(char_desc, scenes, doodle_bytes), "sketch"


def _run_coloring(char_desc, scenes, doodle_bytes, tiny_mode, color_img_bytes):
    """Wrapper around coloring GPU call — returns (outlines, engine)."""
    try:
        from services.coloring import _crispen
        coloring_images = generate_coloring_images_gpu(
            char_desc, scenes, doodle_bytes, BASE_SEED, tiny_mode
        )
        outlines = []
        for img in coloring_images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            outlines.append(_crispen(buf.getvalue()))
        return outlines, "flux-direct-lineart"
    except Exception as e:
        logger.warning(f"Direct FLUX coloring failed ({e}); using traced fallback")
        from services.coloring import derive_coloring_pages
        return derive_coloring_pages(color_img_bytes), "trace-fallback"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_layout(
        load_sample_fn=load_sample_book,
        create_book_fn=create_book,
    )
    demo.queue(default_concurrency_limit=4, max_size=32)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7870,
        share=False,
        allowed_paths=[tempfile.gettempdir()],
    )
