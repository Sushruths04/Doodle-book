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
import struct
import re

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

# ZeroGPU sets SPACES_ZERO_GPU. On the Space we load models on cuda at IMPORT
# (a CUDA-emulation layer makes that work without a real GPU); lazy-loading
# inside @spaces.GPU is explicitly discouraged and was why FLUX kept failing
# → sketch. Guarded so a local/dev import doesn't try to pull ~20GB of weights.
ON_ZEROGPU = bool(os.environ.get("SPACES_ZERO_GPU"))

_FLUX_PIPE = None
_STORY_MODEL = None
_STORY_TOKENIZER = None
_TTS_MODEL = None
_LOAD_ERRORS = {}


def load_flux():
    """FLUX image pipeline placed on cuda at module scope (the ZeroGPU pattern).
    No enable_model_cpu_offload() — that fights ZeroGPU's device management."""
    global _FLUX_PIPE
    if _FLUX_PIPE is None:
        from diffusers import Flux2KleinPipeline
        logger.info(f"Loading image model: {FLUX_MODEL.hub_id}")
        pipe = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL.hub_id, torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
        _FLUX_PIPE = pipe
    return _FLUX_PIPE


def load_story():
    global _STORY_MODEL, _STORY_TOKENIZER
    if _STORY_MODEL is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"Loading story model: {STORY_MODEL.hub_id}")
        _STORY_TOKENIZER = AutoTokenizer.from_pretrained(
            STORY_MODEL.hub_id, trust_remote_code=True,
        )
        _STORY_MODEL = AutoModelForCausalLM.from_pretrained(
            STORY_MODEL.hub_id, torch_dtype=torch.float16, trust_remote_code=True,
        ).to("cuda").eval()
    return _STORY_MODEL, _STORY_TOKENIZER


def load_tts():
    global _TTS_MODEL
    if _TTS_MODEL is None:
        from voxcpm import VoxCPM
        logger.info(f"Loading TTS model: {TTS_MODEL.hub_id}")
        _TTS_MODEL = VoxCPM.from_pretrained(TTS_MODEL.hub_id, load_denoiser=False)
    return _TTS_MODEL


if ON_ZEROGPU:
    for _name, _loader in (("flux", load_flux), ("story", load_story), ("tts", load_tts)):
        try:
            _loader()
        except Exception as _e:                       # keep the Space booting
            _LOAD_ERRORS[_name] = repr(_e)
            logger.exception(f"Module-level load failed for {_name}")

COLOR_ART_STYLE = (
    "hand-drawn crayon children's storybook illustration, soft waxy crayon "
    "texture and visible crayon strokes, warm colorful crayon shading, "
    "simple friendly shapes, looks drawn by hand with crayons"
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

THEME_TEMPLATES = {
    "brave adventure": [
        ("{hero} loved exploring new places.", "{hero} standing at the start of a bright adventure trail"),
        ("One morning, {hero} discovered something glowing nearby.", "{hero} spotting a magical glow in the distance"),
        ("Taking a deep breath, {hero} bravely went closer.", "{hero} walking forward with courage"),
        ("There, a new friend needed help.", "{hero} finding a small friend in trouble"),
        ("{hero} helped with kindness and a clever idea.", "{hero} helping the friend together"),
        ("Everyone cheered, and {hero} felt proud and brave.", "{hero} celebrating at sunset with the new friend"),
    ],
    "making a new friend": [
        ("{hero} was playing alone in a sunny place.", "{hero} playing under a bright sky"),
        ("Then {hero} noticed someone shy nearby.", "{hero} seeing a shy new friend nearby"),
        ("{hero} smiled and said hello.", "{hero} waving with a friendly smile"),
        ("Soon they were sharing stories and laughs.", "{hero} and the new friend laughing together"),
        ("They played games all afternoon.", "{hero} and the new friend playing together"),
        ("By sunset, {hero} had made a wonderful new friend.", "{hero} and the new friend smiling together at sunset"),
    ],
}

FEW_SHOT_EXEMPLAR = """
Write a 6-page children's storybook for age 5 about Luna the cat with theme: brave adventure.

Return ONLY valid JSON:
{
  "title": "Luna's Brave Adventure",
  "character_description": "A small orange tabby cat named Luna with big green eyes, whiskers, and a tiny red scarf",
  "pages": [
    {"page": 1, "text": "Luna was a small orange cat who loved to explore.", "scene": "Luna sitting by the window looking outside"},
    {"page": 2, "text": "One sunny morning, Luna saw something sparkling in the forest.", "scene": "Luna spotting a glow in the trees"},
    {"page": 3, "text": "Bravely, Luna crept into the forest to investigate.", "scene": "Luna walking cautiously through trees"},
    {"page": 4, "text": "It was a tiny fairy stuck in a spider web!", "scene": "Luna discovering a fairy in trouble"},
    {"page": 5, "text": "Luna gently freed the fairy with her paw.", "scene": "Luna carefully helping the fairy"},
    {"page": 6, "text": "The fairy thanked Luna and they became friends forever.", "scene": "Luna and fairy playing together at sunset"}
  ]
}
"""


def build_story_prompt(hero_name: str, theme: str, age: int) -> str:
    return f"""{FEW_SHOT_EXEMPLAR}

Write a 6-page children's storybook for age {age} about {hero_name} with theme: {theme}.

Return ONLY valid JSON:
"""


def _validate_story_structure(story: dict) -> bool:
    required_keys = ["title", "character_description", "pages"]
    if not all(k in story for k in required_keys):
        return False
    pages = story.get("pages", [])
    if not isinstance(pages, list) or len(pages) < 1:
        return False
    first_page = pages[0]
    return all(k in first_page for k in ["page", "text", "scene"])


def _repair_json(json_str: str) -> str:
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
    json_str = re.sub(r'(?<=")\n(?=")', '\\n', json_str)
    json_str = re.sub(r'(\s)(\w+)(\s*:)', r'\1"\2"\3', json_str)
    return json_str


def parse_story_json(raw_output: str) -> dict | None:
    match = re.search(r'\{[\s\S]*\}', raw_output or "")
    if not match:
        return None
    raw_json = match.group(0)
    for candidate in (raw_json, _repair_json(raw_json)):
        try:
            story = json.loads(candidate)
            if _validate_story_structure(story):
                return story
        except Exception:
            continue
    return None


def _normalize_story(story: dict) -> dict:
    pages = list(story.get("pages", []))[:6]
    while len(pages) < 6:
        pages.append({
            "page": len(pages) + 1,
            "text": "And the adventure continued happily.",
            "scene": "Continuing adventure",
        })
    story["pages"] = pages
    story.setdefault("title", "A Wonderful Adventure")
    story.setdefault(
        "character_description",
        "A friendly children's storybook hero with bright colors and cheerful features",
    )
    return story


def build_story_locally(hero_name: str, theme: str) -> dict:
    """Fast, deterministic fallback story that avoids any Modal dependency."""
    hero = (hero_name or "Little Hero").strip() or "Little Hero"
    beats = THEME_TEMPLATES.get(theme, THEME_TEMPLATES["brave adventure"])
    pages = [
        {"page": i + 1, "text": text.format(hero=hero), "scene": scene.format(hero=hero)}
        for i, (text, scene) in enumerate(beats)
    ]
    return {
        "title": f"{hero}'s Storybook Adventure",
        "character_description": (
            f"{hero}, a friendly children's storybook hero with bright colors, "
            "bold outlines, and a cheerful expressive face"
        ),
        "pages": pages,
    }


def silent_wav_bytes(duration_seconds: int = 2, sample_rate: int = 24000) -> bytes:
    """Return a short silent WAV so the UI remains stable if TTS is unavailable."""
    num_samples = sample_rate * duration_seconds
    data_size = num_samples * 2
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    )
    return header + (b"\x00" * data_size)


def _with_heartbeat(blocking_fn, frame_fn, poll=4.0):
    import threading

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
    """Generate a story on ZeroGPU, falling back to a deterministic local story."""
    try:
        model, tok = load_story()
        prompt = build_story_prompt(hero_name, theme, age)
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=GENERATION_PARAMS.max_story_tokens,
                do_sample=False,
            )
        response = tok.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        parsed = parse_story_json(response)
        if parsed:
            return _normalize_story(parsed)
        logger.warning("Story parser failed; using deterministic local fallback")
    except Exception as e:
        logger.warning(f"ZeroGPU story generation failed: {e}")
    return _normalize_story(build_story_locally(hero_name, theme))


@spaces.GPU(duration=150)
def generate_images_gpu(
    character_desc: str,
    scenes: list,
    doodle_bytes: bytes = None,
    seed: int = 42,
) -> list:
    """Generate all story pages with FLUX on ZeroGPU (two-stage: canonical
    character from the doodle, then the same character in each scene)."""
    import io
    from PIL import Image

    pipe = load_flux()
    num_steps, guidance = 6, 1.0

    canonical = None
    if doodle_bytes:
        try:
            ref = Image.open(io.BytesIO(doodle_bytes)).convert("RGB")
            canonical = pipe(
                prompt=(f"Turn this child's drawing into a clean, friendly, full-body cartoon "
                        f"character for a children's storybook. Keep the EXACT same creature, "
                        f"face, and features as the drawing. {COLOR_ART_STYLE}, "
                        f"plain white background, full character visible, centered."),
                image=ref, height=768, width=768, guidance_scale=guidance,
                num_inference_steps=num_steps,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            logger.info("Canonical character built from doodle")
        except Exception as e:
            logger.warning(f"Canonical build failed ({e}); text2img fallback")
            canonical = None

    images = []
    for i, scene in enumerate(scenes):
        if canonical is not None:
            prompt = f"The same character. {scene}. {COLOR_ART_STYLE}, {COLOR_PAGE_SUFFIX}"
            kw = dict(image=canonical, prompt=prompt)
        else:
            prompt = (f"{character_desc}. Scene: {scene}. {COLOR_ART_STYLE}, "
                      f"white background, centered, full character visible")
            kw = dict(prompt=prompt)
        kw.update(height=768, width=768, guidance_scale=guidance,
                  num_inference_steps=num_steps,
                  generator=torch.Generator("cuda").manual_seed(seed + i + 1))
        images.append(pipe(**kw).images[0])
        logger.info(f"Generated page {i+1}/{len(scenes)}")
    return images


@spaces.GPU(duration=150)
def generate_coloring_images_gpu(color_pngs: list, seed: int = 7) -> list:
    """Coloring pages = FLUX redraws each finished COLOR page as clean line art
    (img2img). This MATCHES the storybook composition and avoids the speckly
    look of tracing crayon texture. Caller crispens the result to black-on-white."""
    import io
    from PIL import Image

    pipe = load_flux()
    prompt = f"{LINE_ART_STYLE}, {LINE_ART_SUFFIX}"
    outs = []
    for i, png in enumerate(color_pngs):
        ref = Image.open(io.BytesIO(png)).convert("RGB")
        base = dict(prompt=prompt, image=ref, height=768, width=768,
                    guidance_scale=1.0, num_inference_steps=6,
                    generator=torch.Generator("cuda").manual_seed(seed + i))
        try:                                  # strength may not be accepted
            img = pipe(**base, strength=0.85).images[0]
        except TypeError:
            img = pipe(**base).images[0]
        outs.append(img)
        logger.info(f"Generated coloring page {i+1}/{len(color_pngs)}")
    return outs


@spaces.GPU(duration=120)
def generate_tts_gpu(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Narrate the book with VoxCPM2. Raises on failure so the caller can show
    the real reason instead of silently shipping a silent clip."""
    import io
    import numpy as np

    try:
        model = load_tts()
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
        # Surface the real reason (e.g. missing model) instead of a silent clip
        # that looks like it worked. create_book records this in the trace.
        logger.exception("TTS failed")
        raise


# ============================================================================
# MAIN BOOK CREATION (Generator for streaming)
# ============================================================================

def create_book(doodle_image, character_name, theme, hero_name, voice=DEFAULT_VOICE, make_coloring=False):
    """ZeroGPU book flow: story → images → narration → PDFs → coloring book,
    each a sequential @spaces.GPU call (ZeroGPU has one GPU per request)."""
    t_total = time.perf_counter()
    character_name = (character_name or "").strip() or "Little Hero"
    hero_name = (hero_name or "").strip() or character_name

    trace_data = {
        "backend": "zerogpu",
        "hero_name": hero_name,
        "theme": theme,
        "voice": voice,
        "make_coloring": make_coloring,
        "seed": BASE_SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if _LOAD_ERRORS:
        trace_data["model_load_errors"] = _LOAD_ERRORS
    
    _no = gr.update(visible=False)
    _keep = gr.update()

    yield (
        magic_loader_html("story", hero_name),
        "Writing the story…",
        None, _keep, {}, "", json.dumps(trace_data, indent=2),
        _no, _keep,
    )

    t_story = time.perf_counter()
    try:
        story = generate_story_gpu(hero_name, theme)
    except Exception as e:
        logger.error(f"Story generation failed: {e}")
        yield (
            f"<div class='page-loading'>Error: {e}</div>",
            f"Error: {e}",
            None, _keep, {}, "", "",
            _no, _keep,
        )
        return
    trace_data["story_sec"] = round(time.perf_counter() - t_story, 2)

    pages = story.get("pages", [])
    char_desc = story.get("character_description", "")
    title = story.get("title", "Untitled Story")
    page_texts = [p.get("text", "") for p in pages]
    scenes = [p.get("scene", "") for p in pages]
    
    trace_data["title"] = title
    trace_data["character_description"] = char_desc
    
    yield (
        magic_loader_html("images", hero_name),
        f"{title} — illustrating on ZeroGPU…",
        None, _keep, story, "", json.dumps(trace_data, indent=2),
        _no, _keep,
    )

    doodle_bytes = None
    if doodle_image is not None:
        import io
        from PIL import Image
        img = Image.fromarray(doodle_image)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        doodle_bytes = buf.getvalue()

    full_text = f"{title}. {' '.join(page_texts)}"

    # ---- NARRATION starts NOW, in PARALLEL with the images (it only needs the
    #      story text). The audio is surfaced the moment it's ready — usually
    #      before the illustrations finish — so narration appears first. ----
    import threading
    voice_box = {}
    t_tts = time.perf_counter()

    def _do_voice():
        try:
            voice_box["bytes"] = generate_tts_gpu(full_text, voice)
        except Exception as e:
            voice_box["err"] = e

    voice_thread = threading.Thread(target=_do_voice, daemon=True)
    voice_thread.start()

    def _audio_now():
        """Write the narration to a temp wav once it's ready; return its path."""
        if voice_box.get("bytes") and not voice_box.get("path"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(voice_box["bytes"])
                    voice_box["path"] = tmp.name
            except Exception as e:
                logger.warning(f"writing audio failed: {e}")
        return voice_box.get("path")

    # ---- IMAGES (FLUX on ZeroGPU), surfacing the narration as soon as it lands ----
    img_bytes, engine, images = None, "sketch", None
    t_images = time.perf_counter()
    try:
        for kind, payload in _with_heartbeat(
            lambda: generate_images_gpu(char_desc, scenes, doodle_bytes, BASE_SEED),
            lambda s: (
                magic_loader_html("images", hero_name),
                f"{title} — illustrating… {s}s"
                + ("  ·  narration ready ▶" if _audio_now() else "  ·  recording narration…"),
                _audio_now(), _keep, story, "", json.dumps(trace_data, indent=2), _no, _keep,
            ),
        ):
            if kind == "hb":
                yield payload
            else:
                images = payload
        import io
        img_bytes = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes.append(buf.getvalue())
        engine = "flux"
    except Exception as e:
        logger.exception("Image generation failed")
        trace_data["image_error"] = repr(e)
        from services.images import generate_placeholder_images
        img_bytes = generate_placeholder_images(char_desc, scenes, doodle_bytes)
        engine = "sketch"
    trace_data["images_sec"] = round(time.perf_counter() - t_images, 2)
    trace_data["engine"] = engine

    book_html = build_book_html(img_bytes, page_texts, title, engine)

    # ---- collect the parallel narration (usually already finished) ----
    while voice_thread.is_alive():
        voice_thread.join(timeout=4)
        if voice_thread.is_alive():
            yield (book_html, f"{title} — finishing narration…",
                   _audio_now(), _keep, story, "", json.dumps(trace_data, indent=2),
                   _no, _keep)
    audio_path = _audio_now()
    if voice_box.get("err"):
        logger.warning(f"TTS failed: {voice_box['err']}")
        trace_data["tts_error"] = repr(voice_box["err"])
    trace_data["tts_sec"] = round(time.perf_counter() - t_tts, 2)

    pdf_path = None
    t_pdf = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = export_pdf(img_bytes, page_texts, title, tmp.name)
    except Exception as e:
        logger.warning(f"PDF failed: {e}")
    trace_data["pdf_sec"] = round(time.perf_counter() - t_pdf, 2)

    coloring_html = ""
    coloring_pdf_path = None
    if make_coloring:
        t_coloring = time.perf_counter()
        try:
            from services.coloring import _crispen
            for kind, payload in _with_heartbeat(
                lambda: generate_coloring_images_gpu(img_bytes, 7),
                lambda s: (
                    book_html,
                    f"{title} — building coloring book… {s}s",
                    audio_path,
                    _keep,
                    story,
                    "",
                    json.dumps(trace_data, indent=2),
                    _no,
                    _keep,
                ),
            ):
                if kind == "hb":
                    yield payload
                else:
                    coloring_images = payload
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
            logger.exception("FLUX line-art coloring failed; using local trace fallback")
            trace_data["coloring_error"] = repr(e)
            try:
                from services.coloring import _to_line_art_opencv
                outlines = [_to_line_art_opencv(b) for b in img_bytes]
                coloring_html = build_coloring_html(outlines, page_texts, title)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    coloring_pdf_path = export_coloring_pdf(outlines, page_texts, title, tmp.name)
                trace_data["coloring_book"] = True
                trace_data["coloring_engine"] = "opencv-trace-fallback"
            except Exception as e2:
                logger.exception("Coloring fallback also failed")
                trace_data["coloring_error2"] = repr(e2)
        trace_data["coloring_sec"] = round(time.perf_counter() - t_coloring, 2)

    trace_data["completed"] = True
    trace_data["pages_generated"] = len(img_bytes)
    trace_data["total_sec"] = round(time.perf_counter() - t_total, 2)

    # Reveal the download buttons (they start visible=False) — value alone left
    # them hidden, which is why there was "no download option" (incl. on mobile).
    pdf_update = gr.update(value=pdf_path, visible=True) if pdf_path else _keep
    coloring_pdf_update = (gr.update(value=coloring_pdf_path, visible=True)
                           if coloring_pdf_path else _keep)
    coloring_display_update = (gr.update(visible=True, value=coloring_html) if coloring_html
                               else _no)

    yield (
        book_html,
        f"Complete: {title} — {len(img_bytes)} pages · {'FLUX (ZeroGPU)' if engine == 'flux' else 'local sketch fallback'} · voice: {voice} · total {trace_data['total_sec']}s",
        audio_path,
        pdf_update,
        story,
        f"Pages: {len(img_bytes)} | Seed: {BASE_SEED} | Engine: {engine} | Story {trace_data.get('story_sec', 0)}s | Images {trace_data.get('images_sec', 0)}s | PDF {trace_data.get('pdf_sec', 0)}s | Coloring {trace_data.get('coloring_sec', 0)}s",
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
    demo.queue(default_concurrency_limit=2, max_size=8)
    # design_kwargs (theme/css/js/head) is non-empty on gradio 6 (moved to launch)
    demo.launch(share=False, allowed_paths=[tempfile.gettempdir()], **demo.design_kwargs)
