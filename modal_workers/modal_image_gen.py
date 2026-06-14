"""
Modal image generation — FLUX.2-klein-4B (verified API) on A100.

Verified against the live model card (June 2026):
  from diffusers import Flux2KleinPipeline
  pipe(prompt=..., guidance_scale=1.0, num_inference_steps=4)   # fast distilled

Character consistency (C1): identical character_description on every page +
locked seed S, page i uses S+i. The character_description is produced upstream
by the vision worker (reads the child's doodle) so the hero matches their drawing.
"""

import os
import io
import logging

import modal

logger = logging.getLogger(__name__)

app = modal.App("doodlebook-image-gen")

CACHE = "/cache"
vol = modal.Volume.from_name("doodlebook-hf-cache", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface")

flux_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "diffusers", "transformers", "accelerate",
        "sentencepiece", "pillow", "huggingface_hub",
    )
    .env({"HF_HOME": CACHE})
)

MIN_CONTAINERS = int(os.environ.get("DOODLEBOOK_KEEP_WARM", "0"))
# FLUX.2-klein-4B is ~13GB in bf16 (see config.FLUX_MODEL.vram_gb) — fits an
# A10G (24GB) with room to spare, ~4-5x cheaper than A100-40GB and far more
# available, so books stop exhausting the scarce A100 pool.
GPU = "A10G"
# A real per-call timeout: if a render can't get a GPU slot (account GPU quota
# exhausted) it FAILS instead of queuing for 24h, so services/images.py falls
# back to the local sketch instead of the app spinning forever.
RENDER_TIMEOUT = 300  # 5 min — generous for a cold start + 6-step render
# Short scaledown so idle containers release their GPU quota quickly instead of
# pinning it for 20 min and blocking the next book (this was the deadlock).
SCALEDOWN = 120
FLUX_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_ART_STYLE = (
    "children's crayon storybook illustration, bold black outlines, "
    "flat bright colors, simple shapes"
)
DEFAULT_COLORING_STYLE = (
    "children's coloring book page, pure black ink outlines on pure white paper, "
    "clean contour lines, no color, no gray, no shading, no texture, "
    "no hatching, no pencil marks, open spaces to color"
)
GPU_FN = dict(  # shared decorator kwargs for the FLUX functions
    gpu=GPU, image=flux_image, volumes={CACHE: vol}, secrets=[HF_SECRET],
    timeout=RENDER_TIMEOUT, min_containers=MIN_CONTAINERS, scaledown_window=SCALEDOWN,
)

# loaded once per warm container, reused across calls
_PIPE = None


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import Flux2KleinPipeline
        logger.info("Loading FLUX.2-klein-4B…")
        _PIPE = Flux2KleinPipeline.from_pretrained(
            FLUX_ID, torch_dtype=torch.bfloat16, cache_dir=CACHE,
        )
        _PIPE.enable_model_cpu_offload()
        logger.info("FLUX ready.")
    return _PIPE


@app.function(**GPU_FN)
def generate_book_pages(
    character_desc: str,
    story_beats: list[str],
    doodle: bytes = None,
    art_style: str = "children's crayon storybook illustration, bold black outlines, flat bright colors, simple shapes",
    seed: int = 42,
    lora_repo: str = None,
    tiny: bool = False,
) -> list[bytes]:
    """
    Render all 6 pages so the hero MATCHES THE CHILD'S DRAWING.

    Two-stage when a doodle is provided (FLUX.2-klein image reference):
      Stage 1: doodle  -> canonical full-body character (same creature, colorized)
      Stage 2: canonical -> the SAME character placed into each story scene
    Falls back to text2img from `character_desc` only when no doodle is given.
    """
    import torch
    from PIL import Image

    pipe = _get_pipe()
    if lora_repo:
        try:
            pipe.load_lora_weights(lora_repo)
            logger.info(f"LoRA loaded: {lora_repo}")
        except Exception as e:
            logger.warning(f"LoRA load failed ({e}); base model")

    steps = 4 if tiny else 6

    def _gen(image, prompt, s):
        kw = dict(prompt=prompt, height=768, width=768, guidance_scale=1.0,
                  num_inference_steps=steps,
                  generator=torch.Generator("cuda").manual_seed(s))
        if image is not None:
            kw["image"] = image
        return pipe(**kw).images[0]

    # --- Stage 1: canonical character from the actual drawing ---
    canonical = None
    if doodle:
        try:
            ref = Image.open(io.BytesIO(doodle)).convert("RGB")
            canonical = _gen(
                ref,
                ("Turn this child's drawing into a clean, friendly, full-body cartoon "
                 "character for a children's storybook. Keep the EXACT same creature, "
                 "face, and features as the drawing. " + art_style +
                 ", plain white background, full character visible, centered."),
                seed,
            )
            logger.info("canonical character built from doodle")
        except Exception as e:
            logger.warning(f"canonical build failed ({e}); text2img fallback")
            canonical = None

    # --- Stage 2: place the SAME character into each scene ---
    pages = []
    for i, beat in enumerate(story_beats):
        if canonical is not None:
            prompt = (
                f"The same character. {beat}. {art_style}, "
                f"full colorful background scene, the character clearly visible."
            )
            img = _gen(canonical, prompt, seed + i + 1)
        else:
            prompt = (
                f"{character_desc}. Scene: {beat}. {art_style}, white background, "
                f"centered, full character visible, same character design throughout"
            )
            img = _gen(None, prompt, seed + i + 1)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        pages.append(buf.getvalue())
        logger.info(f"page {i+1}/{len(story_beats)} done")

    if lora_repo:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
    vol.commit()
    return pages


# ============================================================================
# PARALLEL PATH — split into canonical (1 call) + per-page (fan out via .starmap)
# so the 6 scenes render concurrently across warm containers instead of one
# container doing 7 inferences back-to-back. Orchestrated by services/images.py.
# ============================================================================

# canonical runs once per book, so keep at most ONE warm (don't double the bill)
_CANON_FN = {**GPU_FN, "min_containers": min(1, MIN_CONTAINERS)}


@app.function(**_CANON_FN)
def build_canonical(
    doodle: bytes,
    art_style: str = DEFAULT_ART_STYLE,
    seed: int = 42,
    tiny: bool = False,
) -> bytes:
    """Stage 1: child's drawing -> canonical full-body character (PNG bytes).
    Returns b"" when no doodle is given (caller then renders text2img per page)."""
    if not doodle:
        return b""
    import io
    import torch
    from PIL import Image

    pipe = _get_pipe()
    ref = Image.open(io.BytesIO(doodle)).convert("RGB")
    img = pipe(
        prompt=("Turn this child's drawing into a clean, friendly, full-body cartoon "
                "character for a children's storybook. Keep the EXACT same creature, "
                "face, and features as the drawing. " + art_style +
                ", plain white background, full character visible, centered."),
        image=ref, height=768, width=768, guidance_scale=1.0,
        num_inference_steps=4 if tiny else 6,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


@app.function(**GPU_FN)
def render_page(
    canonical: bytes,
    character_desc: str,
    beat: str,
    art_style: str = DEFAULT_ART_STYLE,
    seed: int = 42,
    tiny: bool = False,
) -> bytes:
    """Stage 2: render ONE scene. Uses the canonical character as an image
    reference when provided (consistency), else text2img from character_desc."""
    import io
    import torch
    from PIL import Image

    pipe = _get_pipe()
    if canonical:
        ref = Image.open(io.BytesIO(canonical)).convert("RGB")
        prompt = (f"The same character. {beat}. {art_style}, "
                  f"full colorful background scene, the character clearly visible.")
        kw = dict(prompt=prompt, image=ref)
    else:
        prompt = (f"{character_desc}. Scene: {beat}. {art_style}, white background, "
                  f"centered, full character visible, same character design throughout")
        kw = dict(prompt=prompt)
    kw.update(height=768, width=768, guidance_scale=1.0,
              num_inference_steps=4 if tiny else 6,
              generator=torch.Generator("cuda").manual_seed(seed))
    img = pipe(**kw).images[0]
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


@app.function(**GPU_FN)
def render_coloring_page(
    canonical: bytes,
    character_desc: str,
    beat: str,
    art_style: str = DEFAULT_COLORING_STYLE,
    seed: int = 42,
    tiny: bool = False,
) -> bytes:
    """Stage 2 alternate render: same scene, but directly as clean line art."""
    import io
    import torch
    from PIL import Image

    pipe = _get_pipe()
    if canonical:
        ref = Image.open(io.BytesIO(canonical)).convert("RGB")
        prompt = (
            f"The same character. {beat}. {art_style}, simple clean background shapes, "
            f"same composition, thick readable outlines, no filled black areas, "
            f"no extra sketch marks."
        )
        kw = dict(prompt=prompt, image=ref)
    else:
        prompt = (
            f"{character_desc}. Scene: {beat}. {art_style}, white background, "
            f"centered, full character visible, same character design throughout"
        )
        kw = dict(prompt=prompt)
    kw.update(
        height=768,
        width=768,
        guidance_scale=1.0,
        num_inference_steps=4 if tiny else 6,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    img = pipe(**kw).images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


LINEART_PROMPT = (
    "black and white coloring book line art, clean bold contour lines only on a "
    "pure white background, no shading, no gray, no color, no fill, no crayon "
    "texture, no crosshatching, no tiny details, simple shapes a child can color"
)

LINEART_NEGATIVE_PROMPT = (
    "color, grayscale, shadows, shading, gradients, texture, speckles, noise, "
    "blur, sketch shading, hatch marks, crosshatching, filled shapes, busy background"
)


@app.function(**GPU_FN)
def render_lineart(color_png: bytes, seed: int = 42) -> bytes:
    """Turn a finished COLOR page into clean coloring-book line art.

    img2img from the color page so the coloring page matches the story picture
    (same pose/composition), but FLUX REDRAWS it as outlines — it understands the
    scene semantically (kid + clouds + hills) and traces shape boundaries instead
    of the crayon texture that wrecked the old OpenCV edge-trace.

    `strength` controls how far it departs from the source: high enough to redraw
    as flat line art, low enough to keep the composition. Flux2KleinPipeline may
    not expose `strength` (it's a unified edit/reference pipeline), so we pass it
    when accepted and silently retry without it.
    """
    import io
    import torch
    from PIL import Image

    pipe = _get_pipe()
    ref = Image.open(io.BytesIO(color_png)).convert("RGB")
    base = dict(
        prompt=LINEART_PROMPT, image=ref, height=768, width=768,
        guidance_scale=1.0, num_inference_steps=6,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    try:
        img = pipe(**base, strength=0.68, negative_prompt=LINEART_NEGATIVE_PROMPT).images[0]
    except TypeError:
        logger.info("pipeline rejected `strength`; retrying without it")
        try:
            img = pipe(**base, negative_prompt=LINEART_NEGATIVE_PROMPT).images[0]
        except TypeError:
            img = pipe(**base).images[0]
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


@app.function(image=flux_image, timeout=30)
def health_check() -> str:
    return "image_gen_healthy"
