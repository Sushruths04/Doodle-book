"""
DoodleBook — MODAL-ONLY runner.

Serves the Gradio UI locally but runs ALL heavy generation on Modal's deployed
functions (no local GPU inference):
  - story      -> doodlebook-story / generate_story      (services.story)
  - images     -> doodlebook-image-gen / generate_book_pages (services.images)
  - voice      -> doodlebook-tts / speak_book            (services.tts)

Use this (not app.py) when you want to check real Modal output on this machine.
Run:  python run_modal.py
"""

import io
import json
import time
import tempfile
import logging

import gradio as gr
from PIL import Image

from config import BASE_SEED, DEFAULT_VOICE
from book_builder import (
    build_book_html, export_pdf, magic_loader_html,
    build_coloring_html, export_coloring_pdf,
)
from ui.layout import create_layout

import services.story as story_svc
import services.images as image_svc
import services.tts as tts_svc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doodlebook.modal")


def _doodle_to_png_bytes(doodle_image):
    """Gradio numpy image -> PNG bytes (or None)."""
    if doodle_image is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(doodle_image).save(buf, format="PNG")
    return buf.getvalue()


def _with_heartbeat(blocking_fn, frame_fn, poll=4.0):
    """
    Run blocking_fn() in a thread while keeping the Gradio stream alive.

    A multi-minute Modal call (FLUX ~2-3 min, VoxCPM ~30-60s) blocks the
    generator with no yield, so the browser's SSE stream goes silent, the
    connection drops, and the UI shows "Error". This pumps frame_fn(elapsed)
    into the stream every `poll` seconds until the work finishes.

    Yields ("hb", <frame tuple>) heartbeats, then a final ("done", <return>).
    Re-raises whatever blocking_fn raised.
    """
    import threading
    box = {}

    def _run():
        try:
            box["val"] = blocking_fn()
        except BaseException as e:  # surfaced to the caller below
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


def create_book(doodle_image, character_name, theme, hero_name,
                tiny_mode=False, voice=DEFAULT_VOICE, make_coloring=False):
    """Streaming book creation — everything heavy runs on Modal."""
    t_total = time.perf_counter()
    character_name = (character_name or "").strip() or "Little Hero"
    hero_name = (hero_name or "").strip() or character_name

    trace = {
        "backend": "modal",
        "hero_name": hero_name,
        "theme": theme,
        "voice": voice,
        "tiny_mode": tiny_mode,
        "make_coloring": make_coloring,
        "seed": BASE_SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    _no = gr.update(visible=False)
    _keep = gr.update()  # no-op: leave the (fixed, always-visible) download buttons as-is

    # ---- 1) STORY (Modal MiniCPM, else fast text template) ----
    yield (magic_loader_html("story", hero_name),
           "Writing the story…", None, _keep, {}, "",
           json.dumps(trace, indent=2), _no, _keep)

    t_story = time.perf_counter()
    story = story_svc.generate_story(hero_name, theme)
    trace["story_sec"] = round(time.perf_counter() - t_story, 2)
    title = story.get("title", "Untitled Story")
    char_desc = story.get("character_description", "")
    pages = story.get("pages", [])
    page_texts = [p.get("text", "") for p in pages]
    scenes = [p.get("scene", "") for p in pages]
    trace.update(title=title, character_description=char_desc)

    # ---- 2) VOICE starts NOW, concurrently with images (it only needs the text,
    #         which is ready) so its ~30-60s overlaps the image render for free ----
    import threading
    voice_box = {}
    full_text = f"{title}. {' '.join(page_texts)}"
    t_voice = time.perf_counter()

    def _do_voice():
        try:
            voice_box["bytes"] = tts_svc.speak_book(full_text, voice)
        except Exception as e:
            voice_box["err"] = e

    voice_thread = threading.Thread(target=_do_voice, daemon=True)
    voice_thread.start()

    # ---- 3) IMAGES (Modal FLUX.2-klein — 6 pages rendered in PARALLEL) ----
    yield (magic_loader_html("images", hero_name),
           f"{title} — illustrating on Modal (FLUX)…",
           None, _keep, story, "", json.dumps(trace, indent=2), _no, _keep)

    doodle_bytes = _doodle_to_png_bytes(doodle_image)
    img_bytes, engine = None, "sketch"
    t_images = time.perf_counter()
    for kind, payload in _with_heartbeat(
        lambda: image_svc.generate_book_pages(
            char_desc, scenes, doodle=doodle_bytes, seed=BASE_SEED, tiny=tiny_mode
        ),
        lambda s: (magic_loader_html("images", hero_name),
                   f"{title} — illustrating… {s}s  (voice recording in parallel)",
                   None, _keep, story, "", json.dumps(trace, indent=2), _no, _keep),
    ):
        if kind == "hb":
            yield payload
        else:
            img_bytes, engine = payload
    trace["images_sec"] = round(time.perf_counter() - t_images, 2)
    trace["engine"] = engine
    if engine != "flux":
        logger.warning("Image gen fell back to local sketch — Modal FLUX did not run.")

    book_html = build_book_html(img_bytes, page_texts, title, engine)

    # ---- 4) Collect the parallel VOICE result (usually already finished) ----
    while voice_thread.is_alive():  # only loops if voice somehow outran images
        voice_thread.join(timeout=4)
        if voice_thread.is_alive():
            yield (book_html, f"{title} — finishing narration…",
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

    # ---- 4) PDFs ----
    pdf_path = None
    t_pdf = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_path = export_pdf(img_bytes, page_texts, title, tmp.name)
    except Exception as e:
        logger.warning(f"PDF failed: {e}")
    trace["pdf_sec"] = round(time.perf_counter() - t_pdf, 2)

    # ---- 5) COLORING BOOK ----
    coloring_html = ""
    coloring_pdf_path = None
    if make_coloring:
        t_coloring = time.perf_counter()
        outlines, coloring_engine = None, "failed"
        for kind, payload in _with_heartbeat(
            lambda: image_svc.generate_coloring_pages(
                char_desc,
                scenes,
                doodle=doodle_bytes,
                source_color_imgs=img_bytes,
                seed=BASE_SEED,
                tiny=tiny_mode,
            ),
            lambda s: (
                book_html,
                f"{title} — building coloring book… {s}s",
                audio_path,
                _keep,
                story,
                "",
                json.dumps(trace, indent=2),
                _no,
                _keep,
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

    trace["completed"] = True
    trace["total_sec"] = round(time.perf_counter() - t_total, 2)
    engine_label = "FLUX (Modal)" if engine == "flux" else "local sketch fallback"
    # download buttons stay visible (fixed under the status); just attach the files
    pdf_update = gr.update(value=pdf_path) if pdf_path else _keep
    coloring_pdf_update = gr.update(value=coloring_pdf_path) if coloring_pdf_path else _keep
    coloring_display_update = (gr.update(visible=True, value=coloring_html) if coloring_html
                               else gr.update(visible=False))

    yield (
        book_html,
        f"Complete: {title} — {len(img_bytes)} pages · {engine_label} · voice: {voice} · total {trace['total_sec']}s",
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


if __name__ == "__main__":
    import os

    demo = create_layout(create_book_fn=create_book)
    # Queue so a long (multi-minute) Modal generation doesn't make the whole app
    # unresponsive: allow several concurrent sessions and never time a job out.
    demo.queue(default_concurrency_limit=8, max_size=64)
    # NOTE: when server_port is set, Gradio does NOT auto-pick a free port — it
    # raises OSError and exits if the port is busy. start_app.bat frees the port
    # before launching; if you run this directly, make sure 7880 is free first.
    port = int(os.environ.get("DOODLEBOOK_PORT", "7880"))
    # Bind all interfaces so both 127.0.0.1 and localhost (and LAN/phone) reach it.
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            inbrowser=False,
            show_error=True,
            max_threads=40,
            # PDFs are written to the system temp dir; Gradio won't serve files
            # outside its allowed paths, so the DownloadButton links 404'd and the
            # button looked broken. Allow the temp dir so downloads actually work.
            allowed_paths=[tempfile.gettempdir()],
        )
    except OSError as e:
        logger.error(
            f"Could not bind port {port}: {e}\n"
            f"  Something is already using it — likely a leftover DoodleBook "
            f"instance (app.py on 7870, an old run_modal.py, or test_final.py).\n"
            f"  Fix: close the other window, or run:  "
            f"netstat -ano | findstr :{port}   then  taskkill /f /pid <PID>\n"
            f"  Then relaunch with start_app.bat (it frees the port automatically)."
        )
        raise SystemExit(1)
