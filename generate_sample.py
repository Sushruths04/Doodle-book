"""Generate the loading SAMPLE book (assets/sample_book/) from REAL FLUX output.

Turns the example cat doodle into a consistent crayon storybook hero via the
deployed Modal pipeline, so the sample users see on load is genuine app output
(not the old placeholder blobs). Images are inlined as base64 so sample.html is
self-contained and renders on the HF Space regardless of temp paths.

Run once:  python generate_sample.py
"""

import os
import sys
import base64

sys.path.insert(0, os.path.dirname(__file__))

import io
from PIL import Image

from config import SAMPLE_BOOK_PATH, BASE_SEED
from book_builder import COVER_HTML, PAGE_HTML, ENGINE_BADGES
import services.images as image_svc

HERE = os.path.dirname(__file__)

# Match the app's crayon look (app.py COLOR_ART_STYLE)
CRAYON = (
    "hand-drawn crayon children's storybook illustration, soft waxy crayon "
    "texture and visible crayon strokes, warm colorful crayon shading, "
    "simple friendly shapes, looks drawn by hand with crayons"
)

TITLE = "Ziggy and the Little Lost Star"
CHAR_DESC = (
    "Ziggy, a curious round-faced cartoon cat with big friendly eyes, "
    "pointy ears, whiskers and a long tail"
)
# (page text, scene prompt) — a complete, gentle 6-page arc that fits the cat doodle
PAGES = [
    ("Ziggy the curious little cat loved watching the night sky from the rooftop.",
     "a cute cat sitting on a cozy rooftop at night, gazing up at a big starry sky"),
    ("One quiet night, a tiny star tumbled down and landed softly in the garden.",
     "a tiny glowing star falling into a flower garden while the surprised cat watches"),
    ("The little star was scared and far from home, blinking sadly.",
     "the cat gently looking at a small sad glowing star sitting in the grass"),
    ("\"Don't worry,\" purred Ziggy, \"I'll help you climb back to the sky.\"",
     "the kind cat comforting the little glowing star, cozy and warm at night"),
    ("Together they leapt from the tallest tree to the top of the old windmill.",
     "the cat and the glowing star climbing a tall tree toward a windmill at night"),
    ("With one gentle hop, the star sparkled home, and Ziggy smiled, full of stars.",
     "the happy cat on a hill waving as the star flies back into the starry sky"),
]


def _b64(img_bytes: bytes, max_px: int = 720, quality: int = 84) -> str:
    """Compress to a reasonably small JPEG before inlining — full-res FLUX PNGs
    would make sample.html >10MB and stall the frontend on load."""
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    texts = [t for t, _ in PAGES]
    scenes = [s for _, s in PAGES]
    with open(os.path.join(HERE, "assets", "sample_doodle.jpg"), "rb") as f:
        doodle = f.read()

    print("Generating real FLUX pages via Modal (this calls the deployed app)...")
    images, engine = image_svc.generate_book_pages(
        CHAR_DESC, scenes, doodle=doodle, art_style=CRAYON, seed=BASE_SEED,
    )
    print(f"engine={engine}, pages={len(images)}")
    if engine != "flux" or not images:
        raise SystemExit("FLUX did not run (got sketch); aborting so we never bake a placeholder.")

    os.makedirs(SAMPLE_BOOK_PATH, exist_ok=True)

    # Self-contained HTML: base64-inline the (compressed) images so the baked
    # sample renders on the Space (the live builder uses temp-file URLs, which
    # don't exist for a static file). Only sample.html is used by the app.
    cover = COVER_HTML.format(title=TITLE, badge=ENGINE_BADGES.get("flux", ""))
    pages_html = ""
    for i, (b, text) in enumerate(zip(images, texts)):
        pages_html += PAGE_HTML.format(
            img_b64=_b64(b), text=text, page_num=i + 1,
            alt_text=(text[:50] + "...") if len(text) > 50 else text,
        )
    html = f'<div class="book-container">\n{cover}\n{pages_html}\n</div>'
    with open(os.path.join(SAMPLE_BOOK_PATH, "sample.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved sample.html ({len(html.encode()) // 1024} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
