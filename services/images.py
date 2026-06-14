"""
Image generation service — calls modal_image_gen for FLUX.2-klein inference.

C1 Compliance: Character consistency stack
- Locked seed; page i uses seed+i
- Identical character_description on every page
- Doodle as IP-Adapter image prompt (if provided)
"""

from config import FLUX_MODEL, GENERATION_PARAMS, page_seed
import logging

logger = logging.getLogger(__name__)

DEFAULT_COLOR_ART_STYLE = (
    "children's crayon storybook illustration, bold black outlines, "
    "flat bright colors, simple shapes"
)
DEFAULT_COLORING_ART_STYLE = (
    "children's coloring book page, pure black ink outlines on pure white paper, "
    "clean contour lines, no color, no gray, no shading, no texture, "
    "no hatching, no pencil marks, open spaces to color"
)


def generate_book_pages(
    character_desc: str,
    story_beats: list[str],
    doodle: bytes = None,
    art_style: str = "crayon drawing, children's book, colorful, simple shapes",
    seed: int = None,
    lora_repo: str = None,
    tiny: bool = False
) -> list[bytes]:
    """
    Generate all 6 book pages via FLUX.2-klein on Modal.
    
    Uses one warm container call (C3) for efficiency.
    Character consistency via:
    - Locked seed per page
    - Identical character_description on every page
    - Doodle image as IP-Adapter prompt (if provided)
    
    Args:
        character_desc: Visual description of character
        story_beats: List of scene descriptions
        doodle: Optional doodle image bytes for IP-Adapter
        art_style: Art style prefix
        seed: Base seed (default from config)
        lora_repo: Optional LoRA repo
        tiny: Use Tiny Mode (SD-Turbo, 4 steps)
        
    Returns:
        List of PNG image bytes (6 pages)
    """
    if seed is None:
        seed = GENERATION_PARAMS.num_inference_steps

    # 1) PARALLEL Modal path — build the canonical character once, then fan the 6
    #    scene renders out across warm containers via .starmap (concurrent, fast).
    try:
        import modal
        canon = modal.Function.from_name("doodlebook-image-gen", "build_canonical")
        page = modal.Function.from_name("doodlebook-image-gen", "render_page")
        canonical = canon.remote(doodle or b"", art_style, seed, tiny)
        args = [
            (canonical, character_desc, beat, art_style, seed + i + 1, tiny)
            for i, beat in enumerate(story_beats)
        ]
        images = list(page.starmap(args))
        if images and all(images):
            return images, "flux"
        raise RuntimeError("parallel render returned empty pages")
    except Exception as e:
        logger.warning(f"Modal parallel image gen failed ({e}); trying single-call worker")

    # 2) Single-container Modal worker (older path) as a fallback.
    try:
        import modal
        fn = modal.Function.from_name("doodlebook-image-gen", "generate_book_pages")
        images = fn.remote(
            character_desc, story_beats, doodle, art_style, seed, lora_repo, tiny
        )
        return images, "flux"
    except Exception as e:
        logger.warning(f"Modal generation failed: {e}, using local sketch fallback")
        return generate_placeholder_images(character_desc, story_beats, doodle=doodle), "sketch"


def generate_coloring_pages(
    character_desc: str,
    story_beats: list[str],
    doodle: bytes = None,
    source_color_imgs: list[bytes] = None,
    art_style: str = DEFAULT_COLORING_ART_STYLE,
    seed: int = None,
    tiny: bool = False,
) -> tuple[list[bytes], str]:
    """Generate coloring pages directly with Modal FLUX line-art renders.

    Primary path: build one canonical character, then render each page directly
    as line art. Fallback: derive outlines from the finished color pages.
    """
    if seed is None:
        seed = GENERATION_PARAMS.num_inference_steps

    try:
        import modal
        canon = modal.Function.from_name("doodlebook-image-gen", "build_canonical")
        page = modal.Function.from_name("doodlebook-image-gen", "render_coloring_page")
        canonical = canon.remote(doodle or b"", art_style, seed, tiny)
        args = [
            (canonical, character_desc, beat, art_style, seed + i + 101, tiny)
            for i, beat in enumerate(story_beats)
        ]
        images = list(page.starmap(args))
        if images and all(images):
            from services.coloring import _crispen
            return ([_crispen(img) for img in images], "flux-direct-lineart")
        raise RuntimeError("parallel coloring render returned empty pages")
    except Exception as e:
        logger.warning(f"Modal direct coloring gen failed ({e}); using traced fallback")

    color_imgs = source_color_imgs
    if color_imgs is None:
        color_imgs, _ = generate_book_pages(
            character_desc,
            story_beats,
            doodle=doodle,
            art_style=DEFAULT_COLOR_ART_STYLE,
            seed=seed,
            tiny=tiny,
        )
    from services.coloring import derive_coloring_pages
    return derive_coloring_pages(color_imgs), "trace-fallback"


def generate_placeholder_images(
    character_desc: str = "",
    story_beats: list[str] = None,
    doodle: bytes = None,
) -> list[bytes]:
    """
    Generate crayon-style PREVIEW illustrations (used when no GPU/FLUX is available).

    Key behaviour: if the child's `doodle` is provided, their ACTUAL drawing is cut
    out and composited as the recurring hero on every page — so different drawings
    produce different books (fixing "it builds the same no matter what I draw").
    Without a doodle, a drawn blob-creature is used, varied by the character name.
    Deterministic for a given input.
    """
    from PIL import Image
    import io
    import random

    if story_beats is None:
        story_beats = ["Once upon a time", "A discovery", "The journey",
                       "A little trouble", "Together", "The happy end"]

    # vary palette/character by the character text so different inputs differ
    vkey = sum(ord(c) for c in (character_desc or "hero")) or 7
    hero_palette = ["#ef6a3a", "#e0533a", "#d6517a", "#4a9fd6", "#2ba39a", "#8e63c4"]
    hero = hero_palette[vkey % len(hero_palette)]
    friend = "#2ba39a" if hero != "#2ba39a" else "#ef6a3a"
    ink = "#2e2a26"

    sky_colors = ["#dff1fb", "#fde9d6", "#e6f4e1", "#e3eefc", "#f3e6f6", "#fdf3d6"]
    ground_colors = ["#cfe3a6", "#e9c79a", "#bfe3b0", "#cfe3a6", "#d9c7e8", "#e8dca0"]

    # the child's actual character, traced into a clean cartoon (used on every page)
    doodle_cut = _doodle_to_cartoon(doodle, fill_rgb=_hex(hero)) if doodle else None

    # where the hero stands on each page (cx, cy)
    hero_spots = [(384, 300), (270, 312), (380, 312), (250, 312), (290, 312), (384, 300)]

    def place_hero(img, rng, cx, cy, flip=False):
        if doodle_cut is not None:
            _paste_doodle(img, doodle_cut, cx, cy, target_h=210, flip=flip)
        else:
            _crayon_creature(img, rng, cx, cy + 30, hero, ink)

    images = []
    for i, (sky, ground, beat) in enumerate(zip(sky_colors, ground_colors, story_beats)):
        rng = random.Random(1000 + i + vkey)  # deterministic, but varies by character
        img = Image.new("RGB", (768, 512), sky)

        _crayon_band(img, rng, 0, 360, sky)
        _crayon_band(img, rng, 360, 512, ground)

        cx, cy = hero_spots[i]
        if i == 0:
            _crayon_sun(img, rng, 650, 90)
            place_hero(img, rng, cx, cy)
        elif i == 1:
            place_hero(img, rng, cx, cy)
            _crayon_sparkle(img, rng, 560, 200, "#f4c64a")
            _crayon_sparkle(img, rng, 600, 260, "#d6517a")
        elif i == 2:
            _crayon_tree(img, rng, 140); _crayon_tree(img, rng, 610)
            place_hero(img, rng, cx, cy, flip=True)
        elif i == 3:
            place_hero(img, rng, cx, cy)
            _crayon_blob(img, rng, 560, 360, 70, "#9aa0a6", ink)  # a little obstacle
        elif i == 4:
            place_hero(img, rng, cx, cy)
            _crayon_creature(img, rng, 500, 342, friend, ink)     # a new friend
            _crayon_sparkle(img, rng, 392, 210, "#d6517a")
        else:
            place_hero(img, rng, cx, cy)
            for _ in range(26):
                x, y = rng.randint(80, 690), rng.randint(40, 190)
                c = rng.choice(["#ef6a3a", "#2ba39a", "#f4c64a", "#d6517a", "#4a9fd6"])
                _crayon_sparkle(img, rng, x, y, c, size=rng.randint(6, 12))

        _paper_grain(img, rng, amount=2200)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(buf.getvalue())

    return images


def _doodle_to_cartoon(doodle_bytes: bytes, fill_rgb=(239, 106, 58)):
    """
    Turn a PHOTO of a child's drawing into a clean cartoon cut-out:
      1. find the paper (largest bright region) and crop to it — drops the desk/shadows
      2. trace the crayon/pencil strokes into bold black outlines (adaptive threshold)
      3. drop noise, fill the silhouette with a flat character color, ink the outlines
      4. return a transparent RGBA character — NOT the raw photo

    Falls back to a simple alpha cut-out if OpenCV/processing fails.
    """
    from PIL import Image
    import io
    try:
        import cv2
        import numpy as np

        rgb = np.array(Image.open(io.BytesIO(doodle_bytes)).convert("RGB"))
        h, w = rgb.shape[:2]
        s = 900.0 / max(h, w)
        if s < 1:
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 1) ONLY crop to paper when the background is dark (a photo on a desk).
        #    For clean white-background drawings, cropping/insetting would clip the art.
        corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
        if float(np.mean(corners)) < 150:
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, paper = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c) > 0.12 * gray.size:
                    x, y, wc, hc = cv2.boundingRect(c)
                    pad = int(0.03 * max(wc, hc))
                    gray = gray[y + pad:y + hc - pad, x + pad:x + wc - pad]
                    rgb = rgb[y + pad:y + hc - pad, x + pad:x + wc - pad]
        if gray.size == 0:
            raise ValueError("empty crop")
        H, W = gray.shape

        # 2) trace strokes -> white-on-black mask
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        strokes = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            blockSize=25, C=10,
        )
        strokes = cv2.medianBlur(strokes, 3)

        # drop small noise specks, keep real strokes
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(strokes, 8)
        clean = np.zeros_like(strokes)
        min_area = max(15, int(0.0001 * strokes.size))
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[lbl == i] = 255
        strokes = clean
        if int(strokes.sum()) < 255 * 40:
            raise ValueError("no strokes found")

        # 3) BACKGROUND = the white area connected to the image border.
        #    Temporarily thicken strokes to seal small gaps so the flood can't
        #    leak into the body, then flood from a 1px frame around the image.
        sealed = cv2.dilate(strokes, np.ones((3, 3), np.uint8), iterations=2)
        free = np.where(sealed == 0, 255, 0).astype(np.uint8)   # paintable = non-stroke
        framed = cv2.copyMakeBorder(free, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        ffmask = np.zeros((framed.shape[0] + 2, framed.shape[1] + 2), np.uint8)
        cv2.floodFill(framed, ffmask, (0, 0), 128)             # 128 = outside background
        filled = framed[1:-1, 1:-1]
        body = np.where(filled != 128, 255, 0).astype(np.uint8)  # interior + sealed strokes
        # undo the gap-sealing dilation so the body matches the real outline
        body = cv2.erode(body, np.ones((3, 3), np.uint8), iterations=2)
        body = cv2.medianBlur(body, 5)

        # 4) compose RGBA: flat character color where body, black ink on the strokes
        strokes = cv2.dilate(strokes, np.ones((2, 2), np.uint8), iterations=1)  # bolder ink
        out = np.zeros((H, W, 4), np.uint8)
        out[body > 0] = (fill_rgb[0], fill_rgb[1], fill_rgb[2], 255)
        out[strokes > 0] = (38, 30, 28, 255)
        alpha = np.maximum(body, strokes)
        out[..., 3] = alpha

        ys, xs = np.where(alpha > 0)
        if len(xs) == 0:
            raise ValueError("empty character")
        out = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return Image.fromarray(out, "RGBA")

    except Exception as e:
        logger.warning(f"cartoonify failed ({e}); using simple cut-out")
        im = Image.open(io.BytesIO(doodle_bytes)).convert("RGBA")
        if max(im.size) > 900:
            im.thumbnail((900, 900))
        px = im.getdata()
        im.putdata([(r, g, b, 0) if (r > 232 and g > 232 and b > 232) else (r, g, b, a)
                    for r, g, b, a in px])
        bbox = im.getbbox()
        return im.crop(bbox) if bbox else im


def _paste_doodle(img, cut, cx, cy, target_h=210, flip=False):
    """Scale the cut-out character to target height and paste it centered at (cx, cy)."""
    from PIL import Image
    c = cut
    if flip:
        c = c.transpose(Image.FLIP_LEFT_RIGHT)
    w, h = c.size
    if h == 0:
        return
    scale = target_h / h
    new = c.resize((max(1, int(w * scale)), target_h), Image.LANCZOS)
    x = int(cx - new.width / 2)
    y = int(cy - new.height / 2)
    img.paste(new, (x, y), new)  # use alpha as mask


# --- crayon drawing primitives (jittered strokes => hand-drawn feel) -----------

def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[j:j+2], 16) for j in (0, 2, 4))


def _crayon_band(img, rng, y0, y1, color):
    """Fill a horizontal band with short jittered crayon strokes."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    base = _hex(color)
    for _ in range(900):
        x = rng.randint(0, 768); y = rng.randint(y0, y1)
        ln = rng.randint(8, 22); jit = rng.randint(-12, 12)
        shade = rng.randint(-14, 10)
        col = tuple(max(0, min(255, v + shade)) for v in base)
        d.line([(x, y), (x + ln, y + jit // 3)], fill=col, width=rng.randint(2, 4))


def _crayon_blob(img, rng, cx, cy, r, color, ink):
    """A filled wobbly blob with a crayon outline."""
    from PIL import ImageDraw
    import math
    d = ImageDraw.Draw(img)
    pts = []
    for a in range(0, 360, 18):
        rr = r + rng.randint(-r // 8, r // 8)
        pts.append((cx + rr * math.cos(math.radians(a)),
                    cy + rr * math.sin(math.radians(a))))
    d.polygon(pts, fill=_hex(color))
    # crayon hatch shading
    base = _hex(color)
    for _ in range(int(r * 6)):
        x = rng.randint(cx - r, cx + r); y = rng.randint(cy - r, cy + r)
        if (x - cx) ** 2 + (y - cy) ** 2 <= (r - 4) ** 2:
            shade = rng.randint(-22, 4)
            col = tuple(max(0, min(255, v + shade)) for v in base)
            d.line([(x, y), (x + rng.randint(4, 10), y + rng.randint(-3, 3))], fill=col, width=2)
    _wobble_outline(d, rng, pts, ink)


def _wobble_outline(d, rng, pts, ink, width=4):
    col = _hex(ink)
    for j in range(len(pts)):
        a = pts[j]; b = pts[(j + 1) % len(pts)]
        mx = (a[0] + b[0]) / 2 + rng.randint(-2, 2)
        my = (a[1] + b[1]) / 2 + rng.randint(-2, 2)
        d.line([a, (mx, my), b], fill=col, width=width, joint="curve")


def _crayon_creature(img, rng, cx, cy, color, ink):
    """A friendly rounded blob-creature with a big face — far cuter than a stick figure."""
    from PIL import ImageDraw
    _crayon_blob(img, rng, cx, cy, 64, color, ink)              # body
    _crayon_blob(img, rng, cx, cy - 78, 40, color, ink)        # head
    d = ImageDraw.Draw(img)
    inkc = _hex(ink)
    # legs + arms
    for dx in (-26, 26):
        d.line([(cx + dx, cy + 50), (cx + dx + rng.randint(-4, 4), cy + 92)], fill=inkc, width=6)
    for dx in (-58, 58):
        d.line([(cx, cy - 6), (cx + dx, cy + 18 + rng.randint(-4, 4))], fill=inkc, width=5)
    # eyes
    for dx in (-14, 14):
        d.ellipse([cx + dx - 9, cy - 92, cx + dx + 9, cy - 74], fill="white", outline=inkc, width=2)
        d.ellipse([cx + dx - 3, cy - 86, cx + dx + 3, cy - 80], fill=inkc)
    # rosy cheeks + smile
    for dx in (-22, 22):
        d.ellipse([cx + dx - 7, cy - 72, cx + dx + 7, cy - 60], fill=(246, 170, 150))
    d.arc([cx - 14, cy - 78, cx + 14, cy - 58], 10, 170, fill=inkc, width=4)


def _crayon_sun(img, rng, x, y):
    from PIL import ImageDraw
    import math
    _crayon_blob(img, rng, x, y, 38, "#f4c64a", "#e0a92e")
    d = ImageDraw.Draw(img)
    for a in range(0, 360, 30):
        x2 = x + 58 * math.cos(math.radians(a)); y2 = y + 58 * math.sin(math.radians(a))
        d.line([(x, y), (x2, y2)], fill=(224, 169, 46), width=4)


def _crayon_tree(img, rng, x):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.line([(x, 420), (x + rng.randint(-4, 4), 320)], fill=(120, 84, 60), width=12)
    _crayon_blob(img, rng, x, 300, 56, "#74b85a", "#4f7d3a")


def _crayon_sparkle(img, rng, x, y, color, size=14):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    c = _hex(color)
    d.line([(x, y - size), (x, y + size)], fill=c, width=4)
    d.line([(x - size, y), (x + size, y)], fill=c, width=4)
    d.line([(x - size // 2, y - size // 2), (x + size // 2, y + size // 2)], fill=c, width=3)
    d.line([(x - size // 2, y + size // 2), (x + size // 2, y - size // 2)], fill=c, width=3)


def _paper_grain(img, rng, amount=2000):
    """Sprinkle subtle light/dark specks so it reads as paper, not flat pixels."""
    px = img.load()
    w, h = img.size
    for _ in range(amount):
        x = rng.randint(0, w - 1); y = rng.randint(0, h - 1)
        r, g, b = px[x, y]
        s = rng.choice([-12, -8, 8, 12])
        px[x, y] = (max(0, min(255, r + s)), max(0, min(255, g + s)), max(0, min(255, b + s)))
