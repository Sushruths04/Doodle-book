"""
Book builder — assembles pages into storybook HTML and PDF.

Features:
- Magic Loader: animated wait messages during generation
- Coloring Book: outline pages from same FLUX images
- Styled PDF covers: scrapbook-style cover matching on-screen UI
"""

import io
import logging
import math
import os
import tempfile
import urllib.parse

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from config import COLORS

logger = logging.getLogger(__name__)

FONTS_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")
FONT_GAEGU = os.path.join(FONTS_DIR, "Gaegu-Bold.ttf")
FONT_CAVEAT = os.path.join(FONTS_DIR, "Caveat.ttf")


# ============================================================================
# STORYBOOK HTML
# ============================================================================

PAGE_HTML = """
<div class="book-page" style="page-break-after: always;">
  <div class="page-art">
    <img src="{img_b64}" alt="Page {page_num}: {alt_text}"/>
  </div>
  <p class="page-text">{text}</p>
  <span class="page-num">~ {page_num} ~</span>
</div>
"""

COVER_HTML = """
<div class="book-cover">
  <p class="cover-kicker">a DoodleBook story</p>
  <h2 class="cover-title">{title}</h2>
  {badge}
</div>
"""

ENGINE_BADGES = {
    "flux":   '<span class="engine-badge flux">illustrated by FLUX.2-klein</span>',
    "sketch": '<span class="engine-badge sketch">preview sketch — connect a GPU for FLUX art</span>',
}


def build_book_html(
    pages_images: list,
    pages_texts: list,
    title: str,
    engine: str = "flux",
) -> str:
    """Build storybook HTML from images and texts."""
    badge = ENGINE_BADGES.get(engine, "")
    cover = COVER_HTML.format(title=title, badge=badge)

    pages_html = ""
    for i, (img_bytes, text) in enumerate(zip(pages_images, pages_texts)):
        pages_html += PAGE_HTML.format(
            img_b64=_img_src_for_bytes(img_bytes),
            text=text,
            page_num=i + 1,
            alt_text=(text[:50] + "...") if len(text) > 50 else text,
        )

    return f"""<div class="book-container">
    {cover}
    {pages_html}
    </div>"""


# ============================================================================
# MAGIC LOADER (Feature 1)
# ============================================================================

def magic_loader_html(stage: str = "story", hero_name: str = "your hero") -> str:
    """
    Animated crayon-styled wait panel with rotating messages.

    Pure CSS fade cycle — no JS, no Gradio streaming needed.
    stage: "story" | "images" | "tts" — controls which messages show.
    """
    messages = {
        "story": [
            f"✏️ MiniCPM is dreaming up {hero_name}'s story\u2026",
            "📖 Writing page by page\u2026",
            "💡 Did you know? Your whole storybook runs on tiny models!",
        ],
        "images": [
            f"🎨 FLUX is painting {hero_name}'s 6 pages\u2026",
            "🖌️ Adding crayon details\u2026",
            "💡 Did you know? The brain is only 3B parameters!",
        ],
        "tts": [
            "🔊 VoxCPM is recording the narration\u2026",
            "🎙️ Warming up the storyteller voice\u2026",
            "💡 Did you know? The voice runs on a 2B model!",
        ],
    }
    msgs = messages.get(stage, messages["story"])
    n = len(msgs)
    step = 3.0
    total = step * n

    items = ""
    for i, m in enumerate(msgs):
        delay = i * step
        items += (
            f'<div class="ml-msg" style="animation-delay:{delay:.1f}s;">{m}</div>\n'
        )

    return f"""
    <div class="magic-loader">
      <div class="ml-spinner"></div>
      <div class="ml-stack">
        {items}
      </div>
      <style>
        .magic-loader {{
          text-align:center; padding:36px 20px; font-family:'Gaegu',cursive;
          font-size:22px; color:#2e2a26; position:relative; min-height:120px;
        }}
        .ml-spinner {{
          width:36px; height:36px; margin:0 auto 14px;
          border:4px solid #efe0c2; border-top-color:#ef6a3a;
          border-radius:50%; animation:ml-spin 1s linear infinite;
        }}
        @keyframes ml-spin {{ to {{ transform:rotate(360deg); }} }}
        .ml-stack {{ position:relative; height:64px; }}
        .ml-msg {{
          position:absolute; inset:0; display:flex; align-items:center;
          justify-content:center; text-align:center; padding:0 10px;
          opacity:0; animation:ml-cycle {total:.1f}s infinite;
        }}
        @keyframes ml-cycle {{
          0%,24% {{ opacity:1; }}
          33%,100% {{ opacity:0; }}
        }}
        @media (prefers-reduced-motion:reduce) {{
          .ml-spinner {{ animation:none; }}
          .ml-msg {{ animation:none; opacity:1; position:static; margin:6px 0; }}
        }}
      </style>
    </div>
    """


# ============================================================================
# STYLIZED PDF COVER (Feature 3)
# ============================================================================

def _load_font(path: str, size: int):
    """Load TrueType font with fallback to default."""
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _paper_grain(img, rng_seed=42, amount=2000):
    """Sprinkle subtle specks so cover reads as paper."""
    import random
    rng = random.Random(rng_seed)
    px = img.load()
    w, h = img.size
    for _ in range(amount):
        x = rng.randint(0, w - 1)
        y = rng.randint(0, h - 1)
        r, g, b = px[x, y]
        s = rng.choice([-10, -6, 6, 10])
        px[x, y] = (
            max(0, min(255, r + s)),
            max(0, min(255, g + s)),
            max(0, min(255, b + s)),
        )


def render_cover_image(
    title: str,
    badge_text: str = "illustrated by FLUX.2-klein",
    kind: str = "story",
) -> bytes:
    """
    Render a full-page scrapbook-style cover as PNG bytes.

    Canvas ~1240x1754 (A4 @150dpi). Cream bg, crayon title with layered
    shadow, badge. Used as page 1 of both story and coloring PDFs.
    """
    W, H = 1240, 1754
    cream = (255, 248, 230)
    ink = (46, 42, 38)
    sun = (244, 198, 74)
    berry = (214, 81, 122)
    leaf = (116, 184, 90)

    img = Image.new("RGB", (W, H), cream)
    draw = ImageDraw.Draw(img)

    _paper_grain(img, amount=6000)

    # Kicker
    font_kicker = _load_font(FONT_CAVEAT, 60)
    draw.text((W // 2, 520), "a DoodleBook story", font=font_kicker,
              fill=berry, anchor="mm")

    # Title — layered shadow
    font_title = _load_font(FONT_GAEGU, 120)

    # wrap title to <=2 lines
    words = title.split()
    lines = []
    current = ""
    for w in words:
        test = f"{current} {w}".strip()
        bbox = font_title.getbbox(test)
        if bbox[2] - bbox[0] > W - 200:
            if current:
                lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)

    total_h = len(lines) * 140
    y_start = 680

    for i, line in enumerate(lines):
        cy = y_start + i * 140
        # sun shadow
        draw.text((W // 2 + 4, cy + 4), line, font=font_title,
                  fill=sun, anchor="mm")
        # ink on top
        draw.text((W // 2, cy), line, font=font_title,
                  fill=ink, anchor="mm")

    # Badge — rounded rect
    font_badge = _load_font(FONT_GAEGU, 44)
    bbox = font_badge.getbbox(badge_text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    bx = W // 2 - (tw + 40) // 2
    by = y_start + total_h + 80
    draw.rounded_rectangle(
        [bx, by, bx + tw + 40, by + th + 24],
        radius=16, fill=leaf,
    )
    draw.text((bx + 20 + tw // 2, by + 12 + th // 2), badge_text,
              font=font_badge, fill=(255, 255, 255), anchor="mm")

    # Crayon underline squiggle
    import random
    rng = random.Random(77)
    sx = W // 2 - 180
    sy = y_start + total_h + 40
    pts = [(sx + j * 10, sy + rng.randint(-5, 5)) for j in range(36)]
    draw.line(pts, fill=leaf, width=5, joint="curve")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================================
# COLORING BOOK (Feature 2)
# ============================================================================

COLORING_COVER_HTML = """
<div class="book-cover coloring-cover">
  <p class="cover-kicker">a DoodleBook coloring book</p>
  <h2 class="cover-title">{title}</h2>
  <span class="engine-badge sketch">a coloring book to color in</span>
</div>
"""

COLORING_PAGE_HTML = """
<div class="book-page" style="page-break-after: always;">
  <div class="page-art coloring-art">
    <img src="{img_b64}" alt="Coloring page {page_num}"/>
  </div>
  <p class="page-text coloring-text">{text}</p>
  <span class="page-num">~ {page_num} ~</span>
</div>
"""


def build_coloring_html(
    outline_imgs: list,
    page_texts: list,
    title: str,
) -> str:
    """Build coloring-book HTML — same layout, outline images."""
    cover = COLORING_COVER_HTML.format(title=title)

    pages_html = ""
    for i, (img_bytes, text) in enumerate(zip(outline_imgs, page_texts)):
        pages_html += COLORING_PAGE_HTML.format(
            img_b64=_img_src_for_bytes(img_bytes),
            text=text,
            page_num=i + 1,
        )

    return f"""<div class="book-container">
    {cover}
    {pages_html}
    </div>"""


def _img_src_for_bytes(img_bytes: bytes) -> str:
    """Persist an image to temp and return a Gradio-served file URL.

    Inline base64 made the final HTML payload >10MB for a 6-page book, which
    caused the frontend to stall even though backend generation succeeded.
    File-backed URLs keep the final update small and let the browser fetch page
    images incrementally.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(img_bytes)
        path = tmp.name
    return f"/gradio_api/file={urllib.parse.quote(path)}"


# ============================================================================
# PDF EXPORT
# ============================================================================

def _pdf_cover_page(pdf: FPDF, cover_png_bytes: bytes):
    """Place a styled PNG cover as the current PDF page."""
    img = Image.open(io.BytesIO(cover_png_bytes))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, "PNG")
        pdf.image(tmp.name, x=0, y=0, w=210, h=297)
    os.unlink(tmp.name)


def _safe_text(s: str) -> str:
    """Make text safe for fpdf's Latin-1 core fonts.

    The page text comes from the story generator and can contain Unicode
    punctuation (em-dash —, curly quotes, ellipsis). fpdf's Helvetica is
    Latin-1 only and RAISES on those, which previously killed PDF generation
    and left the download button missing. Map the common ones, then drop any
    remaining non-Latin-1 char so a PDF is ALWAYS produced.
    """
    repl = {
        "—": "-", "–": "-", "‒": "-",
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "…": "...", " ": " ", "​": "",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def export_pdf(
    pages_images: list,
    pages_texts: list,
    title: str,
    output_path: str = None,
) -> str:
    """Export storybook as PDF with styled cover."""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".pdf", prefix="doodlebook_")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)

    # Styled cover
    cover_png = render_cover_image(title, "illustrated by FLUX.2-klein", "story")
    pdf.add_page()
    _pdf_cover_page(pdf, cover_png)

    for i, (img_bytes, text) in enumerate(zip(pages_images, pages_texts)):
        pdf.add_page()
        img = Image.open(io.BytesIO(img_bytes))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name, "PNG")
            pdf.image(tmp.name, x=10, y=10, w=190)
        os.unlink(tmp.name)

        pdf.set_y(210)
        pdf.set_font("Helvetica", "", 16)
        pdf.multi_cell(0, 10, _safe_text(text), align="C")

        pdf.set_y(270)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, f"Page {i+1}", ln=True, align="C")

    pdf.output(output_path)
    return output_path


def export_coloring_pdf(
    outline_imgs: list,
    page_texts: list,
    title: str,
    output_path: str = None,
) -> str:
    """Export coloring book as PDF with styled cover."""
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".pdf", prefix="doodlebook_coloring_")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)

    # Styled cover
    cover_png = render_cover_image(title, "a coloring book to color in", "coloring")
    pdf.add_page()
    _pdf_cover_page(pdf, cover_png)

    for i, (img_bytes, text) in enumerate(zip(outline_imgs, page_texts)):
        pdf.add_page()
        img = Image.open(io.BytesIO(img_bytes))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name, "PNG")
            pdf.image(tmp.name, x=10, y=10, w=190)
        os.unlink(tmp.name)

        pdf.set_y(210)
        pdf.set_font("Helvetica", "", 14)
        pdf.multi_cell(0, 10, _safe_text(text), align="C")

        pdf.set_y(270)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, f"Page {i+1}", ln=True, align="C")

    pdf.output(output_path)
    return output_path
