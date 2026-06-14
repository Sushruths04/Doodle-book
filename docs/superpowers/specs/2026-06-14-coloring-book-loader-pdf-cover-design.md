# DoodleBook — Coloring Book, Magic Loader & Styled PDF Covers

**Date:** 2026-06-14
**Status:** Approved design (pending spec review)

Three user-requested features for the DoodleBook app (`run_modal.py` Modal build,
mirrored in `app.py` HF build):

1. **Magic Loader** — an engaging, on-brand wait screen while generation runs.
2. **Coloring Book** — the SAME FLUX images with the colors removed (outlines
   only) so kids can color the exact same pictures. No second/different image is
   ever generated. Opt-in via a checkbox chosen before generating.
3. **Styled PDF covers** — the PDF front page should match the on-screen scrapbook
   cover (cream paper, crayon title with shadow, green "illustrated by" badge),
   for both the storybook and the coloring-book PDFs.

---

## Feature 1 — Magic Loader

### Goal
Generation takes a few minutes (FLUX is the slow stage). Replace the single plain
status line with a crayon-styled animated panel that tells the user what's
happening and showcases the small-model stack (good for judges too).

### Approach
Pure-CSS rotating messages (no JS, no per-page streaming — reliable inside Gradio
`gr.HTML`). A stack of message `<div>`s fade in/out in sequence via staggered CSS
`animation-delay` on an opacity keyframe.

- Helper `magic_loader_html(stage: str, hero_name: str) -> str` in `book_builder.py`.
- Rotating messages (image stage — the long one):
  - `✏️ MiniCPM is dreaming up {hero}'s story…`
  - `🎨 FLUX is painting your 6 pages…`
  - `🔊 VoxCPM is recording the narration…`
  - `💡 Did you know? Your whole storybook runs on tiny models!`
- `create_book` yields the loader HTML into `book_display` during the story and
  image stages (it already streams stage-by-stage).

### CSS
New `.magic-loader` / `.ml-msg` rules in the `ui/layout.py` CSS string. Each
`.ml-msg` is absolutely stacked; `animation: ml-cycle Ns infinite` with
`animation-delay: i*step`. Respects existing `prefers-reduced-motion` block
(messages still legible, just no fade).

---

## Feature 2 — Coloring Book (checkbox-triggered)

### Goal
When the user opts in, produce printable black-and-white outline pages **of the
exact same FLUX images** (same scenes, same character) for kids to color. The
coloring page is the color page with its colors removed — never a newly generated
or different image.

### Trigger
New checkbox in the input card: `🖍️ Also make a coloring book` (`make_coloring`),
chosen before pressing "Make my book!". Outline pages are produced automatically
with the book when checked.

### Generation (same image, colors removed — instant, free)
Process the already-generated color images locally with OpenCV to strip the fills
and keep the outlines. No extra Modal call, ~seconds, and the result is the SAME
picture as line art. There is no FLUX re-generation and no "HD" alternative — that
guarantees the coloring page always matches the color page exactly.

### New module: `services/coloring.py`
- `to_line_art(png_bytes: bytes) -> bytes`
  - OpenCV: grayscale → light blur → `adaptiveThreshold(GAUSSIAN_C, THRESH_BINARY,
    blockSize≈11, C≈2)` to get black lines on white; remove tiny speck components
    (reuse the cleanup approach from `services/images.py:_doodle_to_cartoon`).
  - Returns a PNG (white background, black outlines).
  - On any failure: fall back to grayscale Otsu threshold; never raises.
- `derive_coloring_pages(color_imgs: list[bytes]) -> list[bytes]`.

### `book_builder.py` additions
- `build_coloring_html(outline_imgs, page_texts, title) -> str` — same scrapbook
  layout as `build_book_html` but with a coloring-book cover badge and outline
  images (text kept small/light beneath each page).
- `export_coloring_pdf(outline_imgs, page_texts, title, path) -> str` — styled
  cover (Feature 3) + outline pages, print-friendly.

### UI (`ui/layout.py`)
- Input card: `make_coloring = gr.Checkbox(...)` (styled like `tiny_mode`).
- Output card (below the storybook + downloads):
  - `coloring_display = gr.HTML(visible=False)`
  - `coloring_pdf_download = gr.DownloadButton("Download Coloring Book (PDF)", visible=False)`

### Data flow
```
create_book(doodle, char_name, theme, hero, tiny, voice, make_coloring):
  yield magic_loader(story)         -> book_display
  story = services.story.generate_story(...)
  yield magic_loader(images)
  color_imgs, engine = services.images.generate_book_pages(...)
  book_html = build_book_html(color_imgs, ...)
  if make_coloring:
      outlines = services.coloring.derive_coloring_pages(color_imgs)  # SAME imgs, colors removed
      coloring_html = build_coloring_html(outlines, ...)
      coloring_pdf = export_coloring_pdf(outlines, ...)
  audio = services.tts.speak_book(...)
  pdf = export_pdf(color_imgs, ...)
  yield final: book_html, status, audio, pdf(visible), story_json, image_info,
               trace, coloring_html(visible if make_coloring),
               coloring_pdf(visible if make_coloring)
```
No `gr.State` and no regenerate handler are needed — the coloring pages are derived
directly from `color_imgs`, so they always match the color book.

---

## Feature 3 — Styled PDF cover

### Goal
Replace the plain Helvetica text title page in `export_pdf` with a cover that
matches the on-screen scrapbook cover.

### Approach
Render the cover as a full-page **PIL image** and place it as PDF page 1 (PIL is
already a dependency; works on HF with no browser).

- `book_builder.render_cover_image(title, badge_text, kind="story") -> bytes`
  - Canvas ~1240×1754 (A4 @150dpi). Cream fill (`#fff8e6`) + subtle speckle.
  - Kicker `a DoodleBook story` — Caveat, berry (`#d6517a`), centered upper third.
  - Title — Gaegu Bold, large, centered, wrapped to ≤2 lines, layered shadow:
    offset draw in crayon-sun (`#f4c64a`) then ink (`#2e2a26`) on top.
  - Badge — rounded rect, crayon-leaf (`#74b85a`), white Gaegu text `badge_text`.
  - `badge_text`: story = `illustrated by FLUX.2-klein`; coloring =
    `a coloring book to color in`.
  - Fonts from `assets/fonts/Gaegu-Bold.ttf` + `Caveat.ttf` via
    `ImageFont.truetype`; fall back to `ImageFont.load_default()` if missing.
- `export_pdf` and `export_coloring_pdf` use `render_cover_image(...)` for page 1.

### Fonts to bundle (OFL, free to redistribute)
- `assets/fonts/Gaegu-Bold.ttf`
- `assets/fonts/Caveat.ttf` (Regular or SemiBold)

---

## Error handling (never crash a generation)
- `to_line_art`: OpenCV failure → Otsu threshold fallback → original image.
- `render_cover_image`: missing font → default font; PIL failure → old text cover.
- `create_book` body wrapped so exceptions yield an error state, not a throw.

## Testing
- Unit:
  - `to_line_art` → output PNG is mostly white with a meaningful fraction of dark
    pixels (line art), valid dimensions.
  - `render_cover_image` → returns a valid PNG of expected size; runs with fonts
    present and with fonts removed (fallback path).
  - `export_coloring_pdf` / `export_pdf` → produce a non-empty PDF whose page 1 is
    the cover image.
- Manual (run_modal.py): checkbox ON → loader animates → coloring section +
  "Download Coloring Book (PDF)" appear; the outline pages are visibly the SAME
  scenes as the color book; both PDFs open with the styled cover.

## Files
- **New:** `services/coloring.py`; `assets/fonts/Gaegu-Bold.ttf`,
  `assets/fonts/Caveat.ttf`; this spec.
- **Changed:** `ui/layout.py` (loader CSS, checkbox, coloring outputs, wiring);
  `run_modal.py` (`create_book`) is the primary target; `app.py` mirrors the same
  three features (loader, coloring from its own color images, styled cover) for HF
  parity; `book_builder.py` (`render_cover_image`, `build_coloring_html`,
  `export_coloring_pdf`, `magic_loader_html`, `export_pdf` cover).
  No Modal worker changes are needed.

## Out of scope (YAGNI)
- Regenerating a separate line-art via FLUX. The user wants the EXACT same image,
  just without color — so the coloring page is always derived from the color image,
  never re-generated. (No "HD lines" button, no second FLUX run.)
- Per-page progress bar (needs Modal streaming; loader covers the need).
- Saving/downloading the narration audio (separate future ask).
- Animated illustrated pipeline loader (chose the lighter CSS version).
</content>
