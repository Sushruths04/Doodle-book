# FLUX-generated line art for coloring pages

**Date:** 2026-06-14
**Status:** Approved, pending implementation

## Problem

Coloring pages are derived from the finished full-color crayon page by
`services/coloring.py:to_line_art()` (OpenCV: bilateral filter → k-means
quantize → region-boundary trace). On textured/busy backgrounds (sand, hills,
crayon-shaded sky) the crayon strokes fragment into hundreds of stray edges, so
the coloring page is speckled and uncolorable (see the "rolling sand dunes" and
"deep breath" pages). The source was never line art, so tracing it cleanly is
fundamentally hard.

The full-COLOR storybook pages look good and are NOT changing.

## Approach (Option B — keep color pipeline, let FLUX draw the outline)

Stop guessing outlines from pixels. Hand the finished color page back to FLUX
and have it **redraw the scene as clean line art** via img2img. FLUX understands
the scene semantically (character + clouds + hills) so it draws shape boundaries
instead of tracing crayon texture.

### Pipeline

```
canonical char ─┐
                ├─► render_page (color, A10G) ──► STORYBOOK page (unchanged)
story beat   ───┘            │
                             └─► render_lineart (FLUX img2img, A10G)
                                          │
                                          └─► threshold B/W + despeckle ──► COLORING page
```

### Components

1. **`modal_workers/modal_image_gen.py` — new `render_lineart(color_png) -> bytes`**
   - A10G FLUX function (uses shared `GPU_FN` decorator → A10G, 300s timeout,
     120s scaledown).
   - img2img with the **color page as the image reference** (so the coloring
     page matches the story picture exactly: same pose, same composition).
   - Prompt: *"black and white coloring book line drawing, clean bold black
     outlines on a pure white background, no shading, no color, no crayon
     texture, simple shapes a child can color in."*
   - Denoising `strength` tuned high enough to redraw as line art but low enough
     to keep composition. **Exact param verified against the live
     Flux2KleinPipeline and tuned on a real render before sign-off.**

2. **`services/coloring.py` — cleanup + orchestration**
   - Keep a small local `_crispen(png)`: threshold the FLUX output to pure
     black-on-white + despeckle (FLUX may leave faint gray; coloring pages must
     be crisp). Fast, no GPU.
   - `derive_coloring_pages(color_imgs)`: fan the pages out via
     `modal.Function.from_name("doodlebook-image-gen", "render_lineart").starmap`
     (concurrent, like `render_page`), then `_crispen` each.
   - **Fallback:** if the Modal call fails, fall back to the existing OpenCV
     `to_line_art` (renamed `_to_line_art_opencv`) — so it degrades gracefully,
     never worse than today.

### Decisions (confirmed)

- **Trace source:** the finished color page (matches the story picture).
- **When it runs:** only when "Also make a coloring book" is checked — no extra
  cost/time otherwise. `run_modal.py:182` already gates `derive_coloring_pages`
  behind `make_coloring`, so no caller change needed.

### Cost / latency

+1 A10G render per page (~6 extra renders/book), opt-in. Cheap on A10G.

## Ships with the GPU/deadlock fix

Already edited (pending `modal deploy`): image-gen GPU A100→A10G,
`scaledown_window` 1200→120s, per-call `timeout` 24h→300s. The deploy that ships
line art also activates these. See the separate deadlock diagnosis (9 idle A100
containers pinning the 10-GPU account quota → queued calls never timed out →
app spun forever).

## Testing

1. Verify Flux2KleinPipeline img2img accepts a `strength`/denoising param;
   confirm the exact name.
2. Render ONE real book locally with coloring enabled; save the color page and
   the line-art coloring page side by side.
3. Confirm: clean colorable regions on the previously-bad busy pages (sand
   dunes, hills), character preserved, pure black-on-white, no speckle.
4. Tune `strength` if texture survives (raise) or composition drifts (lower).

## Out of scope

- Changing the color storybook render (looks good).
- Programmatic flat-fill colorizing (not needed; color pipeline stays).
