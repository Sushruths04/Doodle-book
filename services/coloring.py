"""
Coloring-book service — turn the color pages into clean coloring-book outlines.

Primary path: hand each finished color page back to FLUX (Modal `render_lineart`)
which REDRAWS it as clean line art. FLUX understands the scene (kid + clouds +
hills) and traces shape boundaries, so busy/textured backgrounds no longer
shatter into speckle the way the OpenCV edge-trace did. A tiny local `_crispen`
then forces it to pure black-on-white. Falls back to the OpenCV region tracer,
then to PIL, then to the original — so it never raises and is never worse than
before.
"""

import io
import logging

logger = logging.getLogger(__name__)


def _crispen(png_bytes: bytes) -> bytes:
    """Force FLUX line art to crisp black-on-white: threshold + despeckle.

    FLUX may leave faint gray or off-white; coloring pages must be pure outlines.
    Cheap, local, no GPU. Returns the input unchanged if processing fails."""
    try:
        import cv2
        import numpy as np
        from PIL import Image

        gray = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))
        # anything not near-white becomes a candidate line, then threshold hard
        gray = cv2.medianBlur(gray, 3)
        _, ink = cv2.threshold(gray, 214, 255, cv2.THRESH_BINARY_INV)  # ink=255
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # drop tiny specks, keep real strokes
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        clean = np.zeros_like(ink)
        min_area = max(14, int(0.00025 * ink.size))
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[lbl == i] = 255
        clean = cv2.dilate(clean, np.ones((2, 2), np.uint8), iterations=1)
        out = Image.fromarray(255 - clean).convert("RGB")  # black ink on white
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"_crispen failed ({e}); using FLUX line art as-is")
        return png_bytes


def _to_line_art_opencv(png_bytes: bytes) -> bytes:
    """
    Convert one colored page into CLEAN coloring-book outlines: the OUTLINE of
    each region (character, clouds, mountain, sky, ground) with NO inner crayon
    hatching/texture inside a region.

    Why not edge detection: Canny traces every crayon stroke, so shaded areas
    (mountains, ground) fill with speckly internal lines. Instead we FLATTEN the
    image into a few flat color regions and trace only the boundaries BETWEEN
    regions — giving silhouettes a child can color inside.

    Steps: bilateral-smooth (kill crayon texture) -> k-means quantize to ~6 flat
    colors -> median-clean the label map (dissolve tiny texture regions) -> mark
    pixels where the region label changes -> despeckle. K=6 + median 7 is the
    sweet spot: fewer colors / stronger median dissolves the character too. Fast
    (~0.4s). Falls back to a PIL contour filter, then the original, so it never
    raises.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image

        rgb = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        h, w = rgb.shape[:2]

        # 1) flatten crayon texture/shading (keep the big shape edges)
        sm = cv2.bilateralFilter(rgb, 9, 120, 120)
        sm = cv2.bilateralFilter(sm, 9, 120, 120)

        # 2) quantize colors into a few flat regions
        Z = sm.reshape(-1, 3).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(Z, 6, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lab = labels.reshape(h, w).astype(np.uint8)

        # 3) dissolve tiny speckle regions so only real shapes survive
        lab = cv2.medianBlur(lab, 7)
        lab = cv2.medianBlur(lab, 7)

        # 4) trace ONLY the boundaries between regions
        L = lab.astype(np.int32)
        edges = np.zeros((h, w), np.uint8)
        edges[:, :-1] |= (L[:, :-1] != L[:, 1:]).astype(np.uint8) * 255
        edges[:-1, :] |= (L[:-1, :] != L[1:, :]).astype(np.uint8) * 255
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        # 5) despeckle: drop tiny boundary fragments, keep real outlines
        num, lbl, stats, _ = cv2.connectedComponentsWithStats(edges, 8)
        clean = np.zeros_like(edges)
        min_area = max(10, int(0.0003 * edges.size))
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[lbl == i] = 255

        out = Image.fromarray(255 - clean).convert("RGB")  # invert: black on white
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        logger.warning(f"to_line_art region path failed ({e}); trying PIL contour")
        return _to_line_art_pil(png_bytes)


def _to_line_art_pil(png_bytes: bytes) -> bytes:
    """No-OpenCV fallback: PIL CONTOUR gives dark outlines on a light ground."""
    try:
        from PIL import Image, ImageFilter, ImageOps

        im = Image.open(io.BytesIO(png_bytes)).convert("L")
        contour = im.filter(ImageFilter.CONTOUR)          # light bg, dark edges
        contour = ImageOps.autocontrast(contour)
        out = contour.convert("RGB")
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"PIL line-art fallback failed ({e}); returning original")
        return png_bytes


def derive_coloring_pages(color_imgs: list[bytes]) -> list[bytes]:
    """Clean coloring-book outline for every page.

    Primary: FLUX `render_lineart` on Modal, fanned out across containers via
    .starmap (concurrent), then crispened to pure black-on-white. If Modal is
    unavailable or returns a blank, fall back per-page to the OpenCV tracer.
    """
    # 1) FLUX line art (parallel), then local crispen
    try:
        import modal
        fn = modal.Function.from_name("doodlebook-image-gen", "render_lineart")
        raw = list(fn.starmap([(img, 7 + i) for i, img in enumerate(color_imgs)]))
        if raw and all(raw):
            return [_crispen(r) for r in raw]
        raise RuntimeError("render_lineart returned empty page(s)")
    except Exception as e:
        logger.warning(f"FLUX line art failed ({e}); OpenCV outline fallback")

    # 2) Fallback: derive outlines locally from the color pages
    return [_to_line_art_opencv(img) for img in color_imgs]
