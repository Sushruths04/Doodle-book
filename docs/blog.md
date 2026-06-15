# Field Notes: Cross-Page Character Consistency in DoodleBook

*How we keep the same hand-drawn hero across all six pages of a storybook — from a single child's doodle, with no per-user training — using FLUX.2-klein, a canonical-character pass, seed-locking, and a fixed character description.*

---

## The Challenge

The core problem in AI-generated storybooks is **character consistency**. Generate six pages independently and you get six different characters — different colors, proportions, and style on every page. The magic is gone.

We needed: **the same character, in the same crayon style, across all six pages — and it has to look like the picture the child actually drew.**

A tempting answer is "train a LoRA per child." That is infeasible at demo time (LoRA training is minutes-to-hours on a real GPU), so we deliberately did **not** rely on per-user training. Instead, character identity comes from the child's own drawing, carried through the book by image conditioning.

---

## Our Approach: The Consistency Stack

We layer four complementary techniques. None solves it alone; together they're reliable.

### 1. The canonical-character pass (the real secret sauce)

This is the key idea. We do **two stages**:

1. **Stage 1 — build the canonical hero.** The child's doodle is sent through FLUX.2-klein **img2img** once, turning the rough drawing into one clean, full-body crayon character on a plain background. This is the "model sheet" for the whole book.
2. **Stage 2 — render every page from that canonical.** Each of the six scenes is rendered with the **canonical image itself as the reference** (`image=canonical`), plus the scene description. Because every page is conditioned on the *same* reference image, the hero stays recognizably the same creature the child drew.

```python
canonical = build_canonical(doodle)            # doodle -> one clean hero (img2img)
pages = [render_page(canonical, scene, seed+i)  # every page references the SAME hero
         for i, scene in enumerate(scenes)]
```

Identity comes from the **child's drawing**, not from a trained checkpoint — so a cat doodle becomes a cat hero, a robot becomes a robot, with no training step.

### 2. Seed locking

```python
BASE_SEED = 42
# page i uses BASE_SEED + i + 1  -> reproducible, slight per-page variation
```

Deterministic seeds make a book reproducible (same inputs → same outputs) while still letting each page differ.

### 3. A fixed character description

The same `character_description` string is reused on every page as a text anchor (and is the fallback identity when no doodle is uploaded, via text2img).

### 4. Style in the prompt

A consistent crayon art-style suffix ("hand-drawn crayon storybook illustration, waxy crayon texture…") keeps the medium uniform across pages.

---

## The Coloring Book: redraw, don't trace

A matching coloring book is the second output. Our first attempt **traced** the finished crayon pages with classical edge/region detection — but crayon texture shattered busy backgrounds into speckle, producing pages no child could color.

The fix: hand each finished color page back to FLUX as **img2img** with a "clean black-and-white coloring-book line art" prompt. FLUX understands the scene semantically and redraws clean shape outlines that *match* the story page, instead of tracing texture. A tiny local pass then thresholds it to crisp black-on-white.

---

## Key Learnings

1. **Seed alone isn't enough.** Same seed + different prompt still drifts. You need an identity anchor.
2. **Image conditioning beats per-user training.** Feeding the child's actual drawing (as a canonical reference) gives per-character identity *without* training a LoRA per user — the decisive call for a live demo.
3. **For line art, redraw beats trace.** Generating coloring pages with a model that understands the scene is far cleaner than stripping color out of a finished texture.
4. **4B is the right renderer.** FLUX.2-klein-4B runs fast enough for a live book with strong storybook quality.

---

## Technical Details

**Model stack**
- Image / renderer: `black-forest-labs/FLUX.2-klein-4B`
- Story / brain: `openbmb/MiniCPM5-1B` (1B)
- Voice: `openbmb/VoxCPM2` (2B)

**Inference (storybook pages)**
```yaml
guidance_scale: 1.0
num_inference_steps: 6
width: 768
height: 768
```

**Coloring pages**: FLUX img2img from the color page, `strength≈0.85`, then threshold/despeckle.

---

## Roadmap: a crayon-style LoRA (not yet trained)

A natural next step is a **crayon art-style LoRA** on FLUX.2-klein to make the medium even more uniform and reduce prompt sensitivity. The training pipeline is scaffolded, but **no LoRA has been trained or published yet**, so we make no "Well-Tuned" claim — current consistency is achieved entirely by the canonical-character + seed + description stack above.

---

## Conclusion

Character consistency in AI storybooks doesn't require per-user fine-tuning. A canonical-character pass (drive every page from one img2img render of the child's own drawing), a locked seed, and a fixed description together turn a single crayon doodle into a consistent, narrated, illustrated book — the child's character, their style, brought to life.

---

*Built for Build Small Hackathon 2026 · Adventure in Thousand Token Wood*
