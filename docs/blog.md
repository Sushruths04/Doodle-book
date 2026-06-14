# Field Notes: FLUX + LoRA Character Consistency

*How we achieved cross-page character consistency in DoodleBook using FLUX.2-klein, seed-locking, and a crayon-style LoRA.*

---

## The Challenge

The core problem in AI-generated storybooks is **character consistency**. If you generate 6 pages of a story independently, each page produces a different character — different colors, different proportions, different style. The magic is lost.

We needed: **the same character, in the same art style, across all 6 pages.**

---

## Our Approach: The Consistency Stack

We didn't rely on a single technique. Instead, we layered three complementary approaches:

### 1. Seed Locking

```python
BASE_SEED = 42
def page_seed(page_num):
    return BASE_SEED + page_num  # Page 0: 42, Page 1: 43, ...
```

Each page uses a deterministic seed derived from a locked base. This ensures:
- Reproducible generation (same inputs = same outputs)
- Slight variation between pages (different seeds)
- Consistent "feel" across the book

### 2. Character Description Reuse

Every page uses the **exact same** `character_description` string:

```python
prompt = f"""
{character_description},  # IDENTICAL on every page
{scene_description},      # UNIQUE per page
{art_style}, page {i+1} of children's book
"""
```

The character description acts as an anchor, keeping the model's interpretation consistent.

### 3. LoRA Fine-Tuning (The Secret Sauce)

We trained a **crayon-style LoRA** on FLUX.2-klein:

- **Trigger token:** `[DOODLECHAR]`
- **Training data:** 10-15 crayon-style character images
- **Rank:** 16 (balances quality vs. file size)
- **Steps:** 300

The LoRA teaches FLUX to generate images in a specific art style. Combined with the character description, this creates a consistent visual identity.

---

## The Results

### Before LoRA (Base FLUX)
- Pages look like generic AI art
- Character changes dramatically between pages
- No consistent style

### After LoRA + Consistency Stack
- Same character across all 6 pages
- Consistent crayon art style
- Recognizable as "the same book"

---

## Key Learnings

1. **Seed alone isn't enough.** Different prompts with the same seed produce different characters. You need description consistency too.

2. **LoRA provides style, not identity.** The LoRA teaches the art style (crayon, watercolor, etc.), but the character identity comes from the prompt.

3. **Image conditioning helps.** When available, feeding the child's actual doodle as an image prompt (via img2img) dramatically improves style matching.

4. **Quality vs. speed tradeoff.** FLUX.2-klein-4B (4B params) runs faster than 9B with minimal quality loss for storybook art.

---

## Technical Details

### Model Stack
- **Image:** FLUX.2-klein-4B + crayon-style LoRA
- **Story:** MiniCPM5-1B (1B)
- **TTS:** VoxCPM2 (2B)

### Training Config
```yaml
rank: 16
alpha: 16
learning_rate: 1e-4
steps: 300
resolution: 512
batch_size: 1
```

### Inference Config
```yaml
guidance_scale: 3.5
num_inference_steps: 20  # Standard mode
                       4  # Tiny Mode (SD-Turbo)
width: 768
height: 512
```

---

## Conclusion

Character consistency in AI storybooks requires a multi-layered approach: seed locking for reproducibility, prompt engineering for identity, and LoRA fine-tuning for style. No single technique solves the problem alone, but together they create a reliable system.

The result? A child's crayon drawing becomes a consistent, narrated, illustrated storybook — their character, their style, brought to life by AI.

---

*Built for Build Small Hackathon 2026 · Thousand Token Wood Track*
