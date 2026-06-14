"""
Modal story generation — MiniCPM5-1B on T4 GPU.

C2 Compliance: 3-layer JSON parser + template fallback
- Layer 1: Regex extraction of {...} block
- Layer 2: json-repair / json5 parsing
- Layer 3: Deterministic template fallback (NEVER crashes)

Few-shot prompt with ONE full exemplar for reliable JSON output.
Greedy decode (do_sample=False) for determinism.
"""

import modal
import re
import json
import logging

logger = logging.getLogger(__name__)

app = modal.App("doodlebook-story")

story_env = modal.Image.debian_slim().pip_install(
    "transformers>=4.40", "torch", "accelerate", "sentencepiece"
)


# ============================================================================
# FEW-SHOT EXEMPLAR
# ============================================================================

FEW_SHOT_EXEMPLAR = """
Write a 6-page children's storybook for age 5 about Luna the cat with theme: brave adventure.

Return ONLY valid JSON:
{
  "title": "Luna's Brave Adventure",
  "character_description": "A small orange tabby cat named Luna with big green eyes, whiskers, and a tiny red scarf",
  "pages": [
    {"page": 1, "text": "Luna was a small orange cat who loved to explore.", "scene": "Luna sitting by the window looking outside"},
    {"page": 2, "text": "One sunny morning, Luna saw something sparkling in the forest.", "scene": "Luna spotting a glow in the trees"},
    {"page": 3, "text": "Bravely, Luna crept into the forest to investigate.", "scene": "Luna walking cautiously through trees"},
    {"page": 4, "text": "It was a tiny fairy stuck in a spider web!", "scene": "Luna discovering a fairy in trouble"},
    {"page": 5, "text": "Luna gently freed the fairy with her paw.", "scene": "Luna carefully helping the fairy"},
    {"page": 6, "text": "The fairy thanked Luna and they became friends forever.", "scene": "Luna and fairy playing together at sunset"}
  ]
}
"""


# ============================================================================
# STORY GENERATION PROMPT
# ============================================================================

def build_prompt(hero_name: str, theme: str, age: int) -> str:
    """Build few-shot prompt for story generation."""
    return f"""{FEW_SHOT_EXEMPLAR}

Write a 6-page children's storybook for age {age} about {hero_name} with theme: {theme}.

Return ONLY valid JSON:
"""


# ============================================================================
# 3-LAYER JSON PARSER (C2)
# ============================================================================

def parse_story_json(raw_output: str) -> dict:
    """
    3-layer parser: regex → json5/repair → template fallback.
    
    Layer 1: Extract {...} block with regex
    Layer 2: Parse with json.loads, repair common issues
    Layer 3: Return deterministic template (NEVER crashes)
    
    Args:
        raw_output: Raw model output string
        
    Returns:
        Parsed story dict with keys: title, character_description, pages
    """
    # Layer 1: Regex extraction
    story = _layer1_regex_extract(raw_output)
    if story:
        return story
    
    # Layer 2: JSON repair
    story = _layer2_json_repair(raw_output)
    if story:
        return story
    
    # Layer 3: Template fallback (NEVER crashes)
    logger.warning("All parsing failed, using template fallback")
    return _layer3_template_fallback(raw_output)


def _layer1_regex_extract(text: str) -> dict | None:
    """Layer 1: Extract {...} block with regex."""
    try:
        # Find the outermost {...} block
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return None
        
        json_str = match.group(0)
        story = json.loads(json_str)
        
        # Validate structure
        if _validate_story_structure(story):
            return story
        return None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _layer2_json_repair(text: str) -> dict | None:
    """Layer 2: Repair common JSON issues and parse."""
    try:
        # Find the {...} block
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return None
        
        json_str = match.group(0)
        
        # Common repairs
        json_str = _repair_json(json_str)
        
        story = json.loads(json_str)
        
        if _validate_story_structure(story):
            return story
        return None
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _repair_json(json_str: str) -> str:
    """Repair common JSON issues from 1B model output."""
    # Remove trailing commas before } or ] (with optional whitespace)
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # Remove single-line // comments
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    
    # Remove multi-line comments /* ... */
    json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
    
    # Fix unescaped newlines in strings
    json_str = re.sub(r'(?<=")\n(?=")', '\\n', json_str)
    
    # Fix missing quotes around keys (word before colon)
    json_str = re.sub(r'(\s)(\w+)(\s*:)', r'\1"\2"\3', json_str)
    
    return json_str


def _validate_story_structure(story: dict) -> bool:
    """Validate story has required structure."""
    required_keys = ["title", "character_description", "pages"]
    if not all(k in story for k in required_keys):
        return False
    
    pages = story.get("pages", [])
    if not isinstance(pages, list) or len(pages) < 1:
        return False
    
    # Check first page has required fields
    first_page = pages[0]
    if not all(k in first_page for k in ["page", "text", "scene"]):
        return False
    
    return True


def _layer3_template_fallback(raw_output: str) -> dict:
    """
    Layer 3: Deterministic template fallback.
    NEVER crashes - always returns valid 6-page book.
    """
    # Try to extract any useful text from raw output
    extracted_text = raw_output[:200] if raw_output else "an adventure"
    
    return {
        "title": "A Wonderful Adventure",
        "character_description": f"A friendly character who went on {extracted_text}",
        "pages": [
            {"page": 1, "text": "Once upon a time, there was a character who loved adventures.", "scene": "Character introduction"},
            {"page": 2, "text": "One day, something exciting happened.", "scene": "Inciting incident"},
            {"page": 3, "text": "The character bravely faced the challenge.", "scene": "Rising action"},
            {"page": 4, "text": "With courage and kindness, the character succeeded.", "scene": "Climax"},
            {"page": 5, "text": "Friends gathered to celebrate the victory.", "scene": "Resolution"},
            {"page": 6, "text": "And they all lived happily ever after. The end.", "scene": "Happy ending"}
        ]
    }


# ============================================================================
# TEMPLATE STORY (for testing without Modal)
# ============================================================================

# ---------------------------------------------------------------------------
# Local story generator — theme-accurate, character-aware, and VARIED.
# Each theme has its own 6-beat arc; slots ({place}, {friend}, {thing}, {feeling})
# are filled from per-theme word banks chosen by a seed derived from hero+theme,
# so different heroes/themes produce different books (no more identical text).
# ---------------------------------------------------------------------------

_PLACES = ["whispering forest", "sunny meadow", "sparkling river", "cloud-top hill",
           "hidden garden", "snowy valley", "rolling sand dunes", "moonlit lake"]
_FRIENDS = ["a shy little fox", "a lost baby bird", "a giggling firefly", "a sleepy turtle",
            "a tiny dragon", "a kind old owl", "a bouncing bunny", "a glowing jellyfish"]
_THINGS = ["a glowing key", "a singing flower", "a map of stars", "a tiny golden bell",
           "a magic seed", "a shimmering shell", "a friendly lantern", "a curious door"]


def _theme_arc(theme: str, hero: str, place: str, friend: str, thing: str) -> dict:
    """Return {title, pages[6]} for the given theme, with slots filled in."""
    T = {
        "brave adventure": {
            "title": f"{hero}'s Brave Adventure",
            "pages": [
                (f"{hero} woke up wanting to explore the world.", f"{hero} standing at the edge of a {place}"),
                (f"At the {place}, {hero} found {thing} glowing softly.", f"{hero} discovering {thing}"),
                (f"Taking a deep breath, {hero} bravely followed where it led.", f"{hero} walking bravely into the {place}"),
                (f"There, {friend} was stuck and a little scared.", f"{friend} in trouble, {hero} nearby"),
                (f"{hero} was brave and gently helped {friend} get free.", f"{hero} helping {friend}"),
                (f"Side by side they went home, and {hero} felt brave and proud.", f"{hero} and {friend} heading home at sunset"),
            ],
        },
        "making a new friend": {
            "title": f"{hero} Makes a Friend",
            "pages": [
                (f"{hero} was playing alone in the {place}.", f"{hero} playing alone in a {place}"),
                (f"Nearby, {friend} sat all by itself, looking lonely.", f"{friend} sitting alone"),
                (f"{hero} walked over and said a cheerful hello.", f"{hero} greeting {friend} with a wave"),
                (f"They shared {thing} and laughed together.", f"{hero} and {friend} sharing {thing}"),
                (f"All afternoon they played their favorite games.", f"{hero} and {friend} playing games"),
                (f"Now {hero} knew: a friend is just a hello away.", f"{hero} and {friend} smiling together"),
            ],
        },
        "overcoming a fear": {
            "title": f"{hero} and the Big Brave Day",
            "pages": [
                (f"{hero} felt scared of the dark {place}.", f"{hero} looking nervously at a dark {place}"),
                (f"But {friend} needed {thing} from inside it.", f"{friend} asking {hero} for help"),
                (f"{hero}'s tummy felt wobbly, but {hero} took one small step.", f"{hero} taking a brave first step into the {place}"),
                (f"One step, then another — it wasn't so scary after all.", f"{hero} walking carefully, growing braver"),
                (f"{hero} found {thing} and carried it back proudly.", f"{hero} holding {thing} triumphantly"),
                (f"{hero} learned that being brave means trying, even when you're scared.", f"{hero} and {friend} celebrating"),
            ],
        },
        "helping someone": {
            "title": f"{hero} Lends a Hand",
            "pages": [
                (f"One morning {hero} skipped through the {place}.", f"{hero} walking happily through a {place}"),
                (f"{hero} heard a tiny cry — it was {friend}!", f"{hero} noticing {friend} in need"),
                (f"{friend} had dropped {thing} and couldn't reach it.", f"{friend} reaching for {thing}"),
                (f"{hero} thought hard and came up with a clever plan.", f"{hero} thinking of a plan"),
                (f"Together they got {thing} back, and {friend} cheered.", f"{hero} and {friend} succeeding together"),
                (f"Helping others made {hero}'s heart feel warm and happy.", f"{hero} and {friend} hugging"),
            ],
        },
        "lost and found": {
            "title": f"{hero} and the Lost {thing.split()[-1].title()}",
            "pages": [
                (f"{hero} was playing when {thing} suddenly went missing.", f"{hero} searching for {thing}"),
                (f"{hero} looked all around the {place}.", f"{hero} looking around a {place}"),
                (f"Along the way, {hero} met {friend} who wanted to help.", f"{hero} meeting {friend}"),
                (f"They followed tiny clues together, step by step.", f"{hero} and {friend} following a trail"),
                (f"At last, {thing} was found tucked beneath a leaf!", f"{hero} finding {thing}"),
                (f"{hero} hugged {friend} and thanked them for never giving up.", f"{hero} and {friend} happy together"),
            ],
        },
        "learning something new": {
            "title": f"{hero} Learns to Soar",
            "pages": [
                (f"{hero} really wanted to learn something new today.", f"{hero} curious in a {place}"),
                (f"{friend} offered to teach {hero} a wonderful trick.", f"{friend} teaching {hero}"),
                (f"The first try wobbled and didn't work at all.", f"{hero} trying and stumbling"),
                (f"{hero} practiced again and again, never giving up.", f"{hero} practicing hard"),
                (f"Suddenly it worked, with {thing} sparkling in the air!", f"{hero} succeeding with {thing}"),
                (f"{hero} beamed — trying your best helps you grow.", f"{hero} and {friend} celebrating the win"),
            ],
        },
    }
    return T.get(theme, T["brave adventure"])


def generate_story_local(hero_name: str, theme: str, age: int = 5) -> dict:
    """
    Theme-accurate, varied, character-aware story (no Modal/GPU required).
    Deterministic per (hero, theme) but different across heroes/themes.
    """
    import random
    hero = (hero_name or "Little Hero").strip()
    hero = hero[:1].upper() + hero[1:] if hero else "Little Hero"
    rng = random.Random(hash((hero.lower(), theme)) & 0xFFFFFFFF)
    place = rng.choice(_PLACES)
    friend = rng.choice(_FRIENDS)
    thing = rng.choice(_THINGS)

    arc = _theme_arc(theme, hero, place, friend, thing)
    pages = [{"page": i + 1, "text": t, "scene": s} for i, (t, s) in enumerate(arc["pages"])]

    return {
        "title": arc["title"],
        "character_description": (
            f"{hero}, a friendly storybook character, bright crayon colors, "
            f"bold outlines, simple children's-book style"
        ),
        "pages": pages,
    }


# ============================================================================
# MODAL FUNCTION
# ============================================================================

@app.function(gpu="T4", image=story_env, timeout=120)
def generate_story(character_name: str, theme: str, age: int = 5) -> dict:
    """
    Generate a 6-page children's story via MiniCPM5-1B.
    
    C2 Compliance:
    - Few-shot prompt with ONE full exemplar
    - Greedy decode (do_sample=False)
    - 3-layer parser + template fallback
    - NEVER crashes on bad model output
    
    Args:
        character_name: Main character name
        theme: Story theme
        age: Target age
        
    Returns:
        dict with keys: title, character_description, pages[{page, text, scene}]
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # Load model (MiniCPM ships custom modeling code -> trust_remote_code required)
    model_id = "openbmb/MiniCPM5-1B"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=True
    ).cuda().eval()

    # Build prompt and wrap in the model's chat template (it's an instruct model;
    # a raw prompt generates poorly). enable_thinking=False = no reasoning preamble.
    prompt = build_prompt(character_name, theme, age)
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate with greedy decode for determinism
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=False,  # Greedy for determinism
        )

    response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    # Parse with 3-layer parser (NEVER crashes)
    story = parse_story_json(response)
    
    # Ensure pages list has exactly 6 entries
    while len(story.get("pages", [])) < 6:
        story.setdefault("pages", []).append({
            "page": len(story.get("pages", [])) + 1,
            "text": "And the adventure continued happily.",
            "scene": "Continuing adventure"
        })
    
    return story


@app.function(gpu="T4", image=story_env, timeout=30)
def health_check() -> str:
    """Quick health check for Modal function."""
    return "story_gen_healthy"


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    # Test local generation
    story = generate_story_local("Ziggy", "brave adventure", 5)
    print(json.dumps(story, indent=2))
