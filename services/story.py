"""
Story generation service — calls modal_story_gen for MiniCPM5-1B inference.

C2 Compliance: 3-layer JSON parser + template fallback.
"""

from config import STORY_MODEL, GENERATION_PARAMS
import os
import logging

logger = logging.getLogger(__name__)


def generate_story(hero_name: str, theme: str, age: int = None) -> dict:
    """
    Generate a 6-page children's story.

    Story runs LOCALLY by default: the local generator is instant and
    theme-accurate, whereas the deployed MiniCPM5-1B was slower (T4 cold start)
    and lower quality (the 1B model parroted the few-shot example). Set
    DOODLEBOOK_STORY_MODAL=1 to route the story to the Modal MiniCPM model.

    Args:
        hero_name: Main character name
        theme: Story theme (e.g., "brave adventure")
        age: Target age (default from config)

    Returns:
        dict with keys: title, character_description, pages[{page, text, scene}]
    """
    if age is None:
        age = GENERATION_PARAMS.target_age

    # 1) Real model on Modal (MiniCPM) — opt-in only
    if os.environ.get("DOODLEBOOK_STORY_MODAL", "0") == "1":
        try:
            import modal
            fn = modal.Function.from_name("doodlebook-story", "generate_story")
            story = fn.remote(hero_name, theme, age)
            if story and story.get("pages"):
                logger.info("Story generated via Modal MiniCPM")
                return story
        except Exception as e:
            logger.info(f"Modal story unavailable ({e}); using local generator")

    # 2) Rich local generator (theme-accurate, varied) — DEFAULT
    try:
        from modal_workers.modal_story_gen import generate_story_local
        return generate_story_local(hero_name, theme, age)
    except Exception as e:
        logger.error(f"Local story generation failed: {e}")
        return _fallback_story(hero_name, theme, age)


def _fallback_story(hero_name: str, theme: str, age: int) -> dict:
    """
    Deterministic template fallback (C2 Layer 3).
    NEVER crashes - always returns valid 6-page book.
    """
    return {
        "title": f"{hero_name}'s {theme.title()}",
        "character_description": f"A friendly character named {hero_name}, drawn in crayon style with bright colors, suitable for age {age}",
        "pages": [
            {"page": 1, "text": f"Once upon a time, there was a character named {hero_name}.", "scene": "Character introduction"},
            {"page": 2, "text": f"{hero_name} loved going on adventures.", "scene": "Adventure begins"},
            {"page": 3, "text": f"One day, {hero_name} discovered something magical.", "scene": "Discovery moment"},
            {"page": 4, "text": f"With courage, {hero_name} faced the challenge.", "scene": "Challenge scene"},
            {"page": 5, "text": f"Friends helped {hero_name} succeed.", "scene": "Teamwork scene"},
            {"page": 6, "text": f"And they all lived happily ever after. The end.", "scene": "Happy ending"}
        ]
    }
