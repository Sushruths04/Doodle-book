"""
DoodleBook — Single Source of Truth for Model IDs, Fallbacks, and Generation Params

Innovations:
- License-aware model selection (prevents non-commercial model usage)
- VRAM-based hardware recommendations for Modal
- Dynamic fallback resolution with logging
- Deterministic seed management for character consistency
- Type-safe model registry with frozen dataclasses

Phase 1, Task 0: All model IDs verified on HuggingFace Hub (June 2026)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# LICENSE TRACKING
# ============================================================================

class LicenseType(Enum):
    """License types for model compliance checking."""
    APACHE_2_0 = "apache-2.0"
    MIT = "mit"
    NON_COMMERCIAL = "non-commercial"
    FLUX_DEV = "flux-dev-non-commercial"
    UNKNOWN = "unknown"


# ============================================================================
# MODEL REGISTRY
# ============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable model configuration with license tracking.
    
    Attributes:
        hub_id: HuggingFace Hub model identifier
        params_b: Model parameters in billions
        license: License type for compliance
        vram_gb: Estimated VRAM requirement in GB
        fallback_id: Alternative model if primary fails
        fallback_reason: Why fallback exists
        is_primary: Whether this is a primary or fallback model
        modal_gpu: Recommended Modal GPU type
        modal_memory: Modal memory in MB
    """
    hub_id: str
    params_b: float
    license: LicenseType
    vram_gb: float
    fallback_id: Optional[str] = None
    fallback_reason: Optional[str] = None
    is_primary: bool = True
    modal_gpu: str = "T4"
    modal_memory: int = 8192
    
    @property
    def can_use_commercially(self) -> bool:
        """Check if model can be used for commercial purposes."""
        return self.license in (LicenseType.APACHE_2_0, LicenseType.MIT)
    
    @property
    def license_warning(self) -> Optional[str]:
        """Return warning if license has restrictions."""
        if self.license == LicenseType.NON_COMMERCIAL:
            return f"NON-COMMERCIAL: {self.hub_id} cannot be used in production"
        if self.license == LicenseType.FLUX_DEV:
            return f"FLUX DEV LICENSE: {self.hub_id} has usage restrictions"
        return None


# ============================================================================
# VERIFIED MODEL REGISTRY (Phase 1, Task 0 — June 2026)
# ============================================================================
#
# All primary models verified on HuggingFace Hub.
# Fallbacks selected for license compatibility (Apache 2.0 preferred).
#

# The three sponsor models DoodleBook actually loads. Fallback/variant configs
# were removed so the HF Space links exactly these three (HF auto-links any
# model id it finds in the repo files).

FLUX_MODEL = ModelConfig(
    hub_id="black-forest-labs/FLUX.2-klein-4B",
    params_b=4.0,
    license=LicenseType.APACHE_2_0,
    vram_gb=13.0,
    modal_gpu="A10G",  # 24GB fits the ~13GB model; A100-40GB was overkill
    modal_memory=32768,
)

STORY_MODEL = ModelConfig(
    hub_id="openbmb/MiniCPM5-1B",
    params_b=1.0,
    license=LicenseType.APACHE_2_0,
    vram_gb=4.0,
    modal_gpu="T4",
    modal_memory=8192,
)

TTS_MODEL = ModelConfig(
    hub_id="openbmb/VoxCPM2",
    params_b=2.0,
    license=LicenseType.APACHE_2_0,
    vram_gb=8.0,
    modal_gpu="T4",
    modal_memory=8192,
)


# ============================================================================
# VOICE PRESETS (VoxCPM2 "voice design")
# ============================================================================
#
# Each preset is a VoxCPM2 voice-design prefix — the (parenthetical) is read as
# an instruction describing the speaker, NOT spoken aloud. Ordered youngest →
# oldest. Default is a young child's voice (the previous adult "storyteller"
# voice sounded too old for the kids).
#
# SINGLE SOURCE OF TRUTH for the live app (app.py). modal_workers/modal_tts.py
# keeps a mirror of these values so the Modal deploy stays import-free — keep
# the two in sync if you edit them.

VOICE_PRESETS: Dict[str, Dict[str, str]] = {
    "kid": {
        "label": "🧒 Little kid",
        "design": "(A sweet little girl around seven years old telling a story to "
                  "her friends, bright high-pitched cheerful child's voice, playful, "
                  "giggly and full of wonder)",
    },
    "big_kid": {
        "label": "🎒 Big kid",
        "design": "(A lively young girl about eleven years old reading a fun story "
                  "aloud, bright youthful energetic voice, expressive and excited)",
    },
    "playful": {
        "label": "✨ Playful",
        "design": "(A cheerful, friendly young woman telling a fun children's story, "
                  "bright, animated, smiling, expressive)",
    },
    "storyteller": {
        "label": "🌙 Storyteller",
        "design": "(A warm, gentle female storyteller reading a bedtime story to a "
                  "young child, soft, soothing, slow and expressive, kind and cozy)",
    },
    "grandpa": {
        "label": "👴 Grandpa",
        "design": "(A kind, gentle old grandfather telling a cozy bedtime story, "
                  "warm, slow, soothing)",
    },
}

# Default leans young per user request — the youngest, most kid-friendly voice.
DEFAULT_VOICE: str = "kid"

# (label, value) pairs for a gr.Radio — shows the friendly label, passes the key.
VOICE_CHOICES = [(preset["label"], key) for key, preset in VOICE_PRESETS.items()]


def voice_design(voice: str) -> str:
    """Return the VoxCPM2 design prefix for a voice key (falls back to default)."""
    preset = VOICE_PRESETS.get(voice) or VOICE_PRESETS[DEFAULT_VOICE]
    return preset["design"]


# ============================================================================
# SEED MANAGEMENT
# ============================================================================

BASE_SEED: int = 42  # Locked for character consistency across pages


def page_seed(page_num: int) -> int:
    """
    Deterministic seed per page for character consistency.
    
    Page 0 uses BASE_SEED, page 1 uses BASE_SEED+1, etc.
    This ensures reproducible generation while maintaining
    slight variation between pages.
    
    Args:
        page_num: Zero-indexed page number (0-5)
        
    Returns:
        Deterministic seed for this page
        
    Example:
        >>> page_seed(0)  # 42
        >>> page_seed(5)  # 47
    """
    if not 0 <= page_num <= 5:
        raise ValueError(f"page_num must be 0-5, got {page_num}")
    return BASE_SEED + page_num


# ============================================================================
# GENERATION PARAMETERS
# ============================================================================

@dataclass
class GenerationParams:
    """
    Generation parameters for image and story creation.
    
    These are the "knobs" that control quality vs speed tradeoffs.
    """
    
    # Image generation
    image_width: int = 768
    image_height: int = 512
    num_inference_steps: int = 20      # Standard mode
    num_inference_steps_tiny: int = 4  # Tiny Mode (SD-Turbo)
    guidance_scale: float = 3.5
    
    # LoRA settings
    lora_scale: float = 0.85
    lora_repo: str = "build-small-hackathon/doodlebook-flux-lora"
    
    # Story generation
    max_story_tokens: int = 800
    story_temperature: float = 0.7
    story_top_p: float = 0.9
    
    # TTS settings
    tts_sample_rate: int = 48000
    tts_speed: float = 1.0
    
    # Book settings
    num_pages: int = 6
    target_age: int = 5


GENERATION_PARAMS = GenerationParams()


# ============================================================================
# COLOR PALETTE (Off-Brand Badge)
# ============================================================================

COLORS = {
    "paper": "#FEF9E7",      # App background
    "page": "#FFFDE7",       # Book pages
    "ink": "#3E2723",        # Body text
    "title_brown": "#5D4037", # Headings
    "crayon_orange": "#FF7043", # Primary CTA
    "sky_accent": "#4FC3F7",  # Secondary/links
    "muted": "#BCAAA4",       # Page numbers/meta
}


# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

MODAL_CONFIG = {
    "image_gen": {
        "app_name": "doodlebook-image-gen",
        "gpu": FLUX_MODEL.modal_gpu,
        "memory": FLUX_MODEL.modal_memory,
        "timeout": 300,
        "keep_warm": 1,  # During judging window
    },
    "story_gen": {
        "app_name": "doodlebook-story",
        "gpu": STORY_MODEL.modal_gpu,
        "memory": STORY_MODEL.modal_memory,
        "timeout": 120,
        "keep_warm": 0,
    },
    "tts": {
        "app_name": "doodlebook-tts",
        "gpu": TTS_MODEL.modal_gpu,
        "memory": TTS_MODEL.modal_memory,
        "timeout": 120,
        "keep_warm": 0,
    },
}


# ============================================================================
# SAMPLE BOOK CONFIGURATION
# ============================================================================

SAMPLE_BOOK_PATH = "assets/sample_book"
SAMPLE_DOODLE_PATH = "assets/sample_doodle.jpg"


# ============================================================================
# VALIDATION & COMPLIANCE
# ============================================================================

def validate_model_ids(use_hf_hub: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Validate model IDs exist on HuggingFace Hub.
    
    Args:
        use_hf_hub: If True, actually check HF Hub (slower but thorough)
        
    Returns:
        Dictionary mapping model IDs to validation results
    """
    models = [FLUX_MODEL, STORY_MODEL, TTS_MODEL]
    results = {}
    
    for model in models:
        results[model.hub_id] = {
            "exists": True,
            "checked": False,
            "license": model.license.value,
            "commercial_ok": model.can_use_commercially,
        }
        
        if use_hf_hub:
            try:
                from huggingface_hub import model_info
                model_info(model.hub_id)
                results[model.hub_id]["checked"] = True
                logger.info(f"Verified: {model.hub_id}")
            except Exception as e:
                results[model.hub_id] = {
                    "exists": False,
                    "error": str(e),
                    "fallback": model.fallback_id,
                }
                logger.warning(f"Model not found: {model.hub_id}, fallback: {model.fallback_id}")
    
    return results


def check_license_compliance() -> bool:
    """
    Ensure no non-commercial models are in the primary production path.
    
    Raises:
        ValueError: If a non-commercial model is configured as primary
        
    Returns:
        True if all primary models are license-compliant
    """
    primary_models = [FLUX_MODEL, STORY_MODEL, TTS_MODEL]
    
    for model in primary_models:
        if not model.can_use_commercially:
            raise ValueError(
                f"LICENSE VIOLATION: {model.hub_id} is {model.license.value}. "
                f"This model CANNOT be used commercially. "
                f"Use fallback: {model.fallback_id}"
            )
    
    return True


def get_model_with_fallback(
    model: ModelConfig,
    use_fallback: bool = False
) -> ModelConfig:
    """
    Get model config, optionally using fallback.
    
    Args:
        model: Primary model config
        use_fallback: If True, return fallback model
        
    Returns:
        ModelConfig (primary or fallback)
    """
    # Fallback model configs were removed (the Space links only the 3 primaries);
    # there is no alternate config to swap in, so always return the primary.
    return model


# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN")


def get_modal_token() -> Optional[str]:
    """Get Modal token from environment."""
    return os.environ.get("MODAL_TOKEN_ID") or os.environ.get("MODAL_TOKEN")


# ============================================================================
# STARTUP VALIDATION
# ============================================================================

def startup_check() -> bool:
    """
    Run at app startup to ensure configuration is valid.
    
    Returns:
        True if all checks pass
    """
    try:
        check_license_compliance()
        logger.info("License compliance: PASSED")
        return True
    except ValueError as e:
        logger.error(f"Startup check FAILED: {e}")
        return False


# Run on import if in production mode
if os.environ.get("DOODLEBOOK_ENV") == "production":
    startup_check()
