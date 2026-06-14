"""
Trace logging service — publishes generation metadata to HF Dataset (Open Trace badge).

Logs prompts, seeds, and LoRA version for reproducibility.
"""

from typing import Optional
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

TRACE_DATASET = "build-small-hackathon/doodlebook-traces"


def log_trace(
    hero_name: str,
    theme: str,
    story: dict,
    seed: int,
    lora_version: Optional[str] = None,
    tiny_mode: bool = False,
    character_description: str = "",
    art_style: str = "crayon drawing, children's book"
) -> str:
    """
    Log generation trace to HuggingFace Dataset.
    
    Creates a row in the trace dataset with all generation parameters
    for reproducibility (Open Trace badge).
    
    Args:
        hero_name: Character name used
        theme: Story theme
        story: Generated story dict
        seed: Seed used for generation
        lora_version: LoRA model version (if used)
        tiny_mode: Whether Tiny Mode was used
        character_description: Character description used
        art_style: Art style used
        
    Returns:
        Dataset URL
    """
    trace = {
        "timestamp": datetime.now().isoformat(),
        "hero_name": hero_name,
        "theme": theme,
        "title": story.get("title", ""),
        "character_description": character_description or story.get("character_description", ""),
        "art_style": art_style,
        "seed": seed,
        "lora_version": lora_version or "none",
        "tiny_mode": tiny_mode,
        "num_pages": len(story.get("pages", [])),
        "pages": story.get("pages", []),
        "models": {
            "image": "black-forest-labs/FLUX.2-klein-4B",
            "story": "openbmb/MiniCPM5-1B",
            "tts": "openbmb/VoxCPM2"
        }
    }
    
    # Save trace locally
    trace_dir = "traces"
    os.makedirs(trace_dir, exist_ok=True)
    trace_file = os.path.join(trace_dir, f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(trace_file, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Trace saved: {trace_file}")
    
    # Try to upload to HF Dataset
    try:
        return _upload_to_hf_dataset(trace)
    except Exception as e:
        logger.warning(f"HF Dataset upload failed: {e}. Trace saved locally.")
        return f"Local: {trace_file}"


def _upload_to_hf_dataset(trace: dict) -> str:
    """Upload trace to HuggingFace Dataset."""
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        
        # Check if dataset exists, create if not
        try:
            api.dataset_info(TRACE_DATASET)
        except Exception:
            api.create_repo(TRACE_DATASET, repo_type="dataset", exist_ok=True)
        
        # Upload trace as JSON
        trace_json = json.dumps(trace, indent=2, ensure_ascii=False)
        filename = f"trace_{trace['timestamp'].replace(':', '-').replace('.', '-')}.json"
        
        api.upload_file(
            path_or_fileobj=trace_json.encode(),
            path_in_repo=filename,
            repo_id=TRACE_DATASET,
            repo_type="dataset",
            commit_message=f"Log trace for {trace['hero_name']}"
        )
        
        return f"https://huggingface.co/datasets/{TRACE_DATASET}/blob/main/{filename}"
        
    except ImportError:
        logger.warning("huggingface_hub not installed")
        return "Trace saved locally (no HF upload)"
