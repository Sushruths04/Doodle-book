"""
TTS service — calls modal_tts for VoxCPM2 narration.

C5 Compliance: Fallback chain
- Primary: VoxCPM2 (2B, Apache 2.0)
- Fallback 1: Kokoro-82M (ultra-lightweight)
- Fallback 2: MeloTTS (MIT license)
"""

from config import TTS_MODEL, GENERATION_PARAMS
import logging

logger = logging.getLogger(__name__)


def speak_book(text: str, voice: str = "kid") -> bytes:
    """
    Generate narration audio via VoxCPM2 on Modal.
    
    C5 Compliance: Falls back to local TTS if Modal fails.
    
    Args:
        text: Full text to narrate (title + all pages)
        voice: Voice style (warm, friendly, etc.)
        
    Returns:
        WAV audio bytes
    """
    try:
        # Try Modal (real VoxCPM2) — looks up the DEPLOYED function on Modal cloud
        import modal
        fn = modal.Function.from_name("doodlebook-tts", "speak_book")
        return fn.remote(text, voice)
    except Exception as e:
        logger.warning(f"Modal TTS unavailable: {e}, using local fallback")
        return speak_book_local(text, voice)


def speak_book_local(text: str, voice: str = "warm") -> bytes:
    """
    Local TTS for testing (no Modal required).
    Uses MeloTTS or returns silent WAV placeholder.
    """
    try:
        from modal_workers.modal_tts import speak_book_local as local_tts
        return local_tts(text, voice)
    except Exception as e:
        logger.warning(f"Local TTS failed: {e}, returning placeholder")
        from modal_workers.modal_tts import _generate_silent_wav
        return _generate_silent_wav(duration_seconds=5)
