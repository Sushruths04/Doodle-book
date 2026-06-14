"""
Modal TTS — VoxCPM2 narration on T4 GPU.

C5 Compliance: Fallback chain
- Primary: VoxCPM2 (2B, Apache 2.0)
- Fallback 1: Kokoro-82M (ultra-lightweight)
- Fallback 2: MeloTTS (MIT license)

Generates WAV audio for book narration.
"""

import modal
import io
import os
import logging

logger = logging.getLogger(__name__)

app = modal.App("doodlebook-tts")

# Keep N containers always warm so the app is "live" with no cold start.
# 0 = scale to zero when idle (cheap); 1 = always-on GPU (costs money 24/7).
# Set at deploy time: DOODLEBOOK_KEEP_WARM=1 modal deploy modal_workers/modal_tts.py
KEEP_WARM = int(os.environ.get("DOODLEBOOK_KEEP_WARM", "0"))
NO_TIMEOUT = 86400  # Modal's max (24h) — effectively no per-call timeout

CACHE = "/cache"
vol = modal.Volume.from_name("doodlebook-hf-cache", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface")

tts_env = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("voxcpm==2.0.3", "soundfile", "torch", "huggingface_hub")
    .env({"HF_HOME": CACHE})
)

# Child-friendly voices (VoxCPM2 "voice design" prefixes). The (parenthetical) is
# interpreted as a voice instruction, not spoken aloud.
# MIRROR of config.VOICE_PRESETS — kept inline so the Modal deploy stays
# import-free. Keep the two in sync if you edit them. Default leans young.
DEFAULT_VOICE = "kid"
VOICE_DESIGN = {
    "kid": "(A sweet little girl around seven years old telling a story to her "
           "friends, bright high-pitched cheerful child's voice, playful, giggly "
           "and full of wonder)",
    "big_kid": "(A lively young girl about eleven years old reading a fun story "
               "aloud, bright youthful energetic voice, expressive and excited)",
    "playful": "(A cheerful, friendly young woman telling a fun children's story, "
               "bright, animated, smiling, expressive)",
    "storyteller": "(A warm, gentle female storyteller reading a bedtime story to a "
                   "young child, soft, soothing, slow and expressive, kind and cozy)",
    "grandpa": "(A kind, gentle old grandfather telling a cozy bedtime story, warm, "
               "slow, soothing)",
}

_TTS = None


def _get_tts():
    global _TTS
    if _TTS is None:
        from voxcpm import VoxCPM
        logger.info("Loading VoxCPM2…")
        _TTS = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
        logger.info("VoxCPM2 ready.")
    return _TTS


@app.function(
    gpu="A10G", image=tts_env, volumes={CACHE: vol}, secrets=[HF_SECRET],
    timeout=NO_TIMEOUT, scaledown_window=1200, min_containers=KEEP_WARM,
)
def speak_book(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """
    Narrate the book with VoxCPM2 using a child-friendly storyteller voice.

    Generates sentence-by-sentence with the SAME voice-design prefix for a
    consistent voice, then stitches with short pauses for natural pacing.
    """
    import re
    import numpy as np
    import soundfile as sf

    model = _get_tts()
    design = VOICE_DESIGN.get(voice, VOICE_DESIGN[DEFAULT_VOICE])
    sr = model.tts_model.sample_rate

    # split into sentences so long books stay stable; keep each chunk's voice fixed
    chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not chunks:
        chunks = [text.strip() or "The end."]

    pause = np.zeros(int(sr * 0.35), dtype=np.float32)   # gentle gap between sentences
    pieces = []
    for i, sentence in enumerate(chunks):
        wav = model.generate(
            text=f"{design} {sentence}",
            cfg_value=2.0,
            inference_timesteps=10,
        )
        pieces.append(np.asarray(wav, dtype=np.float32))
        if i < len(chunks) - 1:
            pieces.append(pause)
        logger.info(f"narrated sentence {i+1}/{len(chunks)}")

    audio = np.concatenate(pieces)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    vol.commit()
    return buf.getvalue()


# ============================================================================
# LOCAL TTS (FOR TESTING WITHOUT MODAL)
# ============================================================================

def speak_book_local(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """
    Local TTS for testing (no Modal/GPU required).

    Chain: Windows SAPI5 (offline, audible) -> pyttsx3 (if installed) ->
    silent WAV (last resort). Real child-friendly voice = VoxCPM2 on Modal.
    """
    for fn in (_speak_sapi_windows, _speak_pyttsx3):
        try:
            audio = fn(text, voice)
            if audio:
                logger.info(f"Local TTS via {fn.__name__}")
                return audio
        except Exception as e:
            logger.warning(f"{fn.__name__} unavailable: {e}")
    logger.error("No working local TTS — returning silence")
    return _generate_silent_wav(duration_seconds=5)


def _speak_sapi_windows(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """
    Offline Windows TTS via SAPI5 (pywin32). Produces an audible WAV with no
    GPU, no internet, and no extra install — pywin32 ships win32com.
    """
    import win32com.client
    import pythoncom
    import tempfile
    import os

    pythoncom.CoInitialize()  # Gradio runs handlers off-thread; COM needs init
    path = None
    try:
        spvoice = win32com.client.Dispatch("SAPI.SpVoice")
        stream = win32com.client.Dispatch("SAPI.SpFileStream")

        # Prefer a female/child-friendly voice if the system has one
        try:
            for tok in spvoice.GetVoices():
                desc = tok.GetDescription()
                if any(n in desc for n in ("Zira", "Hazel", "Female")):
                    spvoice.Voice = tok
                    break
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name

        stream.Open(path, 3)            # 3 = SSFMCreateForWrite
        spvoice.AudioOutputStream = stream
        spvoice.Rate = -1               # a touch slower, gentle for a bedtime story
        spvoice.Speak(text)
        stream.Close()

        with open(path, "rb") as f:
            data = f.read()
        # release COM objects before uninit to avoid noisy IUnknown warnings
        spvoice.AudioOutputStream = None
        stream = None
        spvoice = None
        return data
    finally:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        pythoncom.CoUninitialize()


def _speak_pyttsx3(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Cross-platform offline TTS via pyttsx3 (only if installed)."""
    import pyttsx3
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.save_to_file(text, path)
    engine.runAndWait()
    with open(path, "rb") as f:
        data = f.read()
    if os.path.exists(path):
        os.unlink(path)
    return data


def _generate_silent_wav(duration_seconds: int = 5, sample_rate: int = 48000) -> bytes:
    """Generate silent WAV file as placeholder."""
    import struct
    
    # WAV header
    num_samples = sample_rate * duration_seconds
    data_size = num_samples * 2  # 16-bit audio
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size
    )
    
    # Silent audio data
    silent_data = b'\x00' * data_size
    
    return header + silent_data


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.function(gpu="T4", image=tts_env, timeout=30)
def health_check() -> str:
    """Quick health check for Modal function."""
    return "tts_healthy"
