"""Kannada TTS — tier 1: sush0401/IndicF5-Kannada-Bedtime-v2 (fine-tuned).
Tier 2: facebook/mms-tts-kan (VITS, 16kHz, no voice cloning).
Tier 3: gTTS (Google, always works, generic Kannada voice).

ZeroGPU pattern: GPU models loaded to CPU at module scope so ZeroGPU packs their
tensors. Inside inference functions (called from @spaces.GPU), .to("cuda") is
called and ZeroGPU transfers packed tensors to GPU — no re-download needed.
"""
from __future__ import annotations
import logging
import os
import re
import tempfile
import numpy as np
import torch

logger = logging.getLogger(__name__)

_FINETUNE_HUB  = "sush0401/IndicF5-Kannada-Bedtime-v2"
_FALLBACK_HUB  = "facebook/mms-tts-kan"

_indic_model   = None   # IndicF5 fine-tuned (with voice cloning)
_mms_model     = None   # MMS-TTS fallback (no voice cloning)
_mms_tok       = None
_use_indic     = False  # set True after successful IndicF5 load
MMS_SR = 16_000


def _load_indic():
    global _indic_model, _use_indic
    from transformers import AutoModel
    logger.info(f"Loading IndicF5 from {_FINETUNE_HUB}")
    # Load to CPU — ZeroGPU packs these tensors.
    _indic_model = AutoModel.from_pretrained(
        _FINETUNE_HUB, trust_remote_code=True,
    )
    _use_indic = True
    logger.info("IndicF5 loaded OK")


def _load_mms():
    global _mms_model, _mms_tok
    from transformers import VitsModel, AutoTokenizer
    logger.info(f"Loading MMS-TTS from {_FALLBACK_HUB}")
    _mms_tok   = AutoTokenizer.from_pretrained(_FALLBACK_HUB)
    # Load to CPU — ZeroGPU packs these tensors.
    _mms_model = VitsModel.from_pretrained(_FALLBACK_HUB)
    logger.info("MMS-TTS loaded OK")


def _get_model():
    if not _use_indic and _mms_model is None:
        try:
            _load_indic()
        except Exception as e:
            logger.warning(f"IndicF5 load failed ({e}); trying MMS-TTS")
            try:
                _load_mms()
            except Exception as e2:
                logger.warning(f"MMS-TTS load failed too ({e2}); will use gTTS fallback")
    return _use_indic


# Pre-load to CPU at module scope so ZeroGPU packs the tensors.
try:
    _get_model()
except Exception:
    pass


def _split(text: str, max_chars: int = 200):
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > max_chars:
            cut = p.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        out.append(p)
    return out or [text.strip()]


def _narrate_indic(ref_wav: str, kannada_text: str) -> tuple[np.ndarray, int]:
    model = _indic_model.to("cuda")  # ZeroGPU intercepts inside @spaces.GPU
    silence_sr = 24_000
    silence = np.zeros(int(0.55 * silence_sr), dtype=np.float32)
    chunks = []
    for sent in _split(kannada_text):
        kw = dict(ref_audio_path=ref_wav) if ref_wav and os.path.exists(ref_wav) else {}
        audio = model(sent, **kw)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size and float(np.max(np.abs(audio))) > 1.0:
            audio = audio / 32768.0
        if audio.size:
            chunks.append(audio)
            chunks.append(silence)
    return (np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)), silence_sr


def _narrate_mms(kannada_text: str) -> tuple[np.ndarray, int]:
    model = _mms_model.to("cuda")  # ZeroGPU intercepts inside @spaces.GPU
    silence = np.zeros(int(0.55 * MMS_SR), dtype=np.float32)
    chunks = []
    for sent in _split(kannada_text):
        if not sent.strip():
            continue
        inputs = _mms_tok(sent, return_tensors="pt").to(model.device)
        with torch.no_grad():
            wav = model(**inputs).waveform
        audio = wav.squeeze().cpu().float().numpy()
        if audio.size:
            chunks.append(audio)
            chunks.append(silence)
    full = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return full, MMS_SR


def _narrate_gtts(kannada_text: str) -> tuple[np.ndarray, int]:
    """Last-resort: gTTS Kannada (generic Google voice, no GPU needed)."""
    import io
    import librosa
    from gtts import gTTS
    logger.info("Using gTTS Kannada fallback")
    tts = gTTS(text=kannada_text, lang="kn", slow=True)
    fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        tts.save(mp3_path)
        data, sr = librosa.load(mp3_path, sr=22050, mono=True)
    finally:
        try:
            os.unlink(mp3_path)
        except OSError:
            pass
    return data.astype(np.float32), sr


def narrate_kannada(ref_wav: str, ref_text: str, kannada_text: str,
                    mood: str = "", energy: float = 0.45) -> str:
    """Narrate Kannada text. Tries: IndicF5 → MMS-TTS → gTTS. Returns WAV path."""
    text = (kannada_text or "").strip()
    if not text:
        raise ValueError("No Kannada text to narrate.")

    use_indic = _get_model()
    full = np.array([], dtype=np.float32)
    sr   = MMS_SR

    # Tier 1: user's fine-tuned IndicF5
    if use_indic:
        try:
            full, sr = _narrate_indic(ref_wav or "", text)
            logger.info("IndicF5 narration OK")
        except Exception as e:
            logger.warning(f"IndicF5 narration failed ({e}); trying MMS-TTS")

    # Tier 2: MMS-TTS-Kan
    if not full.size and _mms_model is not None:
        try:
            full, sr = _narrate_mms(text)
            logger.info("MMS-TTS narration OK")
        except Exception as e:
            logger.warning(f"MMS-TTS narration failed ({e}); trying gTTS")

    # Tier 3: gTTS (always works, no GPU needed)
    if not full.size:
        try:
            full, sr = _narrate_gtts(text)
            logger.info("gTTS narration OK")
        except Exception as e:
            raise RuntimeError(f"All Kannada TTS tiers failed. Last error: {e}") from e

    if not full.size:
        raise RuntimeError("TTS produced no audio.")

    peak = float(np.max(np.abs(full)))
    if peak > 0:
        full = full / peak * 0.92

    import soundfile as sf
    fd, path = tempfile.mkstemp(prefix="bedtime_kn_", suffix=".wav")
    os.close(fd)
    sf.write(path, full, sr)
    return path
