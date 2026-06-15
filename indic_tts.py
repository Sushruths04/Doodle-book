"""Kannada TTS — primary: sush0401/IndicF5-Kannada-Bedtime-v2 (fine-tuned, non-gated).
Fallback: facebook/mms-tts-kan (VITS, 16kHz, no voice cloning).
"""
from __future__ import annotations
import os
import re
import tempfile
import numpy as np
import torch

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
    _indic_model = AutoModel.from_pretrained(
        _FINETUNE_HUB, trust_remote_code=True,
    ).to("cuda").eval()
    _use_indic = True


def _load_mms():
    global _mms_model, _mms_tok
    from transformers import VitsModel, AutoTokenizer
    _mms_tok   = AutoTokenizer.from_pretrained(_FALLBACK_HUB)
    _mms_model = VitsModel.from_pretrained(_FALLBACK_HUB).to("cuda").eval()


def _get_model():
    if not _use_indic and _mms_model is None:
        try:
            _load_indic()
        except Exception:
            _load_mms()
    elif not _use_indic:
        pass  # already loaded MMS
    return _use_indic


# Pre-load at module scope on ZeroGPU
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


def _narrate_indic(ref_wav: str, kannada_text: str) -> np.ndarray:
    model = _indic_model
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
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32), silence_sr


def _narrate_mms(kannada_text: str) -> tuple[np.ndarray, int]:
    silence = np.zeros(int(0.55 * MMS_SR), dtype=np.float32)
    chunks = []
    for sent in _split(kannada_text):
        inputs = _mms_tok(sent, return_tensors="pt").to("cuda")
        with torch.no_grad():
            wav = _mms_model(**inputs).waveform
        audio = wav.squeeze().cpu().float().numpy()
        if audio.size:
            chunks.append(audio)
            chunks.append(silence)
    full = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    return full, MMS_SR


def narrate_kannada(ref_wav: str, ref_text: str, kannada_text: str,
                    mood: str = "", energy: float = 0.45) -> str:
    """Narrate Kannada text. Uses fine-tuned IndicF5 with voice cloning if available,
    otherwise MMS-TTS-Kan (generic voice). Returns a temp WAV path."""
    text = (kannada_text or "").strip()
    if not text:
        raise ValueError("Please provide Kannada text to narrate.")

    use_indic = _get_model()

    if use_indic:
        try:
            full, sr = _narrate_indic(ref_wav or "", text)
        except Exception:
            if _mms_model is None:
                _load_mms()
            full, sr = _narrate_mms(text)
    else:
        full, sr = _narrate_mms(text)

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
