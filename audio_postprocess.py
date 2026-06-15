"""Minimal audio post-processing for bedtime narration."""
import numpy as np


def postprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    """Normalize bedtime narration to a comfortable listening level."""
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.92
    return audio
