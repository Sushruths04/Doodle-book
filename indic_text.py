"""English → Kannada translation via facebook/nllb-200-distilled-600M (non-gated).

ZeroGPU pattern: model loaded to CPU at module scope so ZeroGPU can pack its
tensors. Inside @spaces.GPU functions, .to("cuda") is called and ZeroGPU
transfers the packed tensors efficiently — no re-download needed.
"""
from __future__ import annotations
import re
import torch

_HUB_ID  = "facebook/nllb-200-distilled-600M"
_SRC     = "eng_Latn"
_TGT     = "kan_Knda"
_model   = None
_tok     = None


def _get_model():
    global _model, _tok
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _tok   = AutoTokenizer.from_pretrained(_HUB_ID, src_lang=_SRC)
        # Load to CPU only — ZeroGPU packs these tensors at startup.
        # .to("cuda") is called inside translate_to_kannada() which runs
        # inside @spaces.GPU, where ZeroGPU intercepts the call.
        _model = AutoModelForSeq2SeqLM.from_pretrained(_HUB_ID)
    return _model, _tok


# Pre-load to CPU at module scope so ZeroGPU packs the tensors.
try:
    _get_model()
except Exception:
    pass


def translate_to_kannada(en_text: str) -> str:
    """Translate an English string to Kannada via NLLB-200."""
    text = (en_text or "").strip()
    if not text:
        raise ValueError("Nothing to translate.")

    model, tok = _get_model()
    # Move to GPU — ZeroGPU intercepts this inside @spaces.GPU and uses the
    # pre-packed tensors, so no re-download is needed.
    model = model.to("cuda")

    tgt_id = tok.lang_code_to_id[_TGT]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]

    parts = []
    for sent in sentences:
        inputs = tok(sent, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=tgt_id, max_length=512)
        parts.append(tok.decode(out[0], skip_special_tokens=True))

    return " ".join(parts)
