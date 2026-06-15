"""English → Kannada translation via facebook/nllb-200-distilled-600M (non-gated)."""
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
        _model = AutoModelForSeq2SeqLM.from_pretrained(_HUB_ID).to("cuda").eval()
    return _model, _tok


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
    tgt_id = tok.lang_code_to_id[_TGT]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]

    parts = []
    for sent in sentences:
        inputs = tok(sent, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=tgt_id, max_length=512)
        parts.append(tok.decode(out[0], skip_special_tokens=True))

    return " ".join(parts)
