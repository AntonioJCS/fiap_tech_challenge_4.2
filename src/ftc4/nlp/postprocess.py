from __future__ import annotations
from typing import Dict, List


def normalized_text(segments: List[dict]) -> str:
    text = " ".join(s["text"].strip() for s in segments)
    # regras simples (expansível)
    return " ".join(text.split())
