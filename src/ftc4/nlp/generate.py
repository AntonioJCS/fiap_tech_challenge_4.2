from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from ..core.config import settings
from ..core.logging import logger
from .prompts import FICHAMENTO_PROMPT, QUIZ_PROMPT


@dataclass
class GenOutputs:
    fichamento_md: str
    quiz_json: List[dict]


def _load_model():
    logger.info(f"Carregando modelo HF: {settings.hf_model_name}")
    tok = AutoTokenizer.from_pretrained(
        settings.hf_model_name, use_auth_token=settings.hf_token
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        settings.hf_model_name, use_auth_token=settings.hf_token
    )
    gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=settings.gen_max_new_tokens,
        do_sample=True,
        temperature=settings.gen_temperature,
        top_p=settings.gen_top_p,
    )
    return gen


def _generate_json_list(gen, prompt: str) -> List[dict]:
    import json

    raw = gen(prompt)[0]["generated_text"]
    # heurística: extrair primeiro bloco JSON válido
    try:
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        json_str = raw[first_brace : last_brace + 1]
        obj = json.loads(json_str)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "quiz" in obj:
            return obj["quiz"]
    except Exception:
        pass
    return []


def generate_outputs(transcript_text: str, n_questions: int = 8) -> GenOutputs:
    gen = _load_model()

    ficha_prompt = FICHAMENTO_PROMPT.format(transcript=transcript_text)
    quiz_prompt = QUIZ_PROMPT.format(transcript=transcript_text, n=n_questions)

    logger.info("Gerando Fichamento.md ...")
    fichamento_md = gen(ficha_prompt)[0]["generated_text"]

    logger.info("Gerando Quiz ...")
    quiz_list = _generate_json_list(gen, quiz_prompt)

    return GenOutputs(fichamento_md=fichamento_md, quiz_json=quiz_list)
