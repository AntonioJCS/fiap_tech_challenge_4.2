from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from transformers import pipeline

from ftc4.core.config import settings
from ftc4.core.logging import logger
from ftc4.core.device import detect_device
from ftc4.nlp.prompts import FICHAMENTO_PROMPT, QUIZ_PROMPT
from ftc4.nlp.model_presets import choose_textgen_preset, load_textgen_pipeline
from ftc4.utils.validators import clamp, clamp_int, safe_float_default, safe_int_default

"""
Geração local 100% offline (Transformers) com:
- Presets de modelo por hardware (CUDA/MPS/CPU) + quantização opcional;
- Sanitização segura de hiperparâmetros vindos do .env (temperature/top_p/max_new_tokens);
- Override opcional via HF_MODEL_NAME (mantendo limites do preset do device).
"""

@dataclass
class GenOutputs:
    fichamento_md: str
    quiz_json: List[dict]


def _resolve_sampling_params(preset) -> Tuple[float, float, int]:
    """
    Lê valores do .env e aplica:
      - coerção segura (safe_float_default/safe_int_default)
      - fallback para defaults do preset
      - clamp dentro de limites do preset
    """
    sb = preset.sampling

    # valores do .env (se inválidos, caem no default do preset)
    env_temp = safe_float_default(getattr(settings, "gen_temperature", None), sb.temperature_default)
    env_topp = safe_float_default(getattr(settings, "gen_top_p", None), sb.top_p_default)
    env_maxn = safe_int_default(getattr(settings, "gen_max_new_tokens", None), sb.max_new_tokens_default)

    # clamp pelos limites do preset
    temperature = clamp(env_temp, sb.temperature_min, sb.temperature_max)
    top_p = clamp(env_topp, sb.top_p_min, sb.top_p_max)
    max_new_tokens = clamp_int(env_maxn, sb.max_new_tokens_min, sb.max_new_tokens_max)

    return temperature, top_p, max_new_tokens


def _load_model_local():
    """
    Seleciona e carrega o modelo local:
      1) Se HF_MODEL_NAME estiver no .env → usa override explícito (device_map=auto).
      2) Caso contrário, escolhe preset por hardware (CUDA/MPS/CPU + VRAM),
         define sampling bounds apropriados e carrega com dtype/quantização.
    Retorna um pipeline "text2text-generation" configurado.
    """
    info = detect_device()

    # Escolha de preset por device (mesmo quando override, usamos os bounds do device)
    preset = choose_textgen_preset(info)

    if settings.hf_model_name:
        # Override explícito do modelo (mantém limites do preset do device)
        model_name = settings.hf_model_name
        logger.info(f"Usando override de modelo: {model_name}")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        # Carregamento adaptativo por preset (dtype/quantização conforme hardware)
        logger.info(f"Preset escolhido: {preset.model_name} ({preset.reason})")
        tok, model = load_textgen_pipeline(preset)

    # Sanitização final dos hiperparâmetros com base nos bounds do preset
    temperature, top_p, max_new_tokens = _resolve_sampling_params(preset)
    logger.info(
        f"Sampling saneado → temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}"
    )

    gen = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    return gen


def _generate_json_list(gen, prompt: str) -> List[dict]:
    """
    Tenta extrair lista de questões em JSON do texto gerado.
    Heurística simples: pega o primeiro bloco entre { ... }.
    """
    import json

    raw = gen(prompt)[0]["generated_text"]
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
    """
    Gera (1) Fichamento Markdown e (2) Quiz em JSON a partir do texto transcrito.
    """
    gen = _load_model_local()

    ficha_prompt = FICHAMENTO_PROMPT.format(transcript=transcript_text)
    quiz_prompt = QUIZ_PROMPT.format(transcript=transcript_text, n=n_questions)

    logger.info("Gerando Fichamento.md ...")
    fichamento_md = gen(ficha_prompt)[0]["generated_text"]

    logger.info("Gerando Quiz ...")
    quiz_list = _generate_json_list(gen, quiz_prompt)

    return GenOutputs(fichamento_md=fichamento_md, quiz_json=quiz_list)
