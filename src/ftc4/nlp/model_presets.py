from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


@dataclass
class SamplingBounds:
    # limites hard e default por preset
    temperature_min: float
    temperature_max: float
    temperature_default: float
    top_p_min: float
    top_p_max: float
    top_p_default: float
    max_new_tokens_min: int
    max_new_tokens_max: int
    max_new_tokens_default: int

@dataclass
class TextGenPreset:
    model_name: str
    use_quant_8bit: bool = False
    use_quant_4bit: bool = False
    torch_dtype: Optional[torch.dtype] = None
    device_map: str = "auto"
    reason: str = ""
    sampling: SamplingBounds = None  # novos limites por preset


def _gpu_bounds_big() -> SamplingBounds:
    # GPUs >=20GB: podemos gerar mais tokens
    return SamplingBounds(
        temperature_min=0.1, temperature_max=2.0, temperature_default=0.7,
        top_p_min=0.05, top_p_max=1.0, top_p_default=0.95,
        max_new_tokens_min=128, max_new_tokens_max=1024, max_new_tokens_default=640
    )

def _gpu_bounds_mid() -> SamplingBounds:
    # 12–19GB
    return SamplingBounds(
        temperature_min=0.1, temperature_max=2.0, temperature_default=0.7,
        top_p_min=0.05, top_p_max=1.0, top_p_default=0.95,
        max_new_tokens_min=128, max_new_tokens_max=768, max_new_tokens_default=512
    )

def _gpu_bounds_small() -> SamplingBounds:
    # 8–11GB
    return SamplingBounds(
        temperature_min=0.1, temperature_max=2.0, temperature_default=0.7,
        top_p_min=0.05, top_p_max=1.0, top_p_default=0.95,
        max_new_tokens_min=96, max_new_tokens_max=640, max_new_tokens_default=384
    )

def _gpu_bounds_tiny() -> SamplingBounds:
    # <8GB ou VRAM no limite
    return SamplingBounds(
        temperature_min=0.1, temperature_max=2.0, temperature_default=0.7,
        top_p_min=0.05, top_p_max=1.0, top_p_default=0.9,
        max_new_tokens_min=64, max_new_tokens_max=512, max_new_tokens_default=256
    )

def _mps_bounds() -> SamplingBounds:
    return SamplingBounds(
        temperature_min=0.1, temperature_max=2.0, temperature_default=0.7,
        top_p_min=0.05, top_p_max=1.0, top_p_default=0.95,
        max_new_tokens_min=96, max_new_tokens_max=512, max_new_tokens_default=320
    )

def _cpu_bounds() -> SamplingBounds:
    return SamplingBounds(
        temperature_min=0.1, temperature_max=2.0, temperature_default=0.7,
        top_p_min=0.05, top_p_max=1.0, top_p_default=0.9,
        max_new_tokens_min=64, max_new_tokens_max=384, max_new_tokens_default=256
    )


def _cuda_presets(vram_mb: int, at_limit: bool) -> TextGenPreset:
    if vram_mb >= 20000 and not at_limit:
        return TextGenPreset("google/mt5-large", torch_dtype=torch.float16, reason="CUDA ≥20GB → mt5-large fp16", sampling=_gpu_bounds_big())
    elif vram_mb >= 12000 and not at_limit:
        return TextGenPreset("google/mt5-base",  torch_dtype=torch.float16, reason="CUDA 12-19GB → mt5-base fp16", sampling=_gpu_bounds_mid())
    elif vram_mb >= 8000 and not at_limit:
        return TextGenPreset("google/byt5-base", torch_dtype=torch.float16, reason="CUDA 8-11GB → byt5-base fp16", sampling=_gpu_bounds_small())
    else:
        if _HAS_BNB:
            return TextGenPreset("google/byt5-base", use_quant_8bit=True, reason="CUDA <8GB/limite → byt5-base 8-bit (bnb)", sampling=_gpu_bounds_tiny())
        return TextGenPreset("google/byt5-small", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, reason="CUDA <8GB/limite sem bnb → byt5-small", sampling=_gpu_bounds_tiny())

def _mps_presets() -> TextGenPreset:
    return TextGenPreset("google/byt5-base", torch_dtype=torch.float16, reason="MPS → byt5-base fp16", sampling=_mps_bounds())

def _cpu_presets() -> TextGenPreset:
    return TextGenPreset("google/byt5-small", torch_dtype=torch.float32, reason="CPU → byt5-small fp32", sampling=_cpu_bounds())

def choose_textgen_preset(device_info) -> TextGenPreset:
    if device_info.device_str == "cuda":
        total = int(device_info.vram_total_mb or 0)
        at_limit = bool(device_info.vram_used_pct is not None and device_info.vram_used_pct >= 90.0)
        return _cuda_presets(total, at_limit)
    if device_info.device_str == "mps":
        return _mps_presets()
    return _cpu_presets()

def load_textgen_pipeline(preset: TextGenPreset):
    tok = AutoTokenizer.from_pretrained(preset.model_name, use_fast=True)
    load_kwargs: Dict[str, Any] = dict(device_map=preset.device_map, low_cpu_mem_usage=True)
    if (preset.use_quant_8bit or preset.use_quant_4bit) and torch.cuda.is_available() and _HAS_BNB:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=preset.use_quant_8bit, load_in_4bit=preset.use_quant_4bit,
                                     bnb_4bit_use_double_quant=True if preset.use_quant_4bit else None,
                                     bnb_4bit_compute_dtype=torch.float16 if preset.use_quant_4bit else None)
        load_kwargs["quantization_config"] = bnb_cfg
    else:
        if preset.torch_dtype is not None:
            load_kwargs["torch_dtype"] = preset.torch_dtype
    model = AutoModelForSeq2SeqLM.from_pretrained(preset.model_name, **load_kwargs)
    return tok, model
