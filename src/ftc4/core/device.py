from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# GPUtil é opcional. Se não houver, seguimos sem VRAM detalhada.
try:
    import GPUtil  # type: ignore

    _HAS_GPUTIL = True
except Exception:
    _HAS_GPUTIL = False


@dataclass
class DeviceInfo:
    device_str: str  # "cuda", "mps", "cpu"
    torch_version: str
    cudnn_version: Optional[int]
    gpu_count: int
    gpu_name: str
    vram_total_mb: Optional[int]
    vram_used_mb: Optional[int]
    vram_used_pct: Optional[float]


def _get_cuda_device_name(index: int = 0) -> str:
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return "Unknown CUDA GPU"


def _get_mps_device_name() -> str:
    # PyTorch não expõe nome MPS; deixamos nominal.
    return "Apple MPS (Metal)"


def _gputil_mem(index: int = 0) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    if not _HAS_GPUTIL:
        return None, None, None
    try:
        gpus = GPUtil.getGPUs()
        if not gpus or index >= len(gpus):
            return None, None, None
        gpu = gpus[index]
        used = int(gpu.memoryUsed)  # MB
        total = int(gpu.memoryTotal)  # MB
        pct = (used / total * 100.0) if total > 0 else None
        return total, used, pct
    except Exception:
        return None, None, None


def detect_device(preferred_index: int = 0) -> DeviceInfo:
    torch_version = torch.__version__
    cudnn_version = None
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None

    if torch.cuda.is_available():
        device_str = "cuda"
        gpu_count = torch.cuda.device_count()
        gpu_name = _get_cuda_device_name(preferred_index)
        total_mb, used_mb, pct = _gputil_mem(preferred_index)
        return DeviceInfo(
            device_str,
            torch_version,
            cudnn_version,
            gpu_count,
            gpu_name,
            total_mb,
            used_mb,
            pct,
        )

    # Apple Silicon (Metal/MPS)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_str = "mps"
        gpu_count = 1
        gpu_name = _get_mps_device_name()
        # GPUtil não funciona no MPS, deixamos None
        return DeviceInfo(
            device_str,
            torch_version,
            cudnn_version,
            gpu_count,
            gpu_name,
            None,
            None,
            None,
        )

    # Fallback CPU
    return DeviceInfo(
        "cpu", torch_version, cudnn_version, 0, "**No GPU available**", None, None, None
    )


def memory_at_the_limit(info: DeviceInfo, threshold_pct: float = 90.0) -> bool:
    if info.vram_used_pct is None:
        return False
    return info.vram_used_pct >= threshold_pct


def vram_resume(info: DeviceInfo) -> str:
    if (
        info.vram_total_mb is None
        or info.vram_used_mb is None
        or info.vram_used_pct is None
    ):
        return "VRAM usage: (n/d)"
    return f"VRAM: {info.vram_used_mb}MB / {info.vram_total_mb}MB ({info.vram_used_pct:.2f}% usado)"


def _pick_gpu_presets(info: DeviceInfo):
    """
    Heurística para Whisper/WhisperX em GPU:
    - 20GB+  : large-v2, batch 32, float16
    - 12-19GB: large-v2, batch 16, float16
    - 8-11GB : medium,   batch 12, float16
    - 6-7GB  : small,    batch 10, float16
    - <6GB   : small,    batch 8,  int8  (salva VRAM)
    Se VRAM crítica (>=90%), reduzimos batch e/or trocamos compute_type para int8.
    """
    total = info.vram_total_mb or 0

    if total >= 20000:
        model_arch, batch_size, compute_type = "large-v2", 32, "float16"
    elif total >= 12000:
        model_arch, batch_size, compute_type = "large-v2", 16, "float16"
    elif total >= 8000:
        model_arch, batch_size, compute_type = "medium", 12, "float16"
    elif total >= 6000:
        model_arch, batch_size, compute_type = "small", 10, "float16"
    else:
        model_arch, batch_size, compute_type = "small", 8, "int8"

    if memory_at_the_limit(info):
        # Alivia pressão de VRAM
        batch_size = max(4, batch_size // 2)
        if compute_type != "int8":
            compute_type = "int8"

    return model_arch, batch_size, compute_type


def _pick_mps_presets():
    """
    Heurística para Apple Silicon (MPS):
    - Usar modelos menores; MPS é rápido, mas economizar memória ajuda.
    """
    return "small", 8, "float16"


def _pick_cpu_presets():
    """
    Heurística para CPU:
    - Evite modelos grandes; priorize 'base' ou 'small'.
    - compute_type 'int8' (faster-whisper) ajuda a velocidade e memória.
    """
    return "base", 4, "int8"


@dataclass
class WhisperRuntimeConfig:
    device: str  # "cuda" | "mps" | "cpu"
    model_arch: str  # "tiny"|"base"|"small"|"medium"|"large-v1"|"large-v2"
    batch_size: int  # 64 | 32 | 16 | 8 | 4
    compute_type: str  # "float16"|"int8"|"float32" etc.


def suggest_whisper_runtime(
    preferred_index: int = 0, force_device: str | None = None
) -> WhisperRuntimeConfig:
    """
    
    """

    # detecta HW
    info = detect_device(preferred_index)

    # se houver override, substitui device e recalcula presets
    if force_device in {"cpu", "cuda", "mps"}:
        if force_device == "cuda":
            # usa heurística de GPU (mas sem VRAM do GPUtil caso não exista)
            model_arch, batch_size, compute_type = _pick_gpu_presets(info)
        elif force_device == "mps":
            model_arch, batch_size, compute_type = _pick_mps_presets()
        else:
            model_arch, batch_size, compute_type = _pick_cpu_presets()
        return WhisperRuntimeConfig(force_device, model_arch, batch_size, compute_type)

    # padrão: baseado no device detectado
    if info.device_str == "cuda":
        model_arch, batch_size, compute_type = _pick_gpu_presets(info)
    elif info.device_str == "mps":
        model_arch, batch_size, compute_type = _pick_mps_presets()
    else:
        model_arch, batch_size, compute_type = _pick_cpu_presets()

    return WhisperRuntimeConfig(info.device_str, model_arch, batch_size, compute_type)


def _as_dict(info: DeviceInfo):
    return {
        "device": info.device_str,
        "torch_version": info.torch_version,
        "cudnn_version": info.cudnn_version,
        "gpu_count": info.gpu_count,
        "gpu_name": info.gpu_name,
        "vram_total_mb": info.vram_total_mb,
        "vram_used_mb": info.vram_used_mb,
        "vram_used_pct": None
        if info.vram_used_pct is None
        else round(info.vram_used_pct, 2),
        "memory_at_the_limit": memory_at_the_limit(info),
    }


def main():
    import json

    info = detect_device()
    rt = suggest_whisper_runtime()

    print("# Device detection")
    print(json.dumps(_as_dict(info), ensure_ascii=False, indent=2))
    print("\n# VRAM")
    print(vram_resume(info))
    print("\n# Whisper presets")
    print(
        json.dumps(
            {
                "device": rt.device,
                "model_arch": rt.model_arch,
                "batch_size": rt.batch_size,
                "compute_type": rt.compute_type,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
