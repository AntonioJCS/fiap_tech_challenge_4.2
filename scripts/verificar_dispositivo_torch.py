from __future__ import annotations
import json
from ftc4.core.device import detect_device, suggest_whisper_runtime, vram_resume, memory_at_the_limit
import torch

def main():
    info = detect_device()
    rt = suggest_whisper_runtime()
    print("# Device detection")
    print(json.dumps({
        "device": info.device_str,
        "torch_version": info.torch_version,
        "cuda_version" : torch.version.cuda,
        "cudnn_version": info.cudnn_version,
        "gpu_count": info.gpu_count,
        "gpu_name": info.gpu_name,
        "vram_total_mb": info.vram_total_mb,
        "vram_used_mb": info.vram_used_mb,
        "vram_used_pct": None if info.vram_used_pct is None else round(info.vram_used_pct, 2),
        "memory_at_the_limit": memory_at_the_limit(info),
        "vram_resume": vram_resume(info),
        "whisper_runtime": {
            "device": rt.device,
            "model_arch": rt.model_arch,
            "batch_size": rt.batch_size,
            "compute_type": rt.compute_type,
        }
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
