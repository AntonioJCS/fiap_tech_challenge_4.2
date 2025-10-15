from __future__ import annotations
import importlib
import pytest

# Utilitário: recarrega módulo com patches aplicados
def reload_device_module(monkeypatch, *, cuda=False, mps=False, vram_total=None, vram_used=None, gpu_name="Mock GPU"):
    import torch

    # Mock torch.cuda
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1 if cuda else 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx=0: gpu_name)

    # Mock torch.backends.cudnn.version
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 90000 if cuda else None)

    # Mock torch.backends.mps
    class _MPS:
        @staticmethod
        def is_available(): return mps
        @staticmethod
        def is_built(): return mps
    monkeypatch.setattr(torch.backends, "mps", _MPS)

    # Importa e patcha o módulo alvo
    import ftc4.core.device as device
    importlib.reload(device)

    # Força que GPUtil é "disponível" e retorne VRAM desejada
    monkeypatch.setattr(device, "_HAS_GPUTIL", True, raising=False)
    def fake_gputil_mem(index=0):
        if vram_total is None or vram_used is None:
            return None, None, None
        pct = (vram_used / vram_total) * 100.0 if vram_total else None
        return int(vram_total), int(vram_used), pct
    monkeypatch.setattr(device, "_gputil_mem", fake_gputil_mem, raising=True)
    return device


def test_cpu_presets(monkeypatch):
    device = reload_device_module(monkeypatch, cuda=False, mps=False)
    info = device.detect_device()
    assert info.device_str == "cpu"
    rt = device.suggest_whisper_runtime()
    assert rt.device == "cpu"
    assert rt.model_arch == "base"
    assert rt.batch_size == 4
    assert rt.compute_type == "int8"


def test_mps_presets(monkeypatch):
    device = reload_device_module(monkeypatch, cuda=False, mps=True)
    info = device.detect_device()
    assert info.device_str == "mps"
    rt = device.suggest_whisper_runtime()
    assert rt.device == "mps"
    assert rt.model_arch == "small"
    assert rt.batch_size == 8
    assert rt.compute_type == "float16"


@pytest.mark.parametrize("vram_total,expect_model,expect_batch,expect_type", [
    (22000, "large-v2", 32, "float16"),  # 20GB+
    (16000, "large-v2", 16, "float16"),  # 12–19GB
    (9000,  "medium",  12, "float16"),   # 8–11GB
    (6500,  "small",   10, "float16"),   # 6–7GB
    (5000,  "small",    8, "int8"),      # <6GB
])
def test_gpu_presets_by_vram(monkeypatch, vram_total, expect_model, expect_batch, expect_type):
    # uso baixo de vram para não disparar "at_the_limit"
    device = reload_device_module(monkeypatch, cuda=True, mps=False, vram_total=vram_total, vram_used=256, gpu_name="Mock 4090")
    info = device.detect_device()
    assert info.device_str == "cuda"
    rt = device.suggest_whisper_runtime()
    assert rt.model_arch == expect_model
    assert rt.batch_size == expect_batch
    assert rt.compute_type == expect_type


def test_gpu_presets_when_memory_at_limit(monkeypatch):
    # 12GB total, 95% usada => reduz batch e força int8
    device = reload_device_module(monkeypatch, cuda=True, mps=False, vram_total=12288, vram_used=11600, gpu_name="Mock 3080 12GB")
    info = device.detect_device()
    assert info.device_str == "cuda"
    # sanity: sem o limite, seria large-v2, 16, float16
    rt = device.suggest_whisper_runtime()
    assert rt.model_arch in ("large-v2", "medium", "small")  # depende da heurística
    assert rt.batch_size <= 16  # reduzido
    assert rt.compute_type == "int8"  # forçado quando limite
