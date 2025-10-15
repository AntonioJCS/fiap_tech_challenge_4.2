from ftc4.core.device import suggest_whisper_runtime
from ftc4.core.config import settings
from ftc4.core.logging import logger
import whisperx
from pathlib import Path
from typing import Dict


def transcribe_wav(path_wav: Path) -> Dict:
    # 1) Sugestão automática baseada no device
    rt = suggest_whisper_runtime()

    # 2) Permite override via .env (se definidos)
    model_arch = settings.whisper_model_arch or rt.model_arch
    compute_type = settings.whisper_compute_type or rt.compute_type
    device = (
        settings.whisper_device
        if settings.whisper_device in {"cpu", "cuda", "mps"}
        else rt.device
    )
    batch_size = (
        settings.whisper_batch_size
        if settings.whisper_batch_size > 0
        else rt.batch_size
    )

    logger.info(
        f"WhisperX runtime → device={device} | model={model_arch} | "
        f"batch={batch_size} | compute_type={compute_type}"
    )

    # 3) Carrega o modelo WhisperX
    model = whisperx.load_model(
        model_arch,
        device=device,
        language=settings.whisper_lang,
        compute_type=compute_type,
    )

    audio = whisperx.load_audio(str(path_wav))
    result = model.transcribe(audio, batch_size=batch_size)

    # alinhamento (opcional)
    model_a, metadata = whisperx.load_align_model(
        language_code=settings.whisper_lang, device=device
    )
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, audio, device
    )

    segments = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()}
        for s in result_aligned["segments"]
    ]
    return {
        "metadata": {"source": str(path_wav), "language": settings.whisper_lang},
        "segments": segments,
    }
