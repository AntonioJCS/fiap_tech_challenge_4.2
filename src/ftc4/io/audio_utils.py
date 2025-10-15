from pathlib import Path
from ffmpeg import FFmpeg
from ftc4.core.config import settings

AUDIO_CODEC = "pcm_s16le"  # WAV PCM 16-bit
SAMPLE_RATE = 16000
CHANNELS = 1


def ensure_wav(path_in: str | Path) -> Path:
    """
    Garante um .wav (PCM 16kHz mono) a partir de qualquer mídia suportada pelo FFmpeg.
    Se já for .wav, apenas retorna. Caso contrário, transcodifica.
    """
    path_in = Path(path_in).resolve()
    if path_in.suffix.lower() == ".wav":
        return path_in

    path_out = path_in.with_suffix(".wav")
    path_out.parent.mkdir(parents=True, exist_ok=True)

    (
        FFmpeg(settings.ffmpeg_exe.as_posix())
        .option("y")  # overwrite
        .input(path_in.as_posix()) # Precisa estar como String
        .output(
            path_out.as_posix(), # Precisa estar como String
            acodec=AUDIO_CODEC,
            ar=SAMPLE_RATE,
            ac=CHANNELS,
        )
        .execute()
    )
    return path_out