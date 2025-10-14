from pathlib import Path
from ffmpeg import FFmpeg

AUDIO_CODEC = "pcm_s16le"  # WAV PCM 16-bit
SAMPLE_RATE = 16000
CHANNELS = 1


def ensure_wav(path_in: Path) -> Path:
    """
    Garante um .wav (PCM 16kHz mono) a partir de qualquer mídia suportada pelo FFmpeg.
    Se já for .wav, apenas retorna. Caso contrário, transcodifica.
    """
    path_in = Path(path_in)
    if path_in.suffix.lower() == ".wav":
        return path_in

    path_out = path_in.with_suffix(".wav")

    # python-ffmpeg: FFmpeg().input(...).output(...).execute()
    (
        FFmpeg()
        .option("y")  # overwrite
        .input(str(path_in))
        .output(
            str(path_out),
            acodec=AUDIO_CODEC,
            ar=SAMPLE_RATE,
            ac=CHANNELS,
        )
        .execute()
    )
    return path_out


if __name__ == "__main__":
    print(ensure_wav(Path("data/inputs/audio_test.mp3")))
