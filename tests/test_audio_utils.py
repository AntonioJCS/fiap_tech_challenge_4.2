from ftc4.core.config import settings
from ftc4.io.audio_utils import ensure_wav
from pathlib import Path

def test_soma_basico():
    dir = settings.inputs_dir / "audio_test.webm"
    assert ensure_wav(dir)==Path("./data/inputs/audio_test.wav").resolve()