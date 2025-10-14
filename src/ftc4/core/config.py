from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "dev")
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data")).resolve()
    inputs_dir: Path = (Path(os.getenv("DATA_DIR", "./data")) / "inputs").resolve()
    outputs_dir: Path = (Path(os.getenv("DATA_DIR", "./data")) / "outputs").resolve()
    datasets_dir: Path = (Path(os.getenv("DATA_DIR", "./data")) / "datasets").resolve()
    models_dir: Path = Path(os.getenv("MODELS_DIR", "./models")).resolve()
    hf_model_name: str = os.getenv("HF_MODEL_NAME", "google/byt5-small")
    gen_max_new_tokens: int = int(os.getenv("GEN_MAX_NEW_TOKENS", 512))
    gen_temperature: float = float(os.getenv("GEN_TEMPERATURE", 0.8))
    gen_top_p: float = float(os.getenv("GEN_TOP_P", 0.95))
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_batch_size: int = int(os.getenv("WHISPER_BATCH_SIZE", 16))
    whisper_lang: str = os.getenv("WHISPER_LANG", "pt")
    hf_token: str | None = os.getenv("HUGGINGFACE_TOKEN")
    whisper_model_arch: str = os.getenv("WHISPER_MODEL_ARCH", "")  # "" => usar auto
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "")  # "" => usar auto


settings = Settings()

for p in [
    settings.data_dir,
    settings.inputs_dir,
    settings.outputs_dir,
    settings.datasets_dir,
    settings.models_dir,
]:
    p.mkdir(parents=True, exist_ok=True)
