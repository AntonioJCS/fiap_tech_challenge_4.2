from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(override=True)


# --- Função de Fail Fast ---
def get_required_env(var_name: str) -> str:
    """
    Tenta obter uma variável de ambiente. Se não for encontrada, levanta um erro.
    Implementa o princípio 'Fail Fast'.
    """
    value = os.getenv(var_name)
    if value is None:
        raise EnvironmentError(
            f"Variável de ambiente obrigatória '{var_name}' não encontrada. "
            f"Verifique seu arquivo .env ou variáveis do sistema."
        )
    return value


# --- Dataclasses com as variaveis ---
@dataclass(frozen=True)
class Settings:
    
    # Ambiente
    app_env: str = os.getenv("APP_ENV", "dev")

    # Diretório raiz do projeto
    base_dir: Path = Path(__file__).parents[3].resolve()

    # Armazenamento
    data_dir: Path =  Path(os.getenv("DATA_DIR", base_dir / "data")).resolve()
    inputs_dir: Path = (Path(os.getenv("DATA_DIR", data_dir)) / "inputs").resolve()
    outputs_dir: Path = (Path(os.getenv("DATA_DIR", data_dir)) / "outputs").resolve()
    datasets_dir: Path = (Path(os.getenv("DATA_DIR", data_dir)) / "datasets").resolve()
    models_dir: Path = Path(os.getenv("MODELS_DIR", base_dir / "models")).resolve()

    # Transformers
    hf_model_name: str = os.getenv("HF_MODEL_NAME", "google/byt5-small")
    gen_max_new_tokens: int = int(os.getenv("GEN_MAX_NEW_TOKENS", 512))
    gen_temperature: float = float(os.getenv("GEN_TEMPERATURE", 0.8))   
    gen_top_p: float = float(os.getenv("GEN_TOP_P", 0.95))
    hf_token: str | None = os.getenv("HUGGINGFACE_TOKEN")

    # WhisperX
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_batch_size: int = int(os.getenv("WHISPER_BATCH_SIZE", 16))
    whisper_lang: str = os.getenv("WHISPER_LANG", "pt")
    whisper_model_arch: str = os.getenv("WHISPER_MODEL_ARCH", "")  # "" => auto
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "")  # "" => auto

    # FFMPEG
    ffmpeg_bin: Path = Path(get_required_env("FFMPEG_BIN")).resolve()
    ffmpeg_exe: Path = Path(get_required_env("FFMPEG_EXE")).resolve()

    # PATH
    path : str = os.getenv("PATH")

settings = Settings()

# Cria diretórios automaticamente
dirs =[
    settings.data_dir,
    settings.inputs_dir,
    settings.outputs_dir,
    settings.datasets_dir,
    settings.models_dir,
]
for p in dirs:
    p.mkdir(parents=True, exist_ok=True)
