from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(override=True)


# --- Função de Fail Fast ---
_EMPTY = {"", " ", "none", "null", "nil", "None", "Null", "Nil"}

def _getenv_str(key: str, default: str) -> str:
    v = os.getenv(key)
    if v is None or v in _EMPTY:
        return default
    return v

def _getenv_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v.strip() in _EMPTY:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _getenv_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or v.strip() in _EMPTY:
        return default
    try:
        return float(v)
    except Exception:
        return default


# --- Função de Fail Fast ---
def _get_required_env(var_name: str) -> str:
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
    # A ausencia da variavel ("" ou 0) habilita a seleção automatica no backend (core/device.py)
    whisper_device: str = os.getenv("WHISPER_DEVICE", "")
    whisper_batch_size: int = int(os.getenv("WHISPER_BATCH_SIZE", 0))
    whisper_lang: str = os.getenv("WHISPER_LANG", "pt")
    whisper_model_arch: str = os.getenv("WHISPER_MODEL_ARCH", "")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "")

    # FFMPEG
    ffmpeg_bin: Path = Path(_get_required_env("FFMPEG_BIN")).resolve()
    ffmpeg_exe: Path = Path(_get_required_env("FFMPEG_EXE")).resolve()

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
