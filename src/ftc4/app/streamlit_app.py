from __future__ import annotations

from pathlib import Path
import json
import streamlit as st
import torch

from ftc4.core.config import settings
from ftc4.core.logging import logger
from ftc4.io.downloader import download_youtube_audio
from ftc4.io.audio_utils import ensure_wav
from ftc4.nlp.postprocess import normalized_text
from ftc4.nlp.transcribe import transcribe_wav
from ftc4.nlp.generate import generate_outputs
from ftc4.core.device import detect_device, suggest_whisper_runtime, vram_resume


# ------------------------------
# Helpers de estado
# ------------------------------
def _init_state():
    if "path_wav" not in st.session_state:
        st.session_state.path_wav = None  # Path | None
    if "source_type" not in st.session_state:
        st.session_state.source_type = "YouTube URL"
    if "last_error" not in st.session_state:
        st.session_state.last_error = None


def _set_error(err: Exception | str | None):
    st.session_state.last_error = str(err) if err else None


def _set_path_wav(p: Path | None):
    st.session_state.path_wav = p


def _reset_audio():
    _set_path_wav(None)
    _set_error(None)


# ------------------------------
# Preparação de áudio
# ------------------------------
def prepare_audio_from_url(url: str) -> Path:
    """Baixa o áudio e garante WAV."""
    audio_path = download_youtube_audio(url, settings.inputs_dir)
    wav_path = ensure_wav(audio_path)
    return Path(wav_path)


def prepare_audio_from_upload(uploaded_file) -> Path:
    """Salva o upload, força WAV e retorna Path final."""
    tmp = Path(settings.inputs_dir, uploaded_file.name)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(uploaded_file.read())
    wav_path = ensure_wav(tmp)
    return Path(wav_path)


# ------------------------------
# Página principal
# ------------------------------
def main():
    _init_state()
    st.set_page_config(page_title="MLET Playground", layout="wide")

    st.sidebar.title("⚙️ Parâmetros")
    st.session_state.source_type = st.sidebar.selectbox(
        "Fonte", ["YouTube URL", "Upload de vídeo/áudio"], index=0
    )
    n_questions = st.sidebar.slider("Nº de questões", 5, 12, 8)

    st.title("🎧→📝 Fichamento + 🧠 Quiz (Transformers)")
    st.caption("Transcrição com WhisperX + Geração com Hugging Face Transformers")


    # --------------------------
    # Bloco de Hardware e Presets
    # --------------------------
    st.sidebar.subheader("🖥️ Device")
    info = detect_device()
    rt = suggest_whisper_runtime()
    st.sidebar.code(
        f"device: {info.device_str}\n"
        f"torch: {info.torch_version}\n"
        f"cuda version : {torch.version.cuda}\n"
        f"gpu: {info.gpu_name}\n"
        f"{vram_resume(info)}\n\n"
        f"model: {rt.model_arch}\n"
        f"batch: {rt.batch_size}\n"
        f"type:  {rt.compute_type}"
    )


    # --------------------------
    # Bloco de preparação do áudio
    # --------------------------
    with st.expander("🎛️ Preparar áudio", expanded=True):
        col1, col2 = st.columns([3, 2])

        with col1:
            if st.session_state.source_type == "YouTube URL":
                url = st.text_input("Cole a URL do YouTube")
                if st.button("Baixar & preparar", type="primary", use_container_width=True, disabled=not url):
                    _reset_audio()
                    try:
                        with st.spinner("Baixando e preparando áudio..."):
                            path_wav = prepare_audio_from_url(url)
                        _set_path_wav(path_wav)
                        st.success(f"Áudio pronto: {path_wav.name}")
                    except Exception as e:
                        logger.exception("Falha ao preparar áudio via URL")
                        _set_error(e)
                        st.error("Falha ao preparar áudio via URL. Verifique o link/FFmpeg.")
                        st.exception(e)

            else:
                up = st.file_uploader(
                    "Envie .mp4, .mkv, .mp3, .wav",
                    type=["mp4", "mkv", "mp3", "wav"],
                    accept_multiple_files=False,
                )
                if st.button("Preparar upload", use_container_width=True, disabled=not up):
                    _reset_audio()
                    try:
                        with st.spinner("Preparando áudio do upload..."):
                            path_wav = prepare_audio_from_upload(up)
                        _set_path_wav(path_wav)
                        st.success(f"Áudio pronto: {path_wav.name}")
                    except Exception as e:
                        logger.exception("Falha ao preparar áudio via upload")
                        _set_error(e)
                        st.error("Falha ao preparar áudio. Verifique o arquivo/FFmpeg.")
                        st.exception(e)

        with col2:
            st.subheader("📂 Status")
            if st.session_state.path_wav:
                st.info(f"Arquivo: **{Path(st.session_state.path_wav).name}**")
                st.write(f"Local: `{st.session_state.path_wav}`")
                st.button("🔁 Trocar arquivo", on_click=_reset_audio, use_container_width=True)
            else:
                st.warning("Nenhum áudio preparado ainda.")

            if st.session_state.last_error:
                st.error("Último erro:")
                st.code(st.session_state.last_error)

    # --------------------------
    # Bloco principal de execução
    # --------------------------
    audio_ready = bool(st.session_state.path_wav)
    transcribe_disabled = not audio_ready

    if st.button("Transcrever & Gerar", type="primary", disabled=transcribe_disabled, use_container_width=True):
        if not audio_ready:
            st.warning("Prepare um áudio primeiro.")
            st.stop()

        path_wav: Path = Path(st.session_state.path_wav)

        try:
            with st.spinner("Transcrevendo..."):
                tr = transcribe_wav(path_wav)
                text = normalized_text(tr["segments"])

            with st.spinner("Gerando fichamento e quiz..."):
                outs = generate_outputs(text, n_questions=n_questions)

            st.subheader("📄 Fichamento (Markdown)")
            st.code(outs.fichamento_md, language="markdown")
            md_path = Path(settings.outputs_dir, f"fichamento_{path_wav.stem}.md")
            md_path.write_text(outs.fichamento_md, encoding="utf-8")
            st.download_button(
                "⬇️ Baixar .md",
                data=outs.fichamento_md,
                file_name=md_path.name,
                use_container_width=True,
            )

            st.subheader("📝 Quiz (JSON)")
            st.code(json.dumps(outs.quiz_json, ensure_ascii=False, indent=2), language="json")
            quiz_path = Path(settings.outputs_dir, f"quiz_{path_wav.stem}.json")
            quiz_path.write_text(json.dumps(outs.quiz_json, ensure_ascii=False, indent=2), encoding="utf-8")
            st.download_button(
                "⬇️ Baixar Quiz.json",
                data=quiz_path.read_text(encoding="utf-8"),
                file_name=quiz_path.name,
                use_container_width=True,
            )

            st.success("Processo concluído!")
        except Exception as e:
            logger.exception("Falha na etapa Transcrever & Gerar")
            _set_error(e)
            st.error("Falha na etapa de transcrição/geração.")
            st.exception(e)


if __name__ == "__main__":
    main()
