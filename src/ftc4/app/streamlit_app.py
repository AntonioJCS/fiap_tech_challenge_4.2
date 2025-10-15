import json
from pathlib import Path
import streamlit as st
from ftc4.core.config import settings
from ftc4.core.logging import logger
from ftc4.io.downloader import download_youtube_audio
from ftc4.io.audio_utils import ensure_wav
from ftc4.nlp.transcribe import transcribe_wav
from ftc4.nlp.postprocess import normalized_text
from ftc4.nlp.generate import generate_outputs


st.set_page_config(page_title="MLET Playground", layout="wide")

# ---- Titulos ----
st.sidebar.title("⚙️ Parâmetros")
source_type = st.sidebar.selectbox("Fonte", ["YouTube URL", "Upload de vídeo/áudio"])
n_questions = st.sidebar.slider("Nº de questões", 5, 12, 8)


# ---- Sidebar ----
st.title("FTC4 - Audio para Texto")
st.caption("Transcrição com WhisperX + Geração com Hugging Face Transformers")


# ---- Youtube Download ----
path_wav: Path | None = None
if source_type == "YouTube URL":
    url = st.text_input("Cole a URL do YouTube")
    if st.button("Baixar & Processar") and url:
        with st.spinner("Baixando áudio..."):
            audio_path = download_youtube_audio(url, settings.inputs_dir)
        path_wav = audio_path

# ---- Upload de Arquivo ----
else:
    up = st.file_uploader(
        "Envie .mp4, .mkv, .mp3, .wav", type=["mp4", "mkv", "mp3", "wav"]
    )
    if up:
        tmp = Path(settings.inputs_dir, up.name)
        with open(tmp, "wb") as f:
            f.write(up.read())
        path_wav = ensure_wav(tmp)
        st.success(f"Arquivo salvo: {path_wav.name}")

# ---- Upload de Arquivo ----
if st.button("Transcrever & Gerar", disabled=path_wav is None):
    if path_wav is None:
        st.warning("Forneça uma fonte primeiro.")
        st.stop()

    with st.spinner("Transcrevendo..."):
        tr = transcribe_wav(path_wav)
        text = normalized_text(tr["segments"])

    with st.spinner("Gerando fichamento e quiz..."):
        outs = generate_outputs(text, n_questions=n_questions)

    st.subheader("📄 Fichamento (Markdown)")
    st.code(outs.fichamento_md, language="markdown")
    md_path = Path(settings.outputs_dir, f"fichamento_{path_wav.stem}.md")
    md_path.write_text(outs.fichamento_md, encoding="utf-8")
    st.download_button("⬇️ Baixar .md", data=outs.fichamento_md, file_name=md_path.name)

    st.subheader("📝 Quiz (JSON)")
    st.code(json.dumps(outs.quiz_json, ensure_ascii=False, indent=2), language="json")
    quiz_path = Path(settings.outputs_dir, f"quiz_{path_wav.stem}.json")
    quiz_path.write_text(
        json.dumps(outs.quiz_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    st.download_button(
        "⬇️ Baixar Quiz.json",
        data=quiz_path.read_text(encoding="utf-8"),
        file_name=quiz_path.name,
    )

    st.success("Processo concluído!")
