requisitos: Windows / Linux / Docker
precisamos instalar o nvidia-smi (GPUutil utiliza ele)
precisamos instalar os binarios o ffmpeg e INCLUIR ELE no .ENV
poetry >= 2.*



# 🎧→📝 MLET Playground: Transcrever, Fichar e Avaliar (Streamlit + Transformers)

Este projeto implementa um **playground em Streamlit** que:
- Recebe link do YouTube **ou** upload de vídeo/áudio,
- **Extrai** e **transcreve** com WhisperX,
- **Gera Fichamento Markdown** e **Quiz** com **Hugging Face Transformers**,
- Oferece **download** dos artefatos.


---

## 🧰 Requisitos

- **Python 3.10+**
- **FFmpeg** (instalado localmente *ou* use **Docker**)
- **Poetry** (gerenciador de dependências)
- (Opcional) **GPU CUDA** para acelerar WhisperX/Transformers
  Execute no terminal para verificar a versão do Cuda em sua máquina
  ```bash
  nvidia-smi
  ```
  Ajuste o pyproject.toml de acordo com o que é indicado em "https://pytorch.org/get-started/locally/"

- (Opcional) Token HF (`HUGGINGFACE_TOKEN`) para alguns modelos

---

## 🚀 Setup (local)

```bash
# 1) Clone
git clone <SEU_REPO>.git
cd mlet-fase4-playground

# 2) Ambiente
pip install -U pip
pip install poetry==2.1.1
poetry install

# 3) Verificação de Hardware
poetry run python -m src.core.device

# 4) Configuração
cp .env.example .env
# edite .env se necessário (modelo HF, device, etc.)

# 5) Executar
poetry run streamlit run src/app/streamlit_app.py
```

Acesse: [http://localhost:8501](http://localhost:8501/)

---

## 🐳 Setup com Docker

```bash
# build
docker compose build

# run
docker compose up
```

Acesse: [http://localhost:8501](http://localhost:8501/)

---

## 🧪 Como usar

1. **Escolha a fonte** (URL do YouTube ou Upload).
2. Clique em **Baixar & Processar** (se URL) ou envie o arquivo.
3. Clique em **Transcrever & Gerar**.
4. Baixe o **`Fichamento.md`** e o **`Quiz.json`**.

> _Dica:_ áudios longos podem demorar. Para MVP, use conteúdos de 5–15 min.

---

## 🤖 Ajuste Fino (opcional)

Prepare um dataset (CSV/JSON/JSONL) com colunas:
- `text` → entrada (trechos da transcrição)
- `target` → saída (fichamento/quiz no formato escolhido)

Execute:

```bash
poetry run python -m src.training.finetune data/datasets/train.jsonl models/finetuned
```

Depois, aponte `HF_MODEL_NAME=models/finetuned` no `.env`.

---

## 🧪 Avaliação Rápida

Exemplo (heurística simples):

```bash
poetry run python scripts/sample_eval.py data/outputs/quiz_*.json
```

(Edite `sample_eval.py` para métricas de cobertura/consistência.)

---

## 📦 Estrutura

```
src/
  app/streamlit_app.py      # UI do Playground
  nlp/transcribe.py         # WhisperX
  nlp/generate.py           # Transformers (fichamento/quiz)
  training/finetune.py      # ajuste fino opcional
  ...
```

---

## 🔐 Observações

- **Diarização é opcional**; requer modelos pyannote (podem exigir token HF).
- Para **GPU**, ajuste `torch`/`accelerate` conforme sua placa.

---

## 🌐 Deploy no Streamlit Cloud

1. Suba o repositório no GitHub.
2. Crie um app no Streamlit Community Cloud apontando para `src/app/streamlit_app.py`.
3. Defina **Secrets** com o conteúdo do seu `.env` (sem aspas).
4. (Opcional) Ajuste variáveis de ambiente para cache de modelos.


---

## 📄 Entrega (conforme prova)

- **Repositório GitHub** (este)
- **Link do Deploy Streamlit**
- **Vídeo (≥ 5 min)** explicando tema, modelo, dados (se houver), ajuste fino, avaliação, demonstração.