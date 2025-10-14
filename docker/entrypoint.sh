#!/usr/bin/env bash
set -e
# Baixa modelo na primeira execução (opcional)
# python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; m='google/byt5-small'; AutoTokenizer.from_pretrained(m); AutoModelForSeq2SeqLM.from_pretrained(m)"
exec streamlit run src/app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
