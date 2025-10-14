FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir uv
RUN uv sync --extra cpu --no-cache

COPY . .
EXPOSE 8501
CMD ["uv","run","streamlit","run","src/ftc4/app/streamlit_app.py","--server.port=8501","--server.address=0.0.0.0"]
