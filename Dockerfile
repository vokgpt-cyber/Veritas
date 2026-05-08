# EPAM VERITAS - Backend Docker image
#
# Runtime shape:
# - FastAPI backend only.
# - Frontend is built and run by frontend/Dockerfile.
# - Ollama stays on the Windows host and is reached through
#   http://host.docker.internal:11434 from docker-compose.yml.

FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

LABEL maintainer="EPAM Systems" \
      description="EPAM VERITAS backend - on-premise transcription and protocols" \
      version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/veritas/.cache/huggingface \
    TORCH_HOME=/home/veritas/.cache/torch \
    XDG_CACHE_HOME=/home/veritas/.cache \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
      libgomp1 \
      libsndfile1 \
      libopenblas0 \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt \
    && python -m pip install --force-reinstall --no-deps \
      https://github.com/salute-developers/GigaAM/archive/refs/heads/main.zip \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /root/.cache /var/lib/apt/lists/*

RUN groupadd --system veritas \
    && useradd --system --gid veritas --home-dir /home/veritas --create-home veritas

COPY backend/ /app/backend/
COPY config/ /app/config/

RUN mkdir -p \
      /app/data \
      /app/shared-data \
      /app/certs \
      /home/veritas/.cache/huggingface \
      /home/veritas/.cache/torch \
      /home/veritas/.cache/ctranslate2 \
    && chown -R veritas:veritas /app /home/veritas

USER veritas

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=5)" || exit 1

CMD ["python", "-m", "uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--limit-concurrency", "20", \
     "--timeout-keep-alive", "30"]
