FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/sulishi2011/InfiniteTalk-API"
LABEL org.opencontainers.image.description="InfiniteTalk API service with GPU Docker deployment"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    git \
    build-essential \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt requirements-api.txt ./

RUN python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
RUN python -m pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
RUN python -m pip install "misaki[en]" ninja psutil packaging
RUN python -m pip install flash_attn==2.7.4.post1 --no-build-isolation
RUN python -m pip install -r requirements-api.txt

COPY . .

RUN chmod +x /app/scripts/entrypoint.sh

EXPOSE 8000

ENV INFINITETALK_API_HOST=0.0.0.0 \
    INFINITETALK_API_PORT=8000 \
    INFINITETALK_DATA_ROOT=/app/runtime_data \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    INFINITETALK_AUTO_DOWNLOAD_MODELS=true \
    INFINITETALK_AUTO_DOWNLOAD_KOKORO=false

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
CMD ["uvicorn", "infinitetalk_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
