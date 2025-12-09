# RunPod Serverless - ViralPro (WHISPER FIX)
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    default-jre \
    ffmpeg \
    git \
    fonts-liberation \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# INSTALAÇÃO ROBUSTA
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir faster-whisper ctranslate2 && \
    pip install --no-cache-dir -r requirements.txt

COPY core/ ./core/
COPY fontes/ ./fontes/
COPY handler.py .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH
ENV ADOBE_FONT_PATH=/app/fontes
ENV RUNPOD_VOLUME_PATH=/runpod-volume

CMD ["python", "-u", "/app/handler.py"]
