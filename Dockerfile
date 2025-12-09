# RunPod Serverless - ViralPro (FINAL)
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Sistema
RUN apt-get update && apt-get install -y \
    default-jre \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# ESTRATÉGIA DE INSTALAÇÃO SEGURA:
# 1. Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 2. Instala requirements simples primeiro
RUN pip install --no-cache-dir -r requirements.txt

# 3. Instala Faster Whisper e CTranslate2 (as mais complexas) separadamente
RUN pip install --no-cache-dir faster-whisper ctranslate2

# 4. Re-garante Torch CUDA (caso algo tenha sobrescrito, embora improvável nessa ordem)
# Opcional, pois a imagem base já tem torch, mas guarantees nunca são demais
# RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

COPY core/ ./core/
COPY fontes/ ./fontes/
COPY handler.py .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH
ENV ADOBE_FONT_PATH=/app/fontes
ENV RUNPOD_VOLUME_PATH=/runpod-volume

CMD ["python", "-u", "/app/handler.py"]


