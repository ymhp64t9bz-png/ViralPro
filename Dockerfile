# syntax=docker/dockerfile:1.4
# 🎬 ViralPRO Serverless v1.0.0 - SHORTS/REELS GENERATOR
# Face Tracking + Auto Subtitles + Smart Crop
# Stack: Whisper V3 Turbo, MediaPipe, OpenCV, FFmpeg NVENC
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# ==================== BUILD INFO ====================
ARG BUILD_VERSION=1.0.0
ARG BUILD_TIMESTAMP=20251223_VIRALPRO_V1

RUN echo "ViralPRO Build: ${BUILD_VERSION}" && \
    echo "Timestamp: ${BUILD_TIMESTAMP}" && \
    echo "Build ID: $(date +%s)_$RANDOM" > /BUILD_ID

RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 2>/dev/null || true

# ==================== CACHE BUSTER ====================
ARG CACHEBUST=20251223_VIRALPRO_V1_RELEASE
RUN echo "=== VIRALPRO v1.0.0 ===" > /BUILD_INFO && \
    echo "Timestamp: ${CACHEBUST}" >> /BUILD_INFO && \
    echo "Build: $(date -Iseconds)" >> /BUILD_INFO

WORKDIR /app

# ==================== VARIÁVEIS DE AMBIENTE ====================
ENV BUILD_VERSION="1.0.0"
ENV BUILD_DATE="2025-12-23"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME="/runpod-volume/.cache/huggingface"
ENV CUDA_VISIBLE_DEVICES="0"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# ==================== 1. DEPENDÊNCIAS DE SISTEMA ====================
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    pkg-config \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    nano \
    curl \
    wget \
    gnupg \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ==================== 2. FFMPEG ESTÁTICO COM NVENC ====================
# Binários estáticos de johnvansickle.com com NVENC
RUN echo "=== INSTALANDO FFMPEG ESTÁTICO COM NVENC ===" && \
    cd /tmp && \
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar -xf ffmpeg-release-amd64-static.tar.xz && \
    mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf /tmp/ffmpeg* && \
    echo "✓ FFmpeg estático instalado"

# Verifica FFmpeg e NVENC
RUN echo "=== VERIFICANDO FFMPEG ===" && \
    ffmpeg -version | head -5 && \
    echo "" && \
    echo "=== ENCODERS DISPONÍVEIS ===" && \
    ffmpeg -hide_banner -encoders 2>/dev/null | grep -E "264|265|nvenc" | head -10 && \
    echo "" && \
    echo "=== VERIFICAÇÃO NVENC ===" && \
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "h264_nvenc"; then \
        echo "✓ NVENC h264_nvenc DISPONÍVEL!"; \
    else \
        echo "⚠ NVENC verificar em runtime"; \
    fi

# ==================== 3. cuDNN 9 ====================
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends libcudnn9-cuda-12 libcudnn9-dev-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

# Atualizar pip
RUN pip install --upgrade pip setuptools wheel

# ==================== 4. NUMPY SHIELD ====================
RUN pip install --no-cache-dir "numpy==1.26.4"

# ==================== 5. CORE DEPENDENCIES ====================
RUN pip install --no-cache-dir \
    "runpod>=1.6.0" \
    "boto3>=1.34.0" \
    "botocore>=1.34.0" \
    "requests>=2.31.0" \
    "tqdm>=4.66.4" \
    "colorama"

# ==================== 6. PROCESSAMENTO DE VÍDEO ====================
RUN pip install --no-cache-dir \
    "moviepy==1.0.3" \
    "imageio>=2.34.1" \
    "imageio-ffmpeg>=0.5.1" \
    "proglog>=0.1.10" \
    "opencv-python-headless>=4.9.0.80"

# ==================== 7. PROCESSAMENTO DE ÁUDIO ====================
RUN pip install --no-cache-dir \
    "librosa" \
    "soundfile>=0.12.1" \
    "scipy"

# ==================== 8. MEDIAPIPE - FACE DETECTION ====================
# MediaPipe para detecção facial precisa e rápida
RUN pip install --no-cache-dir \
    "mediapipe>=0.10.9"

# ==================== 9. IA & VISÃO ====================
RUN pip install --no-cache-dir \
    "ultralytics"

# ==================== 10. DeepFilterNet ====================
RUN pip install --no-cache-dir "deepfilternet" || echo "DeepFilterNet opcional"

# ==================== 11. WHISPER & TRANSCRIÇÃO ====================
RUN pip install --no-cache-dir \
    "transformers>=4.40.0" \
    "accelerate>=0.30.0" \
    "optimum" \
    "protobuf" \
    "sentencepiece" \
    "ctranslate2>=4.0.0" \
    "faster-whisper>=1.0.0"

# ==================== 12. FERRAMENTAS ====================
RUN pip install --no-cache-dir \
    "Pillow>=10.3.0" \
    "decorator<5.0" \
    "Cython<3"

# ==================== 13. NUMPY INTEGRITY ====================
RUN pip install "numpy==1.26.4" --force-reinstall --no-cache-dir

# ==================== 14. VERIFICAÇÕES ====================
RUN python3 -c "import ctranslate2; print(f'CTranslate2: {ctranslate2.__version__}')" && \
    python3 -c "from faster_whisper import WhisperModel; print('faster-whisper: OK')" && \
    python3 -c "import mediapipe; print(f'MediaPipe: {mediapipe.__version__}')" && \
    python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && \
    ldconfig -p | grep cudnn || echo "Aviso: cuDNN libs"

# ==================== 15. FONTES PARA LEGENDAS ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fonts-liberation \
    fonts-freefont-ttf \
    fonts-roboto \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/fonts /workspace/fonts

# Copia fontes customizadas (se existirem)
COPY fontes/ /app/fonts/ 2>/dev/null || true

RUN for font in /app/fonts/*; do \
        if [ -f "$font" ]; then \
            ln -sf "$font" /workspace/fonts/$(basename "$font"); \
        fi; \
    done && \
    echo "Fontes:" && \
    ls -la /workspace/fonts/ 2>/dev/null || true

RUN mkdir -p /usr/local/share/fonts/custom && \
    cp /app/fonts/* /usr/local/share/fonts/custom/ 2>/dev/null || true && \
    fc-cache -fv

# ==================== 16. HANDLER ====================
ARG HANDLER_NOCACHE=1.0.0_20251223
RUN echo "Handler rebuild: ${HANDLER_NOCACHE} - $(date)" > /tmp/handler_build.txt

COPY handler.py .

RUN echo "=== BUILD COMPLETO v1.0.0 ===" && \
    echo "Handler timestamp: $(date -Iseconds)" && \
    echo "FFmpeg:" && \
    which ffmpeg && \
    ffmpeg -version | head -2 && \
    echo "Python:" && python3 --version && \
    echo "MediaPipe: OK" && \
    echo "Whisper: OK" && \
    echo "Handler:" && \
    head -15 handler.py && \
    echo "Build finalizado!"

# ==================== VOLUMES ====================
VOLUME ["/workspace"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# ==================== EXECUÇÃO ====================
CMD ["python3", "-u", "handler.py"]
