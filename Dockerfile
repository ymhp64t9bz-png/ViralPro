# 💎 ViralPro Serverless v2.0 - Dockerfile
# Imagem otimizada para RunPod Serverless com GPU

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ==================== METADADOS ====================
LABEL maintainer="ViralPro Team"
LABEL description="ViralPro Serverless v2.0 - AI-Powered Viral Shorts Generator"
LABEL version="2.0"

# ==================== VARIÁVEIS DE AMBIENTE ====================
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Diretórios de trabalho
ENV APP_DIR=/app
ENV TEMP_DIR=/tmp/viralpro
ENV OUTPUT_DIR=/tmp/viralpro/output

# ==================== INSTALAÇÃO DE DEPENDÊNCIAS DO SISTEMA ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg (essencial para MoviePy)
    ffmpeg \
    # ImageMagick (para TextClip do MoviePy)
    imagemagick \
    # OpenCV dependencies
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    # Audio processing
    libsndfile1 \
    # Networking
    wget \
    curl \
    git \
    # Fonts (para legendas)
    fonts-dejavu-core \
    fonts-liberation \
    fonts-freefont-ttf \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configura política do ImageMagick para permitir conversão de texto
RUN sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml || true

# ==================== CRIAR DIRETÓRIOS ====================
WORKDIR $APP_DIR

RUN mkdir -p \
    $TEMP_DIR \
    $OUTPUT_DIR

# ==================== COPIAR REQUIREMENTS ====================
COPY requirements.txt .

# ==================== INSTALAR DEPENDÊNCIAS PYTHON ====================
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# ==================== COPIAR CÓDIGO DA APLICAÇÃO ====================
COPY handler.py /app/

# ==================== HEALTHCHECK ====================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import runpod; print('OK')" || exit 1

# ==================== EXPOR PORTA ====================
EXPOSE 8000

# ==================== COMANDO DE INICIALIZAÇÃO ====================
CMD ["python3", "-u", "handler.py"]
