# 🎯 ViralPro Serverless - Dockerfile Funcional
# Versão sem HEALTHCHECK (que causava crash)

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# ==================== VARIÁVEIS DE AMBIENTE ====================
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# ==================== DEPENDÊNCIAS DO SISTEMA ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ==================== WORKDIR ====================
WORKDIR /app

# ==================== CRIAR DIRETÓRIOS ====================
RUN mkdir -p /tmp/viralpro /tmp/viralpro/output

# ==================== INSTALAR DEPENDÊNCIAS PYTHON ====================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# ==================== COPIAR CÓDIGO ====================
COPY handler.py .

# ==================== SEM HEALTHCHECK (CAUSAVA CRASH) ====================
# HEALTHCHECK REMOVIDO INTENCIONALMENTE

# ==================== COMANDO ====================
CMD ["python3", "-u", "handler.py"]
