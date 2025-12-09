FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV IMAGEMAGICK_BINARY=/usr/bin/convert
ENV PYTHONPATH=/app

WORKDIR /app

# 1. Dependências de Sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg imagemagick libsndfile1 git wget curl libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/none/read,write/g' /etc/ImageMagick-6/policy.xml

# 2. Python Deps (Force Reinstall para evitar cache corrompido)
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --force-reinstall -r /app/requirements.txt

# 3. Código Fonte
COPY . /app/

# DEBUG: Auditoria de arquivos e pacotes
RUN echo "--- FILES IN /app ---" && ls -la /app && \
    echo "--- INSTALLED PACKAGES ---" && pip list

RUN mkdir -p /app/output /app/temp

CMD [ "python3", "-u", "/app/handler.py" ]
