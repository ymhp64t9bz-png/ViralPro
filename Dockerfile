# RunPod Serverless - KwaiCut (GOLD MASTER)
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

COPY core/ ./core/
COPY handler.py .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:$PYTHONPATH
ENV RUNPOD_VOLUME_PATH=/runpod-volume

CMD ["python", "-u", "/app/handler.py"]

