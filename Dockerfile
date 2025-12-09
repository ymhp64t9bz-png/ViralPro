FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV IMAGEMAGICK_BINARY=/usr/bin/convert
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg imagemagick libsndfile1 git wget curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/none/read,write/g' /etc/ImageMagick-6/policy.xml

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

RUN mkdir -p /app/output /app/temp

CMD [ "python3", "-u", "/app/handler.py" ]
