# RunPod Serverless - AnimeCut
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 1. Instala dependências do sistema (FFmpeg, ImageMagick, Java)
RUN apt-get update && apt-get install -y \
    default-jre \
    ffmpeg \
    imagemagick \
    git \
    fonts-liberation \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. CORREÇÃO CRÍTICA DO IMAGEMAGICK (Para permitir criar textos/legendas)
# Sem isso, o MoviePy dá erro de "security policy"
RUN sed -i 's/none/read,write/g' /etc/ImageMagick-6/policy.xml

WORKDIR /app

# 3. Copia requirements e instala (Aqui entra o MoviePy)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir runpod

# 4. Copia código da aplicação e fontes
COPY core/ ./core/
COPY assets/ ./assets/
COPY fontes/ ./fontes/
COPY handler.py .

# 5. Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
# Aponta para onde as fontes estão (para o ImageMagick achar)
ENV ADOBE_FONT_PATH=/app/fontes
ENV IMAGEMAGICK_BINARY=/usr/bin/convert
ENV RUNPOD_VOLUME_PATH=/runpod-volume

# Comando para iniciar o handler serverless
CMD ["python", "-u", "handler.py"]
