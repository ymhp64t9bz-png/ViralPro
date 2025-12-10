"""
🔥 ViralPro Cloud - Handler de Produção (Final 100% B2)
Compatível com: RunPod Serverless + Agent 3 + Backblaze B2 (Private Bucket)
Funções:
- Download (Signed URL S3v4 do próprio Backblaze)
- Processamento Vertical
- Upload para B2 via b2_storage
- Retorno de Signed URL privada
"""

import runpod
import os
import logging
import requests
import uuid
import boto3
from botocore.client import Config
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings
import b2_storage

# Configura o ImageMagick usado pelo MoviePy
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ViralPro-Cloud")

OUTPUT_DIR = "/app/output"
TEMP_DIR = "/app/temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ------------------------------------------------------------------------------------
# UTILIDADES
# ------------------------------------------------------------------------------------

def normalize_path(path: str) -> str:
    """Remove .private/ e barras desnecessárias."""
    return path.replace(".private/", "").lstrip("/")

def get_s3_client():
    """Client S3 compatível com Backblaze B2 (S3v4)."""
    key_id = os.getenv("B2_KEY_ID")
    app_key = os.getenv("B2_APPLICATION_KEY")
    endpoint = os.getenv("B2_ENDPOINT_URL", "https://s3.us-east-005.backblazeb2.com")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        config=Config(signature_version="s3v4")
    )

def generate_signed_url_s3v4(bucket, file_path):
    """Cria Signed URL S3v4 (Backblaze B2) válida por 5 minutos."""
    try:
        return get_s3_client().generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": file_path},
            ExpiresIn=300,
            HttpMethod="GET"
        )
    except Exception as e:
        logger.error(f"❌ Erro ao gerar Signed URL: {e}")
        return None

def download_video(url):
    """Baixa o vídeo de URL assinada."""
    local_filename = os.path.join(TEMP_DIR, f"source_{uuid.uuid4()}.mp4")
    logger.info(f"⬇️ Baixando: {url[:60]}...")

    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        with requests.get(url, stream=True, headers=headers, timeout=120) as r:
            if r.status_code == 403:
                raise PermissionError("403 Forbidden ao tentar baixar o arquivo.")
            r.raise_for_status()

            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return local_filename

    except Exception as e:
        logger.error(f"❌ Falha no download: {e}")
        return None

# ------------------------------------------------------------------------------------
# EDIÇÃO DO VÍDEO
# ------------------------------------------------------------------------------------

def make_vertical_viral(video_path, title=None):
    """Reformata vídeo horizontal → vertical 1080x1920 com título."""
    logger.info("🎬 Renderizando Vertical Viral...")

    out_path = os.path.join(OUTPUT_DIR, f"viralpro_{uuid.uuid4()}.mp4")

    try:
        clip = VideoFileClip(video_path)

        # background (blur leve)
        bg = clip.resize(height=1920).crop(
            x1=clip.w / 2 - 540,
            x2=clip.w / 2 + 540
        ).fl_image(lambda i: i * 0.4)

        # vídeo principal centralizado
        main = clip.resize(width=1080).set_position("center")

        layers = [bg, main]

        # título opcional
        if title:
            txt = TextClip(
                title.upper(),
                fontsize=70,
                color='white',
                font='DejaVu-Sans-Bold',
                size=(900, None),
                method='caption'
            ).set_position(('center', 200)).set_duration(clip.duration)

            layers.append(txt)

        final = CompositeVideoClip(layers, size=(1080, 1920))
        final = final.set_audio(clip.audio)

        final.write_videofile(
            out_path,
            fps=24,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',
            threads=4,
            logger=None
        )

        clip.close()
        return out_path

    except Exception as e:
        logger.error(f"❌ Erro na renderização: {e}")
        return None

# ------------------------------------------------------------------------------------
# HANDLER PRINCIPAL
# ------------------------------------------------------------------------------------

async def handler(job):
    job_input = job.get("input", {})
    raw_path = job_input.get("video_path") or job_input.get("video_url")
    title = job_input.get("title", "")

    if not raw_path:
        return {"status": "error", "error": "Nenhuma URL recebida."}

    # Resolve link
    if raw_path.startswith("http"):
        url = raw_path
    else:
        path = normalize_path(raw_path)
        url = generate_signed_url_s3v4(os.getenv("B2_BUCKET_NAME"), path)

    if not url:
        return {"status": "error", "error": "Falha ao gerar Signed URL."}

    try:
        # Download
        src = download_video(url)
        if not src:
            return {"status": "error", "error": "Falha no download do vídeo."}

        # Processamento
        final_path = make_vertical_viral(src, title)
        if not final_path:
            return {"status": "error", "error": "Falha no processamento."}

        # Upload para B2
        file_name = f"viralpro/{os.path.basename(final_path)}"
        b2_storage.upload_file(final_path, file_name)

        # URL final assinada
        download_url = b2_storage.generate_signed_download_url(file_name)

        # limpeza
        if os.path.exists(src): os.remove(src)
        if os.path.exists(final_path): os.remove(final_path)

        return {
            "status": "success",
            "file_name": file_name,
            "download_url": download_url
        }

    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
