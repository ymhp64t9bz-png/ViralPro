"""
🔥 ViralPro Cloud - Handler de Produção
Pipeline: Download -> Convert to Vertical (9:16) -> Blurred BG -> Title Overlay -> Upload B2
"""

import runpod
import os
import logging
import requests
import uuid
import b2_storage
from moviepy.editor import (
    VideoFileClip, TextClip, CompositeVideoClip, ColorClip, vfx
)
from moviepy.config import change_settings

# Configuração Docker
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ViralPro-Cloud")

OUTPUT_DIR = "/app/output"
TEMP_DIR = "/app/temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def download_video(url):
    local_filename = os.path.join(TEMP_DIR, f"source_{uuid.uuid4()}.mp4")
    logger.info(f"⬇️ Baixando: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def make_vertical_viral(video_path, title=None):
    """
    Transforma vídeo horizontal em vertical (1080x1920).
    Técnica: Vídeo original centralizado + Fundo desfocado do mesmo vídeo (Blurred Background).
    """
    logger.info("🎬 Iniciando edição 'Viral Vertical'...")
    output_filename = f"viralpro_{uuid.uuid4()}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        clip = VideoFileClip(video_path)
        
        # 1. Background Desfocado (Preenche 1080x1920)
        # Corta o centro para preencher a tela verticalmente
        bg_clip = clip.resize(height=1920)
        bg_clip = bg_clip.crop(x1=bg_clip.w/2 - 540, x2=bg_clip.w/2 + 540)
        # Aplica Blur (simulado com resize down/up se blur for lento, mas vamos tentar blur real)
        # Blur real é muito lento em CPU. Vamos usar dim effect.
        bg_clip = bg_clip.fl_image(lambda image: image * 0.4) # Escurece 60%
        
        # 2. Vídeo Principal (Centralizado, largura 1080)
        main_clip = clip.resize(width=1080)
        main_clip = main_clip.set_position("center")
        
        # 3. Legenda/Título (Opcional)
        final_layers = [bg_clip, main_clip]
        
        if title:
            try:
                txt = TextClip(
                    title.upper(), 
                    fontsize=70, 
                    color='white', 
                    font='DejaVu-Sans-Bold',
                    method='caption',
                    size=(900, None)
                ).set_position(('center', 200)).set_duration(clip.duration)
                final_layers.append(txt)
            except Exception as e:
                logger.warning(f"⚠️ Erro no título (ImageMagick?): {e}")

        # 4. Composição
        final = CompositeVideoClip(final_layers, size=(1080, 1920))
        final = final.set_duration(clip.duration)
        final = final.set_audio(clip.audio) # Mantém áudio original
        
        # 5. Exporta
        final.write_videofile(
            output_path, 
            fps=24, # Cinematic
            codec='libx264', 
            audio_codec='aac',
            preset='ultrafast', # Serverless speed
            threads=4,
            logger=None
        )
        
        clip.close()
        return output_path

    except Exception as e:
        logger.error(f"❌ Erro na renderização: {e}")
        return None

async def handler(job):
    job_input = job.get("input", {})
    video_url = job_input.get("video_url")
    title = job_input.get("title", "") # Título opcional sobre o vídeo
    
    if not video_url:
        return {"status": "error", "error": "No video_url provided"}
    
    try:
        # 1. Download
        source_path = download_video(video_url)
        
        # 2. Edição Viral
        final_video = make_vertical_viral(source_path, title)
        
        if not final_video:
            return {"status": "error", "error": "Render failed"}
            
        # 3. Upload B2
        file_name = f"viralpro/{os.path.basename(final_video)}"
        if b2_storage.upload_file(final_video, file_name):
            url = b2_storage.generate_signed_download_url(file_name)
            
            # Limpeza
            if os.path.exists(source_path): os.remove(source_path)
            
            return {
                "status": "success",
                "b2_key": file_name,
                "download_url": url
            }
        
        return {"status": "error", "error": "Upload failed"}

    except Exception as e:
        logger.error(f"❌ Erro Fatal: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
