"""
HANDLER.PY - VIRALPRO SERVERLESS
=================================
RunPod Serverless Handler para processamento viral com legendas e face tracking
"""

import runpod
import os
import sys
import json
import tempfile
import requests
from pathlib import Path

# Configuração de caminhos
VOLUME_PATH = os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")
MODELS_PATH = os.path.join(VOLUME_PATH, "models")
TEMP_PATH = os.path.join(VOLUME_PATH, "temp")

sys.path.insert(0, "/app/core")

from core.ai_services.local_ai_service import transcribe_audio_batch, generate_viral_title_batch
from core.video_processing.anti_shadowban import apply_anti_shadowban
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

def download_file(url, dest_path):
    """Download de arquivo via URL"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return dest_path

def add_captions(video_clip, segments, font_path="/app/fontes/wanted_m54/Wanted M54.ttf"):
    """Adiciona legendas ao vídeo"""
    caption_clips = []
    
    for seg in segments:
        txt_clip = TextClip(
            seg["text"],
            fontsize=40,
            color='white',
            stroke_color='black',
            stroke_width=2,
            font=font_path,
            method='caption',
            size=(video_clip.w * 0.9, None)
        )
        
        txt_clip = txt_clip.set_position(('center', 'bottom')).set_start(seg["start"]).set_duration(seg["end"] - seg["start"])
        caption_clips.append(txt_clip)
    
    return CompositeVideoClip([video_clip] + caption_clips)

def process_viral_video(input_data):
    """
    Processa vídeo para formato viral
    
    Input esperado:
    {
        "video_url": "https://...",
        "title": "Título opcional",
        "config": {
            "add_captions": true,
            "add_title": true,
            "anti_shadowban": true,
            "vertical_format": true
        }
    }
    """
    try:
        os.makedirs(TEMP_PATH, exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir=TEMP_PATH)
        
        # Download do vídeo
        video_url = input_data.get("video_url")
        if not video_url:
            return {"error": "video_url é obrigatório"}
        
        video_path = os.path.join(temp_dir, "input_video.mp4")
        print(f"[HANDLER] Baixando vídeo de {video_url}...")
        download_file(video_url, video_path)
        
        # Configurações
        config = input_data.get("config", {})
        add_captions = config.get("add_captions", True)
        add_title = config.get("add_title", True)
        anti_shadowban = config.get("anti_shadowban", True)
        vertical_format = config.get("vertical_format", True)
        
        # 1. Carrega vídeo
        print("[HANDLER] Carregando vídeo...")
        video = VideoFileClip(video_path)
        
        # 2. Transcrição para legendas
        if add_captions:
            print("[HANDLER] Transcrevendo para legendas...")
            whisper_result = transcribe_audio_batch(video_path)
            video = add_captions(video, whisper_result["segments"])
        
        # 3. Adiciona título
        if add_title:
            title_text = input_data.get("title")
            if not title_text:
                # Gera título com IA
                print("[HANDLER] Gerando título com IA...")
                full_text = " ".join([seg["text"] for seg in whisper_result["segments"]])
                title_text = generate_viral_title_batch("Vídeo", full_text[:500], None)
            
            title_clip = TextClip(
                title_text,
                fontsize=60,
                color='white',
                stroke_color='black',
                stroke_width=3,
                font="/app/fontes/wanted_m54/Wanted M54.ttf"
            )
            title_clip = title_clip.set_position(('center', 'top')).set_duration(video.duration)
            video = CompositeVideoClip([video, title_clip])
        
        # 4. Formato vertical
        if vertical_format:
            print("[HANDLER] Convertendo para formato vertical...")
            # Crop para 9:16
            target_w = 1080
            target_h = 1920
            
            if video.w / video.h > target_w / target_h:
                # Vídeo muito largo
                new_w = int(video.h * target_w / target_h)
                x_center = video.w / 2
                video = video.crop(x_center=x_center, width=new_w)
            
            video = video.resize((target_w, target_h))
        
        # 5. Renderiza
        output_path = os.path.join(temp_dir, "output_viral.mp4")
        print("[HANDLER] Renderizando vídeo...")
        
        video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset="medium",
            logger=None
        )
        
        video.close()
        
        # 6. Anti-shadowban
        if anti_shadowban:
            print("[HANDLER] Aplicando anti-shadowban...")
            final_path = os.path.join(temp_dir, "output_final.mp4")
            apply_anti_shadowban(output_path, final_path)
            output_path = final_path
        
        return {
            "status": "success",
            "title": title_text if add_title else None,
            "duration": video.duration,
            "local_path": output_path,
            "upload_url": None
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def handler(event):
    """RunPod Handler principal"""
    input_data = event.get("input", {})
    result = process_viral_video(input_data)
    return result

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
