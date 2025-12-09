"""
HANDLER.PY - KWAICUT SERVERLESS
================================
RunPod Serverless Handler para cortes automáticos de vídeo
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

from core.video_processing.scene_extractor import extract_scenes_from_whisper
from core.ai_services.local_ai_service import transcribe_audio_batch
from moviepy.editor import VideoFileClip

def download_file(url, dest_path):
    """Download de arquivo via URL"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return dest_path

def process_kwai_cuts(input_data):
    """
    Processa vídeo para cortes automáticos
    
    Input esperado:
    {
        "video_url": "https://...",
        "config": {
            "max_clips": 10,
            "min_duration": 10,
            "max_duration": 60,
            "threshold": 30.0
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
        max_clips = config.get("max_clips", 10)
        min_duration = config.get("min_duration", 10)
        max_duration = config.get("max_duration", 60)
        threshold = config.get("threshold", 30.0)
        
        # 1. Transcrição
        print("[HANDLER] Transcrevendo áudio...")
        whisper_result = transcribe_audio_batch(video_path)
        
        # 2. Detecção de cenas
        print("[HANDLER] Detectando cenas...")
        scenes = extract_scenes_from_whisper(whisper_result, video_path)
        
        # 3. Filtra por duração
        filtered_scenes = []
        for scene in scenes:
            duration = scene["end"] - scene["start"]
            if min_duration <= duration <= max_duration:
                filtered_scenes.append(scene)
        
        # Limita ao máximo
        filtered_scenes = filtered_scenes[:max_clips]
        
        # 4. Extrai clips
        print("[HANDLER] Extraindo clips...")
        output_dir = os.path.join(temp_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        video = VideoFileClip(video_path)
        results = []
        
        for idx, scene in enumerate(filtered_scenes, 1):
            output_path = os.path.join(output_dir, f"clip_{idx}.mp4")
            
            # Extrai subclip
            subclip = video.subclip(scene["start"], scene["end"])
            subclip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(temp_dir, f"temp_audio_{idx}.m4a"),
                remove_temp=True,
                logger=None
            )
            
            results.append({
                "clip_number": idx,
                "start": scene["start"],
                "end": scene["end"],
                "duration": scene["end"] - scene["start"],
                "text": scene.get("text", ""),
                "local_path": output_path,
                "upload_url": None
            })
        
        video.close()
        
        return {
            "status": "success",
            "total_clips": len(results),
            "clips": results
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
    result = process_kwai_cuts(input_data)
    return result

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
