#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ViralPro Serverless - Handler Completo
Processamento de vídeos virais com Smart Crop, Legendas e Títulos IA
Baseado no ViralPro local
"""

import runpod
import os
import sys
import logging
import tempfile
import requests
import uuid
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ==================== CONFIGURAÇÃO ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ViralPro")

# Diretórios
TEMP_DIR = Path("/tmp/viralpro")
OUTPUT_DIR = Path("/tmp/viralpro/output")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("🎯 ViralPro Serverless - Full Features")
print("=" * 60)

# ==================== IMPORTS CONDICIONAIS ====================
try:
    from moviepy.editor import (
        VideoFileClip, ImageClip, AudioFileClip,
        TextClip, CompositeVideoClip, concatenate_videoclips,
        ColorClip
    )
    from moviepy.video.fx.all import crop
    import numpy as np
    MOVIEPY_AVAILABLE = True
    logger.info("✅ MoviePy disponível")
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    logger.error(f"❌ MoviePy não disponível: {e}")

try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV disponível")
except ImportError as e:
    CV2_AVAILABLE = False
    logger.error(f"❌ OpenCV não disponível: {e}")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
    logger.info("✅ PIL disponível")
except ImportError as e:
    PIL_AVAILABLE = False
    logger.error(f"❌ PIL não disponível: {e}")

try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    MEDIAPIPE_AVAILABLE = True
    logger.info("✅ MediaPipe disponível")
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    logger.warning(f"⚠️ MediaPipe não disponível: {e}")

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    logger.info("✅ Faster-Whisper disponível")
except ImportError as e:
    WHISPER_AVAILABLE = False
    logger.warning(f"⚠️ Faster-Whisper não disponível: {e}")

try:
    import boto3
    from botocore.client import Config
    
    B2_KEY_ID = os.getenv("B2_KEY_ID", "68702c2cbfc6")
    B2_APP_KEY = os.getenv("B2_APP_KEY", "00506496bc1450b6722b672d9a43d00605f17eadd7")
    B2_ENDPOINT = os.getenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    B2_BUCKET = os.getenv("B2_BUCKET_NAME", "KortexClipAI")
    
    if B2_KEY_ID and B2_APP_KEY:
        s3_client = boto3.client(
            "s3",
            endpoint_url=B2_ENDPOINT,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APP_KEY,
            config=Config(signature_version="s3v4")
        )
        B2_AVAILABLE = True
        logger.info("✅ Backblaze B2 configurado")
    else:
        B2_AVAILABLE = False
        logger.warning("⚠️ B2 credentials não configuradas")
except Exception as e:
    B2_AVAILABLE = False
    logger.error(f"❌ Erro ao configurar B2: {e}")

# ==================== DOWNLOAD DE VÍDEO ====================
def download_video(url: str) -> str:
    """Baixa vídeo da URL"""
    try:
        logger.info(f"📥 Baixando vídeo...")
        
        temp_file = TEMP_DIR / f"input_{uuid.uuid4().hex[:8]}.mp4"
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0 and downloaded % (1024 * 1024) == 0:
                    progress = (downloaded / total_size) * 100
                    logger.info(f"📥 Download: {progress:.1f}%")
        
        logger.info(f"✅ Download completo: {temp_file} ({downloaded / 1024 / 1024:.2f} MB)")
        return str(temp_file)
        
    except Exception as e:
        logger.error(f"❌ Erro no download: {e}")
        raise

# ==================== DETECÇÃO DE ROSTO ====================
def detect_face_center(frame, face_detection) -> Optional[int]:
    """Detecta centro do rosto usando MediaPipe"""
    try:
        if not MEDIAPIPE_AVAILABLE:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            frame_width = frame.shape[1]
            face_center_x = int((bbox.xmin + bbox.width / 2) * frame_width)
            
            return face_center_x
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Erro na detecção de rosto: {e}")
        return None

# ==================== SMART CROP ====================
def create_smart_crop(clip) -> VideoFileClip:
    """Aplica Smart Crop 9:16 focado no rosto"""
    try:
        logger.info("🎯 Aplicando Smart Crop...")
        
        if not MEDIAPIPE_AVAILABLE or not CV2_AVAILABLE:
            logger.warning("⚠️ MediaPipe/CV2 não disponível, usando crop centralizado")
            # Crop centralizado simples
            w, h = clip.size
            target_w = int(h * 9 / 16)
            x_center = w / 2
            x1 = max(0, x_center - target_w / 2)
            return clip.fx(crop, x1=x1, width=target_w)
        
        # Análise de frames para encontrar rosto
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            # Amostra 5 frames
            sample_times = [clip.duration * i / 5 for i in range(5)]
            face_centers = []
            
            for t in sample_times:
                frame = clip.get_frame(t)
                face_x = detect_face_center(frame, face_detection)
                if face_x:
                    face_centers.append(face_x)
            
            # Calcula posição média do rosto
            if face_centers:
                avg_face_x = sum(face_centers) / len(face_centers)
                logger.info(f"✅ Rosto detectado em X={avg_face_x:.0f}")
            else:
                avg_face_x = clip.w / 2
                logger.warning("⚠️ Nenhum rosto detectado, usando centro")
            
            # Calcula crop
            target_w = int(clip.h * 9 / 16)
            x1 = max(0, min(avg_face_x - target_w / 2, clip.w - target_w))
            
            logger.info(f"✂️ Crop: x1={x1:.0f}, width={target_w}")
            return clip.fx(crop, x1=x1, width=target_w)
        
    except Exception as e:
        logger.error(f"❌ Erro no Smart Crop: {e}")
        # Fallback para crop centralizado
        w, h = clip.size
        target_w = int(h * 9 / 16)
        x1 = (w - target_w) / 2
        return clip.fx(crop, x1=x1, width=target_w)

# ==================== GERAÇÃO DE LEGENDAS ====================
def generate_subtitles(audio_path: str, model_size: str = "base") -> List[Dict]:
    """Gera legendas usando Faster-Whisper"""
    try:
        if not WHISPER_AVAILABLE:
            logger.warning("⚠️ Whisper não disponível")
            return []
        
        logger.info(f"🎤 Transcrevendo áudio com Whisper ({model_size})...")
        
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        
        segments, info = model.transcribe(audio_path, language="pt")
        
        subtitles = []
        for segment in segments:
            subtitles.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        logger.info(f"✅ {len(subtitles)} legendas geradas")
        return subtitles
        
    except Exception as e:
        logger.error(f"❌ Erro na transcrição: {e}")
        return []

# ==================== ADICIONAR LEGENDAS AO VÍDEO ====================
def add_subtitles_to_video(clip: VideoFileClip, subtitles: List[Dict]) -> VideoFileClip:
    """Adiciona legendas ao vídeo"""
    try:
        if not subtitles or not PIL_AVAILABLE:
            return clip
        
        logger.info("📝 Adicionando legendas...")
        
        subtitle_clips = []
        
        for sub in subtitles:
            # Cria imagem com texto
            img = Image.new('RGBA', (1080, 200), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 50)
            except:
                font = ImageFont.load_default()
            
            text = sub['text']
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x = (1080 - text_width) // 2
            y = 50
            
            # Borda preta
            for adj_x in range(-2, 3):
                for adj_y in range(-2, 3):
                    draw.text((x + adj_x, y + adj_y), text, font=font, fill=(0, 0, 0, 255))
            
            # Texto branco
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
            
            # Converte para clip
            img_array = np.array(img)
            txt_clip = ImageClip(img_array).set_duration(sub['end'] - sub['start'])
            txt_clip = txt_clip.set_start(sub['start']).set_position(('center', 'bottom'))
            
            subtitle_clips.append(txt_clip)
        
        # Composição
        final_clip = CompositeVideoClip([clip] + subtitle_clips, size=clip.size)
        final_clip = final_clip.set_audio(clip.audio)
        
        logger.info(f"✅ {len(subtitle_clips)} legendas adicionadas")
        return final_clip
        
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar legendas: {e}")
        return clip

# ==================== PROCESSAMENTO DE VÍDEO VIRAL ====================
def process_viral_video(
    video_path: str,
    num_clips: int = 3,
    clip_duration: int = 60,
    start_min: int = 0,
    add_subtitles: bool = True
) -> List[str]:
    """Processa vídeo completo com Smart Crop e Legendas"""
    try:
        logger.info("🎬 Iniciando processamento viral...")
        
        if not MOVIEPY_AVAILABLE:
            raise Exception("MoviePy não disponível")
        
        # Carrega vídeo
        video = VideoFileClip(video_path)
        duration = video.duration
        
        logger.info(f"📊 Duração: {duration}s")
        
        # Calcula clips
        start_time = start_min * 60
        clips_output = []
        
        for i in range(num_clips):
            clip_start = start_time + (i * clip_duration)
            clip_end = min(clip_start + clip_duration, duration)
            
            if clip_start >= duration:
                break
            
            logger.info(f"✂️ Processando clip {i+1}/{num_clips}: {clip_start}s - {clip_end}s")
            
            # Extrai clip
            clip = video.subclip(clip_start, clip_end)
            
            # Smart Crop
            clip = create_smart_crop(clip)
            
            # Legendas
            if add_subtitles:
                # Extrai áudio
                audio_path = TEMP_DIR / f"audio_{i}_{uuid.uuid4().hex[:8]}.wav"
                clip.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
                
                # Gera legendas
                subtitles = generate_subtitles(str(audio_path))
                
                # Adiciona legendas
                if subtitles:
                    clip = add_subtitles_to_video(clip, subtitles)
                
                # Remove áudio temporário
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            # Exporta
            output_file = OUTPUT_DIR / f"viral_{i+1}_{uuid.uuid4().hex[:8]}.mp4"
            
            logger.info(f"🎬 Renderizando clip {i+1}...")
            
            clip.write_videofile(
                str(output_file),
                codec='libx264',
                audio_codec='aac',
                preset='fast',
                ffmpeg_params=[
                    '-profile:v', 'high',
                    '-level', '4.1',
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart'
                ],
                verbose=False,
                logger=None
            )
            
            clip.close()
            clips_output.append(str(output_file))
            
            logger.info(f"✅ Clip {i+1} concluído: {output_file}")
        
        video.close()
        
        logger.info(f"✅ Processamento completo: {len(clips_output)} clips gerados")
        return clips_output
        
    except Exception as e:
        logger.error(f"❌ Erro no processamento: {e}")
        raise

# ==================== UPLOAD PARA B2 ====================
def upload_to_b2(file_path: str, object_name: str = None) -> Optional[str]:
    """Upload para Backblaze B2"""
    try:
        if not B2_AVAILABLE:
            logger.warning("⚠️ B2 não disponível, retornando path local")
            return file_path
        
        if object_name is None:
            object_name = f"viralpro/{os.path.basename(file_path)}"
        
        logger.info(f"📤 Uploading para B2: {object_name}")
        
        s3_client.upload_file(file_path, B2_BUCKET, object_name)
        
        # Gera URL assinada
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': B2_BUCKET, 'Key': object_name},
            ExpiresIn=3600
        )
        
        logger.info(f"✅ Upload completo: {object_name}")
        return url
        
    except Exception as e:
        logger.error(f"❌ Erro no upload B2: {e}")
        return file_path

# ==================== HANDLER PRINCIPAL ====================
def handler(event):
    """Handler principal do ViralPro"""
    try:
        logger.info("🚀 ViralPro Handler iniciado")
        logger.info(f"📦 Event: {event.get('id', 'N/A')}")
        
        input_data = event.get("input", {})
        
        # Modo de teste
        if input_data.get("mode") == "test":
            return {
                "status": "success",
                "message": "ViralPro worker funcionando!",
                "version": "1.0",
                "features": {
                    "moviepy": MOVIEPY_AVAILABLE,
                    "cv2": CV2_AVAILABLE,
                    "pil": PIL_AVAILABLE,
                    "mediapipe": MEDIAPIPE_AVAILABLE,
                    "whisper": WHISPER_AVAILABLE,
                    "b2": B2_AVAILABLE
                }
            }
        
        # Validação
        video_url = input_data.get("video_url")
        if not video_url:
            return {
                "status": "error",
                "error": "video_url não fornecido"
            }
        
        # Parâmetros
        num_clips = input_data.get("num_clips", 3)
        clip_duration = input_data.get("clip_duration", 60)
        start_min = input_data.get("start_min", 0)
        add_subtitles = input_data.get("add_subtitles", True)
        
        # Download
        video_path = download_video(video_url)
        
        # Processamento
        clips = process_viral_video(
            video_path,
            num_clips,
            clip_duration,
            start_min,
            add_subtitles
        )
        
        # Upload para B2
        clips_data = []
        for clip_path in clips:
            b2_url = upload_to_b2(clip_path)
            clips_data.append({
                "local_path": clip_path,
                "b2_url": b2_url
            })
        
        # Limpeza
        try:
            os.remove(video_path)
        except:
            pass
        
        # Resultado
        result = {
            "status": "success",
            "message": f"{len(clips)} clips virais gerados",
            "clips": clips_data
        }
        
        logger.info(f"✅ Job completo: {len(clips)} clips")
        return result
        
    except Exception as e:
        logger.error(f"❌ Erro no handler: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "type": type(e).__name__
        }

# ==================== INICIALIZAÇÃO ====================
if __name__ == "__main__":
    logger.info("🎯 Iniciando ViralPro Serverless Worker...")
    logger.info(f"📊 MoviePy: {MOVIEPY_AVAILABLE}")
    logger.info(f"📊 OpenCV: {CV2_AVAILABLE}")
    logger.info(f"📊 PIL: {PIL_AVAILABLE}")
    logger.info(f"📊 MediaPipe: {MEDIAPIPE_AVAILABLE}")
    logger.info(f"📊 Whisper: {WHISPER_AVAILABLE}")
    logger.info(f"📊 B2: {B2_AVAILABLE}")
    
    runpod.serverless.start({"handler": handler})
    logger.info("✅ Worker iniciado!")
