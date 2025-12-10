#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💎 ViralPro Serverless v2.0 - Handler Completo
================================================
Sistema completo de geração de shorts verticais virais para RunPod Serverless.

FUNCIONALIDADES IMPLEMENTADAS:
✅ Smart Crop com rastreamento facial (MediaPipe)
✅ Legendas automáticas com Whisper (GPU)
✅ Títulos virais com Gemini API
✅ Renderização GPU (NVENC) + fallback CPU
✅ Múltiplas fontes estilizadas
✅ Upload automático para Backblaze B2
✅ Processamento em lote
"""

import runpod
import os
import sys
import gc
import logging
import time
import random
import re
from pathlib import Path
from typing import List, Dict, Optional

# ==================== CONFIGURAÇÃO DE LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ViralPro-Serverless")

# ==================== PATHS ====================

TEMP_DIR = Path("/tmp/viralpro")
OUTPUT_DIR = Path("/tmp/viralpro/output")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== IMPORTS ====================

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
    logger.info("✅ OpenCV importado")
except ImportError:
    CV2_AVAILABLE = False
    logger.error("❌ OpenCV não disponível")

try:
    from moviepy.editor import (
        VideoFileClip, AudioFileClip, TextClip,
        CompositeVideoClip, concatenate_videoclips
    )
    from moviepy.video.fx.all import crop
    from moviepy.config import change_settings
    
    change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
    
    MOVIEPY_AVAILABLE = True
    logger.info("✅ MoviePy importado")
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    logger.error(f"❌ Erro ao importar MoviePy: {e}")

try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    MEDIAPIPE_AVAILABLE = True
    logger.info("✅ MediaPipe importado")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("⚠️ MediaPipe não disponível")

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    logger.info("✅ Faster-Whisper importado")
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("⚠️ Faster-Whisper não disponível")

try:
    import google.generativeai as genai
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        logger.info("✅ Gemini API configurado")
    else:
        GEMINI_AVAILABLE = False
        logger.warning("⚠️ GEMINI_API_KEY não configurado")
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("⚠️ Gemini não disponível")

try:
    import boto3
    from botocore.client import Config
    
    B2_KEY_ID = os.getenv("B2_KEY_ID", "68702c2cbfc6")
    B2_APP_KEY = os.getenv("B2_APP_KEY", "00506496bc1450b6722b672d9a43d00605f17eadd7")
    B2_ENDPOINT = os.getenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
    B2_BUCKET = os.getenv("B2_BUCKET_NAME", "autocortes-storage")
    
    s3_client = boto3.client(
        "s3",
        endpoint_url=B2_ENDPOINT,
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APP_KEY,
        config=Config(signature_version="s3v4")
    )
    
    B2_AVAILABLE = True
    logger.info("✅ Backblaze B2 configurado")
except Exception as e:
    B2_AVAILABLE = False
    logger.error(f"❌ Erro ao configurar B2: {e}")

# ==================== FUNÇÕES AUXILIARES ====================

def clean_memory():
    """Limpa memória"""
    gc.collect()
    logger.info("🧹 Memória limpa")

def upload_to_b2(file_path: Path, remote_path: str) -> Optional[str]:
    """Upload para Backblaze B2"""
    if not B2_AVAILABLE:
        logger.error("❌ B2 não disponível")
        return None
    
    try:
        logger.info(f"📤 Uploading para B2: {remote_path}")
        
        with open(file_path, 'rb') as f:
            s3_client.upload_fileobj(f, B2_BUCKET, remote_path)
        
        # Gera URL assinada (válida por 7 dias)
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': B2_BUCKET, 'Key': remote_path},
            ExpiresIn=604800  # 7 dias
        )
        
        logger.info(f"✅ Upload completo: {url}")
        return url
    except Exception as e:
        logger.error(f"❌ Erro no upload B2: {e}")
        return None

# ==================== SMART CROP (MEDIAPIPE) ====================

def detect_face_center(frame, face_detection) -> Optional[int]:
    """
    Detecta o centro do rosto no frame usando MediaPipe
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        center_x = int((bboxC.xmin + bboxC.width / 2) * width)
        return center_x
    
    return None

def create_smart_crop(clip):
    """
    Aplica Smart Crop 9:16 com rastreamento facial
    """
    if not MEDIAPIPE_AVAILABLE or not CV2_AVAILABLE:
        # Fallback: crop centralizado
        w, h = clip.size
        target_width = int(h * 9/16)
        x1 = (w - target_width) // 2
        return crop(clip, x1=x1, y1=0, width=target_width, height=h)
    
    w, h = clip.size
    target_ratio = 9/16
    target_width = int(h * target_ratio)
    
    # Inicializa MediaPipe
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.6
    )
    
    centers = []
    # Analisa 1 frame a cada 2 segundos
    for t in range(0, int(clip.duration), 2):
        frame = clip.get_frame(t)
        center = detect_face_center(frame, face_detection)
        if center:
            centers.append(center)
    
    face_detection.close()
    
    if centers:
        avg_center_x = int(sum(centers) / len(centers))
    else:
        avg_center_x = w // 2
    
    # Calcula coordenadas de corte
    x1 = max(0, avg_center_x - target_width // 2)
    x2 = min(w, x1 + target_width)
    
    if x2 == w:
        x1 = w - target_width
    
    return crop(clip, x1=x1, y1=0, width=target_width, height=h)

# ==================== LEGENDAS (WHISPER) ====================

def generate_subtitles(audio_path: str, model_size: str = "medium") -> List[Dict]:
    """
    Transcreve áudio usando Faster-Whisper
    """
    if not WHISPER_AVAILABLE:
        logger.warning("⚠️ Whisper não disponível, retornando legendas vazias")
        return []
    
    try:
        # Tenta GPU primeiro
        try:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            logger.info("✅ Whisper carregado na GPU")
        except Exception:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info("⚠️ Whisper carregado na CPU (fallback)")
        
        segments, info = model.transcribe(audio_path, beam_size=5, language="pt")
        
        subs = []
        for segment in segments:
            subs.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        logger.info(f"✅ {len(subs)} legendas geradas")
        return subs
        
    except Exception as e:
        logger.error(f"❌ Erro na transcrição: {e}")
        return []

# ==================== TÍTULOS VIRAIS (GEMINI) ====================

def clean_filename(filename: str) -> str:
    """Extrai nome limpo do arquivo"""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[._-]', ' ', name)
    name = re.sub(
        r'\b(scene|cena|corte|parte|cut|temp|mpy|wvf|snd|processed|video)\b',
        '',
        name,
        flags=re.IGNORECASE
    )
    name = re.sub(r'\d+', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def generate_viral_title(
    filename: str,
    scene_index: int,
    platform: str = "geral",
    clip_duration: int = 60
) -> str:
    """
    Gera título viral com Gemini API
    """
    if not GEMINI_AVAILABLE:
        # Fallback titles
        fallback_titles = [
            "VOCÊ NÃO VAI ACREDITAR!",
            "MOMENTO ÉPICO!",
            "OLHA O QUE ACONTECEU!",
            "CENA INCRÍVEL!",
            "VIRAL TOTAL!"
        ]
        return fallback_titles[scene_index % len(fallback_titles)]
    
    movie_name = clean_filename(filename)
    tempo_inicio = scene_index * (clip_duration / 60)  # em minutos
    tempo_fim = tempo_inicio + (clip_duration / 60)
    
    platform_prompts = {
        "tiktok": "Crie um título ULTRA VIRAL para TikTok (máximo 5 palavras, muito impactante)",
        "shorts": "Crie um título EXPLOSIVO para YouTube Shorts (máximo 5 palavras, clickbait forte)",
        "instagram": "Crie um título CHAMATIVO para Instagram Reels (máximo 6 palavras)",
        "geral": "Crie um título curto (máximo 6 palavras), viral e impactante"
    }
    
    platform_instruction = platform_prompts.get(platform.lower(), platform_prompts["geral"])
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""No vídeo "{movie_name}", o que acontece aproximadamente entre os minutos {tempo_inicio:.1f} e {tempo_fim:.1f}?

Com base nisso, {platform_instruction} em Português para essa cena.

Regras OBRIGATÓRIAS:
- Máximo 6 palavras
- Tudo em MAIÚSCULAS
- Extremamente impactante
- Em Português
- Apenas o título, sem explicações
- Deve terminar com ! ou ?

Título:"""
        
        response = model.generate_content(prompt)
        
        if response.text:
            title = response.text.strip()
            title = title.replace('"', '').replace('*', '').replace('`', '')
            title = title.upper()
            
            if not (title.endswith('!') or title.endswith('?')):
                title += '!'
            
            words = title.split()
            if len(words) > 6:
                title = ' '.join(words[:6])
                if not (title.endswith('!') or title.endswith('?')):
                    title += '!'
            
            logger.info(f"✅ Título gerado: {title}")
            return title
        
        return "MOMENTO INCRÍVEL!"
        
    except Exception as e:
        logger.error(f"❌ Erro ao gerar título: {e}")
        return "CENA ÉPICA!"

# ==================== PROCESSAMENTO DE VÍDEO ====================

def process_viral_clip(
    video_path: str,
    clip_start: float,
    clip_end: float,
    clip_index: int,
    config: Dict
) -> Optional[str]:
    """
    Processa um único clip viral
    """
    try:
        logger.info(f"🎬 Processando clip {clip_index + 1}...")
        
        # 1. Extração do subclipe
        full_clip = VideoFileClip(video_path)
        current_clip = full_clip.subclip(clip_start, min(clip_end, full_clip.duration))
        
        # 2. Smart Crop (rastreamento facial)
        logger.info("🤖 Aplicando Smart Crop...")
        cropped_clip = create_smart_crop(current_clip)
        
        # Redimensiona para 1080x1920
        final_clip = cropped_clip.resize(height=1920)
        if final_clip.w != 1080:
            final_clip = crop(final_clip, x1=final_clip.w//2 - 540, width=1080, height=1920)
        
        # 3. Transcrição e Legendas
        if config.get("enable_captions", True):
            logger.info("🎙️ Transcrevendo áudio...")
            
            temp_audio = TEMP_DIR / f"audio_{clip_index}.wav"
            current_clip.audio.write_audiofile(str(temp_audio), logger=None)
            
            subtitles = generate_subtitles(str(temp_audio))
            
            if temp_audio.exists():
                temp_audio.unlink()
        else:
            subtitles = []
        
        # 4. Cria clips de legenda
        subtitle_clips = []
        font_path = config.get("font", "Arial")
        font_color = config.get("caption_color", "white")
        
        for sub in subtitles:
            # Sombra
            shadow_clip = (TextClip(
                            sub['text'].upper(),
                            fontsize=70,
                            font=font_path if os.path.exists(font_path) else "Arial",
                            color='black',
                            method='caption',
                            align='center',
                            size=(900, None)
                        )
                        .set_position(('center', 1405))
                        .set_opacity(0.6)
                        .set_start(sub['start'])
                        .set_end(sub['end']))
            
            # Texto principal
            txt_clip = (TextClip(
                            sub['text'].upper(),
                            fontsize=70,
                            font=font_path if os.path.exists(font_path) else "Arial",
                            color=font_color,
                            method='caption',
                            align='center',
                            size=(900, None)
                        )
                        .set_position(('center', 1400))
                        .set_start(sub['start'])
                        .set_end(sub['end']))
            
            subtitle_clips.append(shadow_clip)
            subtitle_clips.append(txt_clip)
        
        # 5. Título Viral
        if config.get("enable_titles", True):
            logger.info("🔥 Gerando título viral...")
            
            viral_title = generate_viral_title(
                config.get("filename", "video.mp4"),
                clip_index,
                config.get("platform", "geral"),
                config.get("clip_duration", 60)
            )
            
            # Sombra do título
            title_shadow = (TextClip(
                            viral_title,
                            fontsize=60,
                            font=font_path if os.path.exists(font_path) else "Arial",
                            color='black',
                            method='caption',
                            align='center',
                            size=(1000, None)
                        )
                        .set_position(('center', 103))
                        .set_opacity(0.7)
                        .set_duration(final_clip.duration))
            
            # Título principal
            title_main = (TextClip(
                            viral_title,
                            fontsize=60,
                            font=font_path if os.path.exists(font_path) else "Arial",
                            color='#FFD700',
                            method='caption',
                            align='center',
                            size=(1000, None)
                        )
                        .set_position(('center', 100))
                        .set_duration(final_clip.duration))
            
            subtitle_clips.insert(0, title_shadow)
            subtitle_clips.insert(1, title_main)
        
        # 6. Composição final
        if subtitle_clips:
            final_video = CompositeVideoClip([final_clip] + subtitle_clips)
        else:
            final_video = final_clip
        
        # 7. Renderização
        output_filename = OUTPUT_DIR / f"viral_{clip_index + 1}_{int(time.time())}.mp4"
        logger.info(f"🎥 Renderizando vídeo...")
        
        try:
            # Tenta NVENC (GPU)
            final_video.write_videofile(
                str(output_filename),
                codec='h264_nvenc',
                audio_codec='aac',
                bitrate='6000k',
                preset='p6',
                threads=8,
                logger=None
            )
            logger.info("✅ Renderizado com NVENC (GPU)")
        except Exception as e:
            logger.warning(f"⚠️ NVENC falhou: {e}. Usando CPU...")
            final_video.write_videofile(
                str(output_filename),
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                threads=4,
                logger=None
            )
            logger.info("✅ Renderizado com libx264 (CPU)")
        
        # Limpa recursos
        final_video.close()
        current_clip.close()
        full_clip.close()
        
        return str(output_filename)
        
    except Exception as e:
        logger.error(f"❌ Erro ao processar clip {clip_index + 1}: {e}")
        return None

# ==================== HANDLER PRINCIPAL ====================

async def handler(job):
    """
    Handler principal do RunPod Serverless
    
    Input esperado:
    {
        "input": {
            "video_url": "https://example.com/video.mp4",
            "num_clips": 3,
            "clip_duration": 60,
            "start_offset": 0,
            "platform": "tiktok",
            "enable_captions": true,
            "enable_titles": true,
            "caption_color": "#FFD700",
            "font": "Arial"
        }
    }
    """
    
    try:
        input_data = job.get("input", {})
        
        # Parâmetros
        video_url = input_data.get("video_url")
        num_clips = int(input_data.get("num_clips", 3))
        clip_duration = int(input_data.get("clip_duration", 60))
        start_offset = int(input_data.get("start_offset", 0))
        platform = input_data.get("platform", "tiktok")
        enable_captions = input_data.get("enable_captions", True)
        enable_titles = input_data.get("enable_titles", True)
        caption_color = input_data.get("caption_color", "#FFD700")
        font = input_data.get("font", "Arial")
        
        logger.info("=" * 60)
        logger.info("💎 ViralPro Serverless v2.0 - Iniciando")
        logger.info(f"📺 URL: {video_url}")
        logger.info(f"🔢 Clips: {num_clips}")
        logger.info(f"⏱️ Duração: {clip_duration}s")
        logger.info("=" * 60)
        
        # Download do vídeo
        logger.info("📥 Baixando vídeo...")
        import requests
        
        video_filename = f"input_{int(time.time())}.mp4"
        video_path = TEMP_DIR / video_filename
        
        response = requests.get(video_url, stream=True)
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✅ Vídeo baixado: {video_path}")
        
        # Configuração
        config = {
            "filename": video_filename,
            "platform": platform,
            "clip_duration": clip_duration,
            "enable_captions": enable_captions,
            "enable_titles": enable_titles,
            "caption_color": caption_color,
            "font": font
        }
        
        # Processa clips
        generated_files = []
        uploaded_urls = []
        
        for i in range(num_clips):
            clip_start = start_offset + (i * clip_duration)
            clip_end = clip_start + clip_duration
            
            output_path = process_viral_clip(
                str(video_path),
                clip_start,
                clip_end,
                i,
                config
            )
            
            if output_path:
                generated_files.append(output_path)
                
                # Upload para B2
                remote_path = f"viralpro/{Path(output_path).name}"
                signed_url = upload_to_b2(Path(output_path), remote_path)
                
                if signed_url:
                    uploaded_urls.append({
                        "clip_index": i + 1,
                        "url": signed_url,
                        "b2_key": remote_path,
                        "start": clip_start,
                        "end": clip_end
                    })
                    
                    # Remove arquivo local
                    Path(output_path).unlink()
        
        # Remove vídeo de entrada
        if video_path.exists():
            video_path.unlink()
        
        # Limpa memória
        clean_memory()
        
        logger.info("=" * 60)
        logger.info("✅ ViralPro Serverless - Concluído")
        logger.info(f"📊 {len(uploaded_urls)} clips gerados")
        logger.info("=" * 60)
        
        return {
            "status": "success",
            "clips": uploaded_urls,
            "total_clips": len(uploaded_urls),
            "config": config
        }
        
    except Exception as e:
        logger.error(f"❌ Erro fatal no handler: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ==================== INICIALIZAÇÃO ====================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("💎 ViralPro Serverless v2.0")
    logger.info("=" * 60)
    logger.info(f"✅ CV2 Available: {CV2_AVAILABLE}")
    logger.info(f"✅ MoviePy Available: {MOVIEPY_AVAILABLE}")
    logger.info(f"✅ MediaPipe Available: {MEDIAPIPE_AVAILABLE}")
    logger.info(f"✅ Whisper Available: {WHISPER_AVAILABLE}")
    logger.info(f"✅ Gemini Available: {GEMINI_AVAILABLE}")
    logger.info(f"✅ B2 Available: {B2_AVAILABLE}")
    logger.info("=" * 60)
    logger.info("🚀 Iniciando RunPod Serverless Handler...")
    logger.info("=" * 60)
    
    runpod.serverless.start({"handler": handler})
