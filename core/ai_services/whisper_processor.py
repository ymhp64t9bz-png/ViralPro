# -*- coding: utf-8 -*-
"""
WHISPER PROCESSOR - Transcrição Única com Timestamps
====================================================
Transcreve vídeo UMA VEZ com timestamps precisos por palavra
"""

import os
import logging
import gc
from moviepy.editor import VideoFileClip

logger = logging.getLogger(__name__)

# Importa Whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("[WHISPER] Biblioteca não disponível")

# Instância global
_WHISPER_INSTANCE = None

def load_whisper_model(model_size="medium"):
    """Carrega modelo Whisper uma vez"""
    global _WHISPER_INSTANCE
    
    if _WHISPER_INSTANCE is None:
        if not WHISPER_AVAILABLE:
            logger.error("[WHISPER] Biblioteca não instalada")
            return None
        
        logger.info(f"[WHISPER] Carregando modelo {model_size}...")
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _WHISPER_INSTANCE = whisper.load_model(model_size, device=device)
            logger.info(f"[WHISPER] Modelo carregado ({device})")
        except Exception as e:
            logger.error(f"[WHISPER] Erro ao carregar: {e}")
            return None
    
    return _WHISPER_INSTANCE

def extract_audio(video_path, output_path=None):
    """Extrai áudio do vídeo"""
    if output_path is None:
        output_path = video_path.replace('.mp4', '_audio.wav')
    
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_path, logger=None)
        video.close()
        return output_path
    except Exception as e:
        logger.error(f"[WHISPER] Erro ao extrair áudio: {e}")
        return None

def transcribe_with_timestamps(video_path):
    """
    Transcreve vídeo UMA VEZ com timestamps precisos
    
    Args:
        video_path: Caminho do vídeo
    
    Returns:
        {
            'text': 'Transcrição completa...',
            'segments': [
                {
                    'start': 0.0,
                    'end': 5.2,
                    'text': 'Bem-vindo ao quartel',
                    'words': [
                        {'word': 'Bem-vindo', 'start': 0.0, 'end': 0.8},
                        {'word': 'ao', 'start': 0.9, 'end': 1.0},
                        {'word': 'quartel', 'start': 1.1, 'end': 1.5}
                    ]
                },
                ...
            ],
            'duration': 1440.5
        }
    """
    logger.info("[WHISPER] Iniciando transcrição única com timestamps...")
    
    # Carrega modelo
    model = load_whisper_model()
    if model is None:
        return None
    
    # Extrai áudio
    logger.info("[WHISPER] Extraindo áudio...")
    audio_path = extract_audio(video_path)
    if audio_path is None:
        return None
    
    try:
        # Transcreve com word timestamps
        logger.info("[WHISPER] Transcrevendo com timestamps por palavra...")
        result = model.transcribe(
            audio_path,
            language='pt',
            word_timestamps=True,  # ✅ CRÍTICO - timestamps por palavra
            verbose=False,
            fp16=True if model.device.type == 'cuda' else False
        )
        
        # Obtém duração do vídeo
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()
        
        # Adiciona duração ao resultado
        result['duration'] = duration
        
        logger.info(f"[WHISPER] ✅ Transcrição concluída!")
        logger.info(f"[WHISPER] Texto: {len(result['text'])} caracteres")
        logger.info(f"[WHISPER] Segmentos: {len(result['segments'])}")
        logger.info(f"[WHISPER] Duração: {duration:.1f}s")
        
        # Remove arquivo de áudio temporário
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return result
        
    except Exception as e:
        logger.error(f"[WHISPER] Erro na transcrição: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_text_between(segments, start_time, end_time):
    """
    Extrai texto entre timestamps específicos
    
    Args:
        segments: Lista de segmentos do Whisper
        start_time: Tempo inicial (segundos)
        end_time: Tempo final (segundos)
    
    Returns:
        str: Texto concatenado do período
    """
    text_parts = []
    
    for segment in segments:
        # Segmento está dentro do período
        if segment['start'] >= start_time and segment['end'] <= end_time:
            text_parts.append(segment['text'])
        # Segmento começa antes mas termina dentro
        elif segment['start'] < start_time and segment['end'] > start_time:
            text_parts.append(segment['text'])
        # Segmento começa dentro mas termina depois
        elif segment['start'] < end_time and segment['end'] > end_time:
            text_parts.append(segment['text'])
    
    return ' '.join(text_parts).strip()

def unload_whisper_model():
    """Descarrega modelo Whisper da memória"""
    global _WHISPER_INSTANCE
    
    if _WHISPER_INSTANCE is not None:
        logger.info("[WHISPER] Descarregando modelo...")
        del _WHISPER_INSTANCE
        _WHISPER_INSTANCE = None
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        logger.info("[WHISPER] ✅ Modelo descarregado")
