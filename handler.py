"""
🔥 StoryForge AI - Handler de Produção (RunPod)
Pipeline: Topic -> Math Script -> Edge TTS -> MoviePy Video
"""

import runpod
import os
import asyncio
import logging
import time
import random

# Imports de Mídia
import edge_tts
from moviepy.editor import (
    AudioFileClip, TextClip, ColorClip, CompositeVideoClip
)
from moviepy.config import change_settings

# Configurações Críticas para Docker
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StoryForge-Cloud")

# Diretórios
OUTPUT_DIR = "/app/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------- #
#                           LÓGICA DE PRODUÇÃO                                 #
# ---------------------------------------------------------------------------- #

def generate_script_logic(topic, duration_seconds):
    """Gera roteiro respeitando a duração (2.5 palavras/segundo)."""
    target_words = int(duration_seconds * 2.5)
    logger.info(f"📝 Topic: {topic} | Duração: {duration_seconds}s | Alvo: {target_words} palavras")
    
    # Simulação de LLM (Substitua por chamada real se tiver API Key no futuro)
    intro = f"Hoje vamos falar sobre {topic}. "
    fillers = [
        "Isso é algo que muda tudo.", "A ciência por trás disso é fascinante.",
        f"Muitos não sabem a verdade sobre {topic}.", "Imagine as possibilidades.",
        "Os detalhes são surpreendentes.", "Isso impacta nossa vida diariamente."
    ]
    
    text = intro
    while len(text.split()) < target_words:
        text += random.choice(fillers) + " "
        
    # Ajuste fino
    words = text.split()[:target_words]
    final_text = " ".join(words)
    return final_text

async def generate_voice_async(text, voice):
    """Gera arquivo de áudio MP3 usando Edge TTS."""
    filename = f"audio_{int(time.time())}_{random.randint(1000,9999)}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filepath)
    
    logger.info(f"🎙️ Áudio salvo: {filepath}")
    return filepath

def generate_video_render(audio_path, topic):
    """Renderiza MP4 com fundo e legendas."""
    logger.info("🎬 Renderizando vídeo...")
    filename = f"video_{int(time.time())}_{random.randint(1000,9999)}.mp4"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # 1. Carrega Áudio
    audio = AudioFileClip(audio_path)
    dur = audio.duration
    
    # 2. Cria Fundo (Azul Profundo)
    bg = ColorClip(size=(1080, 1920), color=(10, 20, 40), duration=dur)
    
    # 3. Cria Texto (Título)
    try:
        font_name = 'DejaVu-Sans-Bold'
    except:
        font_name = 'Arial'
        
    txt = TextClip(
        topic.upper(), 
        fontsize=80, 
        color='white', 
        font=font_name, 
        size=(900, None), 
        method='caption'
    ).set_position('center').set_duration(dur)
    
    # 4. Compõe
    final = CompositeVideoClip([bg, txt]).set_audio(audio)
    
    # 5. Exporta
    final.write_videofile(
        output_path, 
        fps=24, 
        codec='libx264', 
        audio_codec='aac',
        preset='ultrafast',
        threads=4,
        logger=None
    )
    
    return output_path

# ---------------------------------------------------------------------------- #
#                                RUNPOD HANDLER                                #
# ---------------------------------------------------------------------------- #

async def handler(job):
    """
    Handler ASSÍNCRONO nativo para evitar erros de loop.
    RunPod suporta 'async def handler(job)'.
    """
    job_input = job.get("input", {})
    
    topic = job_input.get("topic", "Tecnologia")
    duration = int(job_input.get("duration", 30))
    voice = job_input.get("voice", "pt-BR-AntonioNeural")
    
    try:
        logger.info(f"🚀 Job Start: {topic}")
        
        # 1. Roteiro (CPU Bound)
        script = generate_script_logic(topic, duration)
        
        # 2. Voz (IO Bound - Async await direto)
        # Não usamos asyncio.run() aqui, usamos await pois já estamos em loop
        audio_path = await generate_voice_async(script, voice)
        
        # 3. Vídeo (CPU Bound)
        video_path = generate_video_render(audio_path, topic)
        
        return {
            "status": "success",
            "video_path": video_path,
            "script_length": len(script.split()),
            "duration": duration
        }
        
    except Exception as e:
        logger.error(f"❌ Erro Fatal: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
