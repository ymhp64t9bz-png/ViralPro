# -*- coding: utf-8 -*-
"""
HYPERCLIP AI - GPU PROCESSING ENGINE
Pipeline de vídeo acelerado por GPU com Smart Crop, Legendas Whisper e Títulos IA.
"""

import os
import sys
import time
import math
import traceback

# Adiciona diretório raiz ao path para importar módulos irmãos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importações com tratamento de erro para dependências pesadas
try:
    import cv2
    import numpy as np
    import moviepy.editor as mp
    from moviepy.video.fx.all import crop
    import mediapipe as mp_face_detection
    from faster_whisper import WhisperModel
    
    # Importa serviço local de IA (Llama 3) para geração de títulos
    from core.ai_services.local_ai_service import generate_viral_title_local, load_llama_model, unload_llama_model
    
    LIBS_AVAILABLE = True
except ImportError as e:
    LIBS_AVAILABLE = False
    print(f"[ERRO] [GPU_PROCESSOR] Erro crítico de importação: {e}")

class ViralProcessor:
    def __init__(self, api_key=None, status_callback=None):
        self.api_key = api_key
        self.status_callback = status_callback or print
        self.output_dir = os.path.join("outputs", "viral_pro")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Inicializa modelos se as libs estiverem disponíveis
        if LIBS_AVAILABLE:
            self._init_models()

    def _log(self, message):
        """Envia logs para o callback (UI) e terminal."""
        formatted_msg = f"[GPU_ENGINE] {message}"
        self.status_callback(formatted_msg)

    def _init_models(self):
        """Inicializa MediaPipe e Whisper na GPU."""
        self._log("Inicializando modelos de IA na GPU...")
        
        # MediaPipe Face Detection
        self.mp_face = mp_face_detection.FaceDetection(
            model_selection=1, # 1 = Long range (melhor para filmes)
            min_detection_confidence=0.6
        )
        
        # Whisper (Tentativa de carregar na GPU)
        try:
            self._log("Carregando Whisper (CUDA)...")
            self.whisper = WhisperModel("medium", device="cuda", compute_type="float16")
        except Exception:
            self._log("[AVISO] CUDA não detectado. Fallback para CPU (Lento).")
            self.whisper = WhisperModel("medium", device="cpu", compute_type="int8")

    def detect_face_center(self, frame):
        """Detecta o centro do rosto principal no frame."""
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb_frame)

        if results.detections:
            # Pega o rosto com maior score
            detection = max(results.detections, key=lambda d: d.score[0])
            bboxC = detection.location_data.relative_bounding_box
            center_x = int((bboxC.xmin + bboxC.width / 2) * width)
            return center_x
        return None

    def smart_crop(self, clip):
        """Aplica corte 9:16 focado no rosto (Rastreamento Suave)."""
        w, h = clip.size
        target_ratio = 9/16
        target_width = int(h * target_ratio)
        
        centers = []
        # Amostragem inteligente (1 frame a cada 1.5s)
        for t in range(0, int(clip.duration), 2):
            try:
                frame = clip.get_frame(t)
                center = self.detect_face_center(frame)
                if center:
                    centers.append(center)
            except Exception:
                pass

        if centers:
            # Média ponderada para evitar 'jitter' da câmera
            avg_center_x = int(sum(centers) / len(centers))
        else:
            avg_center_x = w // 2

        # Calcula crop box
        x1 = max(0, avg_center_x - target_width // 2)
        x2 = min(w, x1 + target_width)
        
        # Correção de bordas
        if x2 > w:
            x1 = w - target_width
            x2 = w
        if x1 < 0:
            x1 = 0
            x2 = target_width

        return crop(clip, x1=x1, y1=0, width=target_width, height=h)

    def generate_subtitles(self, audio_path):
        """Transcreve áudio e gera estrutura de legendas."""
        segments, _ = self.whisper.transcribe(audio_path, language="pt")
        return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]

    def process_video(self, video_path, num_clips=1, clip_duration=60, start_min=0):
        """Pipeline principal de processamento."""
        if not LIBS_AVAILABLE:
            self._log("[ERRO] Erro: Bibliotecas não instaladas.")
            return []

        generated_files = []
        
        try:
            full_clip = mp.VideoFileClip(video_path)
            video_duration = full_clip.duration
            start_sec = start_min * 60
            
            self._log(f"Iniciando processamento: {os.path.basename(video_path)}")
            
            for i in range(num_clips):
                clip_start = start_sec + (i * clip_duration)
                if clip_start >= video_duration:
                    break
                
                clip_end = min(clip_start + clip_duration, video_duration)
                
                self._log(f"--- Processando Clipe {i+1}/{num_clips} ({int(clip_start)}s - {int(clip_end)}s) ---")
                
                # 1. Extração
                current_clip = full_clip.subclip(clip_start, clip_end)
                
                # 2. Smart Crop
                self._log("[IA] Aplicando Smart Crop (Face Tracking)...")
                cropped_clip = self.smart_crop(current_clip)
                final_clip = cropped_clip.resize(height=1920) # Resize para HD Vertical
                
                # Garante largura 1080 (se resize não for exato)
                if final_clip.w != 1080:
                    final_clip = crop(final_clip, x1=final_clip.w//2 - 540, width=1080, height=1920)

                # 3. Áudio e Legendas
                self._log("[EMOJI]️ Transcrevendo Áudio (Whisper GPU)...")
                temp_audio = f"temp_audio_{i}.wav"
                current_clip.audio.write_audiofile(temp_audio, logger=None)
                
                subtitles = self.generate_subtitles(temp_audio)
                if os.path.exists(temp_audio): os.remove(temp_audio)

                # 4. Geração de Título Viral (Gemini)
                self._log("✨ Gerando Título Viral (Gemini AI)...")
                viral_title = generate_viral_title(self.api_key, video_path, i)
                self._log(f"Título Gerado: '{viral_title}'")

                # 5. Overlay Gráfico (Legendas + Título)
                self._log("[DESIGN] Renderizando Elementos Gráficos...")
                
                text_clips = []
                
                # Título no Topo
                title_clip = (mp.TextClip(
                    viral_title, 
                    fontsize=70, color='#FFD700', font='Arial-Bold', 
                    method='caption', size=(900, None), align='center'
                ).set_position(('center', 150)).set_duration(final_clip.duration))
                text_clips.append(title_clip)
                
                # Legendas
                for sub in subtitles:
                    # Sombra
                    shadow = (mp.TextClip(
                        sub['text'].upper(), fontsize=55, color='black', font='Arial-Bold',
                        method='caption', size=(900, None), align='center'
                    ).set_position(('center', 1405)).set_start(sub['start']).set_end(sub['end']))
                    
                    # Texto
                    txt = (mp.TextClip(
                        sub['text'].upper(), fontsize=55, color='white', font='Arial-Bold',
                        method='caption', size=(900, None), align='center'
                    ).set_position(('center', 1400)).set_start(sub['start']).set_end(sub['end']))
                    
                    text_clips.append(shadow)
                    text_clips.append(txt)

                # Composição Final
                final_video = mp.CompositeVideoClip([final_clip] + text_clips)

                # 6. Renderização (NVENC)
                safe_title = "".join([c for c in viral_title if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
                output_filename = os.path.join(self.output_dir, f"{safe_title}.mp4")
                
                self._log(f"[INICIO] Renderizando Final (NVENC): {output_filename}")
                
                # Tenta usar NVENC, fallback para libx264
                try:
                    final_video.write_videofile(
                        output_filename,
                        codec='h264_nvenc',
                        audio_codec='aac',
                        bitrate='6000k',
                        preset='p4',
                        threads=8,
                        logger=None
                    )
                except Exception:
                    self._log("[AVISO] NVENC falhou. Usando CPU (libx264)...")
                    final_video.write_videofile(
                        output_filename,
                        codec='libx264',
                        audio_codec='aac',
                        preset='fast',
                        threads=4,
                        logger=None
                    )
                
                generated_files.append(output_filename)
                final_video.close()
                current_clip.close()

            full_clip.close()
            self._log("[OK] Processamento Concluído!")
            return generated_files

        except Exception as e:
            self._log(f"[ERRO] Erro Fatal no Pipeline: {traceback.format_exc()}")
            return []

# Teste direto
if __name__ == "__main__":
    print("Módulo GPU Processor carregado.")
