import os
import time
from faster_whisper import WhisperModel
from moviepy.editor import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class CaptionEngine:
    def __init__(self, model_size="medium", device="cuda"):
        print(f"🧠 Carregando modelo Whisper ({model_size}) em {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type="float16" if device=="cuda" else "int8")
        except:
            print("⚠️ GPU falhou, usando CPU...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(audio_path, word_timestamps=True, language="pt")
        final_segments = []
        for segment in segments:
            for word in segment.words:
                final_segments.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })
        return final_segments

    def create_text_image(self, text, font_path, font_size, color, stroke_color, highlight=False):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            
        dummy_img = Image.new('RGBA', (10, 10), (0,0,0,0))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        w = (bbox[2] - bbox[0]) + 40
        h = (bbox[3] - bbox[1]) + 40
        
        img = Image.new('RGBA', (int(w), int(h)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        text_color = (255, 255, 0, 255) if highlight else color
        
        # Stroke
        for adj_x in range(-2, 3):
            for adj_y in range(-2, 3):
                draw.text((20+adj_x, 20+adj_y), text, font=font, fill=stroke_color)
        
        draw.text((20, 20), text, font=font, fill=text_color)
        return np.array(img)

    def generate_viral_video(self, video_clip, output_path, font_style="Arial", position_y="center", highlight_color=True):
        temp_audio = f"temp_audio_{int(time.time())}.wav"
        video_clip.audio.write_audiofile(temp_audio, verbose=False, logger=None)
        
        try:
            words_data = self.transcribe(temp_audio)
        except Exception as e:
            print(f"Erro na transcrição: {e}")
            words_data = [] # Continua sem legenda se falhar
        
        if os.path.exists(temp_audio): os.remove(temp_audio)
        
        text_clips = []
        font_size = 70
        fonts = {"Arial": "arial.ttf", "Impact": "impact.ttf", "Roboto": "arialbd.ttf"}
        selected_font = fonts.get(font_style, "arial.ttf")
        
        for word_info in words_data:
            word_text = word_info['word']
            start_t = word_info['start']
            end_t = word_info['end']
            duration = end_t - start_t
            if duration < 0.1: duration = 0.1
            
            img_array = self.create_text_image(
                word_text.upper(), selected_font, font_size, 
                (255, 255, 255, 255), (0, 0, 0, 255), highlight=highlight_color
            )
            
            txt_clip = ImageClip(img_array).set_start(start_t).set_duration(duration)
            
            w, h = video_clip.size
            if position_y == "top": pos = ('center', int(h * 0.2))
            elif position_y == "bottom": pos = ('center', int(h * 0.8))
            else: pos = ('center', 'center')
                
            txt_clip = txt_clip.set_position(pos)
            text_clips.append(txt_clip)

        final = CompositeVideoClip([video_clip] + text_clips)
        final.write_videofile(output_path, codec='libx264', audio_codec='aac', fps=video_clip.fps, preset='fast', logger=None)
        return output_path
