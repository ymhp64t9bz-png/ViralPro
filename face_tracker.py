import cv2
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip
import os

class FaceTracker:
    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection, 
            min_detection_confidence=min_detection_confidence
        )
        self.smoothing_factor = 0.1

    def _get_face_center(self, frame):
        try:
            height, width, _ = frame.shape
            results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not results.detections:
                return None

            best_detection = max(results.detections, key=lambda d: d.score[0])
            bboxC = best_detection.location_data.relative_bounding_box
            center_x = int((bboxC.xmin + bboxC.width / 2) * width)
            return center_x
        except Exception as e:
            print(f"Erro na detecção facial: {e}")
            return None

    def process_video(self, input_path):
        print(f"🎥 Iniciando Smart Crop em: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Não foi possível abrir o vídeo: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_height = height
        target_width = int(target_height * (9/16))
        if width < target_width: target_width = width

        centers = []
        
        # Análise rápida (pula frames para velocidade)
        step = 2 
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            if not success: break
            
            center = self._get_face_center(frame)
            if center:
                centers.append(center)
            else:
                # Fallback: Centro da tela ou último conhecido
                centers.append(centers[-1] if centers else width // 2)
            
            # Repete o valor para os frames pulados
            for _ in range(step - 1):
                centers.append(centers[-1])

        cap.release()
        
        # Preenche frames faltantes no final se houver
        while len(centers) < total_frames:
            centers.append(centers[-1] if centers else width // 2)

        # Suavização
        smoothed_centers = []
        current_center = centers[0] if centers else width // 2
        for center in centers:
            current_center = (current_center * (1 - self.smoothing_factor)) + (center * self.smoothing_factor)
            smoothed_centers.append(int(current_center))

        # Renderização
        def crop_getter(t):
            frame_idx = int(t * fps)
            if frame_idx >= len(smoothed_centers): frame_idx = len(smoothed_centers) - 1
            center_x = smoothed_centers[frame_idx]
            
            x1 = center_x - (target_width // 2)
            if x1 < 0: x1 = 0
            if x1 + target_width > width: x1 = width - target_width
            
            return x1, 0, x1 + target_width, target_height

        clip = VideoFileClip(input_path)
        cropped_clip = clip.fl(lambda gf, t: gf(t)[
            int(crop_getter(t)[1]):int(crop_getter(t)[3]),
            int(crop_getter(t)[0]):int(crop_getter(t)[2])
        ], apply_to=['mask'])

        cropped_clip = cropped_clip.resize(width=target_width, height=target_height)
        return cropped_clip
