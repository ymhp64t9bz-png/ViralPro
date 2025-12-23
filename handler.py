#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViralPRO Serverless v1.0.0
BUILD: 2025-12-23 - SHORTS/REELS GENERATOR
Stack: Whisper V3 Turbo, Face Detection, Voice Activity Detection, FFmpeg NVENC

FUNCIONALIDADES:
- Redimensionamento inteligente 16:9 → 9:16
- Rastreamento facial com detecção de voz ativa
- Legendas automáticas sincronizadas
- Análise de cenas com IA
- Anti-ShadowBan
- Suporte a qualquer tipo de vídeo (podcasts, filmes, séries, etc.)
"""

# ==================== IMPORTAÇÕES ESSENCIAIS ====================
import os
import sys
import logging
import time
import hashlib
import tempfile
import requests
import gc
import json
import uuid
import math
import subprocess
import shutil
import random
import threading
import contextlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from functools import wraps
from datetime import datetime
import html
from collections import deque

# ==================== CONFIGURAÇÃO DO VOLUME ====================
VOLUME_BASE = "/workspace"
VOLUME_PATH = Path(VOLUME_BASE)

# Diretórios dentro do volume
TEMP_DIR = Path("/tmp/viralpro")
OUTPUT_DIR = VOLUME_PATH / "output"
MODELS_DIR = VOLUME_PATH / "models"
FONTS_DIR = VOLUME_PATH / "fonts"
CACHE_DIR = VOLUME_PATH / "cache"

# Caminhos específicos de modelos
FONT_PATH = FONTS_DIR / "impact.ttf"

# Garante que todos os diretórios existam
for directory in [TEMP_DIR, OUTPUT_DIR, MODELS_DIR, FONTS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configuração de logging aprimorada
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(VOLUME_PATH / "viralpro.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ViralPRO")

# ==================== CONFIGURAÇÃO GPU ====================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# ==================== DECORADORES DE SEGURANÇA ====================

def safe_gpu_operation(func):
    """Decorator para operações GPU com limpeza automática"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            gc.collect()
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            result = func(*args, **kwargs)
            
            gc.collect()
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"[GPU ERROR] {func.__name__}: {e}")
            gc.collect()
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
    return wrapper

def retry_on_failure(max_attempts=3, delay=2, backoff=2):
    """Decorator para retry com backoff exponencial"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        logger.error(f"[RETRY] {func.__name__} falhou após {max_attempts} tentativas")
                        raise
                    
                    logger.warning(f"[RETRY] {func.__name__} tentativa {attempts}/{max_attempts}: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator

# ==================== GERENCIADOR DE RECURSOS ====================

class ResourceManager:
    """Gerencia recursos com cleanup automático"""
    
    def __init__(self):
        self.resources = []
        self.lock = threading.Lock()
    
    def register(self, resource, cleanup_func):
        with self.lock:
            self.resources.append((resource, cleanup_func))
    
    def cleanup_all(self):
        with self.lock:
            for resource, cleanup_func in reversed(self.resources):
                try:
                    cleanup_func(resource)
                except Exception as e:
                    logger.warning(f"[CLEANUP] Erro ao limpar recurso: {e}")
            self.resources.clear()

resource_manager = ResourceManager()

# ==================== FUNÇÕES DE SANITIZAÇÃO ====================

def sanitize_input(value: Optional[str], max_len: int = 240, escape_html: bool = True) -> str:
    """Sanitiza entradas de texto"""
    if not value:
        return ""
    value = ''.join(ch for ch in str(value) if ch.isprintable())
    value = value.strip()
    if escape_html:
        value = html.escape(value)
    if len(value) > max_len:
        value = value[:max_len]
    return value

def is_safe_path(base_dir: Path, user_path: str) -> bool:
    try:
        candidate = (Path(user_path)).resolve()
        base = base_dir.resolve()
        allowed_bases = [base, TEMP_DIR.resolve(), CACHE_DIR.resolve()]
        return any(str(candidate).startswith(str(ab) + os.sep) or candidate == ab for ab in allowed_bases)
    except Exception:
        return False

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converte cor hex para RGB"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ==================== IMPORTS COM FALLBACK ====================

class DependencyManager:
    """Gerencia imports de forma robusta"""
    
    def __init__(self):
        self.available_modules = {}
        self.module_errors = {}
        self.module_versions = {}
        
    def safe_import(self, module_name, import_path=None, fallback_names=None, min_version=None):
        start_time = time.time()
        
        names_to_try = [module_name]
        if fallback_names:
            names_to_try.extend(fallback_names)
        
        for name in names_to_try:
            try:
                if import_path:
                    module = __import__(import_path, fromlist=[name])
                else:
                    module = __import__(name)
                
                if min_version and hasattr(module, '__version__'):
                    version = module.__version__
                    self.module_versions[module_name] = version
                
                elapsed = time.time() - start_time
                self.available_modules[module_name] = module
                logger.info(f"[SUCCESS] {module_name} carregado ({elapsed:.2f}s)")
                return module
                
            except ImportError as e:
                self.module_errors[name] = str(e)
                continue
            except Exception as e:
                self.module_errors[name] = str(e)
                continue
        
        logger.warning(f"[WARNING] {module_name} não disponível")
        self.available_modules[module_name] = None
        return None

dep_manager = DependencyManager()

# ==================== CARREGAMENTO DE DEPENDÊNCIAS ====================

# 1. NumPy
np = dep_manager.safe_import("numpy")

# 2. OpenCV
CV2_AVAILABLE = False
try:
    cv2 = dep_manager.safe_import("cv2")
    if cv2 and np:
        CV2_AVAILABLE = True
        logger.info("[SUCCESS] OpenCV disponível")
except Exception as e:
    logger.debug(f"[DEBUG] OpenCV não disponível: {e}")

# 3. MediaPipe para Face Detection
MEDIAPIPE_AVAILABLE = False
mp = None
mp_face_detection = None
mp_face_mesh = None
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
    logger.info("[SUCCESS] MediaPipe disponível")
except ImportError:
    logger.info("[INFO] MediaPipe não disponível, usando fallback OpenCV")

# 4. MoviePy
MOVIEPY_AVAILABLE = False
moviepy_version = "N/A"
moviepy_imports = {}

try:
    moviepy = dep_manager.safe_import("moviepy")
    if moviepy:
        moviepy_version = getattr(moviepy, '__version__', 'N/A')
        
        from moviepy.editor import (
            VideoFileClip, ImageClip, CompositeVideoClip,
            ColorClip, TextClip, AudioFileClip, concatenate_videoclips
        )
        from moviepy.video.fx.all import mirror_x, gamma_corr, colorx, resize
        
        moviepy_imports = {
            'VideoFileClip': VideoFileClip,
            'ImageClip': ImageClip,
            'CompositeVideoClip': CompositeVideoClip,
            'ColorClip': ColorClip,
            'TextClip': TextClip,
            'AudioFileClip': AudioFileClip,
            'concatenate_videoclips': concatenate_videoclips,
            'mirror_x': mirror_x,
            'gamma_corr': gamma_corr,
            'colorx': colorx,
            'resize': resize
        }
        
        MOVIEPY_AVAILABLE = True
        logger.info("[SUCCESS] MoviePy configurado")
except Exception as e:
    logger.error(f"[ERROR] Erro no MoviePy: {e}")

# 5. PyTorch e IA
AI_AVAILABLE = False
GPU_AVAILABLE = False
WHISPER_AVAILABLE = False
WHISPER_TYPE = None
TORCH_DEVICE = None
TORCH_VERSION = None

torch = dep_manager.safe_import("torch")

if torch:
    try:
        TORCH_VERSION = torch.__version__
        GPU_AVAILABLE = torch.cuda.is_available()
        
        if GPU_AVAILABLE:
            TORCH_DEVICE = torch.device("cuda:0")
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"[SUCCESS] GPU: {gpu_name}")
        else:
            TORCH_DEVICE = torch.device("cpu")
            logger.info("[INFO] Executando em CPU")
        
        # Whisper
        try:
            from faster_whisper import WhisperModel
            WHISPER_AVAILABLE = True
            WHISPER_TYPE = "faster_whisper"
            logger.info("[SUCCESS] faster-whisper disponível")
        except ImportError:
            try:
                import whisper
                WHISPER_AVAILABLE = True
                WHISPER_TYPE = "openai_whisper"
                logger.info("[SUCCESS] openai-whisper disponível")
            except ImportError:
                logger.info("[INFO] Whisper não disponível")
        
        AI_AVAILABLE = True
        
    except Exception as e:
        logger.warning(f"[WARNING] Erro na configuração de IA: {e}")

# 6. Pillow
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter
    PIL_AVAILABLE = True
    logger.info("[SUCCESS] Pillow disponível")
except Exception as e:
    logger.debug(f"[DEBUG] Pillow não disponível: {e}")

# 7. DeepFilterNet
DF_AVAILABLE = False
DF_TYPE = None
DF_COMMAND = None

try:
    for cmd_name in ["deepFilter", "df", "deepfilternet"]:
        cmd_path = shutil.which(cmd_name)
        if cmd_path:
            result = subprocess.run([cmd_path, "--help"], capture_output=True, timeout=5)
            if result.returncode == 0:
                DF_AVAILABLE = True
                DF_TYPE = "cli"
                DF_COMMAND = cmd_path
                logger.info(f"[SUCCESS] DeepFilterNet CLI: {cmd_name}")
                break
except Exception:
    pass

# 8. Backblaze B2
B2_AVAILABLE = False
s3_client = None
B2_BUCKET = None

try:
    boto3 = dep_manager.safe_import("boto3")
    if boto3:
        from botocore.client import Config
        
        B2_KEY_ID = os.environ.get("B2_KEY_ID", "")
        B2_APP_KEY = os.environ.get("B2_APPLICATION_KEY", "")
        B2_ENDPOINT = os.environ.get("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
        B2_BUCKET = os.environ.get("B2_BUCKET_NAME", "ViralPRO")
        
        if B2_KEY_ID and B2_APP_KEY and B2_BUCKET:
            s3_client = boto3.client(
                "s3",
                endpoint_url=B2_ENDPOINT,
                aws_access_key_id=B2_KEY_ID,
                aws_secret_access_key=B2_APP_KEY,
                config=Config(signature_version="s3v4", connect_timeout=10, read_timeout=30)
            )
            B2_AVAILABLE = True
            logger.info(f"[SUCCESS] Backblaze B2: {B2_BUCKET}")
except Exception as e:
    logger.debug(f"[DEBUG] B2 não configurado: {e}")

# 9. FFmpeg
FFMPEG_AVAILABLE = False
FFMPEG_VERSION = None

try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        FFMPEG_AVAILABLE = True
        lines = result.stdout.split('\n')
        if lines:
            FFMPEG_VERSION = lines[0].split(' ')[2] if len(lines[0].split(' ')) > 2 else "unknown"
        logger.info(f"[SUCCESS] FFmpeg: {FFMPEG_VERSION}")
except:
    logger.error("[ERROR] FFmpeg não disponível!")

# ==================== GERENCIADOR DE REDE ====================

class NetworkManager:
    """Gerencia operações de rede"""
    
    def __init__(self, max_retries=3, timeout=60):
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = None
        self.lock = threading.Lock()
        
    def get_session(self):
        with self.lock:
            if self.session is None:
                self.session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=100, max_retries=3)
                self.session.mount('http://', adapter)
                self.session.mount('https://', adapter)
            return self.session
    
    def validate_url(self, url):
        if not url or not isinstance(url, str):
            return False
        url_lower = url.lower()
        return url_lower.startswith('http://') or url_lower.startswith('https://')
    
    @retry_on_failure(max_attempts=3, delay=2)
    def download_with_retry(self, url, output_path, headers=None, chunk_size=8192):
        if not self.validate_url(url):
            raise ValueError(f"URL inválida: {url}")
        
        session = self.get_session()
        logger.info(f"[DOWNLOAD] Iniciando: {url[:80]}...")
        
        response = session.get(url, stream=True, timeout=self.timeout, headers=headers)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        elapsed = time.time() - start_time
        file_size = os.path.getsize(output_path) / 1e6
        logger.info(f"[SUCCESS] Download: {output_path.name} ({file_size:.1f} MB)")
        
        return True

network = NetworkManager()

# ==================== CONFIGURAÇÃO DE FONTES ====================

def setup_fonts():
    """Configura fontes para legendas"""
    
    font_sources = [
        FONTS_DIR / "impact.ttf",
        FONTS_DIR / "Roboto-Bold.ttf",
        FONTS_DIR / "Arial-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    
    for font_path in font_sources:
        p = Path(font_path)
        if p.exists():
            logger.info(f"[SUCCESS] Fonte: {p}")
            return str(p)
    
    logger.warning("[WARNING] Nenhuma fonte encontrada")
    return None

FONT_TO_USE = setup_fonts()

# ==================== VALIDAÇÃO DE ÁUDIO ====================

def validate_audio_file(audio_path: Path) -> bool:
    """Valida arquivo de áudio"""
    if not FFMPEG_AVAILABLE:
        return audio_path.exists() and audio_path.stat().st_size > 0
    
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type,duration',
            '-of', 'json',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            return len(streams) > 0
        return False
    except:
        return audio_path.exists() and audio_path.stat().st_size > 0

# ==================== DEEPFILTER PARA LIMPEZA DE ÁUDIO ====================

def clean_audio_deepfilter(audio_path: Path) -> Path:
    """Limpa áudio usando DeepFilterNet"""
    
    if not DF_AVAILABLE:
        logger.info("[AUDIO] DeepFilter não disponível, usando áudio original")
        return audio_path
    
    try:
        output_path = TEMP_DIR / f"clean_{audio_path.stem}.wav"
        
        cmd = [DF_COMMAND, str(audio_path), "-o", str(output_path.parent)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and output_path.exists():
            logger.info("[AUDIO] Áudio limpo com DeepFilter")
            return output_path
        
        return audio_path
    except Exception as e:
        logger.warning(f"[AUDIO] Erro no DeepFilter: {e}")
        return audio_path

# ==================== WHISPER MANAGER ====================

class WhisperManager:
    """Gerenciador thread-safe do Whisper"""
    
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self._type = None
    
    def loaded(self):
        return self._model is not None
    
    def load(self):
        with self._lock:
            if self._model is not None:
                return True
            
            if not WHISPER_AVAILABLE:
                return False
            
            try:
                if WHISPER_TYPE == "faster_whisper":
                    from faster_whisper import WhisperModel
                    
                    compute_type = "float16" if GPU_AVAILABLE else "int8"
                    device = "cuda" if GPU_AVAILABLE else "cpu"
                    
                    self._model = WhisperModel(
                        "large-v3-turbo",
                        device=device,
                        compute_type=compute_type,
                        download_root=str(MODELS_DIR)
                    )
                    self._type = "faster_whisper"
                    logger.info("[WHISPER] Modelo Turbo carregado")
                    
                else:
                    import whisper
                    self._model = whisper.load_model("base", device=str(TORCH_DEVICE))
                    self._type = "openai_whisper"
                    logger.info("[WHISPER] Modelo OpenAI carregado")
                
                return True
                
            except Exception as e:
                logger.error(f"[WHISPER] Erro ao carregar: {e}")
                return False
    
    def transcribe(self, audio_path: str) -> Dict:
        """Transcreve áudio com timestamps palavra por palavra"""
        
        if not self.loaded():
            if not self.load():
                raise RuntimeError("Whisper não disponível")
        
        try:
            if self._type == "faster_whisper":
                segments, info = self._model.transcribe(
                    audio_path,
                    language="pt",
                    word_timestamps=True,  # IMPORTANTE: timestamps por palavra
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                words = []
                chunks = []
                
                for segment in segments:
                    text = getattr(segment, 'text', '').strip()
                    
                    # Processa palavras individuais
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            words.append({
                                "word": word.word.strip(),
                                "start": word.start,
                                "end": word.end,
                                "confidence": getattr(word, 'probability', 0.9)
                            })
                    
                    if text:
                        chunks.append({
                            "text": text,
                            "timestamp": (getattr(segment, 'start', 0), getattr(segment, 'end', 0)),
                            "confidence": float(getattr(segment, 'avg_logprob', 0.0))
                        })
                
                return {"chunks": chunks, "words": words, "info": info}
            
            else:
                result = self._model.transcribe(str(audio_path), language="pt")
                return {"chunks": [{"text": result.get('text', ''), "timestamp": (0, 0)}], "words": [], "info": result}
        
        except Exception as e:
            logger.error(f"[WHISPER] Erro na transcrição: {e}")
            raise

whisper_manager = WhisperManager()

# ==================== FACE TRACKER - RASTREAMENTO FACIAL ====================

class FaceTracker:
    """
    Rastreador facial com suavização e detecção de speaker ativo.
    Acompanha o rosto de quem está falando para fazer o crop dinâmico.
    """
    
    def __init__(self, video_path: str, target_width: int = 1080, target_height: int = 1920):
        self.video_path = video_path
        self.target_width = target_width
        self.target_height = target_height
        self.target_aspect = target_width / target_height  # 0.5625 para 9:16
        
        self.face_detector = None
        self.face_positions = []  # Lista de posições de faces por frame
        self.smoothed_positions = []  # Posições suavizadas
        self.speaker_timeline = []  # Quem está falando em cada momento
        
        self._init_detector()
    
    def _init_detector(self):
        """Inicializa detector de faces"""
        if MEDIAPIPE_AVAILABLE:
            self.face_detector = mp_face_detection.FaceDetection(
                model_selection=1,  # 1 = full range (até 5m de distância)
                min_detection_confidence=0.5
            )
            logger.info("[FACE] Usando MediaPipe")
        elif CV2_AVAILABLE:
            # Fallback para Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            logger.info("[FACE] Usando OpenCV Haar Cascade")
        else:
            logger.warning("[FACE] Nenhum detector disponível")
    
    def detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """Detecta faces em um frame"""
        faces = []
        
        if self.face_detector is None:
            return faces
        
        try:
            if MEDIAPIPE_AVAILABLE:
                # MediaPipe espera RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detector.process(rgb_frame)
                
                if results.detections:
                    h, w = frame.shape[:2]
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        
                        # Converte coordenadas relativas para absolutas
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        
                        # Centro do rosto
                        center_x = x + width // 2
                        center_y = y + height // 2
                        
                        faces.append({
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "center_x": center_x,
                            "center_y": center_y,
                            "confidence": detection.score[0] if detection.score else 0.5,
                            "area": width * height
                        })
            else:
                # OpenCV Haar Cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected = self.face_detector.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in detected:
                    faces.append({
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "center_x": x + w // 2,
                        "center_y": y + h // 2,
                        "confidence": 0.7,
                        "area": w * h
                    })
        
        except Exception as e:
            logger.debug(f"[FACE] Erro na detecção: {e}")
        
        return faces
    
    def analyze_video(self, sample_rate: float = 0.5) -> List[Dict]:
        """
        Analisa o vídeo inteiro para detectar faces.
        sample_rate: intervalo em segundos entre amostras
        """
        logger.info(f"[FACE] Analisando vídeo: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error("[FACE] Não foi possível abrir o vídeo")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"[FACE] Vídeo: {frame_width}x{frame_height}, {fps:.1f} fps, {duration:.1f}s")
        
        sample_interval = int(fps * sample_rate)
        if sample_interval < 1:
            sample_interval = 1
        
        self.face_positions = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / fps
                faces = self.detect_faces_in_frame(frame)
                
                # Seleciona o rosto principal (maior ou mais central)
                main_face = None
                if faces:
                    # Ordena por área (maior primeiro)
                    faces.sort(key=lambda f: f['area'], reverse=True)
                    main_face = faces[0]
                
                self.face_positions.append({
                    "timestamp": timestamp,
                    "frame_idx": frame_idx,
                    "faces": faces,
                    "main_face": main_face,
                    "frame_width": frame_width,
                    "frame_height": frame_height
                })
            
            frame_idx += 1
        
        cap.release()
        
        logger.info(f"[FACE] Análise completa: {len(self.face_positions)} amostras")
        
        # Suaviza posições
        self._smooth_positions()
        
        return self.face_positions
    
    def _smooth_positions(self, window_size: int = 5):
        """Suaviza as posições dos rostos para evitar movimentos bruscos"""
        
        if len(self.face_positions) < window_size:
            self.smoothed_positions = self.face_positions
            return
        
        self.smoothed_positions = []
        
        # Coleta centros dos rostos
        centers_x = []
        centers_y = []
        
        for pos in self.face_positions:
            if pos['main_face']:
                centers_x.append(pos['main_face']['center_x'])
                centers_y.append(pos['main_face']['center_y'])
            else:
                # Se não há rosto, usa o centro do frame
                centers_x.append(pos['frame_width'] // 2)
                centers_y.append(pos['frame_height'] // 2)
        
        # Aplica média móvel
        smoothed_x = []
        smoothed_y = []
        
        for i in range(len(centers_x)):
            start = max(0, i - window_size // 2)
            end = min(len(centers_x), i + window_size // 2 + 1)
            
            avg_x = sum(centers_x[start:end]) / (end - start)
            avg_y = sum(centers_y[start:end]) / (end - start)
            
            smoothed_x.append(int(avg_x))
            smoothed_y.append(int(avg_y))
        
        # Atualiza posições suavizadas
        for i, pos in enumerate(self.face_positions):
            smooth_pos = pos.copy()
            smooth_pos['smooth_center_x'] = smoothed_x[i]
            smooth_pos['smooth_center_y'] = smoothed_y[i]
            self.smoothed_positions.append(smooth_pos)
        
        logger.info("[FACE] Posições suavizadas")
    
    def get_crop_region(self, timestamp: float, frame_width: int, frame_height: int) -> Dict:
        """
        Calcula a região de crop para um determinado timestamp.
        Retorna as coordenadas para cortar o frame 16:9 em 9:16.
        """
        
        # Encontra a posição mais próxima
        closest_pos = None
        min_diff = float('inf')
        
        positions = self.smoothed_positions if self.smoothed_positions else self.face_positions
        
        for pos in positions:
            diff = abs(pos['timestamp'] - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_pos = pos
        
        # Centro padrão (centro do frame)
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        if closest_pos:
            if 'smooth_center_x' in closest_pos:
                center_x = closest_pos['smooth_center_x']
                center_y = closest_pos['smooth_center_y']
            elif closest_pos['main_face']:
                center_x = closest_pos['main_face']['center_x']
                center_y = closest_pos['main_face']['center_y']
        
        # Calcula dimensões do crop para 9:16
        # Mantém a altura total e calcula a largura necessária
        crop_height = frame_height
        crop_width = int(crop_height * self.target_aspect)
        
        # Se a largura necessária for maior que disponível, ajusta
        if crop_width > frame_width:
            crop_width = frame_width
            crop_height = int(crop_width / self.target_aspect)
        
        # Calcula posição do crop centralizado no rosto
        crop_x = center_x - crop_width // 2
        crop_y = center_y - crop_height // 2
        
        # Limita às bordas do frame
        crop_x = max(0, min(crop_x, frame_width - crop_width))
        crop_y = max(0, min(crop_y, frame_height - crop_height))
        
        return {
            "x": crop_x,
            "y": crop_y,
            "width": crop_width,
            "height": crop_height,
            "center_x": center_x,
            "center_y": center_y
        }
    
    def get_ffmpeg_crop_filter(self, frame_width: int, frame_height: int) -> str:
        """
        Gera o filtro FFmpeg para crop dinâmico baseado nas posições detectadas.
        Para vídeos sem faces ou com faces estáticas, retorna crop simples.
        """
        
        if not self.smoothed_positions:
            # Crop central simples
            crop_height = frame_height
            crop_width = int(crop_height * self.target_aspect)
            if crop_width > frame_width:
                crop_width = frame_width
                crop_height = int(crop_width / self.target_aspect)
            
            crop_x = (frame_width - crop_width) // 2
            crop_y = (frame_height - crop_height) // 2
            
            return f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}"
        
        # Verifica se há movimento significativo
        centers_x = [p.get('smooth_center_x', frame_width // 2) for p in self.smoothed_positions]
        x_variance = max(centers_x) - min(centers_x)
        
        if x_variance < frame_width * 0.1:  # Menos de 10% de variação
            # Posição média estática
            avg_x = sum(centers_x) // len(centers_x)
            
            crop_height = frame_height
            crop_width = int(crop_height * self.target_aspect)
            if crop_width > frame_width:
                crop_width = frame_width
                crop_height = int(crop_width / self.target_aspect)
            
            crop_x = avg_x - crop_width // 2
            crop_x = max(0, min(crop_x, frame_width - crop_width))
            crop_y = (frame_height - crop_height) // 2
            
            return f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}"
        
        # Para movimento significativo, usa crop dinâmico
        # (implementação avançada com sendcmd do FFmpeg)
        logger.info("[FACE] Detectado movimento de câmera, usando crop adaptativo")
        
        # Por enquanto, retorna crop baseado na posição média
        avg_x = sum(centers_x) // len(centers_x)
        
        crop_height = frame_height
        crop_width = int(crop_height * self.target_aspect)
        if crop_width > frame_width:
            crop_width = frame_width
            crop_height = int(crop_width / self.target_aspect)
        
        crop_x = avg_x - crop_width // 2
        crop_x = max(0, min(crop_x, frame_width - crop_width))
        crop_y = (frame_height - crop_height) // 2
        
        return f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}"

# ==================== GERADOR DE LEGENDAS ====================

class SubtitleGenerator:
    """
    Gerador de legendas automáticas estilo TikTok/Reels.
    Cria legendas animadas palavra por palavra.
    """
    
    def __init__(self, words: List[Dict], video_width: int = 1080, video_height: int = 1920):
        self.words = words
        self.video_width = video_width
        self.video_height = video_height
        self.style = {
            "font_size": 60,
            "font_color": "#FFFFFF",
            "stroke_color": "#000000",
            "stroke_width": 3,
            "highlight_color": "#FFFF00",
            "position_y": 0.75,  # 75% da altura (parte inferior)
            "max_words_per_line": 4,
            "animation": "highlight"  # highlight, fade, pop
        }
    
    def set_style(self, style: Dict):
        """Atualiza estilo das legendas"""
        self.style.update(style)
    
    def generate_ass_subtitles(self, output_path: Path) -> bool:
        """
        Gera arquivo de legendas no formato ASS (Advanced SubStation Alpha).
        Esse formato permite animações e estilos avançados.
        """
        
        if not self.words:
            logger.warning("[SUBTITLE] Nenhuma palavra para gerar legendas")
            return False
        
        try:
            # Cabeçalho ASS
            ass_content = """[Script Info]
Title: ViralPRO Subtitles
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{stroke},2,2,50,50,80,1
Style: Highlight,Arial,{font_size},&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{stroke},2,2,50,50,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
                width=self.video_width,
                height=self.video_height,
                font_size=self.style['font_size'],
                stroke=self.style['stroke_width']
            )
            
            # Agrupa palavras em linhas
            lines = self._group_words_into_lines()
            
            # Gera eventos de legenda
            for line in lines:
                start_time = self._format_ass_time(line['start'])
                end_time = self._format_ass_time(line['end'])
                text = line['text'].upper()
                
                # Adiciona animação de destaque palavra por palavra
                if self.style['animation'] == 'highlight' and 'words' in line:
                    animated_text = self._create_word_animation(line['words'], line['start'])
                    ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{animated_text}\n"
                else:
                    ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"
            
            # Salva arquivo
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            logger.info(f"[SUBTITLE] Legendas ASS geradas: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"[SUBTITLE] Erro ao gerar legendas: {e}")
            return False
    
    def generate_srt_subtitles(self, output_path: Path) -> bool:
        """Gera arquivo de legendas no formato SRT (mais simples)"""
        
        if not self.words:
            return False
        
        try:
            lines = self._group_words_into_lines()
            
            srt_content = ""
            for i, line in enumerate(lines, 1):
                start_time = self._format_srt_time(line['start'])
                end_time = self._format_srt_time(line['end'])
                text = line['text'].upper()
                
                srt_content += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            logger.info(f"[SUBTITLE] Legendas SRT geradas: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"[SUBTITLE] Erro ao gerar SRT: {e}")
            return False
    
    def _group_words_into_lines(self) -> List[Dict]:
        """Agrupa palavras em linhas de legenda"""
        
        lines = []
        current_line = {
            "words": [],
            "text": "",
            "start": 0,
            "end": 0
        }
        
        max_words = self.style['max_words_per_line']
        max_duration = 3.0  # Máximo de 3 segundos por linha
        
        for word in self.words:
            word_text = word.get('word', '').strip()
            if not word_text:
                continue
            
            # Verifica se precisa criar nova linha
            should_break = False
            
            if len(current_line['words']) >= max_words:
                should_break = True
            elif current_line['words'] and (word['start'] - current_line['start']) > max_duration:
                should_break = True
            elif word_text.endswith(('.', '!', '?', ',')):
                # Quebra após pontuação
                current_line['words'].append(word)
                current_line['text'] += (' ' if current_line['text'] else '') + word_text
                current_line['end'] = word['end']
                should_break = True
                word = None  # Já adicionou
            
            if should_break and current_line['words']:
                lines.append(current_line)
                current_line = {"words": [], "text": "", "start": 0, "end": 0}
            
            if word:
                if not current_line['words']:
                    current_line['start'] = word['start']
                
                current_line['words'].append(word)
                current_line['text'] += (' ' if current_line['text'] else '') + word_text
                current_line['end'] = word['end']
        
        # Adiciona última linha
        if current_line['words']:
            lines.append(current_line)
        
        return lines
    
    def _create_word_animation(self, words: List[Dict], line_start: float) -> str:
        """Cria animação de destaque palavra por palavra no formato ASS"""
        
        animated_parts = []
        
        for word in words:
            word_text = word.get('word', '').strip().upper()
            word_start = word['start'] - line_start
            word_end = word['end'] - line_start
            
            # Tempo em centésimos de segundo para ASS
            start_cs = int(word_start * 100)
            end_cs = int(word_end * 100)
            
            # Adiciona tag de transformação de cor
            # Começa branco, fica amarelo durante a palavra
            animated_parts.append(
                f"{{\\t({start_cs},{end_cs},\\c&H00FFFF&)}}{word_text}"
            )
        
        return ' '.join(animated_parts)
    
    def _format_ass_time(self, seconds: float) -> str:
        """Formata tempo para ASS (h:mm:ss.cc)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
    
    def _format_srt_time(self, seconds: float) -> str:
        """Formata tempo para SRT (hh:mm:ss,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# ==================== ANTI-SHADOWBAN ====================

def apply_antishadowban(clip, config: Dict = None):
    """
    Aplica técnicas anti-shadowban no vídeo.
    Modificações sutis que tornam o vídeo único.
    """
    
    if config is None:
        config = {
            "enabled": True,
            "mirror": False,
            "gamma": True,
            "gamma_value": 1.02,
            "color_shift": True,
            "color_value": 1.01,
            "noise": False
        }
    
    if not config.get("enabled", True):
        return clip
    
    try:
        # Espelhamento horizontal
        if config.get("mirror", False):
            clip = moviepy_imports['mirror_x'](clip)
        
        # Ajuste de gamma (brilho)
        if config.get("gamma", True):
            gamma = config.get("gamma_value", 1.02)
            gamma += random.uniform(-0.01, 0.01)  # Variação aleatória
            clip = moviepy_imports['gamma_corr'](clip, gamma)
        
        # Ajuste de cor
        if config.get("color_shift", True):
            color = config.get("color_value", 1.01)
            color += random.uniform(-0.005, 0.005)
            clip = moviepy_imports['colorx'](clip, color)
        
        logger.info("[ANTISHADOW] Filtros aplicados")
        
    except Exception as e:
        logger.warning(f"[ANTISHADOW] Erro: {e}")
    
    return clip

# ==================== ANÁLISE DE CENAS ====================

class SceneAnalyzer:
    """Analisador de cenas para identificar melhores momentos"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.scenes = []
        self.audio_energy = []
    
    def analyze(self, transcription: Dict = None) -> List[Dict]:
        """
        Analisa o vídeo e identifica os melhores momentos para corte.
        Considera: energia do áudio, mudanças de cena, conteúdo da transcrição.
        """
        
        logger.info("[SCENE] Iniciando análise de cenas")
        
        # Obtém duração do vídeo
        duration = self._get_video_duration()
        
        # Analisa energia do áudio
        self._analyze_audio_energy()
        
        # Identifica momentos importantes baseado na transcrição
        important_moments = []
        
        if transcription and transcription.get('chunks'):
            important_moments = self._find_important_moments(transcription['chunks'])
        
        # Combina análise de áudio e transcrição
        self.scenes = self._merge_analyses(duration, important_moments)
        
        logger.info(f"[SCENE] {len(self.scenes)} cenas identificadas")
        
        return self.scenes
    
    def _get_video_duration(self) -> float:
        """Obtém duração do vídeo"""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                self.video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except:
            return 0
    
    def _analyze_audio_energy(self):
        """Analisa energia do áudio para detectar momentos de alta intensidade"""
        
        try:
            # Extrai áudio e analisa volume
            temp_audio = TEMP_DIR / f"energy_{uuid.uuid4().hex[:6]}.wav"
            
            cmd = [
                'ffmpeg', '-i', self.video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '8000', '-ac', '1',
                str(temp_audio), '-y',
                '-hide_banner', '-loglevel', 'error'
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=120)
            
            if temp_audio.exists():
                # Análise com FFmpeg volumedetect
                cmd = [
                    'ffmpeg', '-i', str(temp_audio),
                    '-af', 'volumedetect',
                    '-f', 'null', '-',
                    '-hide_banner'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                # Parse do resultado
                for line in result.stderr.split('\n'):
                    if 'max_volume' in line:
                        try:
                            vol = float(line.split(':')[1].strip().split()[0])
                            self.audio_energy.append(vol)
                        except:
                            pass
                
                temp_audio.unlink()
        
        except Exception as e:
            logger.warning(f"[SCENE] Erro na análise de áudio: {e}")
    
    def _find_important_moments(self, chunks: List[Dict]) -> List[Dict]:
        """Encontra momentos importantes baseado na transcrição"""
        
        moments = []
        
        # Palavras que indicam momentos interessantes
        impact_words = [
            "incrível", "impressionante", "nunca", "sempre", "importante",
            "olha", "veja", "caramba", "nossa", "uau", "então",
            "porque", "por que", "como", "quando", "quem",
            "primeiro", "melhor", "pior", "maior", "menor",
            "problema", "solução", "segredo", "dica", "truque"
        ]
        
        for chunk in chunks:
            text = chunk.get('text', '').lower()
            start, end = chunk.get('timestamp', (0, 0))
            
            if not text or start is None:
                continue
            
            score = 0
            
            # Pontuação por palavras de impacto
            for word in impact_words:
                if word in text:
                    score += 10
            
            # Pontuação por emoção (pontuação)
            score += text.count('!') * 5
            score += text.count('?') * 3
            
            # Pontuação por tamanho (frases mais longas são mais informativas)
            word_count = len(text.split())
            if word_count > 10:
                score += 5
            
            if score > 10:
                moments.append({
                    "start": start,
                    "end": end,
                    "text": text,
                    "score": score,
                    "type": "dialogue"
                })
        
        # Ordena por score
        moments.sort(key=lambda x: x['score'], reverse=True)
        
        return moments
    
    def _merge_analyses(self, duration: float, moments: List[Dict]) -> List[Dict]:
        """Combina diferentes análises para criar lista de cenas"""
        
        scenes = []
        
        # Se há momentos identificados, usa eles
        if moments:
            for moment in moments[:20]:  # Top 20 momentos
                scenes.append({
                    "start": max(0, moment['start'] - 2),  # 2s antes
                    "end": min(duration, moment['end'] + 2),  # 2s depois
                    "score": moment['score'],
                    "type": moment['type'],
                    "text": moment.get('text', '')
                })
        
        # Se não há momentos, divide o vídeo em segmentos regulares
        if not scenes:
            segment_duration = 60  # 60 segundos por segmento
            current = 0
            
            while current < duration:
                end = min(current + segment_duration, duration)
                scenes.append({
                    "start": current,
                    "end": end,
                    "score": 50,
                    "type": "auto",
                    "text": ""
                })
                current = end
        
        return scenes

# ==================== PROCESSADOR PRINCIPAL ====================

class ViralPROProcessor:
    """Processador principal do ViralPRO"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.video_path = None
        self.output_dir = OUTPUT_DIR
        self.temp_files = []
        
        # Configurações padrão
        self.target_width = 1080
        self.target_height = 1920
        self.min_duration = config.get('cutDuration', {}).get('min', 30)
        self.max_duration = config.get('cutDuration', {}).get('max', 90)
    
    def process(self, video_url: str) -> Dict:
        """Processa o vídeo e gera os cortes"""
        
        start_time = time.time()
        results = {
            "status": "processing",
            "cuts": [],
            "errors": []
        }
        
        try:
            # 1. Download do vídeo
            logger.info("=" * 60)
            logger.info("[VIRALPRO] INICIANDO PROCESSAMENTO")
            logger.info(f"  URL: {video_url[:80]}...")
            logger.info("=" * 60)
            
            self.video_path = self._download_video(video_url)
            if not self.video_path:
                raise Exception("Falha no download do vídeo")
            
            # 2. Extrai e transcreve áudio
            logger.info("[STEP 2] Extraindo e transcrevendo áudio...")
            transcription = self._transcribe_video()
            
            # 3. Analisa vídeo para faces
            logger.info("[STEP 3] Analisando faces no vídeo...")
            face_tracker = FaceTracker(str(self.video_path), self.target_width, self.target_height)
            face_tracker.analyze_video(sample_rate=1.0)
            
            # 4. Analisa cenas
            logger.info("[STEP 4] Analisando cenas...")
            scene_analyzer = SceneAnalyzer(str(self.video_path))
            scenes = scene_analyzer.analyze(transcription)
            
            # 5. Seleciona melhores cortes
            logger.info("[STEP 5] Selecionando melhores cortes...")
            selected_cuts = self._select_cuts(scenes, transcription)
            
            # 6. Processa cada corte
            logger.info("[STEP 6] Processando cortes...")
            
            for i, cut in enumerate(selected_cuts):
                try:
                    logger.info(f"\n[CUT {i+1}/{len(selected_cuts)}] {cut['start']:.1f}s - {cut['end']:.1f}s")
                    
                    output_path = self._process_cut(
                        cut_data=cut,
                        cut_number=i + 1,
                        face_tracker=face_tracker,
                        transcription=transcription
                    )
                    
                    if output_path and output_path.exists():
                        # Upload para B2
                        public_url = self._upload_to_b2(output_path, i + 1)
                        
                        results['cuts'].append({
                            "number": i + 1,
                            "start": cut['start'],
                            "end": cut['end'],
                            "duration": cut['end'] - cut['start'],
                            "url": public_url,
                            "local_path": str(output_path)
                        })
                        
                        logger.info(f"[SUCCESS] Corte {i+1} processado")
                    
                except Exception as e:
                    logger.error(f"[ERROR] Corte {i+1}: {e}")
                    results['errors'].append({
                        "cut": i + 1,
                        "error": str(e)
                    })
            
            # Finaliza
            elapsed = time.time() - start_time
            results['status'] = "success"
            results['processing_time'] = round(elapsed, 2)
            results['total_cuts'] = len(results['cuts'])
            
            logger.info("=" * 60)
            logger.info(f"[VIRALPRO] PROCESSAMENTO CONCLUÍDO")
            logger.info(f"  Cortes: {len(results['cuts'])}")
            logger.info(f"  Tempo: {elapsed:.1f}s")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"[FATAL] {e}")
            import traceback
            results['status'] = "error"
            results['error'] = str(e)
            results['traceback'] = traceback.format_exc()
        
        finally:
            self._cleanup()
        
        return results
    
    def _download_video(self, url: str) -> Optional[Path]:
        """Faz download do vídeo"""
        
        try:
            video_path = TEMP_DIR / f"input_{uuid.uuid4().hex[:8]}.mp4"
            self.temp_files.append(video_path)
            
            if network.download_with_retry(url, video_path):
                logger.info(f"[DOWNLOAD] Vídeo salvo: {video_path}")
                return video_path
            
            return None
            
        except Exception as e:
            logger.error(f"[DOWNLOAD] Erro: {e}")
            return None
    
    def _transcribe_video(self) -> Dict:
        """Extrai áudio e transcreve"""
        
        if not WHISPER_AVAILABLE:
            logger.warning("[TRANSCRIBE] Whisper não disponível")
            return {"chunks": [], "words": []}
        
        try:
            # Extrai áudio
            audio_path = TEMP_DIR / f"audio_{uuid.uuid4().hex[:8]}.wav"
            self.temp_files.append(audio_path)
            
            cmd = [
                'ffmpeg', '-i', str(self.video_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                str(audio_path), '-y',
                '-hide_banner', '-loglevel', 'error'
            ]
            
            subprocess.run(cmd, capture_output=True, timeout=300)
            
            if not audio_path.exists():
                raise Exception("Falha ao extrair áudio")
            
            # Limpa áudio
            clean_audio = clean_audio_deepfilter(audio_path)
            if clean_audio != audio_path:
                self.temp_files.append(clean_audio)
            
            # Transcreve
            logger.info("[TRANSCRIBE] Iniciando transcrição...")
            
            if not whisper_manager.loaded():
                whisper_manager.load()
            
            result = whisper_manager.transcribe(str(clean_audio))
            
            logger.info(f"[TRANSCRIBE] {len(result.get('chunks', []))} segmentos, {len(result.get('words', []))} palavras")
            
            return result
            
        except Exception as e:
            logger.error(f"[TRANSCRIBE] Erro: {e}")
            return {"chunks": [], "words": []}
    
    def _select_cuts(self, scenes: List[Dict], transcription: Dict) -> List[Dict]:
        """Seleciona os melhores cortes baseado nas cenas e transcrição"""
        
        cuts = []
        
        # Ordena cenas por score
        sorted_scenes = sorted(scenes, key=lambda x: x.get('score', 0), reverse=True)
        
        # Obtém duração do vídeo
        try:
            video = moviepy_imports['VideoFileClip'](str(self.video_path))
            video_duration = video.duration
            video.close()
        except:
            video_duration = 3600  # Fallback
        
        # Limites de corte
        max_cuts = self.config.get('maxCuts', 10)
        
        for scene in sorted_scenes:
            if len(cuts) >= max_cuts:
                break
            
            start = scene['start']
            end = scene['end']
            duration = end - start
            
            # Ajusta duração
            if duration < self.min_duration:
                # Expande o corte
                extra = (self.min_duration - duration) / 2
                start = max(0, start - extra)
                end = min(video_duration, end + extra)
                duration = end - start
            
            if duration > self.max_duration:
                # Reduz o corte
                end = start + self.max_duration
                duration = self.max_duration
            
            # Verifica sobreposição com cortes existentes
            overlap = False
            for existing in cuts:
                if not (end <= existing['start'] or start >= existing['end']):
                    overlap = True
                    break
            
            if not overlap and duration >= self.min_duration:
                cuts.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "duration": round(duration, 2),
                    "score": scene.get('score', 50),
                    "text": scene.get('text', '')
                })
        
        # Ordena por tempo
        cuts.sort(key=lambda x: x['start'])
        
        logger.info(f"[SELECT] {len(cuts)} cortes selecionados")
        
        return cuts
    
    def _process_cut(self, cut_data: Dict, cut_number: int, 
                     face_tracker: FaceTracker, transcription: Dict) -> Optional[Path]:
        """Processa um corte individual"""
        
        start = cut_data['start']
        end = cut_data['end']
        duration = end - start
        
        output_path = self.output_dir / f"viral_{cut_number:03d}_{uuid.uuid4().hex[:6]}.mp4"
        
        try:
            # Obtém informações do vídeo
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'json',
                str(self.video_path)
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            probe_data = json.loads(probe_result.stdout)
            
            frame_width = probe_data['streams'][0]['width']
            frame_height = probe_data['streams'][0]['height']
            
            logger.info(f"[CUT] Vídeo original: {frame_width}x{frame_height}")
            
            # Gera filtro de crop baseado no face tracking
            crop_filter = face_tracker.get_ffmpeg_crop_filter(frame_width, frame_height)
            
            # Gera legendas
            subtitle_path = None
            words_in_range = self._get_words_in_range(transcription, start, end)
            
            if words_in_range:
                # Ajusta timestamps das palavras
                adjusted_words = []
                for word in words_in_range:
                    adjusted_words.append({
                        "word": word['word'],
                        "start": word['start'] - start,
                        "end": word['end'] - start,
                        "confidence": word.get('confidence', 0.9)
                    })
                
                subtitle_gen = SubtitleGenerator(adjusted_words, self.target_width, self.target_height)
                subtitle_gen.set_style(self.config.get('subtitleStyle', {}))
                
                subtitle_path = TEMP_DIR / f"subs_{cut_number}_{uuid.uuid4().hex[:6]}.ass"
                self.temp_files.append(subtitle_path)
                
                if subtitle_gen.generate_ass_subtitles(subtitle_path):
                    logger.info(f"[CUT] Legendas geradas: {len(adjusted_words)} palavras")
                else:
                    subtitle_path = None
            
            # Monta comando FFmpeg
            # Verifica NVENC
            nvenc_available = False
            try:
                result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                                       capture_output=True, text=True, timeout=10)
                nvenc_available = 'h264_nvenc' in result.stdout
            except:
                pass
            
            # Filtro de vídeo
            video_filter = f"{crop_filter},scale={self.target_width}:{self.target_height}"
            
            # Adiciona legendas se disponível
            if subtitle_path and subtitle_path.exists():
                # Escape do caminho para FFmpeg
                sub_path_escaped = str(subtitle_path).replace('\\', '/').replace(':', '\\:')
                video_filter += f",ass='{sub_path_escaped}'"
            
            # Aplica anti-shadowban via filtro
            if self.config.get('antiShadowban', {}).get('enabled', True):
                gamma = 1.02 + random.uniform(-0.01, 0.01)
                video_filter += f",eq=gamma={gamma}"
            
            # Encoder
            if nvenc_available:
                encoder = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23', '-b:v', '6M']
                logger.info("[CUT] Usando NVENC")
            else:
                encoder = ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23']
                logger.info("[CUT] Usando libx264")
            
            # Comando completo
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-t', str(duration),
                '-i', str(self.video_path),
                '-vf', video_filter,
                '-map', '0:v:0',
                '-map', '0:a:0?',
            ] + encoder + [
                '-c:a', 'aac', '-b:a', '128k',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                str(output_path)
            ]
            
            logger.info(f"[CUT] Processando com FFmpeg...")
            
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 50000:
                file_size = output_path.stat().st_size / 1e6
                logger.info(f"[CUT] Sucesso: {file_size:.1f} MB")
                return output_path
            else:
                logger.error(f"[CUT] FFmpeg falhou: {proc.stderr[-500:]}")
                return None
            
        except Exception as e:
            logger.error(f"[CUT] Erro: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _get_words_in_range(self, transcription: Dict, start: float, end: float) -> List[Dict]:
        """Obtém palavras dentro de um intervalo de tempo"""
        
        words = transcription.get('words', [])
        result = []
        
        for word in words:
            word_start = word.get('start', 0)
            word_end = word.get('end', 0)
            
            if word_start >= start and word_end <= end:
                result.append(word)
        
        return result
    
    def _upload_to_b2(self, file_path: Path, cut_number: int) -> str:
        """Faz upload do arquivo para Backblaze B2"""
        
        if not B2_AVAILABLE or not s3_client:
            return str(file_path)
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            object_key = f"viralpro/{timestamp}/cut_{cut_number:03d}.mp4"
            
            s3_client.upload_file(
                str(file_path),
                B2_BUCKET,
                object_key,
                ExtraArgs={'ContentType': 'video/mp4'}
            )
            
            # Gera URL pública
            url = f"https://{B2_BUCKET}.s3.us-east-005.backblazeb2.com/{object_key}"
            
            logger.info(f"[UPLOAD] {object_key}")
            
            return url
            
        except Exception as e:
            logger.warning(f"[UPLOAD] Erro: {e}")
            return str(file_path)
    
    def _cleanup(self):
        """Limpa arquivos temporários"""
        
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except:
                pass
        
        self.temp_files.clear()
        gc.collect()

# ==================== HANDLER PRINCIPAL ====================

def handler(event: Dict) -> Dict:
    """Handler principal do RunPod"""
    
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info(f"[REQUEST {request_id}] ViralPRO v1.0.0")
    logger.info("=" * 70)
    
    try:
        input_data = event.get("input", {})
        
        # Modo de teste
        if input_data.get("mode") == "test":
            return {
                "status": "ok",
                "version": "1.0.0",
                "gpu": GPU_AVAILABLE,
                "whisper": WHISPER_AVAILABLE,
                "mediapipe": MEDIAPIPE_AVAILABLE,
                "ffmpeg": FFMPEG_AVAILABLE,
                "b2": B2_AVAILABLE,
                "request_id": request_id
            }
        
        # Validação
        video_url = input_data.get("video_url")
        if not video_url:
            return {
                "status": "error",
                "error": "video_url é obrigatório",
                "request_id": request_id
            }
        
        # Configuração
        config = {
            "contentName": input_data.get("contentName", "Video"),
            "cutDuration": input_data.get("cutDuration", {"min": 30, "max": 90}),
            "maxCuts": input_data.get("maxCuts", 10),
            "antiShadowban": input_data.get("antiShadowban", {"enabled": True}),
            "subtitleStyle": input_data.get("subtitleStyle", {
                "font_size": 60,
                "font_color": "#FFFFFF",
                "stroke_color": "#000000",
                "stroke_width": 3,
                "highlight_color": "#FFFF00",
                "position_y": 0.75
            }),
            "faceTracking": input_data.get("faceTracking", {"enabled": True}),
            "debug": input_data.get("debug", False)
        }
        
        # Processa
        processor = ViralPROProcessor(config)
        result = processor.process(video_url)
        
        # Adiciona metadados
        result['request_id'] = request_id
        result['processing_time'] = round(time.time() - start_time, 2)
        
        return result
        
    except Exception as e:
        logger.error(f"[ERROR] {e}")
        import traceback
        
        return {
            "status": "error",
            "request_id": request_id,
            "error": str(e),
            "traceback": traceback.format_exc() if input_data.get("debug") else None,
            "processing_time": round(time.time() - start_time, 2)
        }

def safe_handler(event):
    """Wrapper seguro"""
    try:
        return handler(event)
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# ==================== INICIALIZAÇÃO ====================

if __name__ == "__main__":
    try:
        print("\n" + "=" * 70)
        print("╔═══════════════════════════════════════════════════════════════════╗")
        print("║   VIRALPRO SERVERLESS v1.0.0 - SHORTS/REELS GENERATOR           ║")
        print("║   🚀 Face Tracking + Auto Subtitles + Smart Crop                 ║")
        print("╚═══════════════════════════════════════════════════════════════════╝")
        print(f"Volume: {VOLUME_BASE}")
        print(f"Cache: {CACHE_DIR}")
        print("=" * 70)
        
        print("\n[SYSTEM STATUS]")
        print(f"  GPU: {'✓' if GPU_AVAILABLE else '✗'}")
        print(f"  CUDA: {'✓' if GPU_AVAILABLE else '✗'}")
        print(f"  MoviePy: {'✓' if MOVIEPY_AVAILABLE else '✗'}")
        print(f"  PyTorch: {'✓' if AI_AVAILABLE else '✗'}")
        print(f"  Whisper: {'✓' if WHISPER_AVAILABLE else '✗'} {WHISPER_TYPE if WHISPER_AVAILABLE else ''}")
        print(f"  FFmpeg: {'✓' if FFMPEG_AVAILABLE else '✗'}")
        print(f"  MediaPipe: {'✓' if MEDIAPIPE_AVAILABLE else '✗'}")
        print(f"  OpenCV: {'✓' if CV2_AVAILABLE else '✗'}")
        print(f"  Pillow: {'✓' if PIL_AVAILABLE else '✗'}")
        print(f"  B2 Storage: {'✓' if B2_AVAILABLE else '✗'}")
        print("=" * 70 + "\n")
        
        # RunPod
        try:
            import runpod
            logger.info("[RUNPOD] Iniciando servidor...")
            runpod.serverless.start({
                "handler": safe_handler,
                "concurrency_modifier": lambda x: 1,
                "return_aggregate_stream": True
            })
        except ImportError:
            print("Modo de teste local...")
            test_result = safe_handler({"input": {"mode": "test"}})
            print(json.dumps(test_result, indent=2))
            
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Servidor interrompido")
        resource_manager.cleanup_all()
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
