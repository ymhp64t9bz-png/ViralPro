"""
HANDLER.PY - VIRALPRO SERVERLESS (SECURE STORAGE)
=================================================
RunPod Handler com suporte a Signed URLs e S3 Privado.
"""

import runpod
import os
import sys
import json
import requests
import boto3
import shutil
from pathlib import Path
from botocore.exceptions import NoCredentialsError
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Configuração de Paths
TEMP_ROOT = "/tmp/viralpro"
MODELS_PATH = os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume") + "/models"

sys.path.insert(0, "/app/core")

from core.ai_services.local_ai_service import transcribe_audio_batch, generate_viral_title_batch
from core.video_processing.anti_shadowban import apply_anti_shadowban
from core.video_processing.pipeline_professional import process_video_professional # se necessário

# ==================== S3 UTILS ====================

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        endpoint_url=os.getenv('S3_ENDPOINT_URL')
    )

def download_secure_file(url, dest_path):
    print(f"[HANDLER] Baixando input seguro...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return dest_path

def upload_secure_file(local_path, bucket, s3_key):
    s3 = get_s3_client()
    try:
        print(f"[HANDLER] Uploading para s3://{bucket}/{s3_key}")
        s3.upload_file(local_path, bucket, s3_key)
        return s3_key
    except Exception as e:
        print(f"[ERROR] Upload falhou: {e}")
        raise e

def generate_presigned_url(bucket, s3_key, expiration=3600):
    s3 = get_s3_client()
    try:
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': s3_key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        return None

# ==================== PROCESSAMENTO ====================

def add_captions(video_clip, segments, font_path="/app/fontes/wanted_m54/Wanted M54.ttf"):
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

def process_viral_video(input_data, job_id):
    try:
        video_url = input_data.get("video_url")
        dest_bucket = os.getenv("S3_BUCKET_NAME")
        
        if not dest_bucket:
            return {"error": "S3_BUCKET_NAME não configurado"}
        if not video_url:
            return {"error": "video_url obrigatório"}

        # Setup Temp
        job_dir = os.path.join(TEMP_ROOT, job_id)
        if os.path.exists(job_dir): shutil.rmtree(job_dir)
        os.makedirs(job_dir, exist_ok=True)
        
        video_path = os.path.join(job_dir, "input.mp4")
        download_secure_file(video_url, video_path)
        
        # Lógica ViralPro (mesma de antes)
        config = input_data.get("config", {})
        add_captions_flag = config.get("add_captions", True)
        add_title_flag = config.get("add_title", True)
        vertical_format = config.get("vertical_format", True)
        anti_shadowban = config.get("anti_shadowban", True)
        
        video = VideoFileClip(video_path)
        whisper_result = None
        
        if add_captions_flag or (add_title_flag and not input_data.get("title")):
            print("[HANDLER] Transcrevendo...")
            whisper_result = transcribe_audio_batch(video_path)
            
        if add_captions_flag and whisper_result:
            video = add_captions(video, whisper_result["segments"])
            
        if add_title_flag:
            title_text = input_data.get("title")
            if not title_text and whisper_result:
                full_text = " ".join([s["text"] for s in whisper_result["segments"]])
                title_text = generate_viral_title_batch("Vídeo", full_text[:500], None)
            
            if title_text:
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
                
        if vertical_format:
            target_w, target_h = 1080, 1920
            if video.w / video.h > target_w / target_h:
                new_w = int(video.h * target_w / target_h)
                video = video.crop(x_center=video.w/2, width=new_w)
            video = video.resize((target_w, target_h))
            
        # Renderiza
        output_filename = f"viral_{job_id}.mp4"
        local_output = os.path.join(job_dir, output_filename)
        video.write_videofile(local_output, codec="libx264", audio_codec="aac", fps=30, logger=None)
        video.close()
        
        # Anti-Shadowban
        if anti_shadowban:
            final_output = os.path.join(job_dir, f"asb_{output_filename}")
            apply_anti_shadowban(local_output, final_output)
            local_output = final_output
            output_filename = f"asb_{output_filename}"

        # Upload Seguro
        s3_key = f"outputs/viralpro/{job_id}/{output_filename}"
        upload_secure_file(local_output, dest_bucket, s3_key)
        signed_url = generate_presigned_url(dest_bucket, s3_key)
        
        # Limpeza
        shutil.rmtree(job_dir)
        
        return {
            "status": "success",
            "job_id": job_id,
            "s3_key": s3_key,
            "signed_url": signed_url
        }

    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

def handler(event):
    input_data = event.get("input", {})
    job_id = event.get("id", "local_viral")
    return process_viral_video(input_data, job_id)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
