"""
🔥 ViralPro Cloud - Handler de Produção (Patched v2)
Fix: Normalização de Path (.private) + Signed URL S3v4
"""

import runpod
import os
import logging
import requests
import uuid
import boto3
from botocore.client import Config
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings
import b2_storage

change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ViralPro-Cloud")

OUTPUT_DIR = "/app/output"
TEMP_DIR = "/app/temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- PATCH 403 FIX ---
def normalize_path(path: str) -> str:
    return path.replace(".private/", "").lstrip("/")

def get_s3_client():
    key_id = os.getenv("BACKBLAZE_KEY_ID") or os.getenv("B2_KEY_ID")
    app_key = os.getenv("BACKBLAZE_APP_KEY") or os.getenv("B2_APPLICATION_KEY")
    endpoint = os.getenv("B2_ENDPOINT_URL", "https://s3.us-east-005.backblazeb2.com")
    
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        config=Config(signature_version="s3v4")
    )

def generate_signed_url_s3v4(bucket, file_path):
    try:
        return get_s3_client().generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": file_path},
            ExpiresIn=300,
            HttpMethod="GET"
        )
    except Exception as e:
        logger.error(f"❌ Signed URL Error: {e}")
        return None
# --- END PATCH ---

def download_video(url):
    local_filename = os.path.join(TEMP_DIR, f"source_{uuid.uuid4()}.mp4")
    logger.info(f"⬇️ Baixando: {url[:50]}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            if r.status_code == 403: raise PermissionError("403 Forbidden on Input")
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return local_filename
    except Exception as e:
        logger.error(f"❌ Download Fail: {e}")
        return None

def make_vertical_viral(video_path, title=None):
    logger.info("🎬 Editando ViralPro...")
    out_path = os.path.join(OUTPUT_DIR, f"viralpro_{uuid.uuid4()}.mp4")
    try:
        clip = VideoFileClip(video_path)
        bg = clip.resize(height=1920).crop(x1=clip.w/2-540, x2=clip.w/2+540).fl_image(lambda i: i*0.4)
        main = clip.resize(width=1080).set_position("center")
        layers = [bg, main]
        
        if title:
            txt = TextClip(title.upper(), fontsize=70, color='white', font='DejaVu-Sans-Bold', size=(900,None), method='caption').set_position(('center',200)).set_duration(clip.duration)
            layers.append(txt)
            
        final = CompositeVideoClip(layers, size=(1080,1920)).set_duration(clip.duration).set_audio(clip.audio)
        final.write_videofile(out_path, fps=24, codec='libx264', audio_codec='aac', preset='ultrafast', threads=4, logger=None)
        clip.close()
        return out_path
    except Exception as e:
        logger.error(f"❌ Render Fail: {e}")
        return None

async def handler(job):
    job_input = job.get("input", {})
    raw_path = job_input.get("video_path") or job_input.get("video_url")
    title = job_input.get("title", "")
    
    if not raw_path: return {"status": "error", "error": "No input"}
    
    # Resolve URL
    if raw_path.startswith("http"):
        url = raw_path
    else:
        path = normalize_path(raw_path)
        url = generate_signed_url_s3v4(os.getenv("B2_BUCKET_NAME"), path)
        if not url: return {"status": "error", "error": "Sign URL failed"}
    
    try:
        src = download_video(url)
        if not src: return {"status": "error", "error": "Download failed"}
        
        final = make_vertical_viral(src, title)
        if not final: return {"status": "error", "error": "Render failed"}
        
        # Upload
        fname = f"viralpro/{os.path.basename(final)}"
        if b2_storage.upload_file(final, fname):
             d_url = b2_storage.generate_signed_download_url(fname)
             if os.path.exists(src): os.remove(src)
             if os.path.exists(final): os.remove(final)
             return {"status": "success", "download_url": d_url}
             
        return {"status": "error", "error": "Upload failed"}
        
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

