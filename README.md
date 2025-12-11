# 🎯 ViralPro Serverless

Processamento automático de vídeos virais com Smart Crop (Face Detection) e Legendas Automáticas para RunPod Serverless.

---

## 📁 Arquivos

```
ViralPro/
├── Dockerfile          # Build sem HEALTHCHECK (corrigido para evitar crash)
├── handler.py          # Handler completo (Smart Crop + Faster-Whisper + B2)
└── requirements.txt    # Dependências (MediaPipe, Faster-Whisper, etc.)
```

---

## 🚀 Deploy no RunPod

### 1. Criar Repositório no GitHub

Se ainda não existe, crie um repositório:
- Nome: `ViralPro`
- Visibilidade: Public ou Private

### 2. Fazer Upload dos Arquivos

**Opção A: Via GitHub Web Interface**
1. Acesse seu repositório do ViralPro
2. Upload os 3 arquivos da pasta `ViralPro`:
   - `Dockerfile`
   - `handler.py`
   - `requirements.txt`

**Opção B: Via Git**
```bash
git add Dockerfile handler.py requirements.txt
git commit -m "fix: serverless deployment with smart crop and whisper"
git push origin main
```

### 3. Configurar Endpoint no RunPod

1. **RunPod Console** → **Serverless** → **New Endpoint**
2. **Configurações:**
   - **Name:** ViralPro
   - **Repository:** `https://github.com/SEU_USUARIO/ViralPro.git`
   - **Branch:** `main`
   - **Dockerfile Path:** `Dockerfile`
   - **Container Disk:** 20 GB (Modelos de IA ocupam espaço)
   - **GPU:** RTX 3090 ou superior (Recomendado para Whisper + MediaPipe)

3. **Environment Variables** (opcional para Upload):
   ```
   B2_KEY_ID=your_key_id
   B2_APP_KEY=your_app_key
   B2_BUCKET_NAME=your_bucket_name
   B2_ENDPOINT=https://s3.us-east-005.backblazeb2.com
   ```

4. **Deploy**

---

## 🧪 Testar

### Teste Básico (Healthcheck)
```json
{
  "input": {
    "mode": "test"
  }
}
```

**Resposta esperada:**
```json
{
  "status": "success",
  "message": "ViralPro worker funcionando!",
  "features": {
    "moviepy": true,
    "mediapipe": true,
    "whisper": true,
    "b2": true
    ...
  }
}
```

### Processar Vídeo
```json
{
  "input": {
    "video_url": "https://link-para-seu-video.mp4",
    "num_clips": 3,
    "clip_duration": 60,
    "start_min": 0,
    "add_subtitles": true
  }
}
```

---

## 🎯 Funcionalidades

### ✅ Smart Crop (9:16)
- **Face Detection:** Usa MediaPipe para identificar rostos.
- **Enquadramento Dinâmico:** Mantém o rosto centralizado no vídeo vertical.
- **Fallback:** Crop centralizado se nenhum rosto for detectado.

### ✅ Legendas Automáticas
- **Faster-Whisper:** Transcrição ultra-rápida via GPU.
- **Estilo:** Legendas centralizadas na parte inferior com fundo translúcido.
- **Sincronia:** Timing preciso baseado no áudio.

### ✅ Upload Automático (B2)
- Upload dos cortes gerados para Backblaze B2.
- Geração de URLs assinadas.

---

## 🔧 Troubleshooting

### Worker dá exit code 1
- **Causa:** Healthcheck do Docker nativo conflitando com o RunPod.
- **Solução:** O Dockerfile fornecido já removeu o HEALTHCHECK problemático.

### Erro de Memória (OOM)
- O modelo Whisper e o processamento de vídeo consomem RAM.
- **Solução:** Use um worker com pelo menos 24GB de VRAM/RAM (RTX 3090/4090).

---

**Desenvolvido para RunPod Serverless** 🎯
