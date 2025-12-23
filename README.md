# 🎬 ViralPRO - Gerador de Shorts/Reels Viral

Sistema inteligente de criação de vídeos curtos (Shorts/Reels/TikTok) a partir de vídeos longos como podcasts, filmes, séries, jornais e novelas.

## ✨ Funcionalidades Principais

### 🎯 Redimensionamento Inteligente (16:9 → 9:16)
- Converte vídeos horizontais para formato vertical
- Crop dinâmico baseado em rastreamento facial
- Mantém o sujeito principal sempre enquadrado

### 👤 Rastreamento Facial com Detecção de Voz
- Detecta e rastreia faces em tempo real usando MediaPipe
- Identifica quem está falando baseado na posição
- Transições suaves ao mudar de speaker
- Suavização de movimentos para evitar tremores

### 📝 Legendas Automáticas
- Transcrição com Whisper V3 Turbo
- Legendas palavra por palavra sincronizadas
- Estilo TikTok com destaque animado
- Suporte a formato ASS (animações avançadas) e SRT

### 🛡️ Anti-ShadowBan
- Modificações sutis que tornam cada vídeo único
- Variação de gamma e cor
- Evita detecção de conteúdo duplicado

### 🤖 Análise de Cenas com IA
- Identifica automaticamente os melhores momentos
- Análise de energia do áudio
- Detecção de palavras de impacto na transcrição

## 📦 Stack Tecnológica

| Componente | Tecnologia |
|------------|------------|
| IA/Transcrição | Whisper V3 Turbo (faster-whisper) |
| Face Detection | MediaPipe + OpenCV |
| Processamento | FFmpeg com NVENC (GPU) |
| Vídeo | MoviePy 1.0.3 |
| Runtime | RunPod Serverless |
| Storage | Backblaze B2 |

## 🚀 Como Usar

### Requisitos
- Docker
- GPU NVIDIA com CUDA 12.1+
- Conta RunPod (para serverless)
- Conta Backblaze B2 (para storage)

### Build do Docker

```bash
docker build -t viralpro:latest .
```

### Execução Local (Teste)

```bash
docker run --gpus all -it viralpro:latest
```

### API de Processamento

**Endpoint:** `POST /`

**Request Body:**
```json
{
  "input": {
    "video_url": "https://exemplo.com/video.mp4",
    "contentName": "Meu Podcast",
    "cutDuration": {
      "min": 30,
      "max": 90
    },
    "maxCuts": 10,
    "antiShadowban": {
      "enabled": true
    },
    "subtitleStyle": {
      "font_size": 60,
      "font_color": "#FFFFFF",
      "stroke_color": "#000000",
      "stroke_width": 3,
      "highlight_color": "#FFFF00",
      "position_y": 0.75
    },
    "faceTracking": {
      "enabled": true
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "request_id": "abc123",
  "processing_time": 245.5,
  "total_cuts": 5,
  "cuts": [
    {
      "number": 1,
      "start": 0,
      "end": 60,
      "duration": 60,
      "url": "https://bucket.b2.com/viralpro/cut_001.mp4"
    }
  ]
}
```

## ⚙️ Configurações

### Estilo de Legendas

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `font_size` | Tamanho da fonte | 60 |
| `font_color` | Cor do texto | #FFFFFF |
| `stroke_color` | Cor do contorno | #000000 |
| `stroke_width` | Espessura do contorno | 3 |
| `highlight_color` | Cor de destaque | #FFFF00 |
| `position_y` | Posição vertical (0-1) | 0.75 |
| `max_words_per_line` | Palavras por linha | 4 |

### Anti-ShadowBan

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `enabled` | Ativa/desativa | true |
| `gamma` | Ajuste de gamma | true |
| `gamma_value` | Valor base gamma | 1.02 |
| `color_shift` | Ajuste de cor | true |
| `color_value` | Valor base cor | 1.01 |

### Duração dos Cortes

| Parâmetro | Descrição | Padrão |
|-----------|-----------|--------|
| `min` | Duração mínima (segundos) | 30 |
| `max` | Duração máxima (segundos) | 90 |

## 📁 Estrutura do Projeto

```
viralpro/
├── handler.py          # Código principal
├── Dockerfile          # Container Docker
├── fontes/             # Fontes customizadas
│   └── .gitkeep
└── README.md           # Esta documentação
```

## 🔧 Variáveis de Ambiente

| Variável | Descrição | Obrigatório |
|----------|-----------|-------------|
| `B2_KEY_ID` | Backblaze Key ID | Sim |
| `B2_APPLICATION_KEY` | Backblaze App Key | Sim |
| `B2_BUCKET_NAME` | Nome do bucket | Sim |
| `B2_ENDPOINT` | Endpoint B2 | Não |

## 📊 Performance

- **GPU NVENC:** Encoding 2-3x mais rápido que CPU
- **Whisper Turbo:** Transcrição em tempo real
- **Face Detection:** ~30 FPS em GPU
- **Processamento típico:** 2-5 minutos por vídeo de 30 minutos

## 🆚 Diferenças do AnimeCut

| Feature | AnimeCut | ViralPRO |
|---------|----------|----------|
| Formato saída | Moldura (16:9 em 9:16) | Crop dinâmico (9:16 real) |
| Foco | Animes | Qualquer vídeo |
| Texto | Títulos | Legendas automáticas |
| Enquadramento | Fixo/central | Face tracking |
| Speaker | N/A | Detecta quem fala |

## 📝 Changelog

### v1.0.0 (2025-12-23)
- Versão inicial
- Face tracking com MediaPipe
- Legendas automáticas com Whisper
- Redimensionamento inteligente
- Anti-shadowban
- Suporte a NVENC

## 🤝 Contribuição

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto é proprietário. Todos os direitos reservados.

---

**ViralPRO** - Transforme vídeos longos em conteúdo viral 🚀
