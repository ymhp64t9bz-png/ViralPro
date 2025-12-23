#!/bin/bash
# ViralPRO - Script de Teste Local
# Uso: ./test_local.sh [URL_DO_VIDEO]

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║   VIRALPRO - TESTE LOCAL                                          ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

# Verifica se Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "❌ Docker não encontrado. Instale o Docker primeiro."
    exit 1
fi

# Verifica se NVIDIA Docker está disponível
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "⚠️  NVIDIA Docker runtime não detectado. GPU pode não funcionar."
fi

# Build da imagem
echo ""
echo "🔨 Fazendo build da imagem..."
docker build -t viralpro:test . || {
    echo "❌ Falha no build"
    exit 1
}

echo ""
echo "✅ Build concluído!"
echo ""

# Se URL foi fornecida, processa
if [ -n "$1" ]; then
    echo "🎬 Processando vídeo: $1"
    echo ""
    
    docker run --gpus all -it --rm \
        -v "$(pwd)/output:/workspace/output" \
        -e VIDEO_URL="$1" \
        viralpro:test python3 -c "
import json
from handler import safe_handler

result = safe_handler({
    'input': {
        'video_url': '$1',
        'contentName': 'Teste Local',
        'cutDuration': {'min': 30, 'max': 60},
        'maxCuts': 3,
        'debug': True
    }
})

print(json.dumps(result, indent=2, default=str))
"
else
    # Modo de teste simples
    echo "🧪 Executando teste de sistema..."
    echo ""
    
    docker run --gpus all -it --rm viralpro:test python3 -c "
import json
from handler import safe_handler

result = safe_handler({'input': {'mode': 'test'}})
print(json.dumps(result, indent=2))
"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Para processar um vídeo:"
echo "  ./test_local.sh https://exemplo.com/video.mp4"
echo "═══════════════════════════════════════════════════════════════════"
