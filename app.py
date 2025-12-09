# -*- coding: utf-8 -*-
"""
VIRAL PRO - Processamento Profissional com Smart Crop e Legendas
Motor GPU com Face Tracking, Whisper Captions e Títulos IA
"""

import streamlit as st
import os
import sys
import tempfile
import time
from pathlib import Path

# Adiciona path dos processadores
sys.path.append(os.path.dirname(__file__))

# Imports condicionais
LIBS_AVAILABLE = False
try:
    from gpu_processor import ViralProcessor
    LIBS_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Erro ao importar gpu_processor: {e}")

try:
    from caption_engine import CaptionEngine
    CAPTION_AVAILABLE = True
except ImportError:
    CAPTION_AVAILABLE = False

try:
    from face_tracker import FaceTracker
    FACE_TRACKER_AVAILABLE = True
except ImportError:
    FACE_TRACKER_AVAILABLE = False

def main():
    st.title("🚀 Viral Pro - Processamento Profissional")
    st.caption("Smart Crop com Face Tracking + Legendas Whisper + Títulos IA")
    
    # Verifica dependências
    if not LIBS_AVAILABLE:
        st.error("""
        ⚠️ **Dependências Não Instaladas**
        
        O Viral Pro requer as seguintes bibliotecas:
        - `mediapipe` (Face Tracking)
        - `faster-whisper` (Legendas)
        - `opencv-python` (Processamento de vídeo)
        
        **Instalação:**
        ```bash
        pip install mediapipe faster-whisper opencv-python
        ```
        """)
        return
    
    # Sidebar - Configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        st.subheader("🎬 Processamento")
        num_clips = st.number_input("Número de Clips", 1, 10, 1,
                                    help="Quantos clips gerar do vídeo")
        clip_duration = st.number_input("Duração por Clip (s)", 15, 300, 60,
                                       help="Duração de cada clip em segundos")
        start_min = st.number_input("Início (minutos)", 0, 120, 0,
                                    help="De onde começar a extrair clips")
        
        st.subheader("🎯 Smart Crop")
        use_face_tracking = st.checkbox("Ativar Face Tracking", value=True,
                                       disabled=not FACE_TRACKER_AVAILABLE,
                                       help="Rastreamento de rosto para crop inteligente")
        
        if not FACE_TRACKER_AVAILABLE:
            st.caption("⚠️ mediapipe não instalado")
        
        st.subheader("📝 Legendas")
        use_captions = st.checkbox("Gerar Legendas (Whisper)", value=False,
                                  disabled=not CAPTION_AVAILABLE,
                                  help="Transcrição automática com Whisper")
        
        if not CAPTION_AVAILABLE:
            st.caption("⚠️ faster-whisper não instalado")
        
        if use_captions and CAPTION_AVAILABLE:
            font_style = st.selectbox("Estilo de Fonte", ["Arial", "Impact", "Roboto"])
            caption_position = st.selectbox("Posição", ["center", "top", "bottom"])
            highlight_color = st.checkbox("Destacar Palavra Atual", value=True)
        
        st.subheader("🤖 Títulos IA")
        use_ai_titles = st.checkbox("Gerar Títulos com IA", value=False)
        
        api_key = None
        if use_ai_titles:
            api_key = st.text_input("API Key (Gemini)", type="password",
                                   help="Obtenha em: https://makersuite.google.com/app/apikey")
    
    # Área Principal
    st.write("### 📤 Upload de Vídeo")
    uploaded_video = st.file_uploader("Arraste seu vídeo aqui", type=['mp4', 'mkv', 'avi', 'mov'])
    
    if uploaded_video:
        # Salva temporariamente
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        # Info do vídeo
        try:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                width, height = clip.size
                fps = clip.fps
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duração", f"{duration:.1f}s")
                with col2:
                    st.metric("Resolução", f"{width}x{height}")
                with col3:
                    st.metric("FPS", f"{fps:.1f}")
        except Exception as e:
            st.warning(f"Não foi possível ler informações do vídeo: {e}")
        
        # Botão de processamento
        if st.button("🚀 INICIAR PROCESSAMENTO", type="primary"):
            
            # Callback para status
            status_container = st.empty()
            progress_bar = st.progress(0)
            
            def status_callback(message):
                status_container.info(f"📊 {message}")
            
            try:
                # Inicializa processador
                processor = ViralProcessor(
                    api_key=api_key if use_ai_titles else None,
                    status_callback=status_callback
                )
                
                # Processa vídeo
                with st.spinner("Processando vídeo..."):
                    results = processor.process_video(
                        video_path=video_path,
                        num_clips=num_clips,
                        clip_duration=clip_duration,
                        start_min=start_min
                    )
                
                # Sucesso
                if results and len(results) > 0:
                    st.success(f"✅ {len(results)} clips processados com sucesso!")
                    st.balloons()
                    
                    # Mostra resultados
                    st.write("### 📥 Downloads")
                    
                    for i, result_path in enumerate(results):
                        if os.path.exists(result_path):
                            filename = os.path.basename(result_path)
                            file_size = os.path.getsize(result_path) / (1024 * 1024)  # MB
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{i+1}. {filename}** ({file_size:.1f} MB)")
                            with col2:
                                with open(result_path, 'rb') as f:
                                    st.download_button(
                                        label="⬇️ Baixar",
                                        data=f,
                                        file_name=filename,
                                        mime="video/mp4",
                                        key=f"download_{i}"
                                    )
                else:
                    st.warning("Nenhum clip foi gerado. Verifique as configurações.")
                
            except Exception as e:
                st.error(f"❌ Erro durante o processamento: {e}")
                import traceback
                with st.expander("Detalhes do Erro"):
                    st.code(traceback.format_exc())
    
    # Informações e Tutorial
    with st.expander("ℹ️ Como Funciona o Viral Pro"):
        st.markdown("""
        ## 🎯 Funcionalidades
        
        ### 1. **Smart Crop com Face Tracking**
        - Detecta rostos automaticamente usando MediaPipe
        - Rastreamento suave frame-a-frame
        - Crop inteligente 9:16 focado no rosto principal
        - Fallback para centro da tela se não detectar rosto
        
        ### 2. **Legendas Automáticas (Whisper)**
        - Transcrição de áudio com Whisper (GPU acelerado)
        - Legendas palavra-por-palavra sincronizadas
        - Múltiplos estilos de fonte
        - Destaque da palavra atual (amarelo)
        - Posicionamento configurável
        
        ### 3. **Títulos Virais com IA**
        - Geração de títulos contextuais com Gemini
        - Análise do conteúdo do vídeo
        - Nomes de arquivo otimizados para SEO
        
        ### 4. **Processamento GPU**
        - Aceleração por GPU (CUDA) quando disponível
        - Fallback automático para CPU
        - Otimização de memória VRAM
        
        ## 📋 Requisitos
        
        **Mínimo:**
        - Python 3.8+
        - 8 GB RAM
        - GPU NVIDIA (opcional, mas recomendado)
        
        **Dependências:**
        ```bash
        pip install mediapipe faster-whisper opencv-python moviepy google-generativeai
        ```
        
        ## 🚀 Fluxo de Trabalho
        
        1. **Upload** - Envie seu vídeo (MP4, MKV, AVI, MOV)
        2. **Configure** - Ajuste número de clips, duração, legendas
        3. **Processe** - Aguarde o processamento (pode demorar)
        4. **Download** - Baixe os clips processados
        
        ## 💡 Dicas
        
        - Para vídeos longos, use menos clips ou menor duração
        - Face Tracking funciona melhor com rostos bem iluminados
        - Legendas consomem mais tempo de processamento
        - GPU acelera significativamente o processo
        
        ## ⚡ Performance
        
        **Com GPU (RTX 4060):**
        - Clip de 60s: ~2-3 minutos
        - Com legendas: ~4-5 minutos
        
        **Sem GPU (CPU):**
        - Clip de 60s: ~8-10 minutos
        - Com legendas: ~15-20 minutos
        """)
    
    # Status das dependências
    with st.expander("🔧 Status das Dependências"):
        st.write("**Bibliotecas Instaladas:**")
        st.write(f"- gpu_processor: {'✅' if LIBS_AVAILABLE else '❌'}")
        st.write(f"- caption_engine: {'✅' if CAPTION_AVAILABLE else '❌'}")
        st.write(f"- face_tracker: {'✅' if FACE_TRACKER_AVAILABLE else '❌'}")
        
        if not all([LIBS_AVAILABLE, CAPTION_AVAILABLE, FACE_TRACKER_AVAILABLE]):
            st.warning("""
            **Algumas funcionalidades estão desabilitadas.**
            
            Instale as dependências faltantes:
            ```bash
            pip install mediapipe faster-whisper opencv-python moviepy
            ```
            """)

if __name__ == "__main__":
    main()
