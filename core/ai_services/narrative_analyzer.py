# -*- coding: utf-8 -*-
"""
NARRATIVE ANALYZER - Análise Narrativa com Qwen 2.5/Qwen
=========================================================
Analisa roteiro para identificar temas, conflitos e momentos virais
NÃO gera timestamps, NÃO escolhe cenas
"""

import logging
import json
import re

logger = logging.getLogger(__name__)

# Importa modelo (usa mesma instância do Qwen)
from .local_ai_service import load_llama_model, _LLAMA_INSTANCE

def analyze_narrative(full_transcript):
    """
    Analisa narrativa usando Qwen 2.5
    NÃO gera timestamps, NÃO escolhe cenas
    
    Args:
        full_transcript: Transcrição completa do Whisper
    
    Returns:
        {
            'themes': ['ação', 'suspense', 'revelação'],
            'conflicts': ['herói vs vilão', 'dilema moral'],
            'viral_moments': [
                'transformação épica',
                'plot twist chocante',
                'batalha final'
            ],
            'emotional_peaks': ['raiva', 'surpresa', 'triunfo'],
            'narrative_arc': 'crescente com clímax no final',
            'tone': 'épico e dramático'
        }
    """
    logger.info("[NARRATIVE] Iniciando análise narrativa...")
    
    # Carrega modelo
    load_llama_model()
    if _LLAMA_INSTANCE is None:
        logger.error("[NARRATIVE] Modelo não disponível")
        return get_default_analysis()
    
    # Limita transcrição para análise
    max_chars = 8000
    transcript_sample = full_transcript[:max_chars]
    
    # Prompt para análise narrativa
    system_prompt = """Você é um analista narrativo especializado em conteúdo viral.
Analise o roteiro e identifique elementos narrativos importantes.
Responda APENAS em JSON válido, sem explicações adicionais."""
    
    user_prompt = f"""Analise este roteiro de anime/série e identifique:

1. Temas principais (3-5 temas)
2. Conflitos centrais (2-4 conflitos)
3. Momentos com potencial viral (5-10 momentos DIVERSOS, SEM timestamps)
   - Incluir: lutas, momentos engraçados, românticos, euforia, piadas, humor, revelações, transformações
4. Picos emocionais (5-8 emoções variadas)
5. Arco narrativo (descrição breve)
6. Tom geral (descrição breve)

Roteiro:
{transcript_sample}

Responda em JSON com esta estrutura:
{{
    "themes": ["tema1", "tema2", "tema3"],
    "conflicts": ["conflito1", "conflito2"],
    "viral_moments": ["momento1", "momento2", "momento3", "momento4", "momento5"],
    "emotional_peaks": ["emoção1", "emoção2", "emoção3", "emoção4", "emoção5"],
    "narrative_arc": "descrição do arco",
    "tone": "descrição do tom"
}}"""
    
    try:
        logger.info("[NARRATIVE] Gerando análise...")
        
        response = _LLAMA_INSTANCE.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.3,  # Baixa para análise consistente
            stop=["\n\n"]
        )
        
        raw_response = response['choices'][0]['message']['content'].strip()
        logger.info(f"[NARRATIVE] Resposta: {raw_response[:200]}...")
        
        # Tenta extrair JSON
        analysis = parse_narrative_json(raw_response)
        
        if analysis:
            logger.info("[NARRATIVE] ✅ Análise concluída!")
            logger.info(f"[NARRATIVE] Temas: {', '.join(analysis['themes'])}")
            logger.info(f"[NARRATIVE] Conflitos: {', '.join(analysis['conflicts'])}")
            logger.info(f"[NARRATIVE] Momentos virais: {len(analysis['viral_moments'])}")
            return analysis
        else:
            logger.warning("[NARRATIVE] Falha ao parsear JSON, usando análise padrão")
            return get_default_analysis()
            
    except Exception as e:
        logger.error(f"[NARRATIVE] Erro na análise: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return get_default_analysis()

def parse_narrative_json(raw_response):
    """
    Extrai JSON da resposta do modelo
    """
    try:
        # Tenta parsear diretamente
        return json.loads(raw_response)
    except:
        pass
    
    try:
        # Tenta encontrar JSON no texto
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except:
        pass
    
    return None

def get_default_analysis():
    """
    Retorna análise padrão em caso de falha (mínimo 5 momentos virais)
    """
    return {
        'themes': ['ação', 'aventura', 'conflito', 'drama', 'suspense'],
        'conflicts': ['protagonista vs antagonista', 'desafio pessoal', 'dilema moral'],
        'viral_moments': [
            'momento de transformação épica',
            'revelação importante e chocante',
            'confronto épico e intenso',
            'momento engraçado e cômico',
            'cena romântica e emocional',
            'piada ou humor inesperado',
            'momento de euforia e vitória'
        ],
        'emotional_peaks': ['tensão', 'surpresa', 'triunfo', 'alegria', 'raiva', 'medo', 'esperança'],
        'narrative_arc': 'progressão com momentos de tensão e resolução',
        'tone': 'dinâmico e envolvente'
    }

def get_narrative_context(analysis):
    """
    Formata análise narrativa para uso em prompts
    
    Args:
        analysis: Dicionário de análise narrativa
    
    Returns:
        str: Contexto formatado
    """
    context = f"""Contexto Narrativo:
- Temas: {', '.join(analysis['themes'])}
- Conflitos: {', '.join(analysis['conflicts'])}
- Momentos Virais: {', '.join(analysis['viral_moments'])}
- Emoções: {', '.join(analysis['emotional_peaks'])}
- Arco: {analysis['narrative_arc']}
- Tom: {analysis['tone']}"""
    
    return context
