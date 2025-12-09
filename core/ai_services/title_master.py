# -*- coding: utf-8 -*-
"""
TITLE MASTER - Geracao de Titulos com Qwen 2.5 7B
===================================================
Gera 3-5 variacoes de titulo usando analise narrativa
Ortografia perfeita com LanguageTool automatico
"""

import logging

logger = logging.getLogger(__name__)

# Importa modelo e corretor
from .local_ai_service import load_llama_model, _LLAMA_INSTANCE, clean_llama_response
from .narrative_analyzer import get_narrative_context
from .dicionario_pt_br import correct_title as dict_correct_title

try:
    from .languagetool_corrector import corrigir_titulo_completo
    LANGUAGETOOL_AVAILABLE = True
except ImportError:
    LANGUAGETOOL_AVAILABLE = False
    logger.warning("[TITLE MASTER] LanguageTool nao disponivel")

def generate_title_variations(scene_text, narrative_analysis, anime_name="Anime", num_variations=3):
    """Gera multiplas variacoes de titulo"""
    logger.info(f"[TITLE MASTER] Gerando {num_variations} variacoes...")
    
    load_llama_model()
    if _LLAMA_INSTANCE is None:
        return [f"{anime_name.upper()} MOMENTO EPICO"]
    
    context = get_narrative_context(narrative_analysis)
    scene_sample = scene_text[:500]
    
    system_prompt = f"""Voce e um especialista em titulos virais.

INSTRUCAO INTERNA: Crie um título envolvente, claro, chamativo e fiel ao conteúdo, usando exclusivamente o resumo geral + os resumos segmentados. Não invente informações fora do resumo. Capture o tema principal e o conflito central.
Crie titulos em PORTUGUES BRASILEIRO com ORTOGRAFIA PERFEITA.
Gere EXATAMENTE {num_variations} variacoes diferentes."""
    
    user_prompt = f"""{context}
Anime: {anime_name}
Dialogo: "{scene_sample}"
Gere {num_variations} titulos virais."""
    
    try:
        response = _LLAMA_INSTANCE.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.8
        )
        
        raw_response = response['choices'][0]['message']['content'].strip()
        titles = extract_titles(raw_response, num_variations)
        
        corrected_titles = []
        for i, title in enumerate(titles):
            title_clean = clean_llama_response(title).upper()
            title_dict = dict_correct_title(title_clean)
            
            if LANGUAGETOOL_AVAILABLE:
                title_corrected = corrigir_titulo_completo(title_dict).upper()
                corrected_titles.append(title_corrected)
            else:
                corrected_titles.append(title_dict)
        
        valid_titles = validate_titles(corrected_titles, anime_name)
        return valid_titles
        
    except Exception as e:
        logger.error(f"[TITLE MASTER] Erro: {e}")
        return [f"{anime_name.upper()} MOMENTO EPICO"]

def extract_titles(raw_response, num_variations):
    """Extrai titulos da resposta"""
    lines = raw_response.split('\n')
    titles = []
    
    for line in lines:
        line_clean = line.strip().lstrip('0123456789.-) ').strip('"\'')
        if line_clean and len(line_clean) > 10:
            titles.append(line_clean)
        if len(titles) >= num_variations:
            break
    
    return titles[:num_variations] if titles else ["TITULO VIRAL"]

def validate_titles(titles, anime_name):
    """Valida e ajusta titulos"""
    valid_titles = []
    
    for title in titles:
        words = title.split()
        
        if len(words) < 4:
            title = f"{title} EM {anime_name.upper()}"
            words = title.split()
        
        if len(words) > 9:
            title = ' '.join(words[:9])
        
        if len(title) > 80:
            title = title[:80].rsplit(' ', 1)[0].strip()
        
        title = title.rstrip('.,!?;:')
        valid_titles.append(title)
    
    return valid_titles

def select_best_title(titles, criteria='first'):
    """Seleciona melhor titulo"""
    if not titles:
        return "TITULO VIRAL"
    
    if criteria == 'first':
        return titles[0]
    elif criteria == 'longest':
        return max(titles, key=len)
    elif criteria == 'shortest':
        return min(titles, key=len)
    elif criteria == 'random':
        import random
        return random.choice(titles)
    else:
        return titles[0]
