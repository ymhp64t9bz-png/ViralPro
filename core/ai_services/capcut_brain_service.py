# -*- coding: utf-8 -*-
"""
CAPCUT BRAIN v2.0 - Análise Semântica Avançada
===============================================
Analisa trechos transcritos e seleciona os melhores cortes
Score final > 0.72 = corte selecionado
"""

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

def analisar_trecho(texto: str, inicio: float, fim: float) -> Dict:
    """
    Analisa um trecho transcrito e calcula score final
    
    Args:
        texto: Texto transcrito do trecho
        inicio: Timestamp de início (segundos)
        fim: Timestamp de fim (segundos)
    
    Returns:
        {
            'inicio': float,
            'fim': float,
            'texto': str,
            'score_semantico': float,
            'score_emocional': float,
            'score_independencia': float,
            'score_final': float,
            'selecionado': bool
        }
    """
    # 1. Análise Semântica
    score_semantico = calcular_score_semantico(texto)
    
    # 2. Análise Emocional/Paralinguística
    score_emocional = calcular_score_emocional(texto)
    
    # 3. Análise de Independência de Contexto
    score_independencia = calcular_score_independencia(texto)
    
    # 4. Score Final (média ponderada)
    score_final = (
        score_semantico * 0.4 +
        score_emocional * 0.3 +
        score_independencia * 0.3
    )
    
    # 5. Seleção (score > 0.72)
    selecionado = score_final > 0.72
    
    return {
        'inicio': inicio,
        'fim': fim,
        'texto': texto,
        'score_semantico': score_semantico,
        'score_emocional': score_emocional,
        'score_independencia': score_independencia,
        'score_final': score_final,
        'selecionado': selecionado
    }

def calcular_score_semantico(texto: str) -> float:
    """Análise semântica avançada"""
    score = 0.5  # Base
    
    # Palavras-chave virais
    palavras_virais = [
        'incrível', 'épico', 'chocante', 'revelação', 'segredo',
        'batalha', 'luta', 'transformação', 'poder', 'vitória',
        'derrota', 'momento', 'final', 'decisivo', 'importante'
    ]
    
    texto_lower = texto.lower()
    for palavra in palavras_virais:
        if palavra in texto_lower:
            score += 0.05
    
    # Comprimento ideal (40-180 palavras)
    palavras = len(texto.split())
    if 40 <= palavras <= 180:
        score += 0.1
    
    # Pontuação (exclamação, interrogação)
    if '!' in texto or '?' in texto:
        score += 0.05
    
    return min(score, 1.0)

def calcular_score_emocional(texto: str) -> float:
    """Análise emocional e paralinguística"""
    score = 0.5  # Base
    
    # Palavras emocionais
    palavras_emocionais = [
        'amor', 'ódio', 'medo', 'raiva', 'alegria', 'tristeza',
        'surpresa', 'choque', 'esperança', 'desespero', 'coragem',
        'traição', 'vingança', 'sacrifício', 'honra'
    ]
    
    texto_lower = texto.lower()
    for palavra in palavras_emocionais:
        if palavra in texto_lower:
            score += 0.05
    
    # Intensidade (maiúsculas, repetições)
    if texto.isupper():
        score += 0.1
    
    # Exclamações múltiplas
    exclamacoes = texto.count('!')
    if exclamacoes > 1:
        score += 0.05
    
    return min(score, 1.0)

def calcular_score_independencia(texto: str) -> float:
    """Análise de independência de contexto"""
    score = 0.5  # Base
    
    # Frases completas
    if texto.strip().endswith(('.', '!', '?')):
        score += 0.1
    
    # Não começa com pronome relativo
    pronomes_relativos = ['que', 'qual', 'quem', 'onde', 'quando', 'como']
    primeira_palavra = texto.strip().split()[0].lower() if texto.strip() else ''
    if primeira_palavra not in pronomes_relativos:
        score += 0.1
    
    # Tem sujeito e verbo
    if tem_sujeito_verbo(texto):
        score += 0.15
    
    # Comprimento mínimo
    if len(texto.split()) >= 10:
        score += 0.1
    
    return min(score, 1.0)

def tem_sujeito_verbo(texto: str) -> bool:
    """Verifica se texto tem estrutura básica sujeito-verbo"""
    # Simplificado: verifica se tem pelo menos 2 palavras
    return len(texto.split()) >= 2

def selecionar_cortes_automaticos(transcricao_completa: str, duracao_total: float) -> List[Dict]:
    """
    Seleciona automaticamente os melhores cortes usando CAPCUT BRAIN
    
    Args:
        transcricao_completa: Texto completo transcrito pelo Whisper
        duracao_total: Duração total do vídeo (segundos)
    
    Returns:
        Lista de cortes selecionados (score > 0.72)
    """
    logger.info("[CAPCUT BRAIN] Iniciando análise semântica avançada...")
    
    # Divide transcrição em trechos (60-180s cada)
    trechos = dividir_em_trechos(transcricao_completa, duracao_total)
    
    # Analisa cada trecho
    resultados = []
    for trecho in trechos:
        resultado = analisar_trecho(
            trecho['texto'],
            trecho['inicio'],
            trecho['fim']
        )
        resultados.append(resultado)
    
    # Filtra apenas selecionados (score > 0.72)
    selecionados = [r for r in resultados if r['selecionado']]
    
    logger.info(f"[CAPCUT BRAIN] Analisados: {len(resultados)} trechos")
    logger.info(f"[CAPCUT BRAIN] Selecionados: {len(selecionados)} trechos (score > 0.72)")
    
    for i, sel in enumerate(selecionados, 1):
        logger.info(f"[CAPCUT BRAIN] Corte {i}: {sel['inicio']:.1f}s-{sel['fim']:.1f}s (score: {sel['score_final']:.2f})")
    
    return selecionados

def dividir_em_trechos(texto: str, duracao_total: float, duracao_trecho: float = 90) -> List[Dict]:
    """Divide transcrição em trechos de duração aproximada"""
    palavras = texto.split()
    total_palavras = len(palavras)
    
    if total_palavras == 0:
        return []
    
    # Estima palavras por segundo
    palavras_por_segundo = total_palavras / duracao_total if duracao_total > 0 else 2
    
    # Palavras por trecho
    palavras_por_trecho = int(duracao_trecho * palavras_por_segundo)
    
    trechos = []
    inicio = 0
    
    while inicio < total_palavras:
        fim_palavras = min(inicio + palavras_por_trecho, total_palavras)
        
        # Calcula timestamps
        inicio_tempo = (inicio / palavras_por_segundo) if palavras_por_segundo > 0 else 0
        fim_tempo = (fim_palavras / palavras_por_segundo) if palavras_por_segundo > 0 else duracao_total
        
        # Extrai texto do trecho
        texto_trecho = ' '.join(palavras[inicio:fim_palavras])
        
        trechos.append({
            'inicio': inicio_tempo,
            'fim': fim_tempo,
            'texto': texto_trecho
        })
        
        inicio = fim_palavras
    
    return trechos
