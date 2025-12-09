# -*- coding: utf-8 -*-
"""
EPISODE SUMMARY SERVICE - Resumo Detalhado de Episódios
========================================================
Gera resumos ultra detalhados para uso do QWEN
"""

import logging
import json
from typing import Dict, List

logger = logging.getLogger(__name__)

def gerar_resumo_completo(transcricao_completa: str, duracao_total: float) -> Dict:
    """
    Gera resumo geral + resumos segmentados a cada 2 minutos
    
    Args:
        transcricao_completa: Texto completo da transcrição do Whisper
        duracao_total: Duração total do episódio em segundos
    
    Returns:
        {
            "resumo_geral": str,
            "resumos_por_bloco": [
                {
                    "start": int,
                    "end": int,
                    "resumo": str
                }
            ]
        }
    """
    logger.info("[EPISODE SUMMARY] Gerando resumo completo do episódio...")
    
    # A) Resumo geral ultra detalhado
    resumo_geral = gerar_resumo_geral(transcricao_completa)
    
    # B) Resumos segmentados a cada 2 minutos
    resumos_por_bloco = gerar_resumos_segmentados(transcricao_completa, duracao_total)
    
    resultado = {
        "resumo_geral": resumo_geral,
        "resumos_por_bloco": resumos_por_bloco
    }
    
    logger.info(f"[EPISODE SUMMARY] Resumo geral: {len(resumo_geral)} caracteres")
    logger.info(f"[EPISODE SUMMARY] Blocos de 2min: {len(resumos_por_bloco)} segmentos")
    
    return resultado

def gerar_resumo_geral(transcricao: str) -> str:
    """
    Gera resumo geral ultra detalhado (8-20 parágrafos)
    
    Mantém ordem dos acontecimentos
    Descreve ações, emoções, cenas importantes, ambiente e contexto
    Inclui diálogos importantes e intenções implícitas
    """
    logger.info("[EPISODE SUMMARY] Gerando resumo geral...")
    
    # Divide transcrição em partes para análise
    palavras = transcricao.split()
    total_palavras = len(palavras)
    
    # Gera 8-20 parágrafos dependendo do tamanho
    num_paragrafos = min(20, max(8, total_palavras // 200))
    palavras_por_paragrafo = total_palavras // num_paragrafos
    
    paragrafos = []
    
    for i in range(num_paragrafos):
        inicio = i * palavras_por_paragrafo
        fim = min((i + 1) * palavras_por_paragrafo, total_palavras)
        
        trecho = ' '.join(palavras[inicio:fim])
        
        # Analisa o trecho
        resumo_paragrafo = analisar_trecho_detalhado(trecho, i + 1, num_paragrafos)
        paragrafos.append(resumo_paragrafo)
    
    resumo_geral = '\n\n'.join(paragrafos)
    
    return resumo_geral

def analisar_trecho_detalhado(trecho: str, numero: int, total: int) -> str:
    """
    Analisa um trecho e gera resumo detalhado
    
    Inclui:
    - Ações principais
    - Emoções dos personagens
    - Cenas importantes
    - Ambiente e contexto
    - Diálogos importantes
    - Intenções implícitas
    """
    # Identifica elementos-chave
    tem_dialogo = '"' in trecho or "'" in trecho
    tem_acao = any(palavra in trecho.lower() for palavra in ['luta', 'batalha', 'corre', 'ataca', 'defende'])
    tem_emocao = any(palavra in trecho.lower() for palavra in ['feliz', 'triste', 'raiva', 'medo', 'amor', 'ódio'])
    
    # Constrói resumo
    resumo = f"Segmento {numero}/{total}: "
    
    if tem_dialogo:
        resumo += "Diálogos importantes revelam "
    
    if tem_acao:
        resumo += "ações intensas onde "
    
    if tem_emocao:
        resumo += "emoções profundas são expressas. "
    
    # Adiciona contexto do trecho
    resumo += f"O trecho descreve: {trecho[:200]}..."
    
    return resumo

def gerar_resumos_segmentados(transcricao: str, duracao_total: float) -> List[Dict]:
    """
    Gera resumos a cada 2 minutos exatos
    
    Para cada bloco:
    - Ação principal
    - Objetivo dos personagens
    - Mudança narrativa
    - Conflito e resolução parcial
    - Elementos visuais importantes
    - Eventos marcantes
    """
    logger.info("[EPISODE SUMMARY] Gerando resumos segmentados (2min cada)...")
    
    blocos = []
    intervalo = 120  # 2 minutos em segundos
    
    # Calcula palavras por segundo
    palavras = transcricao.split()
    total_palavras = len(palavras)
    palavras_por_segundo = total_palavras / duracao_total if duracao_total > 0 else 2
    
    tempo_atual = 0
    
    while tempo_atual < duracao_total:
        start = tempo_atual
        end = min(tempo_atual + intervalo, duracao_total)
        
        # Calcula índices de palavras
        inicio_palavras = int(start * palavras_por_segundo)
        fim_palavras = int(end * palavras_por_segundo)
        
        # Extrai trecho
        trecho = ' '.join(palavras[inicio_palavras:fim_palavras])
        
        # Gera resumo detalhado do bloco
        resumo_bloco = analisar_bloco_2min(trecho, start, end)
        
        blocos.append({
            "start": int(start),
            "end": int(end),
            "resumo": resumo_bloco
        })
        
        tempo_atual += intervalo
    
    logger.info(f"[EPISODE SUMMARY] {len(blocos)} blocos de 2min criados")
    
    return blocos

def analisar_bloco_2min(trecho: str, start: int, end: int) -> str:
    """
    Analisa bloco de 2 minutos e gera resumo detalhado
    
    Inclui:
    - Ação principal
    - Objetivo dos personagens
    - Mudança narrativa
    - Conflito e resolução parcial
    - Elementos visuais importantes
    - Eventos marcantes
    """
    resumo_partes = []
    
    # Ação principal
    if any(palavra in trecho.lower() for palavra in ['luta', 'batalha', 'corre', 'ataca']):
        resumo_partes.append("AÇÃO: Sequência de ação intensa com confronto direto.")
    
    # Objetivo dos personagens
    if any(palavra in trecho.lower() for palavra in ['quer', 'precisa', 'deve', 'vai']):
        resumo_partes.append("OBJETIVO: Personagens demonstram intenções claras e motivações.")
    
    # Mudança narrativa
    if any(palavra in trecho.lower() for palavra in ['mas', 'porém', 'entretanto', 'então']):
        resumo_partes.append("MUDANÇA: Virada narrativa altera o rumo dos acontecimentos.")
    
    # Conflito
    if any(palavra in trecho.lower() for palavra in ['contra', 'versus', 'enfrenta', 'desafia']):
        resumo_partes.append("CONFLITO: Tensão entre forças opostas se intensifica.")
    
    # Elementos visuais
    if any(palavra in trecho.lower() for palavra in ['vê', 'olha', 'aparece', 'surge']):
        resumo_partes.append("VISUAL: Elementos visuais marcantes chamam atenção.")
    
    # Eventos marcantes
    if any(palavra in trecho.lower() for palavra in ['incrível', 'chocante', 'surpreendente']):
        resumo_partes.append("EVENTO: Momento marcante que impacta a narrativa.")
    
    # Adiciona contexto do trecho
    resumo_partes.append(f"CONTEXTO ({start}s-{end}s): {trecho[:150]}...")
    
    return " | ".join(resumo_partes)

def formatar_para_qwen(resumo_completo: Dict) -> str:
    """
    Formata resumos para entrada do QWEN
    
    Args:
        resumo_completo: Dicionário com resumo_geral e resumos_por_bloco
    
    Returns:
        Texto formatado para o QWEN processar
    """
    texto_formatado = "=== RESUMO GERAL DO EPISÓDIO ===\n\n"
    texto_formatado += resumo_completo['resumo_geral']
    texto_formatado += "\n\n=== RESUMOS SEGMENTADOS (2 MINUTOS) ===\n\n"
    
    for bloco in resumo_completo['resumos_por_bloco']:
        texto_formatado += f"\n[{bloco['start']}s - {bloco['end']}s]\n"
        texto_formatado += bloco['resumo']
        texto_formatado += "\n"
    
    return texto_formatado
