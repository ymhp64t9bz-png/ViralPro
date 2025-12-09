# -*- coding: utf-8 -*-
"""
CORRETOR ORTOGRÁFICO COM LANGUAGETOOL
======================================
Corretor profissional para títulos gerados por LLaMA.
Versão Robusta: Fallback para API pública se Java não estiver instalado.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Variável global para instância do LanguageTool
_LANGUAGETOOL_INSTANCE = None
_LIB_AVAILABLE = False

# Tenta importação no nível do módulo
try:
    import language_tool_python
    _LIB_AVAILABLE = True
except ImportError as e:
    logger.error(f"[LANGUAGETOOL] ❌ Biblioteca 'language-tool-python' não encontrada: {e}")
    _LIB_AVAILABLE = False
except Exception as e:
    logger.error(f"[LANGUAGETOOL] ❌ Erro ao importar biblioteca: {e}")
    _LIB_AVAILABLE = False

def load_languagetool():
    """Carrega LanguageTool com fallback para API pública"""
    global _LANGUAGETOOL_INSTANCE
    
    if _LANGUAGETOOL_INSTANCE is not None:
        return _LANGUAGETOOL_INSTANCE
    
    if not _LIB_AVAILABLE:
        logger.error("[LANGUAGETOOL] ❌ Biblioteca não disponível.")
        return None
    
    # 1. Tenta rodar localmente (precisa de Java)
    try:
        logger.info("[LANGUAGETOOL] Tentando carregar corretor local (precisa de Java)...")
        _LANGUAGETOOL_INSTANCE = language_tool_python.LanguageTool('pt-BR')
        logger.info("[LANGUAGETOOL] ✅ Corretor local carregado!")
        return _LANGUAGETOOL_INSTANCE
        
    except Exception as e_local:
        # Se falhar (provavelmente sem Java), tenta API pública
        logger.warning(f"[LANGUAGETOOL] ⚠️ Falha ao carregar local (provavelmente sem Java): {e_local}")
        logger.info("[LANGUAGETOOL] 🌍 Tentando usar API pública como fallback...")
        
        try:
            # Fallback para API pública
            _LANGUAGETOOL_INSTANCE = language_tool_python.LanguageTool('pt-BR', remote_server='https://api.languagetool.org/v2/')
            logger.info("[LANGUAGETOOL] ✅ Conectado à API pública com sucesso!")
            return _LANGUAGETOOL_INSTANCE
            
        except Exception as e_remote:
            logger.error(f"[LANGUAGETOOL] ❌ Falha total (Local e Remoto): {e_remote}")
            return None

def corrigir_titulo_languagetool(titulo: str) -> Dict[str, any]:
    """Corrige título usando LanguageTool com tratamento de erros robusto"""
    try:
        tool = load_languagetool()
        
        if tool is None:
            return {
                "original": titulo,
                "corrigido": titulo,
                "erros_encontrados": 0,
                "correcoes": []
            }
        
        logger.info(f"[LANGUAGETOOL] Analisando: {titulo}")
        
        # Verifica erros
        matches = tool.check(titulo)
        
        if not matches:
            logger.info("[LANGUAGETOOL] ✅ Nenhum erro encontrado")
            return {
                "original": titulo,
                "corrigido": titulo,
                "erros_encontrados": 0,
                "correcoes": []
            }
        
        # Aplica correções
        titulo_corrigido = tool.correct(titulo)
        
        # Lista de correções para log
        correcoes = []
        for match in matches:
            if match.replacements:
                correcao = f"{match.context} → {match.replacements[0]}"
                correcoes.append(correcao)
        
        logger.info(f"[LANGUAGETOOL] ✅ {len(matches)} erro(s) corrigido(s)")
        logger.info(f"[LANGUAGETOOL] Original: {titulo}")
        logger.info(f"[LANGUAGETOOL] Corrigido: {titulo_corrigido}")
        
        return {
            "original": titulo,
            "corrigido": titulo_corrigido,
            "erros_encontrados": len(matches),
            "correcoes": correcoes
        }
        
    except Exception as e:
        logger.error(f"[LANGUAGETOOL] ❌ Erro durante correção: {e}")
        return {
            "original": titulo,
            "corrigido": titulo,
            "erros_encontrados": 0,
            "correcoes": []
        }

def corrigir_titulo_completo(titulo: str) -> str:
    """Função simplificada"""
    resultado = corrigir_titulo_languagetool(titulo)
    return resultado["corrigido"]
