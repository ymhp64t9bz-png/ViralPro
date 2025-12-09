#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORRETOR DE TÍTULOS - AUTOCORTES
=================================
Sistema de validação e correção automática de títulos gerados por LLaMA.

Solução para problemas de:
- Letras trocadas
- Letras faltando
- Erros gramaticais
- Palavras incorretas

Estratégia:
1. Gerar título com LLaMA
2. Validar ortografia
3. Se tiver erros, tentar corrigir
4. Se não conseguir corrigir, gerar novamente
5. Máximo 3 tentativas
6. Fallback para Gemini se falhar
"""

import re
import logging
from typing import Optional, Dict, List
import gc

logger = logging.getLogger(__name__)

# ==================== DICIONÁRIO DE PALAVRAS COMUNS ====================

PALAVRAS_VIRAIS = {
    # Ações
    "BATALHA", "LUTA", "COMBATE", "GUERRA", "ATAQUE", "DEFESA",
    "VITÓRIA", "DERROTA", "VINGANÇA", "CONFRONTO", "DUELO",
    
    # Adjetivos
    "ÉPICO", "ÉPICA", "INCRÍVEL", "INSANO", "BRUTAL", "PODEROSO",
    "PODEROSA", "FORTE", "MORTAL", "LENDÁRIO", "LENDÁRIA",
    
    # Substantivos
    "HERÓI", "VILÃO", "PODER", "FORÇA", "MAGIA", "TÉCNICA",
    "HABILIDADE", "TRANSFORMAÇÃO", "EVOLUÇÃO", "DESPERTAR",
    
    # Verbos
    "DERROTA", "VENCE", "DESTROI", "ANIQUILA", "DOMINA",
    "REVELA", "DESCOBRE", "CONQUISTA", "LIBERA", "ATIVA",
    
    # Conectores
    "CONTRA", "VERSUS", "VS", "COM", "SEM", "PARA", "POR",
    
    # Intensificadores
    "MUITO", "SUPER", "ULTRA", "MEGA", "HIPER", "EXTREMO",
    "TOTAL", "COMPLETO", "ABSOLUTO", "FINAL", "DEFINITIVO"
}

PALAVRAS_PROIBIDAS = {
    # Palavras que indicam erro
    "UNDEFINED", "NULL", "ERROR", "NONE", "NAN",
    
    # Caracteres estranhos
    "###", "***", "...", "???", "!!!"
}

# ==================== VALIDADOR DE ORTOGRAFIA ====================

class TitleValidator:
    """Valida e corrige títulos"""
    
    def __init__(self):
        self.max_tentativas = 3
        
    def validar_titulo(self, titulo: str) -> Dict[str, any]:
        """
        Valida título e retorna diagnóstico
        
        Returns:
            {
                "valido": bool,
                "erros": List[str],
                "score": float (0-100)
            }
        """
        erros = []
        score = 100.0
        
        # 1. Verificar se está vazio
        if not titulo or len(titulo.strip()) == 0:
            return {
                "valido": False,
                "erros": ["Título vazio"],
                "score": 0.0
            }
        
        titulo = titulo.strip().upper()
        
        # 2. Verificar tamanho (4-9 palavras)
        palavras = titulo.split()
        num_palavras = len(palavras)
        
        if num_palavras < 4:
            erros.append(f"Muito curto ({num_palavras} palavras)")
            score -= 30
        elif num_palavras > 9:
            erros.append(f"Muito longo ({num_palavras} palavras)")
            score -= 20
        
        # 3. Verificar palavras proibidas
        for palavra_proibida in PALAVRAS_PROIBIDAS:
            if palavra_proibida in titulo:
                erros.append(f"Palavra proibida: {palavra_proibida}")
                score -= 50
        
        # 4. Verificar caracteres estranhos
        if re.search(r'[^A-ZÁÀÂÃÉÊÍÓÔÕÚÇ\s!?]', titulo):
            erros.append("Caracteres estranhos detectados")
            score -= 30
        
        # 5. Verificar palavras com letras repetidas (erro comum do LLaMA)
        for palavra in palavras:
            # Detectar padrões como "BATAAALHA", "HERÓÓÓI"
            if re.search(r'(.)\1{2,}', palavra):
                erros.append(f"Letras repetidas em: {palavra}")
                score -= 20
        
        # 6. Verificar palavras muito curtas (possível letra faltando)
        for palavra in palavras:
            if len(palavra) == 1 and palavra not in ['E', 'O', 'A']:
                erros.append(f"Palavra muito curta: {palavra}")
                score -= 15
        
        # 7. Verificar se tem pelo menos uma palavra viral conhecida
        tem_palavra_viral = any(pv in titulo for pv in PALAVRAS_VIRAIS)
        if not tem_palavra_viral:
            erros.append("Nenhuma palavra viral conhecida")
            score -= 10
        
        # 8. Verificar espaços duplos
        if '  ' in titulo:
            erros.append("Espaços duplos detectados")
            score -= 5
        
        score = max(0.0, score)
        valido = score >= 70.0 and len(erros) == 0
        
        return {
            "valido": valido,
            "erros": erros,
            "score": score
        }
    
    def corrigir_titulo(self, titulo: str) -> str:
        """
        Tenta corrigir erros simples no título
        """
        if not titulo:
            return titulo
        
        titulo = titulo.strip().upper()
        
        # 1. Remover espaços duplos
        titulo = re.sub(r'\s+', ' ', titulo)
        
        # 2. Remover caracteres estranhos
        titulo = re.sub(r'[^A-ZÁÀÂÃÉÊÍÓÔÕÚÇ\s!?]', '', titulo)
        
        # 3. Corrigir letras repetidas (BATAAALHA -> BATALHA)
        palavras = titulo.split()
        palavras_corrigidas = []
        
        for palavra in palavras:
            # Reduzir letras repetidas (máximo 2)
            palavra_corrigida = re.sub(r'(.)\1{2,}', r'\1\1', palavra)
            
            # Se ficou muito diferente, tentar encontrar palavra similar
            if len(palavra_corrigida) < len(palavra) * 0.7:
                # Buscar palavra similar no dicionário
                palavra_similar = self._encontrar_palavra_similar(palavra_corrigida)
                if palavra_similar:
                    palavra_corrigida = palavra_similar
            
            palavras_corrigidas.append(palavra_corrigida)
        
        titulo_corrigido = ' '.join(palavras_corrigidas)
        
        # 4. Limitar a 9 palavras
        palavras_finais = titulo_corrigido.split()[:9]
        titulo_final = ' '.join(palavras_finais)
        
        return titulo_final
    
    def _encontrar_palavra_similar(self, palavra: str) -> Optional[str]:
        """Encontra palavra similar no dicionário"""
        palavra = palavra.upper()
        
        # Busca exata
        if palavra in PALAVRAS_VIRAIS:
            return palavra
        
        # Busca por substring (palavra pode estar incompleta)
        for palavra_viral in PALAVRAS_VIRAIS:
            if palavra in palavra_viral or palavra_viral in palavra:
                return palavra_viral
        
        # Busca por similaridade (Levenshtein simplificado)
        melhor_match = None
        menor_distancia = float('inf')
        
        for palavra_viral in PALAVRAS_VIRAIS:
            distancia = self._levenshtein_distance(palavra, palavra_viral)
            if distancia < menor_distancia and distancia <= 2:
                menor_distancia = distancia
                melhor_match = palavra_viral
        
        return melhor_match
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calcula distância de Levenshtein (simplificado)"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

# ==================== GERADOR COM VALIDAÇÃO ====================

def generate_viral_title_validated(
    llama_model,
    prompt: str,
    max_tokens: int = 50,
    max_tentativas: int = 3,
    gemini_fallback: bool = True
) -> str:
    """
    Gera título com validação e correção automática
    
    Args:
        llama_model: Modelo LLaMA carregado
        prompt: Prompt para geração
        max_tokens: Máximo de tokens
        max_tentativas: Máximo de tentativas
        gemini_fallback: Usar Gemini como fallback
    
    Returns:
        str: Título validado e corrigido
    """
    validator = TitleValidator()
    
    logger.info("[TÍTULO] Gerando título com validação...")
    
    for tentativa in range(1, max_tentativas + 1):
        logger.info(f"[TÍTULO] Tentativa {tentativa}/{max_tentativas}")
        
        # Gerar título com LLaMA
        titulo = _gerar_com_llama(llama_model, prompt, max_tokens)
        
        logger.info(f"[TÍTULO] Gerado: {titulo}")
        
        # Validar
        resultado = validator.validar_titulo(titulo)
        
        logger.info(f"[TÍTULO] Score: {resultado['score']:.1f}%")
        
        if resultado['erros']:
            logger.warning(f"[TÍTULO] Erros: {', '.join(resultado['erros'])}")
        
        # Se válido, retornar
        if resultado['valido']:
            logger.info(f"[TÍTULO] ✅ Título válido!")
            return titulo
        
        # Se não válido, tentar corrigir
        titulo_corrigido = validator.corrigir_titulo(titulo)
        
        logger.info(f"[TÍTULO] Corrigido: {titulo_corrigido}")
        
        # Validar correção
        resultado_corrigido = validator.validar_titulo(titulo_corrigido)
        
        if resultado_corrigido['valido']:
            logger.info(f"[TÍTULO] ✅ Correção bem-sucedida!")
            return titulo_corrigido
        
        # Se ainda não válido, tentar novamente
        logger.warning(f"[TÍTULO] ⚠️  Tentativa {tentativa} falhou")
        
        # Limpar memória entre tentativas
        gc.collect()
    
    # Se todas as tentativas falharam, usar fallback
    logger.error(f"[TÍTULO] ❌ Todas as {max_tentativas} tentativas falharam")
    
    if gemini_fallback:
        logger.info("[TÍTULO] Usando Gemini como fallback...")
        try:
            titulo_gemini = _gerar_com_gemini(prompt)
            logger.info(f"[TÍTULO] ✅ Gemini: {titulo_gemini}")
            return titulo_gemini
        except Exception as e:
            logger.error(f"[TÍTULO] ❌ Gemini falhou: {e}")
    
    # Último recurso: título genérico
    logger.warning("[TÍTULO] Usando título genérico")
    return "MOMENTO ÉPICO DO ANIME"

def _gerar_com_llama(model, prompt: str, max_tokens: int) -> str:
    """Gera título com LLaMA"""
    try:
        # Prompt otimizado para evitar erros
        system_prompt = """Você é um especialista em criar títulos virais.

REGRAS OBRIGATÓRIAS:
1. Máximo 9 palavras
2. TUDO EM MAIÚSCULAS
3. Sem emojis
4. Português correto
5. Palavras completas (não abrevie)
6. Verifique a ortografia

IMPORTANTE: Escreva cada palavra CORRETAMENTE, sem trocar ou omitir letras."""

        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Gerar com parâmetros otimizados
        response = model(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.7,  # Menos criativo = menos erros
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.2,  # Evitar repetições
            stop=["<|eot_id|>", "<|end_of_text|>", "\n", "."],
            echo=False
        )
        
        titulo = response['choices'][0]['text'].strip().upper()
        
        # Limpar
        titulo = titulo.replace('"', '').replace("'", '').strip()
        
        return titulo
        
    except Exception as e:
        logger.error(f"[LLAMA] Erro: {e}")
        return ""

def _gerar_com_gemini(prompt: str) -> str:
    """Gera título com Gemini (fallback)"""
    try:
        import google.generativeai as genai
        import os
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise Exception("GEMINI_API_KEY não configurado")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        gemini_prompt = f"""Crie um título viral para vídeo baseado em: {prompt}

REGRAS:
- Máximo 9 palavras
- TUDO EM MAIÚSCULAS
- Português do Brasil
- Sem emojis
- Impactante e chamativo

Responda APENAS com o título, nada mais."""
        
        response = model.generate_content(gemini_prompt)
        titulo = response.text.strip().upper()
        
        # Limpar
        titulo = titulo.replace('"', '').replace("'", '').strip()
        
        return titulo
        
    except Exception as e:
        logger.error(f"[GEMINI] Erro: {e}")
        raise

# ==================== EXEMPLO DE USO ====================

if __name__ == "__main__":
    # Teste
    validator = TitleValidator()
    
    # Títulos com erros
    titulos_teste = [
        "BATAAALHA ÉPICA CONTRA VILÃO",  # Letras repetidas
        "HERÓÓI VENCE",  # Muito curto + letras repetidas
        "LUTA INCRÍVEL SUPER PODEROSA DEFINITIVA TOTAL ABSOLUTA FINAL",  # Muito longo
        "BATALHA ÉPICA CONTRA VILÃO PODEROSO",  # Correto
        "",  # Vazio
        "UNDEFINED ERROR NULL",  # Palavras proibidas
    ]
    
    for titulo in titulos_teste:
        print(f"\nTítulo: {titulo}")
        resultado = validator.validar_titulo(titulo)
        print(f"Válido: {resultado['valido']}")
        print(f"Score: {resultado['score']:.1f}%")
        print(f"Erros: {resultado['erros']}")
        
        if not resultado['valido']:
            corrigido = validator.corrigir_titulo(titulo)
            print(f"Corrigido: {corrigido}")
            resultado_corrigido = validator.validar_titulo(corrigido)
            print(f"Score corrigido: {resultado_corrigido['score']:.1f}%")
