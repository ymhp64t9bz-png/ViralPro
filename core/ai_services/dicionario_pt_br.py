# -*- coding: utf-8 -*-
"""
DICIONÁRIO PORTUGUÊS BRASILEIRO - Para Correção Ortográfica
============================================================
Dicionário com palavras comuns em português para ajudar o Qwen
a gerar títulos com ortografia perfeita
"""

# Palavras comuns que o Qwen deve conhecer
DICIONARIO_PT_BR = {
    # Verbos comuns
    'verbos': [
        'ENFRENTA', 'VENCE', 'DERROTA', 'LUTA', 'BATALHA',
        'DESCOBRE', 'REVELA', 'TRANSFORMA', 'PROTEGE', 'SALVA',
        'ATACA', 'DEFENDE', 'CONQUISTA', 'DOMINA', 'CONTROLA',
        'CHEGA', 'PARTE', 'VOLTA', 'RETORNA', 'ENCONTRA',
        'PERDE', 'GANHA', 'RECEBE', 'DÁ', 'TOMA',
        'FOGE', 'PERSEGUE', 'ESCAPA', 'CAPTURA', 'LIBERTA'
    ],
    
    # Substantivos comuns
    'substantivos': [
        'HERÓI', 'VILÃO', 'GUERREIRO', 'MESTRE', 'ALUNO',
        'BATALHA', 'LUTA', 'COMBATE', 'DUELO', 'CONFRONTO',
        'PODER', 'FORÇA', 'ENERGIA', 'MAGIA', 'TÉCNICA',
        'SEGREDO', 'MISTÉRIO', 'REVELAÇÃO', 'VERDADE', 'MENTIRA',
        'AMIGO', 'INIMIGO', 'RIVAL', 'ALIADO', 'TRAIDOR',
        'MUNDO', 'REINO', 'IMPÉRIO', 'CIDADE', 'VILA',
        'DESTINO', 'FUTURO', 'PASSADO', 'PRESENTE', 'TEMPO',
        'VIDA', 'MORTE', 'ALMA', 'CORAÇÃO', 'MENTE'
    ],
    
    # Adjetivos comuns
    'adjetivos': [
        'ÉPICO', 'INCRÍVEL', 'PODEROSO', 'FORTE', 'FRACO',
        'GRANDE', 'PEQUENO', 'ENORME', 'GIGANTE', 'MINÚSCULO',
        'RÁPIDO', 'LENTO', 'VELOZ', 'ÁGIL', 'PESADO',
        'NOVO', 'VELHO', 'ANTIGO', 'MODERNO', 'ATUAL',
        'BOM', 'MAU', 'PERFEITO', 'IMPERFEITO', 'IDEAL',
        'VERDADEIRO', 'FALSO', 'REAL', 'IRREAL', 'IMPOSSÍVEL',
        'DIFÍCIL', 'FÁCIL', 'SIMPLES', 'COMPLEXO', 'COMPLICADO',
        'FELIZ', 'TRISTE', 'ALEGRE', 'SOMBRIO', 'ESCURO'
    ],
    
    # Palavras compostas e expressões
    'compostas': [
        'BEM-VINDO', 'BEM-VINDA', 'MAL-ESTAR', 'AUTO-ESTIMA',
        'SUPER-HERÓI', 'ANTI-HERÓI', 'SEMI-DEUS', 'EX-ALUNO',
        'VICE-LÍDER', 'CO-FUNDADOR', 'PRÉ-HISTÓRIA', 'PÓS-GUERRA'
    ],
    
    # Palavras específicas de anime
    'anime': [
        'JUTSU', 'CHAKRA', 'NINJA', 'SAMURAI', 'SENSEI',
        'SENPAI', 'KOUHAI', 'OTAKU', 'MANGA', 'ANIME',
        'YOKAI', 'SHINIGAMI', 'HOLLOW', 'QUINCY', 'BANKAI',
        'SHARINGAN', 'BYAKUGAN', 'RINNEGAN', 'SUSANOO', 'RASENGAN'
    ],
    
    # Correções específicas (palavra errada → palavra correta)
    'correcoes': {
        'GELADORES': 'ZELADORES',
        'BEMVINDO': 'BEM-VINDO',
        'BEMVINDA': 'BEM-VINDA',
        'HERÓÓI': 'HERÓI',
        'HEROI': 'HERÓI',
        'VILÃÃÃÃO': 'VILÃO',
        'VILAO': 'VILÃO',
        'INCRÍVEEL': 'INCRÍVEL',
        'INCRIVEL': 'INCRÍVEL',
        'ÉPICA': 'ÉPICA',
        'EPICA': 'ÉPICA',
        'ÉPICO': 'ÉPICO',
        'EPICO': 'ÉPICO',
        'MÁGICA': 'MÁGICA',
        'MAGICA': 'MÁGICA',
        'MÁGICO': 'MÁGICO',
        'MAGICO': 'MÁGICO',
        'ÚLTIMO': 'ÚLTIMO',
        'ULTIMO': 'ÚLTIMO',
        'ÚLTIMA': 'ÚLTIMA',
        'ULTIMA': 'ÚLTIMA',
        'PRÓXIMO': 'PRÓXIMO',
        'PROXIMO': 'PRÓXIMO',
        'PRÓXIMA': 'PRÓXIMA',
        'PROXIMA': 'PRÓXIMA',
        'HISTÓRIA': 'HISTÓRIA',
        'HISTORIA': 'HISTÓRIA',
        'VITÓRIA': 'VITÓRIA',
        'VITORIA': 'VITÓRIA',
        'GLÓRIA': 'GLÓRIA',
        'GLORIA': 'GLÓRIA',
        'MEMÓRIA': 'MEMÓRIA',
        'MEMORIA': 'MEMÓRIA'
    }
}

def get_all_words():
    """Retorna todas as palavras do dicionário"""
    all_words = set()
    
    for category in ['verbos', 'substantivos', 'adjetivos', 'compostas', 'anime']:
        all_words.update(DICIONARIO_PT_BR[category])
    
    return sorted(list(all_words))

def correct_word(word):
    """
    Corrige palavra usando dicionário de correções
    
    Args:
        word: Palavra em maiúsculas
    
    Returns:
        Palavra corrigida ou original se não houver correção
    """
    word_upper = word.upper()
    
    # Verifica se tem correção específica
    if word_upper in DICIONARIO_PT_BR['correcoes']:
        corrected = DICIONARIO_PT_BR['correcoes'][word_upper]
        return corrected
    
    return word

def correct_title(title):
    """
    Corrige título palavra por palavra usando dicionário
    
    Args:
        title: Título em maiúsculas
    
    Returns:
        Título corrigido
    """
    words = title.split()
    corrected_words = []
    
    for word in words:
        corrected = correct_word(word)
        corrected_words.append(corrected)
    
    return ' '.join(corrected_words)

def is_valid_word(word):
    """
    Verifica se palavra está no dicionário
    
    Args:
        word: Palavra em maiúsculas
    
    Returns:
        True se palavra é válida
    """
    word_upper = word.upper()
    
    # Verifica em todas as categorias
    for category in ['verbos', 'substantivos', 'adjetivos', 'compostas', 'anime']:
        if word_upper in DICIONARIO_PT_BR[category]:
            return True
    
    # Verifica se é uma correção conhecida
    if word_upper in DICIONARIO_PT_BR['correcoes'].values():
        return True
    
    return False

def get_suggestions(word):
    """
    Retorna sugestões de correção para palavra
    
    Args:
        word: Palavra possivelmente errada
    
    Returns:
        Lista de sugestões
    """
    word_upper = word.upper()
    suggestions = []
    
    # Verifica correções diretas
    if word_upper in DICIONARIO_PT_BR['correcoes']:
        suggestions.append(DICIONARIO_PT_BR['correcoes'][word_upper])
        return suggestions
    
    # Busca palavras similares (distância de edição simples)
    all_words = get_all_words()
    
    for dict_word in all_words:
        # Similaridade simples: mesma primeira letra e tamanho similar
        if (dict_word[0] == word_upper[0] and 
            abs(len(dict_word) - len(word_upper)) <= 2):
            suggestions.append(dict_word)
    
    return suggestions[:5]  # Retorna até 5 sugestões

# Exporta dicionário como texto para uso do Qwen
def export_dictionary_text():
    """
    Exporta dicionário como texto formatado para uso em prompts
    
    Returns:
        str: Dicionário formatado
    """
    text = "DICIONÁRIO PORTUGUÊS BRASILEIRO:\n\n"
    
    text += "Palavras corretas (use estas):\n"
    all_words = get_all_words()
    text += ", ".join(all_words[:50])  # Primeiras 50 palavras
    text += "\n\n"
    
    text += "Correções importantes:\n"
    for wrong, correct in list(DICIONARIO_PT_BR['correcoes'].items())[:20]:
        text += f"- {wrong} → {correct}\n"
    
    return text

if __name__ == "__main__":
    # Testes
    print("=== DICIONÁRIO PORTUGUÊS BRASILEIRO ===\n")
    
    print(f"Total de palavras: {len(get_all_words())}\n")
    
    print("Testando correções:")
    test_words = ['GELADORES', 'BEMVINDO', 'HERÓÓI', 'INCRÍVEEL']
    for word in test_words:
        corrected = correct_word(word)
        print(f"  {word} → {corrected}")
    
    print("\nTestando título completo:")
    title = "BEMVINDO AO QUARTEL DOS GELADORES"
    corrected_title = correct_title(title)
    print(f"  Original: {title}")
    print(f"  Corrigido: {corrected_title}")
    
    print("\nSugestões para 'GELADORES':")
    suggestions = get_suggestions('GELADORES')
    for sug in suggestions:
        print(f"  - {sug}")
