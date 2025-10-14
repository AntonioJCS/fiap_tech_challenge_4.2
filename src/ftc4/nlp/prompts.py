FICHAMENTO_PROMPT = """Você é um assistente didático. Dado o conteúdo transcrito abaixo, gere um fichamento em Markdown com as seções:
# Título
## Resumo
## Tópicos-Chave
## Exemplos/Metáforas
## Glossário
## Ações/Aplicações Práticas

Regras:
- Linguagem clara e objetiva em pt-BR.
- Não invente conteúdo fora da transcrição.
- Seja conciso e organizado.

CONTEÚDO:
{transcript}
"""

QUIZ_PROMPT = """Crie um quiz de {n} questões de múltipla escolha (4 alternativas), em pt-BR, sobre o conteúdo abaixo.
Para cada questão, retorne JSON com:
- pergunta
- alternativas [A, B, C, D]
- correta ("A"/"B"/"C"/"D")
- explicacao (uma linha)

Use apenas informações do conteúdo.

CONTEÚDO:
{transcript}
"""
