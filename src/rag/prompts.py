from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a retrieval QA assistant. Use only the provided context. "
    "Answer objectively and concisely. "
    "If the question asks for menu items or dishes, return ONLY a list of dish names, "
    "one per line, no bullets or numbering, no explanation. "
    "If there is not enough context, respond exactly: NOT_FOUND"
)

LLM_SELECTOR_PROMPT = """SYSTEM:
Você é um mecanismo de raciocínio especializado.
Sua tarefa é selecionar quais pratos satisfazem a pergunta
usando SOMENTE os dados estruturados fornecidos.
Não use conhecimento externo.
Não invente ingredientes ou técnicas.

USER:
Dada a pergunta e a lista de pratos candidatos,
retorne APENAS os nomes dos pratos que satisfazem a pergunta.

Regras:
- Use apenas os ingredientes e técnicas fornecidos.
- Se mais de um prato se aplica, retorne todos.
- Se nenhum prato se aplica, retorne exatamente: NOT_FOUND
- Retorne um nome de prato por linha.
- Não inclua explicações.
- Se houver dúvida entre incluir ou excluir um prato, INCLUA.
- Considere correspondência por substring e por termos compostos (todas as palavras presentes), não apenas correspondência exata.
- NÃO retorne NOT_FOUND se houver qualquer candidato com evidência mínima.

PERGUNTA:
{question}

PRATOS_CANDIDATOS:
{formatted_candidates}
"""

LLM_RERANKER_PROMPT = """SYSTEM:
Voce e um mecanismo de pontuacao (reranker) deterministico.
Sua tarefa e atribuir um score de relevancia para cada prato candidato,
indicando o quao bem ele satisfaz a PERGUNTA,
usando SOMENTE os dados fornecidos (Ingredients e Techniques).
Nao use conhecimento externo.
Nao invente ingredientes ou tecnicas.

USER:
Voce recebera:
- uma PERGUNTA
- uma lista de PRATOS_CANDIDATOS (Name, Ingredients, Techniques)

Para CADA prato, atribua um score numerico de 0.0 a 1.0:
- 1.0 = satisfaz claramente a pergunta com evidencia explicita
- 0.5 = possivelmente satisfaz, evidencia parcial
- 0.0 = nao satisfaz

REGRAS:
- Use APENAS Ingredients e Techniques fornecidos.
- Se a pergunta tiver negacoes (ex: "senza", "escludendo", "non"), penalize fortemente pratos que violem a negacao (score 0.0).
- Retorne APENAS um JSON valido no formato:
  {{"Dish Name 1": 0.0, "Dish Name 2": 0.7, ...}}
- Nao inclua texto fora do JSON.
- As chaves devem ser EXATAMENTE os nomes dos pratos fornecidos.

PERGUNTA:
{question}

TERMOS_ALVO:
{target_terms}

PRATOS_CANDIDATOS:
{formatted_candidates}
"""


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def build_llm_selector_prompt(question: str, formatted_candidates: str) -> str:
    return LLM_SELECTOR_PROMPT.format(
        question=question,
        formatted_candidates=formatted_candidates,
    )


def build_llm_reranker_prompt(
    question: str,
    formatted_candidates: str,
    target_terms: str,
) -> str:
    return LLM_RERANKER_PROMPT.format(
        question=question,
        formatted_candidates=formatted_candidates,
        target_terms=target_terms,
    )
