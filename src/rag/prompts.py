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
