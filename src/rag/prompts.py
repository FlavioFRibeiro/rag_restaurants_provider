from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a retrieval QA assistant. Use only the provided context. "
    "Answer objectively and concisely. "
    "If the question asks for menu items or dishes, return ONLY a list of dish names, "
    "one per line, no bullets or numbering, no explanation. "
    "If there is not enough context, respond exactly: NOT_FOUND"
)


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
