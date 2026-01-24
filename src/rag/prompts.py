from __future__ import annotations

SYSTEM_PROMPT = (
    "Sei un assistente QA basato su retrieval. Usa solo il contesto fornito. "
    "Rispondi in modo oggettivo e conciso. "
    "Se la domanda riguarda voci di menu o piatti, restituisci SOLO un elenco di nomi di piatti, "
    "uno per riga, senza punti elenco o numerazione, senza spiegazioni. "
    "Se il contesto non e sufficiente, rispondi esattamente: NOT_FOUND"
)

LLM_SELECTOR_PROMPT = """SYSTEM:
Sei un motore di ragionamento specializzato.
Il tuo compito e selezionare quali piatti soddisfano la domanda
usando SOLO i dati strutturati forniti.
Non usare conoscenza esterna.
Non inventare ingredienti o tecniche.

USER:
Data la domanda e la lista di piatti candidati,
restituisci SOLO i nomi dei piatti che soddisfano la domanda.

Regole:
- Usa solo gli ingredienti e le tecniche fornite.
- Se piu di un piatto si applica, restituiscili tutti.
- Se nessun piatto si applica, restituisci esattamente: NOT_FOUND
- Restituisci un nome di piatto per riga.
- Non includere spiegazioni.
- Se hai dubbi tra includere o escludere un piatto, INCLUDI.
- Considera corrispondenza per substring e per termini composti (tutte le parole presenti), non solo corrispondenza esatta.
- NON restituire NOT_FOUND se esiste almeno un candidato con evidenza minima.

DOMANDA:
{question}

PIATTI_CANDIDATI:
{formatted_candidates}
"""


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(contexts)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Contesto:\n{context_block}\n\n"
        f"Domanda: {question}\n"
        "Risposta:"
    )


def build_llm_selector_prompt(question: str, formatted_candidates: str) -> str:
    return LLM_SELECTOR_PROMPT.format(
        question=question,
        formatted_candidates=formatted_candidates,
    )
