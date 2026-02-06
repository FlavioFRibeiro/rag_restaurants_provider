### Technical Documentation
- Autore : Flavio Ribeiro
- Versione: 1.0.0
- data dell'ultima modifica: 25/01/2026
---

Questo documento descrive l’architettura del sistema, le scelte tecniche adottate, i trade-off e i miglioramenti possibili.  
L’obiettivo è rendere il sistema spiegabile, riproducibile e facilmente debuggabile.

---

## File per categoria

- **Ingestion**: `build_index.py`, `pdf_loader.py`, `io.py`
- **Chunking**: `chunker.py`, `text.py`
- **Parsing (menu)**: `menu_parser.py`
- **Build index**: `build_index.py`
- **Retrieval**: `retriever.py`, `bm25_retriever.py`, `pipeline.py`
- **Matching deterministico (Easy-first)**: `pipeline.py`
- **LLM selector**: `llm.py`, `prompts.py`, `pipeline.py`
- **Normalizzazione / sanitizzazione**: `text.py`, `pipeline.py`, `metrics.py`
- **Valutazione**: `run_eval.py`, `metrics.py`
- **Strumenti di diagnostica (fuori dal pipeline)**:
  - `debug_easy.py`
  - `debug_llm_reasoning.py`
  - `debug_title_selection.py`
  - `generate_easy_diagnostics.py`
  - `smoke_titles.py`
- **Esperimento non integrato**:
  - `easy_query_parser.py` (non viene chiamato dal pipeline attuale)

---

## Visione generale del pipeline

- `data/raw` → `build_index.py` → `data/processed/index`  
  (FAISS, `documents.jsonl`, `embed_config.json`) + `menu_dishes.jsonl`

- `.csv` + `.csv` → `run_eval.py` → `pipeline.py` →  
  `predictions.csv` + `eval_report.json`

- L’uso dell’LLM è **opzionale**:
  - `ENABLE_LLM_REASONING` controlla il selector basato su LLM
  - `--use-llm` controlla il retrieval denso (RAG) quando non esiste una risposta deterministica
---

## Decisioni e Trade-off

| Decisione | Implementazione | Trade-off |
|---------|------------------|-----------|
| Parsing deterministico dei menu | `menu_parser.py` | Più spiegabile; sensibile alla formattazione |
| BM25 per i menu | `bm25_retriever.py` | Migliore per termini rari; assenza di semantica |
| FAISS per altri documenti | `retriever.py` | Buona copertura generale; richiede embedding |
| Easy-first deterministico | `pipeline.py` | Riproducibile; non copre parafrasi |
| LLM selector con gating | `pipeline.py`, `prompts.py` | Utile in presenza di rumore; dipende da prompt e cache |
| F1 basato su overlap | `metrics.py` | Semplice; non misura rilevanza parziale fuori da risposte a lista |

---

## Ingestion

**Cosa esiste nel codice**  
Lettura dei PDF tramite la funzione `load_pdf_text` in `pdf_loader.py` (utilizzando `pypdf` con fallback su `pdfplumber`), lettura di file di testo e CSV tramite `read_text` in `io.py`, scansione dei file e instradamento del flusso in `build_index.py`.

**Decisione tecnica (e motivazione)**  
È stato scelto un parser locale e leggero con meccanismo di fallback; i file CSV vengono trattati come testo non strutturato per mantenere il sistema semplice e robusto in modalità offline.

**Pro**
- Facile da eseguire in locale  
- Nessuna dipendenza da servizi esterni  
- Fallback disponibile per PDF problematici  

**Contro**
- I CSV non vengono trattati come dati strutturati  
- La qualità dell’estrazione dipende dalla qualità dei PDF  
- Identificazione del tipo di documento basata su euristiche (nome del file)  

**Rischi e mitigazione**  
PDF vuoti o di bassa qualità possono produrre testo vuoto; il problema è mitigato tramite logging e skip dei file troppo corti.

**Miglioramento piccolo e reversibile**  
Introduzione di una flag `INGEST_MIN_CHARS` per definire un numero minimo di caratteri prima di indicizzare un documento.

---

## Chunking

**Cosa esiste nel codice**  
Il chunking viene effettuato per numero di caratteri in `text.py`, con un wrapper in `chunker.py`. I valori di default sono 800 caratteri per chunk e 120 caratteri di overlap.

**Decisione tecnica (e motivazione)**  
È stato scelto un chunking semplice e deterministico, adeguato ai tempi ridotti del progetto e facile da debuggare.

**Pro**
- Comportamento prevedibile  
- Basso costo computazionale  
- Applicabile a qualsiasi tipo di testo  

**Contro**
- Può interrompere frasi  
- Non tiene conto della struttura semantica  
- Può spezzare tabelle o liste  

**Rischi e mitigazione**  
Possibile perdita di contesto ai bordi dei chunk; il rischio è mitigato tramite overlap e validazioni sui parametri.

**Miglioramento piccolo e reversibile**  
Introduzione di una flag `CHUNK_STRIP_HEADERS=1` per rimuovere intestazioni e piè di pagina ripetuti prima del chunking.

---

## Parsing (menu)

**Cosa esiste nel codice**  
È presente una state machine in `menu_parser.py` basata su header come *Ingredienti* e *Tecniche*; in assenza di questi, viene utilizzato un fallback narrativo con liste di `known_ingredients` e `known_techniques`.  
Il comportamento è controllato da flag di ambiente come `PARSER_TITLE_SCORING`, `PARSER_JOIN_TITLE` e `NORMALIZE_TRAILING_VOWEL`.

**Decisione tecnica (e motivazione)**  
È stato adottato un parsing deterministico con euristiche controllate per estrarre una struttura coerente da menu in formato narrativo.

**Pro**
- Approccio spiegabile  
- Funziona offline  
- Separazione stabile tra piatti, ingredienti e tecniche  

**Contro**
- Sensibile a variazioni di formattazione  
- Dipendenza dalla presenza di header espliciti  
- Scarta piatti privi di ingredienti identificabili  

**Rischi e mitigazione**  
Possibili falsi positivi dovuti a titoli non corretti; il rischio è mitigato tramite filtri sui titoli, limiti di lunghezza e meccanismi di scoring.

**Miglioramento piccolo e reversibile**  
Introduzione di una flag `PARSER_SKIP_PRICE_LINES=1` per ignorare righe che sembrano indicare prezzi (ad esempio pattern `^prezzo`).

---

## Build Index

**Cosa esiste nel codice**  
Il file `build_index.py` genera gli artefatti `documents.jsonl`, `faiss.index`, `embed_config.json` e `menu_dishes.jsonl`.  
Gli embeddings vengono calcolati tramite `SentenceTransformer` e indicizzati con un `IndexFlatIP` normalizzato.

**Decisione tecnica (e motivazione)**  
È stato adottato un indice ibrido semplice: da un lato i menu strutturati, dall’altro chunk densi per gli altri documenti.

**Pro**
- Artefatti chiari e ben separati  
- Menu e documenti generici non vengono mescolati  
- Processo riproducibile  

**Contro**
- Rilevamento dei menu basato su nome o cartella  
- Rebuild completo dell’indice a ogni esecuzione  

**Rischi e mitigazione**  
Possibile presenza di piatti duplicati; il rischio è mitigato tramite deduplicazione basata su `dish_name` e `source_file`.  
I menu privi di header vengono tracciati tramite log.

**Miglioramento piccolo e reversibile**  
Introduzione di una flag `MENU_HINTS` per fornire una lista di token aggiuntivi a supporto del rilevamento dei menu.

---

## Retrieval

**Cosa esiste nel codice**  
Il retrieval denso è implementato tramite FAISS in `retriever.py` (con `top_k=6` di default), mentre per i piatti viene utilizzato BM25 in `bm25_retriever.py`.  
La deduplicazione dei nomi avviene nella funzione `_get_bm25_top_dishes` in `pipeline.py`.

**Decisione tecnica (e motivazione)**  
BM25 viene utilizzato per gestire termini rari e specifici dei menu, mentre FAISS è impiegato per il recupero di contesto più generale quando necessario.

**Pro**
- BM25 gestisce bene termini rari e nomi specifici  
- FAISS è veloce ed eseguibile localmente  
- Percorsi di retrieval separati riducono ambiguità  

**Contro**
- Presenza di due indici distinti  
- L’indice BM25 viene ricostruito in memoria a ogni esecuzione  

**Rischi e mitigazione**  
Possibili divergenze tra il modello di embedding e l’indice; il rischio è mitigato tramite `embed_config.json`.  
La deduplicazione riduce la ripetizione dei risultati.

**Miglioramento piccolo e reversibile**  
Introduzione di una flag `BM25_TOP_K` per regolare il numero di candidati nel bucket Easy senza modificare l’algoritmo.

---

## Matching deterministico (Easy-first)

**Cosa esiste nel codice**  
La funzione `answer_easy` in `pipeline.py` utilizza termini estratti da ingredienti e tecniche (con lunghezza ≥ 4 caratteri), filtra i candidati tramite BM25 (`top_k=30`) e applica gestori di negazione (*senza*, *escludendo*, *evitando*, *non*).

**Decisione tecnica (e motivazione)**  
Sono state adottate regole semplici e deterministiche per garantire la massima spiegabilità e stabilità.

**Pro**
- Comportamento deterministico  
- Elevata velocità  
- Facile da controllare  

**Contro**
- Non copre parafrasi  
- Ignora termini corti  
- Dipende dai termini presenti nella domanda  

**Rischi e mitigazione**  
Possibili falsi positivi dovuti a termini generici; mitigato tramite `required_terms` e filtri di negazione.

**Miglioramento piccolo e reversibile**  
Introduzione di una flag `EASY_USE_QUERY_PARSER=1` per integrare `easy_query_parser.py` (bigrammi e trigrammi).

---

## LLM selector

**Cosa esiste nel codice**  
Il gating dell’LLM avviene in `answer_question` tramite `ENABLE_LLM_REASONING`.  
I candidati sono costruiti per unione in `_build_llm_candidates`.  
Prompt in `prompts.py`, cache in `llm_cache.jsonl`.  
Non è presente una troncatura esplicita delle liste di ingredienti e tecniche.

**Decisione tecnica (e motivazione)**  
L’LLM viene usato solo come selettore in casi rumorosi, preservando il baseline deterministico.

**Pro**
- Riduce il rumore con molti candidati  
- Fallback sicuro  
- Cache evita chiamate ripetute  

**Contro**
- Cache senza versionamento del prompt  
- Prompt potenzialmente lungo  
- L’LLM non aggiunge elementi oltre il deterministico  

**Rischi e mitigazione**  
Risposte non valide dell’LLM mitigate tramite validazione rigorosa e fallback.

**Miglioramento piccolo e reversibile**  
Flag `LLM_TRUNCATE_LISTS=1` e aggiunta di `PROMPT_VERSION` nella chiave di cache.

---

## Normalizzazione / Sanitizzazione

**Cosa esiste nel codice**  
`normalize_text` in `text.py` (lowercase e rimozione punteggiatura), `_postprocess_answer` e `_split_answer_lines` in `pipeline.py`, `_clean_dish_name` in `menu_parser.py`.

**Decisione tecnica (e motivazione)**  
Normalizzazione leggera e locale per evitare NLP pesante.

**Pro**
- Confronto coerente delle risposte  
- Riduzione del rumore  
- Preservazione di termini speciali  

**Contro**
- Validazione LLM richiede match esatto  
- Normalizzazione non applicata ai candidati  

**Rischi e mitigazione**  
Mismatch da errori minori mitigati parzialmente con `NORMALIZE_TRAILING_VOWEL`.

**Miglioramento piccolo e reversibile**  
Flag `LLM_VALIDATE_NORMALIZED=1` per validare su stringhe normalizzate e mappare ai nomi originali.

---

## Valutazione

**Cosa esiste nel codice**  
`run_eval.py` esegue il loop delle domande, invoca `answer_question`, calcola F1 in `metrics.py` e salva `predictions.csv` e `eval_report.json`.

**Decisione tecnica (e motivazione)**  
La metrica F1 basata sull’overlap è semplice e adatta a risposte in lista.

**Pro**
- Veloce  
- Riproducibile  
- Tollerante a formati diversi  

**Contro**
- Nessun breakdown per difficoltà nel core  
- Nessuna metrica di retrieval  

**Rischi e mitigazione**  
CSV con header diversi mitigati tramite euristiche e default.

**Miglioramento piccolo e reversibile**  
Flag `--diagnostics` per integrare `generate_easy_diagnostics.py` nel flusso di eval.
