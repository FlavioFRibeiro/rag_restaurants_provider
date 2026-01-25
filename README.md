# Datapizza RAG MVP
_Soluzione per il Datapizza AI Engineer Technical Test_

## Panoramica del progetto
Il progetto privilegia un baseline funzionale rispetto a un’architettura complessa. Considerando il tempo limitato che avevo, l’obiettivo è fornire una soluzione RAG end-to-end pulita che:
- costruisca l’indice localmente
- risponda alle domande in modo consistente
- valuti automaticamente i risultati
Questo approccio mantiene il sistema facilmente debuggabile ed evita l’overengineering.

## Architettura di alto livello
- Ingestion: lettura di PDF/CSV e raccolta dei file sorgente.
- Parsing: estrazione strutturata dei menu (piatti, ingredienti, tecniche).
- Indexing: creazione di artifacts per retrieval (FAISS + menu_dishes).
- Retrieval: BM25 per i piatti dei menu e dense retrieval per il resto.
- Answering: matching deterministico per il livello Easy, LLM opzionale e conservativo.
- Evaluation: calcolo F1 overlap e salvataggio degli outputs di run.

## Principi di progettazione
- Explainability: regole e parsing deterministici per ridurre ambiguita.
- Robustness: fallback, validazioni e output controllati.
- Offline & reproducible: pipeline locale, artifacts versionabili, cache LLM.
- Conservative use of LLMs: LLM usato solo come supporto selettivo.

## Struttura del repository
```
data/
  raw/            # input PDFs e CSV
  questions/      # questions.csv
  ground_truth/   # ground truth CSV
  processed/      # index + menu_dishes + cache
runs/             # output per esecuzione
scripts/          # strumenti di debug e diagnosi
src/              # codice sorgente
  cli.py          # entrypoint CLI
  config.py       # env + paths
  ingest/         # ingestion + parsing + build_index
  rag/            # retrieval + pipeline + prompts + LLM
  eval/           # evaluation + metriche
  utils/          # I/O, logging, text utils
tests/            # placeholder per test leggeri
.env.example      # template variabili ambiente
.env              # configurazione locale (non committare chiavi)
pyproject.toml    # metadata e dipendenze
requirements.txt  # dipendenze per installazione rapida
TECHNICAL_DOC.md  # documentazione tecnica dettagliata (separata)
```

## Come fare il Run
Creare un virtualenv e installare le dipendenze:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Build index:
```bash
python -m src.ingest.build_index --input-dir data/raw --out-dir data/processed/index
```

Run evaluation:
```bash
python -m src.eval.run_eval --questions data/questions/<questions.csv> --ground-truth data/ground_truth/<ground_truth.csv> --index-dir data/processed/index --out-dir runs/run_001
```

Outputs attesi:
- `data/processed/index/faiss.index`
- `data/processed/index/documents.jsonl`
- `data/processed/index/embed_config.json`
- `data/processed/menu_dishes.jsonl`
- `runs/<run_id>/predictions.csv`
- `runs/<run_id>/eval_report.json`

## Configurazione
Il progetto usa variabili in `.env` per abilitare provider LLM o flag di comportamento. I dettagli completi sono in `TECHNICAL_DOC.md`.

## Filosofia della valutazione
La valutazione usa F1 overlap tra liste di risposte. L'obiettivo e dimostrare ragionamento tecnico e stabilita del sistema, non una corsa al leaderboard.

## Note, limitazioni
- Focus sul subset Easy e sulla stabilita del parsing dei menu.
- L'estrazione PDF dipende dalla qualita del text layer.
- Componenti sperimentali esistono nei `scripts/`, ma non fanno parte del flusso standard.
- Parsing basato su heuristics: buono per spiegabilita, meno flessibile su formati nuovi.
- Chunking semplice: efficace ma non semantico.
- Retrieval ibrido: BM25 copre termini rari, ma non risolve sinonimi complessi.

## Prossimi passi
- Aggiungere un flag per truncare ingredienti/tecniche nel prompt LLM (riduce errori senza cambiare la logica).
- Integrare un report diagnostico leggero nel run di eval (per capire velocemente dove si perdono risposte).

## Documentatzzione
La documentazione tecnica dettagliata e in `TECHNICAL_DOC.md`.
