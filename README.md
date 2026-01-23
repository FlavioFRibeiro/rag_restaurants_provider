# Datapizza RAG MVP

Simple RAG pipeline to answer questions from a CSV using PDF (and optional CSV) context.

## Setup
1) Python 3.11+
2) Create and activate a virtualenv
3) Install dependencies:
   pip install -e .

## Configure .env
Copy `.env.example` to `.env` and set one provider key:
- OPENAI_API_KEY=...
- GEMINI_API_KEY=...

## Build index
python -m src.ingest.build_index --input-dir data/raw --out-dir data/processed/index

## Run evaluation
python -m src.eval.run_eval --questions data/questions/questions.csv --ground-truth data/ground_truth/ground_truth.csv --index-dir data/processed/index --out-dir runs/run_001

## Outputs
- data/processed/index/faiss.index
- data/processed/index/documents.jsonl
- data/processed/index/embed_config.json
- runs/<timestamp>/predictions.csv
- runs/<timestamp>/eval_report.json

## Limitations and next steps
- Naive character chunking and simple top-k retrieval.
- Retrieval-only mode is used when no API key is configured.
- PDF extraction depends on an existing text layer.
- Next step (v2): add reranking.
