# Datapizza RAG MVP

A simple, reliable RAG pipeline to answer CSV questions using PDF (and optional CSV) context.
The focus is a working, local-first system that is easy to run and iterate.

## Why a simple RAG
The project prioritizes a functional baseline over a complex architecture.
Given limited time, the goal is to deliver a clean end-to-end RAG that:
- builds an index locally
- answers questions consistently
- evaluates results automatically

This keeps the system debuggable and avoids overengineering.

## Architecture (v1)
- Ingest: PDFs -> text (pypdf with pdfplumber fallback).
- Chunking: fixed-size character chunks with overlap.
- Menus: parsed into structured dishes, retrieved via BM25 + deterministic matching.
- Other docs: embedded with SentenceTransformers and indexed with FAISS.
- LLM: optional; used only when enabled and a key is available.
- Eval: F1 overlap scoring, runs saved under `runs/`.

## Setup
1) Python 3.11+
2) Create and activate a virtualenv
3) Install dependencies:
   pip install -r requirements.txt

## Configure .env
Copy `.env.example` to `.env` and set one provider key:
- OPENAI_API_KEY=...
- GEMINI_API_KEY=...

Optional:
- OPENAI_MODEL=...
- GEMINI_MODEL=...
- ENABLE_LLM_REASONING=0/1

## Build index
python -m src.ingest.build_index --input-dir data/raw --out-dir data/processed/index

## Run evaluation
python -m src.eval.run_eval --questions data/questions/domande.csv --ground-truth data/ground_truth/ground_truth_mapped.csv --index-dir data/processed/index --out-dir runs/run_001

## Outputs
- data/processed/index/faiss.index
- data/processed/index/documents.jsonl
- data/processed/index/embed_config.json
- data/processed/menu_dishes.jsonl
- runs/<run_id>/predictions.csv
- runs/<run_id>/eval_report.json

### Limitations and Next Steps
- Chunking is naive and retrieval is top-k only.
- PDF parsing depends on a usable text layer.
- Menu parsing is heuristic and may miss edge cases.
- Next steps: add lightweight reranking and improve menu parsing diagnostics.
