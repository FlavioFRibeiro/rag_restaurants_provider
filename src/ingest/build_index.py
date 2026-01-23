from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from ..utils.io import ensure_dir, read_text, write_json, write_jsonl
from ..utils.logging import get_logger
from .chunker import chunk_document
from .pdf_loader import load_pdf_text

logger = get_logger(__name__)


def detect_doc_type(path: Path) -> str:
    name = path.stem.lower()
    ext = path.suffix.lower()
    if "menu" in name or "cardapio" in name:
        return "menu"
    if "manual" in name or "guide" in name or "handbook" in name:
        return "manual"
    if ext in {".py", ".js", ".ts", ".java", ".rs", ".go", ".cpp", ".c", ".cs", ".md"}:
        return "code"
    return "other"


def iter_source_files(input_dir: Path) -> list[Path]:
    exts = {".pdf", ".txt", ".md", ".csv"}
    return [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def load_text_from_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return load_pdf_text(path)
    return read_text(path)


def build_index(
    input_dir: Path,
    out_dir: Path,
    model_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> None:
    ensure_dir(out_dir)

    files = iter_source_files(input_dir)
    if not files:
        raise RuntimeError(f"No supported files found in {input_dir}")

    documents: list[dict] = []
    texts: list[str] = []

    for file_path in files:
        raw_text = load_text_from_file(file_path)
        if not raw_text or len(raw_text.strip()) < 5:
            logger.info("Skipping empty file: %s", file_path)
            continue

        doc_type = detect_doc_type(file_path)
        try:
            source_rel = file_path.relative_to(input_dir).as_posix()
        except Exception:
            source_rel = str(file_path)

        chunks = chunk_document(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk in chunks:
            metadata = {
                "source_file": source_rel,
                "doc_type": doc_type,
                "chunk_id": chunk.chunk_id,
            }
            documents.append({"text": chunk.text, "metadata": metadata})
            texts.append(chunk.text)

    if not texts:
        raise RuntimeError("No text chunks produced. Check your input files.")

    logger.info("Embedding %s chunks with %s", len(texts), model_name)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    write_jsonl(out_dir / "documents.jsonl", documents)

    embed_config = {
        "model_name": model_name,
        "normalize": True,
        "index_type": "IndexFlatIP",
        "embedding_dim": int(dim),
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "num_chunks": len(documents),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    write_json(out_dir / "embed_config.json", embed_config)

    logger.info("Index saved to %s", out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from documents.")
    parser.add_argument("--input-dir", required=True, help="Directory with raw PDFs/txt/csv.")
    parser.add_argument("--out-dir", required=True, help="Output directory for index artifacts.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_index(
        input_dir=Path(args.input_dir),
        out_dir=Path(args.out_dir),
        model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
