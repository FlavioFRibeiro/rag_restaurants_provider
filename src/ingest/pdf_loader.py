from __future__ import annotations

from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        PdfReader = None

    if PdfReader is not None:
        try:
            reader = PdfReader(str(path))
            texts = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(texts).strip()
        except Exception as exc:
            logger.warning("pypdf failed on %s: %s", path, exc)

    try:
        import pdfplumber
    except Exception as exc:
        raise RuntimeError(
            "No PDF parser available. Install pypdf (preferred) or pdfplumber."
        ) from exc

    try:
        with pdfplumber.open(str(path)) as pdf:
            texts = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(texts).strip()
    except Exception as exc:
        logger.warning("pdfplumber failed on %s: %s", path, exc)
        return ""
