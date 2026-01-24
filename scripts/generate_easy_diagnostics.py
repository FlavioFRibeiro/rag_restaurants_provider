from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from src.config import load_env
from src.eval.metrics import f1_overlap, parse_items
from src.eval.run_eval import load_ground_truth, load_questions
from src.rag.llm import get_llm_provider
from src.rag.pipeline import (
    _build_llm_candidates,
    _env_flag,
    _get_bm25_top_dishes,
    _load_menu_dishes,
    _split_answer_lines,
    _llm_select,
    answer_easy,
)
from src.rag.retriever import Retriever
from src.utils.io import ensure_dir, write_json
from src.utils.logging import get_logger
from src.utils.text import normalize_text

logger = get_logger(__name__)


_LIST_SPLIT_RE = re.compile(r"[\n,;|]+")


def _split_output_items(text: str) -> list[str]:
    if not text:
        return []
    if text.strip().upper() == "NOT_FOUND":
        return []
    items: list[str] = []
    for part in _LIST_SPLIT_RE.split(text):
        cleaned = part.strip()
        if cleaned:
            items.append(cleaned)
    return items


def _pipe_join(items: list[str]) -> str:
    return "|".join(items) if items else "NOT_FOUND"


def _normalize_items(items: list[str]) -> set[str]:
    normalized: set[str] = set()
    for item in items:
        norm = normalize_text(item)
        if norm:
            normalized.add(norm)
    return normalized


def _sanitize_field(text: str) -> str:
    if not text:
        return ""
    return text.replace("\r", "\\r").replace("\n", "\\n")


def _load_question_difficulties(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    fieldnames = rows[0].keys()
    question_key = None
    for key in ["question", "pergunta", "query", "text", "domanda"]:
        if key in fieldnames:
            question_key = key
            break
    if question_key is None:
        question_key = list(fieldnames)[0]

    id_key = None
    for key in ["question_id", "id"]:
        if key in fieldnames:
            id_key = key
            break

    difficulty_key = None
    for key in fieldnames:
        lowered = key.lower()
        if "diffic" in lowered or "difficulty" in lowered or "level" in lowered:
            difficulty_key = key
            break

    mapping: dict[str, str] = {}
    for idx, row in enumerate(rows):
        question = (row.get(question_key) or "").strip()
        if not question:
            continue
        qid = row.get(id_key) if id_key else str(idx)
        difficulty = (row.get(difficulty_key) or "").strip() if difficulty_key else ""
        mapping[str(qid)] = difficulty
    return mapping


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Easy diagnostics artifacts.")
    parser.add_argument("--questions", required=True, help="Path to questions CSV.")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth CSV.")
    parser.add_argument("--index-dir", required=True, help="Index directory with FAISS artifacts.")
    parser.add_argument("--out-dir", required=True, help="Output directory for artifacts.")
    parser.add_argument("--use-llm", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    questions = load_questions(Path(args.questions))
    difficulty_map = _load_question_difficulties(Path(args.questions))
    ground_truth = load_ground_truth(Path(args.ground_truth))

    retriever = Retriever(index_dir=Path(args.index_dir))
    dishes = _load_menu_dishes(retriever)
    llm_provider = get_llm_provider() if args.use_llm else None
    llm_reasoning = _env_flag("ENABLE_LLM_REASONING", False)

    easy_rows: list[dict] = []
    bucket_stats = {
        "0": {"count": 0, "correct": 0, "llm_calls": 0},
        "1-3": {"count": 0, "correct": 0, "llm_calls": 0},
        "4-8": {"count": 0, "correct": 0, "llm_calls": 0},
        ">8": {"count": 0, "correct": 0, "llm_calls": 0},
    }

    for item in questions:
        qid = item["question_id"]
        question = item["question"]
        difficulty = (difficulty_map.get(qid) or "").strip().lower()
        if difficulty != "easy":
            continue

        deterministic_answer = answer_easy(question, retriever, top_k=30)
        deterministic_names = _split_answer_lines(deterministic_answer)
        n_det = len(deterministic_names)

        bm25_results = _get_bm25_top_dishes(question, retriever, dishes, top_k=20)
        bm25_top20_names = [item.item.dish_name for item in bm25_results[:20]]
        bm25_top5_names = bm25_top20_names[:5]

        call_reason = "NONE"
        if not deterministic_names:
            call_reason = "NOT_FOUND"
        elif len(deterministic_names) > 8:
            call_reason = "MANY_RESULTS"

        llm_called = False
        llm_raw = ""
        llm_validated: list[str] = []
        final_output = deterministic_answer

        if llm_reasoning and call_reason != "NONE":
            candidates = _build_llm_candidates(
                dishes,
                deterministic_names,
                bm25_results,
                max_candidates=20,
            )
            if candidates and llm_provider is not None:
                llm_validated, llm_raw, _ = _llm_select(
                    question,
                    candidates,
                    retriever,
                    llm_provider,
                    question_id=qid,
                )
                llm_called = True
            else:
                llm_validated = []

            if not deterministic_names:
                final_output = "\n".join(llm_validated) if llm_validated else "NOT_FOUND"
            elif not llm_validated:
                final_output = deterministic_answer
            else:
                det_set = set(deterministic_names)
                llm_set = set(llm_validated)
                if llm_set.issubset(det_set):
                    if len(deterministic_names) > 8:
                        final_output = "\n".join(llm_validated)
                    else:
                        final_output = deterministic_answer
                else:
                    final_output = deterministic_answer

        truth = ground_truth.get(qid, "")
        f1, _, _ = f1_overlap(final_output, truth)
        is_correct = abs(f1 - 1.0) < 1e-9

        truth_items_raw = _split_output_items(truth)
        prediction_items_raw = _split_output_items(final_output)

        truth_items = set(parse_items(truth))
        bm25_norm = _normalize_items(bm25_top20_names)
        det_norm = _normalize_items(deterministic_names)
        union_norm = bm25_norm | det_norm
        truth_in_bm25 = bool(truth_items & bm25_norm)
        truth_in_union = bool(truth_items & union_norm)

        easy_rows.append(
            {
                "question_id": qid,
                "question_text": _sanitize_field(question),
                "ground_truth": _pipe_join(truth_items_raw),
                "deterministic_answer": _pipe_join(deterministic_names),
                "n_deterministic": n_det,
                "bm25_top5_names": _pipe_join(bm25_top5_names),
                "bm25_top20_names": _pipe_join(bm25_top20_names),
                "llm_called": llm_called,
                "llm_call_reason": call_reason,
                "llm_raw_output": _sanitize_field(llm_raw),
                "llm_validated_output": _pipe_join(llm_validated),
                "final_output": _pipe_join(prediction_items_raw),
                "is_correct": is_correct,
                "truth_in_bm25_top20": truth_in_bm25,
                "truth_in_union_candidates": truth_in_union,
            }
        )

        if n_det <= 0:
            bucket = "0"
        elif n_det <= 3:
            bucket = "1-3"
        elif n_det <= 8:
            bucket = "4-8"
        else:
            bucket = ">8"
        bucket_stats[bucket]["count"] += 1
        bucket_stats[bucket]["correct"] += 1 if is_correct else 0
        bucket_stats[bucket]["llm_calls"] += 1 if llm_called else 0

    diagnostics_path = out_dir / "easy_diagnostics.csv"
    fields = [
        "question_id",
        "question_text",
        "ground_truth",
        "deterministic_answer",
        "n_deterministic",
        "bm25_top5_names",
        "bm25_top20_names",
        "llm_called",
        "llm_call_reason",
        "llm_raw_output",
        "llm_validated_output",
        "final_output",
        "is_correct",
        "truth_in_bm25_top20",
        "truth_in_union_candidates",
    ]
    with diagnostics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(easy_rows)

    bucket_summary: dict[str, dict] = {}
    for bucket, stats in bucket_stats.items():
        count = stats["count"]
        correct = stats["correct"]
        llm_calls = stats["llm_calls"]
        accuracy = (correct / count) if count else 0.0
        bucket_summary[bucket] = {
            "count": count,
            "accuracy": accuracy,
            "llm_calls": llm_calls,
        }
    write_json(out_dir / "easy_bucket_summary.json", bucket_summary)

    logger.info("Easy diagnostics written to %s", diagnostics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
