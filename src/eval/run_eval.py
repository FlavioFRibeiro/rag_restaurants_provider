from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from ..config import load_env
from ..rag.llm import get_llm_provider
from ..rag.pipeline import answer_question
from ..rag.retriever import Retriever
from ..utils.io import ensure_dir, write_json
from ..utils.logging import get_logger
from .metrics import f1_overlap

logger = get_logger(__name__)


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_questions(path: Path) -> list[dict]:
    rows = _read_csv(path)
    if not rows:
        raise RuntimeError(f"No questions found in {path}")

    fieldnames = rows[0].keys()
    question_key = None
    for key in ["question", "pergunta", "query", "text"]:
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

    questions: list[dict] = []
    for idx, row in enumerate(rows):
        question = (row.get(question_key) or "").strip()
        if not question:
            continue
        qid = row.get(id_key) if id_key else str(idx)
        questions.append({"question_id": str(qid), "question": question})
    return questions


def load_ground_truth(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if not rows:
        return {}

    fieldnames = rows[0].keys()
    gt_key = None
    for key in ["ground_truth", "answer", "expected", "gold"]:
        if key in fieldnames:
            gt_key = key
            break
    if gt_key is None:
        gt_key = list(fieldnames)[-1]

    id_key = None
    for key in ["question_id", "id"]:
        if key in fieldnames:
            id_key = key
            break

    mapping: dict[str, str] = {}
    for idx, row in enumerate(rows):
        qid = row.get(id_key) if id_key else str(idx)
        mapping[str(qid)] = (row.get(gt_key) or "").strip()
    return mapping


def run_eval(
    questions_path: Path,
    ground_truth_path: Path,
    index_dir: Path,
    out_dir: Path,
    top_k: int = 6,
    no_llm: bool = False,
) -> None:
    load_env()
    ensure_dir(out_dir)

    questions = load_questions(questions_path)
    ground_truth = load_ground_truth(ground_truth_path)

    retriever = Retriever(index_dir=index_dir, top_k=top_k)
    llm_provider = None if no_llm else get_llm_provider()
    if llm_provider is None:
        logger.info("No LLM provider configured; using retrieval-only mode.")

    predictions: list[dict] = []
    score_rows: list[dict] = []
    error_examples: list[dict] = []
    scores: list[float] = []

    for item in questions:
        qid = item["question_id"]
        question = item["question"]
        prediction = answer_question(question, retriever, llm_provider, top_k=top_k)
        truth = ground_truth.get(qid, "")
        f1, precision, recall = f1_overlap(prediction, truth)

        predictions.append(
            {
                "question_id": qid,
                "question": question,
                "prediction": prediction,
            }
        )
        score_rows.append(
            {
                "question_id": qid,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }
        )
        error_examples.append(
            {
                "question_id": qid,
                "question": question,
                "prediction": prediction,
                "ground_truth": truth,
                "f1": f1,
            }
        )
        scores.append(f1)

    score_total = (sum(scores) / len(scores)) * 100 if scores else 0.0
    errors_top_examples = sorted(error_examples, key=lambda x: x["f1"])[:5]

    predictions_path = out_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question", "prediction"])
        writer.writeheader()
        writer.writerows(predictions)

    report = {
        "score_total": score_total,
        "score_per_question": score_rows,
        "errors_top_examples": errors_top_examples,
        "num_questions": len(questions),
    }
    write_json(out_dir / "eval_report.json", report)

    logger.info("Eval complete. Score: %.2f", score_total)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation over a question set.")
    parser.add_argument("--questions", required=True, help="Path to questions CSV.")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth CSV.")
    parser.add_argument("--index-dir", required=True, help="Index directory with FAISS artifacts.")
    parser.add_argument("--out-dir", default="", help="Output directory for run artifacts.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--no-llm", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / ts

    run_eval(
        questions_path=Path(args.questions),
        ground_truth_path=Path(args.ground_truth),
        index_dir=Path(args.index_dir),
        out_dir=out_dir,
        top_k=args.top_k,
        no_llm=args.no_llm,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
