from __future__ import annotations

import re

from ..utils.text import normalize_text

_SPLIT_RE = re.compile(r"[\n,;|]+")


def parse_items(text: str) -> list[str]:
    if not text:
        return []
    items: list[str] = []
    for part in _SPLIT_RE.split(text):
        norm = normalize_text(part)
        if norm:
            items.append(norm)
    return items


def f1_overlap(pred_text: str, true_text: str) -> tuple[float, float, float]:
    pred_items = set(parse_items(pred_text))
    true_items = set(parse_items(true_text))

    if not pred_items and not true_items:
        return 1.0, 1.0, 1.0
    if not pred_items or not true_items:
        return 0.0, 0.0, 0.0

    tp = len(pred_items & true_items)
    precision = tp / len(pred_items) if pred_items else 0.0
    recall = tp / len(true_items) if true_items else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall
