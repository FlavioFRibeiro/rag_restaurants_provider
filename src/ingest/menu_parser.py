from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

_INGREDIENTI_RE = re.compile(r"\bIngredienti\b\s*:?", re.IGNORECASE)
_TECNICHE_RE = re.compile(r"\bTecniche\b\s*:?", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{2,}")

_BAD_TITLES = {
    "menu",
    "atto",
    "prezzo",
    "chef",
    "ingredienti",
    "tecniche",
    "ristorante",
}

_ARTICLE_ONLY = {"la", "il", "lo", "l'", "un", "una", "un'"}

_NARRATIVE_HINTS = {
    "piatto",
    "piatti",
    "esperienza",
    "chef",
    "benvenuti",
    "lasciatevi",
    "imbarcatevi",
    "scoprite",
    "immaginate",
    "ospiti",
    "presentiamo",
    "invit",
}

_TECHNIQUE_PREFIXES = (
    "affettamento",
    "affumicatura",
    "amalgamazione",
    "bollitura",
    "cottura",
    "cryo-tessitura",
    "decostruzione",
    "ebollizione",
    "fermentazione",
    "grigliatura",
    "idro-cristallizzazione",
    "impasto",
    "marinatura",
    "modellatura",
    "saltare",
    "sferificazione",
    "sinergia",
    "surgelamento",
    "congelamento",
    "taglio",
)


@dataclass(frozen=True)
class MenuItem:
    dish_name: str
    ingredients: List[str]
    techniques: List[str]
    source_file: str
    doc_type: str = "menu_dish"
    restaurant_name: str | None = None


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _INGREDIENTI_RE.sub("\nIngredienti\n", normalized)
    normalized = _TECNICHE_RE.sub("\nTecniche\n", normalized)
    normalized = _MULTISPACE_RE.sub(" ", normalized)
    normalized = _MULTINEWLINE_RE.sub("\n", normalized)
    return normalized


def _clean_line(line: str) -> str:
    return _MULTISPACE_RE.sub(" ", line).strip()


def _is_bad_title(line: str) -> bool:
    lowered = line.lower()
    if lowered in _BAD_TITLES:
        return True
    for token in _BAD_TITLES:
        if lowered.startswith(f"{token} "):
            return True
    return False


def _is_title_candidate(line: str) -> bool:
    if len(line) < 2 or len(line) > 120:
        return False
    if line.endswith((".", ",", ";")):
        return False
    if "," in line:
        return False
    if _is_bad_title(line):
        return False
    lowered = line.lower()
    if lowered in _ARTICLE_ONLY:
        return False
    if any(hint in lowered for hint in _NARRATIVE_HINTS):
        return False
    words = [word for word in line.split() if word]
    if len(words) > 12:
        return False
    stripped = line.lstrip(" \"'“”«»([{")
    if not stripped:
        return False
    first = stripped[0]
    if not (first.isupper() or first.isdigit()):
        return False
    return True


def _is_technique_line(line: str) -> bool:
    if len(line) > 120:
        return False
    lowered = line.lower()
    return lowered.startswith(_TECHNIQUE_PREFIXES)


def _pick_dish_name(buffer: List[str]) -> str:
    for idx in range(len(buffer) - 1, -1, -1):
        line = buffer[idx]
        if not _is_title_candidate(line):
            continue
        parts = [line]
        prev = idx - 1
        while prev >= 0 and _is_title_candidate(buffer[prev]):
            if buffer[prev].lower() in _ARTICLE_ONLY:
                break
            combined = f"{buffer[prev]} {parts[0]}".strip()
            combined_words = [word for word in combined.split() if word]
            if len(combined) <= 120 and len(combined_words) <= 12:
                parts.insert(0, buffer[prev])
                prev -= 1
                continue
            break
        return " ".join(parts).strip()
    return ""


def _extract_restaurant_name(source_file: str) -> str | None:
    if not source_file:
        return None
    name = Path(source_file).stem.replace("_", " ").strip()
    return name or None


def parse_menu_items(text: str, source_file: str) -> List[MenuItem]:
    normalized = _normalize_text(text)
    lines = [_clean_line(line) for line in normalized.splitlines()]
    lines = [line for line in lines if line]

    items: List[MenuItem] = []
    buffer: List[str] = []
    current_name = ""
    current_ingredients: List[str] = []
    current_techniques: List[str] = []
    state = "idle"
    restaurant_name = _extract_restaurant_name(source_file)

    def flush_current() -> None:
        nonlocal current_name, current_ingredients, current_techniques
        if current_name and current_ingredients:
            items.append(
                MenuItem(
                    dish_name=current_name,
                    ingredients=current_ingredients,
                    techniques=current_techniques,
                    source_file=source_file,
                    restaurant_name=restaurant_name,
                )
            )
        current_name = ""
        current_ingredients = []
        current_techniques = []

    for line in lines:
        lowered = line.lower()
        if lowered == "ingredienti":
            flush_current()
            current_name = _pick_dish_name(buffer)
            current_ingredients = []
            current_techniques = []
            if not current_name:
                state = "idle"
            else:
                state = "ingredients"
            buffer.append(line)
            if len(buffer) > 25:
                buffer = buffer[-25:]
            continue

        if lowered == "tecniche":
            state = "techniques"
            buffer.append(line)
            if len(buffer) > 25:
                buffer = buffer[-25:]
            continue

        if state == "ingredients":
            if len(line) <= 80:
                current_ingredients.append(line)
        elif state == "techniques":
            if _is_technique_line(line):
                current_techniques.append(line)
            else:
                state = "idle"
                buffer = [line]
                continue

        buffer.append(line)
        if len(buffer) > 25:
            buffer = buffer[-25:]

    flush_current()
    return items
