from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable

_INGREDIENTI_RE = re.compile(r"^ingredienti\\s*:?$", re.IGNORECASE)
_TECNICHE_RE = re.compile(r"^tecniche\\s*:?$", re.IGNORECASE)
_INGREDIENTI_INLINE_RE = re.compile(r"^ingredienti\\s*:?(.*)$", re.IGNORECASE)
_TECNICHE_INLINE_RE = re.compile(r"^tecniche\\s*:?(.*)$", re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"[ \t]+")
_MULTINEWLINE_RE = re.compile(r"\n{2,}")
_ARTICLE_TITLE_RE = re.compile(r"^[a-z]['’][A-Z]")

_BAD_TITLES = {
    "menu",
    "atto",
    "prezzo",
    "chef",
    "ingredienti",
    "tecniche",
    "techniques",
    "ristorante",
}

_ARTICLE_ONLY = {"la", "il", "lo", "l'", "un", "una", "un'"}
_TITLE_END_STOPWORDS = {
    "di",
    "del",
    "della",
    "dello",
    "dei",
    "delle",
    "al",
    "alla",
    "allo",
    "alle",
    "agli",
    "con",
    "per",
    "tramite",
    "sotto",
    "su",
    "da",
}

_NARRATIVE_HINTS = {
    "questa",
    "questo",
    "questi",
    "queste",
    "nostro",
    "nostra",
    "nostri",
    "nostre",
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
    "successivamente",
    "viene",
    "vengono",
    "venire",
    "aggiunge",
    "aggiungono",
    "accompagna",
    "completa",
    "infine",
    "presenta",
    "preparati",
    "preparato",
    "servito",
    "servita",
    "condito",
    "condita",
    "conditi",
    "adagiato",
    "adagiata",
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
    lines = normalized.splitlines()
    out_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _INGREDIENTI_RE.match(stripped):
            out_lines.append("Ingredienti")
            continue
        if _TECNICHE_RE.match(stripped):
            out_lines.append("Tecniche")
            continue
        inline_ing = _INGREDIENTI_INLINE_RE.match(stripped)
        if inline_ing and inline_ing.group(1).strip():
            out_lines.append("Ingredienti")
            out_lines.append(inline_ing.group(1).strip())
            continue
        inline_tec = _TECNICHE_INLINE_RE.match(stripped)
        if inline_tec and inline_tec.group(1).strip():
            out_lines.append("Tecniche")
            out_lines.append(inline_tec.group(1).strip())
            continue
        out_lines.append(line)
    normalized = "\n".join(out_lines)
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
    if _is_technique_line(line):
        return False
    lowered = line.lower()
    if lowered in _ARTICLE_ONLY:
        return False
    if any(hint in lowered for hint in _NARRATIVE_HINTS):
        return False
    words = [word for word in line.split() if word]
    if len(words) > 12:
        return False
    if words and words[-1].lower().strip(".,;:!?") in _TITLE_END_STOPWORDS:
        return False
    stripped = line.lstrip(" \"'“”«»([{")
    if not stripped:
        return False
    first = stripped[0]
    if first.isupper() or first.isdigit():
        return True
    if _ARTICLE_TITLE_RE.match(stripped):
        return True
    return False


def _is_technique_line(line: str) -> bool:
    if len(line) > 120:
        return False
    lowered = line.lower()
    return lowered.startswith(_TECHNIQUE_PREFIXES)


def _is_ingredient_line(line: str) -> bool:
    if len(line) > 80:
        return False
    stripped = line.lstrip(" \"'“”«»([{")
    if not stripped:
        return False
    first = stripped[0]
    return first.isupper() or first.isdigit()


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


def _normalize_term(text: str) -> str:
    return " ".join(text.lower().split())


def _parse_narrative(
    lines: list[str],
    source_file: str,
    restaurant_name: str | None,
    known_ingredients: Iterable[str],
    known_techniques: Iterable[str],
) -> List[MenuItem]:
    items: List[MenuItem] = []
    known_ing = sorted({_normalize_term(term) for term in known_ingredients if term}, key=len, reverse=True)
    known_tec = sorted({_normalize_term(term) for term in known_techniques if term}, key=len, reverse=True)
    known_ing_set = set(known_ing)
    known_tec_set = set(known_tec)

    def is_title(line: str) -> bool:
        if not _is_title_candidate(line):
            return False
        norm = _normalize_term(line)
        if norm in known_ing_set or norm in known_tec_set:
            return False
        if len(norm) >= 6 and any(norm in tech for tech in known_tec_set):
            return False
        return True

    def technique_fragments(term: str) -> list[str]:
        parts = re.split(
            r"\\b(?:con|a|al|alla|allo|alle|agli|dei|delle|del|della|per|tramite|in|su|sotto|da)\\b",
            term,
        )
        fragments = [part.strip() for part in parts if part.strip()]
        return [frag for frag in fragments if len(frag) >= 6]

    tech_signatures = [
        (term, technique_fragments(term)) for term in known_tec if term
    ]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not is_title(line):
            i += 1
            continue
        dish_name = line
        j = i + 1
        while j < len(lines) and not is_title(lines[j]) and lines[j].lower() not in {"ingredienti", "tecniche"}:
            j += 1
        block_text = _normalize_term(" ".join(lines[i + 1 : j]))
        ingredients = [term for term in known_ing if term and term in block_text]
        techniques = [term for term in known_tec if term and term in block_text]
        if tech_signatures:
            for term, fragments in tech_signatures:
                if not fragments or len(fragments) < 2:
                    continue
                if term in techniques:
                    continue
                if all(fragment in block_text for fragment in fragments):
                    techniques.append(term)
        if "sferificazione" in block_text and "campi magnetici entropici" in block_text:
            if "sferificazione con campi magnetici entropici" not in techniques:
                techniques.append("sferificazione con campi magnetici entropici")
        if len(ingredients) >= 2 or len(techniques) >= 1:
            items.append(
                MenuItem(
                    dish_name=dish_name,
                    ingredients=ingredients,
                    techniques=techniques,
                    source_file=source_file,
                    restaurant_name=restaurant_name,
                )
            )
        i = j
    return items


def parse_menu_items(
    text: str,
    source_file: str,
    known_ingredients: Iterable[str] | None = None,
    known_techniques: Iterable[str] | None = None,
) -> List[MenuItem]:
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

    if not any(line.lower() == "ingredienti" for line in lines):
        if known_ingredients and known_techniques:
            return _parse_narrative(lines, source_file, restaurant_name, known_ingredients, known_techniques)
        return items

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
            if _is_ingredient_line(line):
                current_ingredients.append(line)
        elif state == "techniques":
            if _is_technique_line(line):
                current_techniques.append(line)
            else:
                state = "idle"
                buffer = [line]
                continue
        if state != "techniques":
            buffer.append(line)
            if len(buffer) > 25:
                buffer = buffer[-25:]

    flush_current()
    return items
