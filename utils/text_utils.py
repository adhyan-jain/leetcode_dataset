from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, List, Sequence

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency
    fuzz = None


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")
_SENTENCE_RE = re.compile(r"[.!?]\s")


def normalize_whitespace(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def to_ascii(text: str | None) -> str:
    if text is None:
        return ""
    return (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii", "ignore")
    )


def snake_case(text: str | None) -> str:
    if text is None:
        return ""
    text = to_ascii(text)
    text = normalize_whitespace(text)
    if not text:
        return ""
    text = _CAMEL_RE_1.sub(r"\1_\2", text)
    text = _CAMEL_RE_2.sub(r"\1_\2", text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = _NON_ALNUM_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def singularize_label(label: str) -> str:
    label = snake_case(label)
    if not label:
        return label

    parts = label.split("_")
    last = parts[-1]
    if last.endswith("ies") and len(last) > 4:
        parts[-1] = last[:-3] + "y"
    elif last.endswith("ses") and len(last) > 4:
        parts[-1] = last[:-2]
    elif last.endswith("s") and len(last) > 3 and not last.endswith(("ss", "us", "is")):
        parts[-1] = last[:-1]
    return "_".join(parts)


def normalize_label(label: str | None) -> str:
    label = snake_case(label)
    label = singularize_label(label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label


def looks_like_sentence(label: str | None) -> bool:
    if not label:
        return False
    text = normalize_whitespace(str(label))
    if len(text) > 40 and " " in text:
        return True
    return bool(_SENTENCE_RE.search(text))


def fuzzy_ratio(a: str, b: str) -> int:
    a_norm = normalize_label(a)
    b_norm = normalize_label(b)
    if not a_norm or not b_norm:
        return 0
    if fuzz is not None:
        return int(fuzz.ratio(a_norm, b_norm))
    return int(SequenceMatcher(None, a_norm, b_norm).ratio() * 100)


def best_match(label: str, candidates: Sequence[str]) -> tuple[str | None, int]:
    best_label = None
    best_score = -1
    for candidate in candidates:
        score = fuzzy_ratio(label, candidate)
        if score > best_score:
            best_score = score
            best_label = candidate
    return best_label, best_score


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def is_probably_long_label(label: str | None, threshold: int = 40) -> bool:
    if not label:
        return False
    return len(normalize_whitespace(str(label))) >= threshold


def clamp01(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return max(0.0, min(1.0, value))


def clamp_int(value, default: int = 0, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        value = int(round(float(value)))
    except Exception:
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def is_single_word(label: str | None) -> bool:
    if not label:
        return False
    return "_" not in normalize_label(label) and " " not in normalize_whitespace(str(label))

