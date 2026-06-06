from __future__ import annotations
from utils.logging_utils import log_execution, setup_global_logger

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from .text_utils import normalize_whitespace


@log_execution
def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@log_execution
def read_json_file(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@log_execution
def write_json_file(path: str | Path, data: Any, indent: int = 2) -> None:
    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


import threading
_APPEND_LOCK = threading.Lock()

@log_execution
def append_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = ensure_parent_dir(path)
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@log_execution
def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


@log_execution
def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return iter(())
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@log_execution
def load_input_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return list(read_jsonl(path))
    if suffix == ".json":
        data = read_json_file(path)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("data", "records", "problems", "questions", "items"):
                if isinstance(data.get(key), list):
                    return [x for x in data[key] if isinstance(x, dict)]
            # Fallback: treat dict values as records if they look like row dictionaries.
            if all(isinstance(v, dict) for v in data.values()):
                return list(data.values())
        raise ValueError(f"Unsupported JSON input structure in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported input format: {path.suffix}")


@log_execution
def read_existing_ids(path: str | Path, id_field_candidates: Sequence[str] = ("problem_id", "id")) -> set[str]:
    ids: set[str] = set()
    p = Path(path)
    if not p.exists():
        return ids
    for row in read_jsonl(p):
        for field in id_field_candidates:
            value = row.get(field)
            if value is not None and str(value) != "":
                ids.add(str(value))
                break
    return ids


@log_execution
def first_present(record: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


@log_execution
def deep_get(record: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


@log_execution
def extract_text_value(record: Dict[str, Any], keys: Sequence[str]) -> str:
    value = first_present(record, keys, default="")
    if isinstance(value, (list, tuple)):
        parts = [normalize_whitespace(str(item)) for item in value if normalize_whitespace(str(item))]
        return "\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return normalize_whitespace(str(value))


@log_execution
def infer_problem_id(record: Dict[str, Any], fallback: str) -> str:
    candidates = (
        "problem_id",
        "id",
        "question_id",
        "leetcode_id",
        "slug",
        "title_slug",
        "name",
    )
    value = first_present(record, candidates)
    if value is None or str(value).strip() == "":
        return fallback
    return str(value)


@log_execution
def unwrap_raw_and_metadata(record: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else record
    metadata = record.get("pass1_metadata")
    if isinstance(metadata, dict):
        return raw, metadata
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return raw, metadata
    return raw, {}

