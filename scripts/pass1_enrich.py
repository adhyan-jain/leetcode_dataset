from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.io_utils import append_jsonl, ensure_parent_dir, extract_text_value, infer_problem_id, load_input_records, read_existing_ids, unwrap_raw_and_metadata
from utils.ollama_client import generate
from utils.text_utils import normalize_label
from utils.validation import sanitize_metadata_shape


logger = logging.getLogger("pass1_enrich")


PROMPT_TEMPLATE = """You are enriching DSA problem metadata for a personalized recommendation engine.

The recommender uses question metadata to estimate:

* what the problem requires
* what the problem trains
* how difficult it is
* what mistakes users may make
* how similar it is to other problems
* whether it is useful for learning, practice, review, recovery, or challenge

Return ONLY valid JSON.

Problem:
ID: {id}
Title: {title}
Difficulty: {difficulty}
Tags: {tags}
Statement: {statement}
Examples: {examples}
Constraints: {constraints}
Hints: {hints}
Solution/Editorial if available: {solution}

Return JSON exactly in this shape:

{{
"taxonomy": {{
"domains": {{}},
"algorithms": {{}},
"patterns": {{}},
"micro_skills": {{}}
}},
"difficulty_vector": {{
"overall": 0.0,
"conceptual": 0.0,
"implementation": 0.0,
"edge_cases": 0.0
}},
"skill_model": {{
"requires": {{}},
"trains": {{}},
"tests": {{}}
}},
"failure_model": {{
"common_failures": {{}}
}},
"learning_role": {{
"first_exposure": 0.0,
"practice": 0.0,
"review": 0.0,
"assessment": 0.0,
"recovery": 0.0,
"challenge": 0.0
}},
"pattern_cluster": "",
"estimated_time_minutes": 0,
"solution_dna_summary": "",
"metadata_confidence": 0.0
}}

Rules:

* Use snake_case labels.
* Taxonomy values must be weighted from 0 to 1.
* Difficulty values must be from 0 to 1.
* "requires" means prerequisite mastery needed before attempting.
* "trains" means skills improved by solving.
* "tests" means skills checked but not heavily taught.
* Common failures should map to affected skills and severity.
* Prefer solving-pattern labels over broad topic labels.
* Do not add labels just because they appear in the title.
* If multiple approaches exist, prefer the most educational/intended approach.
* If unsure, lower metadata_confidence.
* Do not output markdown.
* Do not include commentary outside JSON.
"""


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_problem_fields(raw: Dict[str, Any]) -> Dict[str, str]:
    return {
        "id": str(raw.get("id") or raw.get("problem_id") or raw.get("question_id") or raw.get("slug") or ""),
        "title": extract_text_value(raw, ("title", "name", "question_title", "problem_title")),
        "difficulty": extract_text_value(raw, ("difficulty", "level", "difficulty_level")),
        "tags": extract_text_value(raw, ("tags", "tag", "topic_tags", "category")),
        "statement": extract_text_value(raw, ("statement", "problem", "description", "content", "question", "body")),
        "examples": extract_text_value(raw, ("examples", "example", "sample_input_output", "sample")),
        "constraints": extract_text_value(raw, ("constraints", "constraint", "limits")),
        "hints": extract_text_value(raw, ("hints", "hint", "guidance")),
        "solution": extract_text_value(raw, ("solution", "editorial", "analysis", "explanation", "approach")),
    }


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def enrich_row(
    raw: Dict[str, Any],
    *,
    model: str,
    ollama_url: str,
    timeout: int,
    retries: int,
    temperature: float,
    sleep_between_requests: float,
) -> tuple[Dict[str, Any], Dict[str, Any] | None, str | None, str | None]:
    fields = build_problem_fields(raw)
    prompt = PROMPT_TEMPLATE.format(**fields)
    last_raw_response = None

    for attempt in range(1, retries + 1):
        try:
            response = generate(
                prompt=prompt,
                model=model,
                ollama_url=ollama_url,
                timeout=timeout,
                retries=1,
                temperature=temperature,
                sleep_between_requests=sleep_between_requests,
                prefer_json=True,
            )
            last_raw_response = response.response_text
            parsed = extract_json(response.response_text)
            metadata, issues = sanitize_metadata_shape(parsed)
            if issues:
                logger.debug("Pass1 metadata sanitation issues for %s: %s", fields["id"], issues)
            return metadata, response.raw, None, None
        except Exception as exc:
            logger.warning(
                "Failed to parse pass1 JSON for problem %s on attempt %s/%s: %s",
                fields["id"],
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                continue
            return None, response.raw, f"invalid_json: {exc}", last_raw_response
    return None, None, "unexpected_failure", last_raw_response


def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run pass 1 DSA metadata enrichment.")
    parser.add_argument("--input", required=True, help="Raw dataset file (.json, .jsonl, or .csv)")
    parser.add_argument("--output", required=True, help="JSONL output path")
    parser.add_argument("--failures", required=True, help="JSONL failures output path")
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = ensure_parent_dir(args.output)
    failures_path = ensure_parent_dir(args.failures)

    if args.force:
        for path in (output_path, failures_path):
            if path.exists():
                path.unlink()

    records = load_input_records(input_path)
    existing_ids = read_existing_ids(output_path) if args.resume and not args.force else set()

    processed = 0
    successes = 0
    failures = 0
    skipped = 0

    logger.info("Loaded %s input rows from %s", len(records), input_path)

    for index, record in enumerate(records):
        if index < args.start_index:
            continue
        if args.limit is not None and processed >= args.limit:
            break

        raw, _ = unwrap_raw_and_metadata(record)
        problem_id = infer_problem_id(raw, fallback=f"row_{index}")
        if not args.force and problem_id in existing_ids:
            skipped += 1
            continue

        processed += 1
        logger.info("Processing %s (%s/%s)", problem_id, processed, args.limit if args.limit is not None else "all")

        try:
            metadata, response_raw, error, raw_response_text = enrich_row(
                raw,
                model=args.model,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                retries=args.retries,
                temperature=args.temperature,
                sleep_between_requests=args.sleep_between_requests,
            )
            if metadata is not None:
                row_out = {
                    "problem_id": problem_id,
                    "raw": raw,
                    "pass1_metadata": metadata,
                    "pass1_status": "success",
                    "pass1_error": None,
                }
                append_jsonl(output_path, [row_out])
                successes += 1
                continue

            failures += 1
            row_out = {
                "problem_id": problem_id,
                "raw": raw,
                "pass1_metadata": None,
                "pass1_status": "failed",
                "pass1_error": error,
            }
            append_jsonl(output_path, [row_out])
            append_jsonl(
                failures_path,
                [
                    {
                        "problem_id": problem_id,
                        "raw": raw,
                        "error": error,
                        "raw_response": response_raw,
                        "response_text": raw_response_text,
                    }
                ],
            )
        except Exception as exc:
            failures += 1
            logger.exception("Unexpected error while processing %s", problem_id)
            append_jsonl(
                output_path,
                [
                    {
                        "problem_id": problem_id,
                        "raw": raw,
                        "pass1_metadata": None,
                        "pass1_status": "failed",
                        "pass1_error": str(exc),
                    }
                ],
            )
            append_jsonl(
                failures_path,
                [
                    {
                        "problem_id": problem_id,
                        "raw": raw,
                        "error": str(exc),
                        "raw_response": None,
                        "response_text": None,
                    }
                ],
            )

    logger.info(
        "Finished. processed=%s success=%s failed=%s skipped=%s",
        processed,
        successes,
        failures,
        skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
