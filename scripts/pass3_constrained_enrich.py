from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.io_utils import append_jsonl, ensure_parent_dir, extract_text_value, infer_problem_id, load_input_records, read_existing_ids, unwrap_raw_and_metadata
from utils.ollama_client import generate
from utils.text_utils import clamp01, clamp_int, normalize_label, normalize_whitespace
from utils.validation import (
    filter_to_allowed_weight_map,
    sanitize_metadata_shape,
    status_from_issues,
    validate_pass3_metadata,
)


logger = logging.getLogger("pass3_constrained_enrich")


PROMPT_TEMPLATE = """You are normalizing DSA problem metadata using a fixed taxonomy.

You MUST use only the allowed labels in the main fields.
If a necessary label is missing, put it in proposed_new_labels, but do not use it in the main metadata.

Allowed domains:
{allowed_domains}

Allowed algorithms:
{allowed_algorithms}

Allowed patterns:
{allowed_patterns}

Allowed micro_skills:
{allowed_micro_skills}

Allowed failure_types:
{allowed_failure_types}

Allowed pattern_clusters:
{allowed_pattern_clusters}

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

Previous messy metadata:
{pass1_metadata}

Return ONLY valid JSON:

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
"metadata_confidence": 0.0,
"proposed_new_labels": {{
"domains": [],
"algorithms": [],
"patterns": [],
"micro_skills": [],
"failure_types": [],
"pattern_clusters": []
}}
}}

Rules:

* Main fields must use only allowed labels.
* Use 1 to 3 domains.
* Use 1 to 3 algorithms.
* Use 1 to 4 patterns.
* Use 3 to 10 micro_skills.
* Weights should reflect importance, not just presence.
* Do not add labels just because they appear in the title.
* If multiple approaches exist, prefer the most educational/intended approach.
* difficulty_vector values must be between 0 and 1.
* metadata_confidence must be between 0 and 1.
* common_failures must use only allowed failure_types.
* affected_skills inside common_failures must use only allowed micro_skills or patterns.
* pattern_cluster must use only allowed pattern_clusters.
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


def load_taxonomy(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("taxonomy must be a JSON object")
    for key in ("domains", "algorithms", "patterns", "micro_skills", "failure_types", "pattern_clusters"):
        values = data.get(key, [])
        if isinstance(values, list):
            data[key] = [normalize_label(value) for value in values if normalize_label(value)]
        else:
            data[key] = []
    return data


def load_label_mapping(path: Path) -> Dict[str, Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("label mapping must be a JSON object")
    for key in ("domains", "algorithms", "patterns", "micro_skills", "failure_types", "pattern_clusters"):
        values = data.get(key, {})
        if isinstance(values, dict):
            data[key] = {normalize_label(k): normalize_label(v) for k, v in values.items() if normalize_label(k) and normalize_label(v)}
        else:
            data[key] = {}
    return data


def format_allowed_list(values: Sequence[str]) -> str:
    if not values:
        return "[]"
    return json.dumps(list(values), ensure_ascii=False, indent=2)


def select_top_items(weight_map: Dict[str, float], max_items: int) -> Dict[str, float]:
    items = sorted(weight_map.items(), key=lambda item: (-float(item[1]), item[0]))
    selected = items[:max_items]
    return {label: clamp01(weight) for label, weight in selected if label}


def remap_with_lookup(
    values: Dict[str, float],
    *,
    category: str,
    mapping: Dict[str, Dict[str, str]],
    allowed: Sequence[str],
) -> tuple[Dict[str, float], List[str]]:
    category_map = mapping.get(category, {})
    allowed_set = {normalize_label(x) for x in allowed}
    remapped: Dict[str, float] = {}
    proposed: List[str] = []
    for label, weight in values.items():
        normalized = normalize_label(label)
        if not normalized:
            continue
        mapped = category_map.get(normalized, normalized)
        if mapped in allowed_set:
            remapped[mapped] = max(remapped.get(mapped, 0.0), clamp01(weight))
        else:
            proposed.append(mapped)
    return remapped, sorted(set(proposed))


def remap_label(
    label: str,
    *,
    category: str,
    mapping: Dict[str, Dict[str, str]],
    allowed: Sequence[str],
) -> tuple[str, bool]:
    normalized = normalize_label(label)
    mapped = mapping.get(category, {}).get(normalized, normalized)
    allowed_set = {normalize_label(x) for x in allowed}
    return (mapped if mapped in allowed_set else normalized), mapped in allowed_set


def remap_skill_label(
    label: str,
    *,
    mapping: Dict[str, Dict[str, str]],
    allowed_micro_skills: Sequence[str],
    allowed_patterns: Sequence[str],
) -> tuple[str, bool]:
    normalized = normalize_label(label)
    candidate = mapping.get("micro_skills", {}).get(normalized, normalized)
    if candidate in {normalize_label(x) for x in allowed_micro_skills}:
        return candidate, True
    candidate = mapping.get("patterns", {}).get(normalized, candidate)
    if candidate in {normalize_label(x) for x in allowed_patterns}:
        return candidate, True
    return normalized, False


def remap_skill_weight_map(
    weight_map: Dict[str, float],
    *,
    mapping: Dict[str, Dict[str, str]],
    allowed_micro_skills: Sequence[str],
    allowed_patterns: Sequence[str],
) -> tuple[Dict[str, float], List[str]]:
    remapped: Dict[str, float] = {}
    proposed: List[str] = []
    for label, weight in weight_map.items():
        mapped, allowed = remap_skill_label(
            label,
            mapping=mapping,
            allowed_micro_skills=allowed_micro_skills,
            allowed_patterns=allowed_patterns,
        )
        if allowed:
            remapped[mapped] = max(remapped.get(mapped, 0.0), clamp01(weight))
        else:
            proposed.append(mapped)
    return remapped, sorted(set(proposed))


def sanitize_proposed_labels(data: Any) -> Dict[str, List[str]]:
    empty = {
        "domains": [],
        "algorithms": [],
        "patterns": [],
        "micro_skills": [],
        "failure_types": [],
        "pattern_clusters": [],
    }
    if not isinstance(data, dict):
        return empty
    for key in empty.keys():
        items = data.get(key, [])
        if isinstance(items, list):
            empty[key] = [normalize_label(x) for x in items if normalize_label(x)]
    return empty


def merge_unique_lists(*lists: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for item in lst:
            item = normalize_label(item)
            if item and item not in seen:
                seen.add(item)
                out.append(item)
    return out


def enrich_row(
    raw: Dict[str, Any],
    previous_metadata: Dict[str, Any],
    *,
    taxonomy: Dict[str, List[str]],
    mapping: Dict[str, Dict[str, str]],
    model: str,
    ollama_url: str,
    timeout: int,
    retries: int,
    temperature: float,
    sleep_between_requests: float,
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None, str | None, List[str]]:
    fields = build_problem_fields(raw)
    prompt = PROMPT_TEMPLATE.format(
        allowed_domains=format_allowed_list(taxonomy["domains"]),
        allowed_algorithms=format_allowed_list(taxonomy["algorithms"]),
        allowed_patterns=format_allowed_list(taxonomy["patterns"]),
        allowed_micro_skills=format_allowed_list(taxonomy["micro_skills"]),
        allowed_failure_types=format_allowed_list(taxonomy["failure_types"]),
        allowed_pattern_clusters=format_allowed_list(taxonomy["pattern_clusters"]),
        pass1_metadata=json.dumps(previous_metadata, ensure_ascii=False, indent=2),
        **fields,
    )

    last_error = None
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
            parsed = extract_json(response.response_text)
            cleaned, _ = sanitize_metadata_shape(parsed, include_proposed=True)
            proposed = sanitize_proposed_labels(cleaned.get("proposed_new_labels"))

            taxonomy_payload = cleaned["taxonomy"]
            filtered_taxonomy: Dict[str, Dict[str, float]] = {}
            proposed_additions: Dict[str, List[str]] = {k: list(v) for k, v in proposed.items()}

            for category, max_items in (
                ("domains", 3),
                ("algorithms", 3),
                ("patterns", 4),
                ("micro_skills", 10),
            ):
                remapped, extra = remap_with_lookup(
                    taxonomy_payload.get(category, {}),
                    category=category,
                    mapping=mapping,
                    allowed=taxonomy[category],
                )
                filtered_taxonomy[category] = select_top_items(remapped, max_items)
                proposed_additions[category] = merge_unique_lists(proposed_additions.get(category, []), extra)

            skill_model = cleaned["skill_model"]
            remapped_requires, extra_requires = remap_skill_weight_map(
                skill_model.get("requires", {}),
                mapping=mapping,
                allowed_micro_skills=taxonomy["micro_skills"],
                allowed_patterns=taxonomy["patterns"],
            )
            remapped_trains, extra_trains = remap_skill_weight_map(
                skill_model.get("trains", {}),
                mapping=mapping,
                allowed_micro_skills=taxonomy["micro_skills"],
                allowed_patterns=taxonomy["patterns"],
            )
            remapped_tests, extra_tests = remap_skill_weight_map(
                skill_model.get("tests", {}),
                mapping=mapping,
                allowed_micro_skills=taxonomy["micro_skills"],
                allowed_patterns=taxonomy["patterns"],
            )
            proposed_additions["micro_skills"] = merge_unique_lists(
                proposed_additions.get("micro_skills", []),
                extra_requires,
                extra_trains,
                extra_tests,
            )

            failure_model = cleaned["failure_model"]
            failure_payload = failure_model.get("common_failures", {})
            allowed_failure_types = {normalize_label(x) for x in taxonomy["failure_types"]}
            allowed_affected = merge_unique_lists(taxonomy["micro_skills"], taxonomy["patterns"])
            remapped_failures: Dict[str, Dict[str, Any]] = {}
            for failure_name, payload in failure_payload.items():
                failure_key, allowed_failure = remap_label(
                    failure_name,
                    category="failure_types",
                    mapping=mapping,
                    allowed=taxonomy["failure_types"],
                )
                if not allowed_failure:
                    proposed_additions["failure_types"] = merge_unique_lists(proposed_additions.get("failure_types", []), [failure_key])
                    continue
                affected = []
                if isinstance(payload, dict):
                    raw_affected = payload.get("affected_skills", [])
                    if isinstance(raw_affected, (list, tuple)):
                        for item in raw_affected:
                            mapped, allowed_skill = remap_skill_label(
                                item,
                                mapping=mapping,
                                allowed_micro_skills=taxonomy["micro_skills"],
                                allowed_patterns=taxonomy["patterns"],
                            )
                            if allowed_skill and mapped in allowed_affected:
                                affected.append(mapped)
                            else:
                                proposed_additions["micro_skills"] = merge_unique_lists(proposed_additions.get("micro_skills", []), [mapped])
                remapped_failures[failure_key] = {
                    "affected_skills": merge_unique_lists(affected),
                    "severity": clamp01(payload.get("severity", 0.0)) if isinstance(payload, dict) else 0.0,
                }

            pattern_cluster = normalize_label(cleaned.get("pattern_cluster", ""))
            mapped_cluster = mapping.get("pattern_clusters", {}).get(pattern_cluster, pattern_cluster)
            if mapped_cluster not in {normalize_label(x) for x in taxonomy["pattern_clusters"]}:
                if mapped_cluster:
                    proposed_additions["pattern_clusters"] = merge_unique_lists(
                        proposed_additions.get("pattern_clusters", []),
                        [mapped_cluster],
                    )
                mapped_cluster = ""

            metadata = {
                "taxonomy": filtered_taxonomy,
                "difficulty_vector": cleaned["difficulty_vector"],
                "skill_model": {
                    "requires": select_top_items(remapped_requires, 10),
                    "trains": select_top_items(remapped_trains, 10),
                    "tests": select_top_items(remapped_tests, 10),
                },
                "failure_model": {
                    "common_failures": remapped_failures,
                },
                "learning_role": cleaned["learning_role"],
                "pattern_cluster": mapped_cluster,
                "estimated_time_minutes": clamp_int(cleaned.get("estimated_time_minutes", 0), default=0, minimum=0, maximum=240),
                "solution_dna_summary": str(cleaned.get("solution_dna_summary", "") or "").strip(),
                "metadata_confidence": clamp01(cleaned.get("metadata_confidence", 0.0)),
                "proposed_new_labels": {
                    "domains": merge_unique_lists(proposed_additions.get("domains", [])),
                    "algorithms": merge_unique_lists(proposed_additions.get("algorithms", [])),
                    "patterns": merge_unique_lists(proposed_additions.get("patterns", [])),
                    "micro_skills": merge_unique_lists(proposed_additions.get("micro_skills", [])),
                    "failure_types": merge_unique_lists(proposed_additions.get("failure_types", [])),
                    "pattern_clusters": merge_unique_lists(proposed_additions.get("pattern_clusters", [])),
                },
            }

            # Ensure domains/algorithms/patterns/micro_skills never exceed the requested caps.
            metadata["taxonomy"]["domains"] = select_top_items(metadata["taxonomy"]["domains"], 3)
            metadata["taxonomy"]["algorithms"] = select_top_items(metadata["taxonomy"]["algorithms"], 3)
            metadata["taxonomy"]["patterns"] = select_top_items(metadata["taxonomy"]["patterns"], 4)
            metadata["taxonomy"]["micro_skills"] = select_top_items(metadata["taxonomy"]["micro_skills"], 10)

            issues = validate_pass3_metadata(
                metadata,
                raw_difficulty=fields["difficulty"],
                allowed_domains=taxonomy["domains"],
                allowed_algorithms=taxonomy["algorithms"],
                allowed_patterns=taxonomy["patterns"],
                allowed_micro_skills=taxonomy["micro_skills"],
                allowed_failure_types=taxonomy["failure_types"],
                allowed_pattern_clusters=taxonomy["pattern_clusters"],
            )
            status = status_from_issues(issues)
            if metadata["proposed_new_labels"]:
                if any(metadata["proposed_new_labels"].values()) and "proposed_new_labels_present" not in issues:
                    issues.append("proposed_new_labels_present")
                status = status_from_issues(issues)
            return metadata, response.raw, None, issues if isinstance(issues, list) else []
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Failed to parse/validate pass3 JSON for %s on attempt %s/%s: %s", fields["id"], attempt, retries, exc)
            if attempt < retries:
                continue
            return None, response.raw, last_error, [f"invalid_json:{last_error}"]
    return None, None, last_error or "unexpected_failure", [last_error or "unexpected_failure"]


def build_report(
    *,
    total: int,
    valid_count: int,
    needs_review_count: int,
    failed_count: int,
    proposed_counter: Counter,
    issue_counter: Counter,
    lowest_confidence_rows: List[Dict[str, Any]],
    domain_counter: Counter,
    algorithm_counter: Counter,
    pattern_counter: Counter,
    difficulty_buckets: Counter,
) -> str:
    lines = [
        "# Final Validation Report",
        "",
        f"- Total processed: {total}",
        f"- Valid: {valid_count}",
        f"- Needs review: {needs_review_count}",
        f"- Failed: {failed_count}",
        "",
        "## Top Proposed New Labels",
    ]
    if proposed_counter:
        for label, freq in proposed_counter.most_common(50):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")
    lines.extend(["", "## Common Validation Issues"])
    if issue_counter:
        for label, freq in issue_counter.most_common(50):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")

    lines.extend(["", "## Lowest Metadata Confidence Rows"])
    if lowest_confidence_rows:
        for item in lowest_confidence_rows[:20]:
            lines.append(
                f"- `{item['problem_id']}`: confidence={item['confidence']:.3f}, status={item['status']}, issues={item['issues']}"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Domain Distribution"])
    if domain_counter:
        for label, freq in domain_counter.most_common(50):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")

    lines.extend(["", "## Algorithm Distribution"])
    if algorithm_counter:
        for label, freq in algorithm_counter.most_common(50):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")

    lines.extend(["", "## Pattern Distribution"])
    if pattern_counter:
        for label, freq in pattern_counter.most_common(50):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")

    lines.extend(["", "## Difficulty Distribution"])
    for bucket, freq in difficulty_buckets.items():
        lines.append(f"- `{bucket}`: {freq}")

    return "\n".join(lines) + "\n"


def bucket_difficulty(value: float) -> str:
    if value < 0.2:
        return "0.0-0.2"
    if value < 0.4:
        return "0.2-0.4"
    if value < 0.6:
        return "0.4-0.6"
    if value < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run constrained pass 3 enrichment.")
    parser.add_argument("--input", required=True, help="Input raw or pass1 JSONL")
    parser.add_argument("--taxonomy", required=True, help="Clean taxonomy JSON")
    parser.add_argument("--label-mapping", required=True, help="Canonical label mapping JSON")
    parser.add_argument("--output", required=True, help="Final JSONL output")
    parser.add_argument("--failures", required=True, help="Failure JSONL output")
    parser.add_argument("--report", required=True, help="Final validation report markdown")
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
    taxonomy_path = Path(args.taxonomy)
    mapping_path = Path(args.label_mapping)
    output_path = ensure_parent_dir(args.output)
    failures_path = ensure_parent_dir(args.failures)
    report_path = ensure_parent_dir(args.report)

    if args.force:
        for path in (output_path, failures_path, report_path):
            if path.exists():
                path.unlink()

    taxonomy = load_taxonomy(taxonomy_path)
    mapping = load_label_mapping(mapping_path)
    records = load_input_records(input_path)
    existing_ids = read_existing_ids(output_path) if args.resume and not args.force else set()

    total = 0
    valid_count = 0
    needs_review_count = 0
    failed_count = 0
    proposed_counter: Counter = Counter()
    issue_counter: Counter = Counter()
    domain_counter: Counter = Counter()
    algorithm_counter: Counter = Counter()
    pattern_counter: Counter = Counter()
    difficulty_buckets: Counter = Counter()
    lowest_confidence_rows: List[Dict[str, Any]] = []

    logger.info("Loaded %s rows from %s", len(records), input_path)

    for index, record in enumerate(records):
        if index < args.start_index:
            continue
        if args.limit is not None and total >= args.limit:
            break

        raw, previous_metadata = unwrap_raw_and_metadata(record)
        problem_id = infer_problem_id(raw, fallback=f"row_{index}")
        if not args.force and problem_id in existing_ids:
            continue

        total += 1
        logger.info("Processing %s (%s/%s)", problem_id, total, args.limit if args.limit is not None else "all")

        try:
            metadata, response_raw, error, parse_issues = enrich_row(
                raw,
                previous_metadata,
                taxonomy=taxonomy,
                mapping=mapping,
                model=args.model,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                retries=args.retries,
                temperature=args.temperature,
                sleep_between_requests=args.sleep_between_requests,
            )
            if metadata is None:
                failed_count += 1
                issues = parse_issues if parse_issues else [error or "unknown_error"]
                append_jsonl(
                    failures_path,
                    [
                        {
                            "problem_id": problem_id,
                            "raw": raw,
                            "error": error,
                            "issues": issues,
                            "raw_response": response_raw,
                        }
                    ],
                )
                append_jsonl(
                    output_path,
                    [
                        {
                            "problem_id": problem_id,
                            "raw": raw,
                            "metadata": None,
                            "validation": {
                                "status": "failed",
                                "issues": issues,
                            },
                        }
                    ],
                )
                continue

            issues = validate_pass3_metadata(
                metadata,
                raw_difficulty=extract_text_value(raw, ("difficulty", "level", "difficulty_level")),
                allowed_domains=taxonomy["domains"],
                allowed_algorithms=taxonomy["algorithms"],
                allowed_patterns=taxonomy["patterns"],
                allowed_micro_skills=taxonomy["micro_skills"],
                allowed_failure_types=taxonomy["failure_types"],
                allowed_pattern_clusters=taxonomy["pattern_clusters"],
            )
            status = status_from_issues(issues)
            if metadata.get("proposed_new_labels"):
                if any(metadata["proposed_new_labels"].values()):
                    proposed_counter.update(
                        label
                        for labels in metadata["proposed_new_labels"].values()
                        for label in labels
                    )
                    if "proposed_new_labels_present" not in issues:
                        issues.append("proposed_new_labels_present")
                status = status_from_issues(issues)

            if status == "valid":
                valid_count += 1
            elif status == "needs_review":
                needs_review_count += 1
            else:
                failed_count += 1

            for label, weight in metadata.get("taxonomy", {}).get("domains", {}).items():
                domain_counter[label] += 1
            for label, weight in metadata.get("taxonomy", {}).get("algorithms", {}).items():
                algorithm_counter[label] += 1
            for label, weight in metadata.get("taxonomy", {}).get("patterns", {}).items():
                pattern_counter[label] += 1

            diff = metadata.get("difficulty_vector", {})
            overall = float(diff.get("overall", 0.0) or 0.0) if isinstance(diff, dict) else 0.0
            difficulty_buckets[bucket_difficulty(overall)] += 1

            confidence = float(metadata.get("metadata_confidence", 0.0) or 0.0)
            lowest_confidence_rows.append(
                {
                    "problem_id": problem_id,
                    "confidence": confidence,
                    "status": status,
                    "issues": issues[:10],
                }
            )
            lowest_confidence_rows = sorted(lowest_confidence_rows, key=lambda item: item["confidence"])[:50]
            issue_counter.update(issues)

            append_jsonl(
                output_path,
                [
                    {
                        "problem_id": problem_id,
                        "raw": raw,
                        "metadata": metadata,
                        "validation": {
                            "status": status,
                            "issues": issues,
                        },
                    }
                ],
            )

            if status != "valid":
                append_jsonl(
                    failures_path,
                    [
                        {
                            "problem_id": problem_id,
                            "raw": raw,
                            "error": None,
                            "issues": issues,
                            "raw_response": response_raw,
                            "validation_status": status,
                        }
                    ],
                )
        except Exception as exc:
            failed_count += 1
            logger.exception("Unexpected error while processing %s", problem_id)
            issue_counter.update([str(exc)])
            append_jsonl(
                failures_path,
                [
                    {
                        "problem_id": problem_id,
                        "raw": raw,
                        "error": str(exc),
                        "issues": [str(exc)],
                        "raw_response": None,
                    }
                ],
            )
            append_jsonl(
                output_path,
                [
                    {
                        "problem_id": problem_id,
                        "raw": raw,
                        "metadata": None,
                        "validation": {
                            "status": "failed",
                            "issues": [str(exc)],
                        },
                    }
                ],
            )

    report = build_report(
        total=total,
        valid_count=valid_count,
        needs_review_count=needs_review_count,
        failed_count=failed_count,
        proposed_counter=proposed_counter,
        issue_counter=issue_counter,
        lowest_confidence_rows=lowest_confidence_rows,
        domain_counter=domain_counter,
        algorithm_counter=algorithm_counter,
        pattern_counter=pattern_counter,
        difficulty_buckets=difficulty_buckets,
    )
    report_path.write_text(report, encoding="utf-8")

    logger.info(
        "Finished. total=%s valid=%s needs_review=%s failed=%s",
        total,
        valid_count,
        needs_review_count,
        failed_count,
    )
    logger.info("Wrote report to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
