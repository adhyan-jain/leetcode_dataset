from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.io_utils import ensure_parent_dir, read_jsonl, read_json_file, write_json_file
from utils.text_utils import (
    best_match,
    is_probably_long_label,
    looks_like_sentence,
    normalize_label,
    normalize_whitespace,
)


logger = logging.getLogger("build_taxonomy")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


BUILT_IN_ALIAS_MAP = {
    "domains": {
        "arr": "array",
        "arrays": "array",
        "array_data_structure": "array",
        "string": "string",
        "strings": "string",
        "linked_list": "linked_list",
        "linked_lists": "linked_list",
        "tree": "tree",
        "trees": "tree",
        "graph": "graph",
        "graphs": "graph",
        "hash_map": "hash_map",
        "hash_table": "hash_map",
        "hash_tables": "hash_map",
        "heap": "heap",
        "heaps": "heap",
        "queue": "queue",
        "queues": "queue",
        "stack": "stack",
        "stacks": "stack",
        "math": "math",
        "greedy": "greedy",
        "dynamic_programming": "dynamic_programming",
        "dp": "dynamic_programming",
    },
    "algorithms": {
        "binary_search_answer": "binary_search_on_answer",
        "binary_search_on_answers": "binary_search_on_answer",
        "parametric_search": "binary_search_on_answer",
        "monotonic_predicate": "monotonic_predicate_design",
        "feasibility_check": "monotonic_predicate_design",
        "feasible_function": "monotonic_predicate_design",
        "feasibility_predicate": "monotonic_predicate_design",
        "sliding_window_var": "sliding_window_variable",
        "variable_window": "sliding_window_variable",
        "dynamic_sliding_window": "sliding_window_variable",
        "two_pointer": "two_pointers",
        "dfs": "depth_first_search",
        "bfs": "breadth_first_search",
        "union_find": "disjoint_set_union",
        "uf": "disjoint_set_union",
        "toposort": "topological_sort",
    },
    "patterns": {
        "off_by_one": "boundary_handling",
        "off_by_one_error": "boundary_handling",
        "boundary_bug": "boundary_handling",
        "wrong_boundary": "boundary_handling",
        "boundary_error": "boundary_handling",
        "monotonic_predicate": "monotonic_predicate_design",
        "feasibility_check": "monotonic_predicate_design",
        "feasible_function": "monotonic_predicate_design",
        "feasibility_predicate": "monotonic_predicate_design",
        "binary_search_answer": "binary_search_on_answer",
        "binary_search_on_answers": "binary_search_on_answer",
        "parametric_search": "binary_search_on_answer",
        "sliding_window_var": "sliding_window_variable",
        "variable_window": "sliding_window_variable",
        "dynamic_sliding_window": "sliding_window_variable",
        "two_pointer": "two_pointers",
        "two_pointers": "two_pointers",
        "prefix sum": "prefix_sum",
        "prefix_sums": "prefix_sum",
    },
    "micro_skills": {
        "off_by_one": "boundary_handling",
        "off_by_one_error": "boundary_handling",
        "boundary_bug": "boundary_handling",
        "wrong_boundary": "boundary_handling",
        "boundary_error": "boundary_handling",
        "index_management": "indexing",
        "pointer_movement": "pointer_movement",
        "state_tracking": "state_tracking",
        "invariant_maintenance": "invariant_maintenance",
        "prefix_sum": "prefix_sum",
        "prefix_sums": "prefix_sum",
    },
    "failure_types": {
        "off_by_one": "boundary_handling",
        "off_by_one_error": "boundary_handling",
        "boundary_bug": "boundary_handling",
        "wrong_boundary": "boundary_handling",
        "boundary_error": "boundary_handling",
        "wrong_predicate": "incorrect_predicate",
        "incorrect_predicate": "incorrect_predicate",
        "wrong_transition": "incorrect_transition",
    },
    "pattern_clusters": {
        "off_by_one": "boundary_handling",
        "boundary_bug": "boundary_handling",
        "monotonic_predicate": "monotonic_predicate_design",
        "binary_search_answer": "binary_search_on_answer",
        "sliding_window_var": "sliding_window_variable",
    },
}


def load_manual_overrides(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    data = read_json_file(path)
    if not isinstance(data, dict):
        return {}
    overrides: Dict[str, Dict[str, str]] = {}
    for category, mapping in data.items():
        if not isinstance(mapping, dict):
            continue
        overrides[category] = {}
        for raw_label, canonical in mapping.items():
            raw_norm = normalize_label(raw_label)
            canonical_norm = normalize_label(canonical)
            if raw_norm and canonical_norm:
                overrides[category][raw_norm] = canonical_norm
    return overrides


def setup_collector() -> Dict[str, Dict[str, Any]]:
    categories = ("domains", "algorithms", "patterns", "micro_skills", "failure_types", "pattern_clusters")
    return {
        category: {
            "counts": Counter(),
            "confidence_sum": defaultdict(float),
            "confidence_count": defaultdict(int),
            "source_rows": defaultdict(list),
        }
        for category in categories
    }


def add_label(bucket: Dict[str, Any], label: str, confidence: float, row_index: int) -> None:
    label = normalize_label(label)
    if not label:
        return
    bucket["counts"][label] += 1
    bucket["confidence_sum"][label] += confidence
    bucket["confidence_count"][label] += 1
    bucket["source_rows"][label].append(row_index)


def collect_from_row(row: Dict[str, Any], collector: Dict[str, Dict[str, Any]], row_index: int) -> None:
    metadata = row.get("pass1_metadata") if isinstance(row.get("pass1_metadata"), dict) else row.get("metadata")
    if not isinstance(metadata, dict):
        return
    confidence = float(metadata.get("metadata_confidence", 0.0) or 0.0)

    taxonomy = metadata.get("taxonomy", {})
    if isinstance(taxonomy, dict):
        for category in ("domains", "algorithms", "patterns", "micro_skills"):
            labels = taxonomy.get(category, {})
            if isinstance(labels, dict):
                for label in labels.keys():
                    add_label(collector[category], label, confidence, row_index)

    skill_model = metadata.get("skill_model", {})
    if isinstance(skill_model, dict):
        for category in ("requires", "trains", "tests"):
            labels = skill_model.get(category, {})
            if isinstance(labels, dict):
                for label in labels.keys():
                    add_label(collector["micro_skills"], label, confidence, row_index)

    failure_model = metadata.get("failure_model", {})
    if isinstance(failure_model, dict):
        common_failures = failure_model.get("common_failures", {})
        if isinstance(common_failures, dict):
            for failure_name, payload in common_failures.items():
                add_label(collector["failure_types"], failure_name, confidence, row_index)
                if isinstance(payload, dict):
                    affected = payload.get("affected_skills", [])
                    if isinstance(affected, (list, tuple)):
                        for skill in affected:
                            add_label(collector["micro_skills"], skill, confidence, row_index)

    pattern_cluster = normalize_label(metadata.get("pattern_cluster", ""))
    if pattern_cluster:
        add_label(collector["pattern_clusters"], pattern_cluster, confidence, row_index)


def canonicalize_label(category: str, label: str, known_canonicals: List[str], threshold: int) -> str:
    label = normalize_label(label)
    if not label:
        return label
    alias = BUILT_IN_ALIAS_MAP.get(category, {}).get(label)
    if alias:
        return alias
    if known_canonicals:
        match, score = best_match(label, known_canonicals)
        if match and score >= threshold:
            return match
    return label


def build_category_mapping(
    category: str,
    collector: Dict[str, Any],
    *,
    threshold: int,
    overrides: Dict[str, Dict[str, str]],
) -> tuple[List[str], Dict[str, str], List[Dict[str, Any]]]:
    counts: Counter = collector["counts"]
    confidences = collector["confidence_sum"]
    confidence_counts = collector["confidence_count"]
    raw_labels_sorted = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    mapping: Dict[str, str] = {}
    canonical_counts: Counter = Counter()
    canonical_order: List[str] = []

    override_map = overrides.get(category, {})

    for raw_label, freq in raw_labels_sorted:
        canonical = override_map.get(raw_label)
        if not canonical:
            canonical = canonicalize_label(category, raw_label, canonical_order, threshold)
        if canonical not in canonical_order:
            canonical_order.append(canonical)
        mapping[raw_label] = canonical
        canonical_counts[canonical] += freq

    canonical_labels = sorted(canonical_counts.keys(), key=lambda lab: (-canonical_counts[lab], lab))
    suspicious = []
    for raw_label, freq in raw_labels_sorted:
        avg_conf = confidences[raw_label] / max(confidence_counts[raw_label], 1)
        if (
            freq <= 1
            or is_probably_long_label(raw_label)
            or looks_like_sentence(raw_label)
            or avg_conf < 0.6
        ):
            suspicious.append(
                {
                    "label": raw_label,
                    "frequency": freq,
                    "canonical": mapping.get(raw_label, raw_label),
                    "avg_confidence": round(avg_conf, 3),
                    "rows": collector["source_rows"][raw_label][:10],
                }
            )
    return canonical_labels, mapping, suspicious


def build_report_section(
    category: str,
    collector: Dict[str, Any],
    canonical_labels: List[str],
    mapping: Dict[str, str],
    suspicious: List[Dict[str, Any]],
) -> str:
    counts: Counter = collector["counts"]
    top_raw = counts.most_common(50)
    lines = [f"## {category}", ""]
    lines.append(f"- Unique raw labels: {len(counts)}")
    lines.append(f"- Canonical labels: {len(canonical_labels)}")
    lines.append("")
    lines.append("### Top 50 raw labels")
    for label, freq in top_raw:
        lines.append(f"- `{label}`: {freq} -> `{mapping.get(label, label)}`")
    lines.append("")
    lines.append("### Suspicious labels")
    if suspicious:
        for item in suspicious[:50]:
            lines.append(
                f"- `{item['label']}` (freq={item['frequency']}, canonical=`{item['canonical']}`, avg_conf={item['avg_confidence']}, rows={item['rows']})"
            )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def manual_review_suggestions(
    all_mappings: Dict[str, Dict[str, str]],
    collectors: Dict[str, Dict[str, Any]],
) -> List[str]:
    suggestions: List[str] = []
    for category, collector in collectors.items():
        for label, freq in collector["counts"].items():
            canonical = all_mappings.get(category, {}).get(label, label)
            if label == canonical:
                if freq <= 2:
                    suggestions.append(f"{category}:{label}")
    return suggestions[:100]


def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description="Build a cleaned taxonomy from pass1 output.")
    parser.add_argument("--input", required=True, help="Input JSONL from pass1_enriched")
    parser.add_argument("--taxonomy-output", required=True, help="Output taxonomy JSON path")
    parser.add_argument("--mapping-output", required=True, help="Output label mapping JSON path")
    parser.add_argument("--report-output", required=True, help="Output taxonomy report markdown path")
    parser.add_argument("--fuzzy-threshold", type=int, default=88)
    parser.add_argument("--manual-overrides", default="data/processed/manual_label_overrides.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    taxonomy_output = ensure_parent_dir(args.taxonomy_output)
    mapping_output = ensure_parent_dir(args.mapping_output)
    report_output = ensure_parent_dir(args.report_output)

    collector = setup_collector()
    rows = list(read_jsonl(input_path))
    logger.info("Loaded %s rows from %s", len(rows), input_path)

    for row_index, row in enumerate(rows):
        collect_from_row(row, collector, row_index)

    overrides = load_manual_overrides(Path(args.manual_overrides))
    if overrides:
        logger.info("Loaded manual overrides from %s", args.manual_overrides)

    taxonomy: Dict[str, List[str]] = {}
    mappings: Dict[str, Dict[str, str]] = {}
    suspicious_by_category: Dict[str, List[Dict[str, Any]]] = {}

    for category in collector.keys():
        canonical_labels, mapping, suspicious = build_category_mapping(
            category,
            collector[category],
            threshold=args.fuzzy_threshold,
            overrides=overrides,
        )
        taxonomy[category] = canonical_labels
        mappings[category] = mapping
        suspicious_by_category[category] = suspicious

    write_json_file(taxonomy_output, taxonomy)
    write_json_file(mapping_output, mappings)

    report_lines = [
        "# Taxonomy Build Report",
        "",
        f"- Source rows: {len(rows)}",
        f"- Fuzzy threshold: {args.fuzzy_threshold}",
        "",
    ]
    for category in collector.keys():
        report_lines.append(
            build_report_section(
                category,
                collector[category],
                taxonomy[category],
                mappings[category],
                suspicious_by_category[category],
            )
        )
    report_lines.append("## Suggested labels for manual review")
    suggestions = manual_review_suggestions(mappings, collector)
    if suggestions:
        for suggestion in suggestions:
            report_lines.append(f"- `{suggestion}`")
    else:
        report_lines.append("- None")

    report_output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    logger.info("Wrote taxonomy to %s", taxonomy_output)
    logger.info("Wrote mapping to %s", mapping_output)
    logger.info("Wrote report to %s", report_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

