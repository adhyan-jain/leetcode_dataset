from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .text_utils import clamp01, clamp_int, normalize_label


PASS1_REQUIRED_KEYS = {
    "taxonomy",
    "difficulty_vector",
    "skill_model",
    "failure_model",
    "learning_role",
    "pattern_cluster",
    "estimated_time_minutes",
    "solution_dna_summary",
    "metadata_confidence",
}

PASS3_REQUIRED_KEYS = PASS1_REQUIRED_KEYS | {"proposed_new_labels"}


def _sanitize_weight_map(value: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(value, dict):
        for key, raw_weight in value.items():
            label = normalize_label(key)
            if not label:
                continue
            weight = clamp01(raw_weight)
            if weight <= 0:
                continue
            out[label] = weight
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                label = normalize_label(item)
                if label:
                    out[label] = 1.0
            elif isinstance(item, dict):
                label = normalize_label(item.get("label") or item.get("name"))
                if not label:
                    continue
                weight = clamp01(item.get("weight", 1.0))
                if weight > 0:
                    out[label] = weight
    return out


def _sanitize_common_failures(value: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(value, dict):
        return out
    for failure_name, payload in value.items():
        failure_key = normalize_label(failure_name)
        if not failure_key:
            continue
        if isinstance(payload, dict):
            affected = payload.get("affected_skills", [])
            if isinstance(affected, (list, tuple)):
                affected_skills = [normalize_label(x) for x in affected if normalize_label(x)]
            else:
                affected_skills = []
            out[failure_key] = {
                "affected_skills": affected_skills,
                "severity": clamp01(payload.get("severity", 0.0)),
            }
        else:
            out[failure_key] = {"affected_skills": [], "severity": 0.0}
    return out


def sanitize_metadata_shape(metadata: Any, *, include_proposed: bool = False) -> tuple[Dict[str, Any], List[str]]:
    issues: List[str] = []
    if not isinstance(metadata, dict):
        return {}, ["metadata_not_object"]

    taxonomy = metadata.get("taxonomy") if isinstance(metadata.get("taxonomy"), dict) else {}
    skill_model = metadata.get("skill_model") if isinstance(metadata.get("skill_model"), dict) else {}
    failure_model = metadata.get("failure_model") if isinstance(metadata.get("failure_model"), dict) else {}
    learning_role = metadata.get("learning_role") if isinstance(metadata.get("learning_role"), dict) else {}
    difficulty_vector = metadata.get("difficulty_vector") if isinstance(metadata.get("difficulty_vector"), dict) else {}

    cleaned = {
        "taxonomy": {
            "domains": _sanitize_weight_map(taxonomy.get("domains")),
            "algorithms": _sanitize_weight_map(taxonomy.get("algorithms")),
            "patterns": _sanitize_weight_map(taxonomy.get("patterns")),
            "micro_skills": _sanitize_weight_map(taxonomy.get("micro_skills")),
        },
        "difficulty_vector": {
            "overall": clamp01(difficulty_vector.get("overall", 0.0)),
            "conceptual": clamp01(difficulty_vector.get("conceptual", 0.0)),
            "implementation": clamp01(difficulty_vector.get("implementation", 0.0)),
            "edge_cases": clamp01(difficulty_vector.get("edge_cases", 0.0)),
        },
        "skill_model": {
            "requires": _sanitize_weight_map(skill_model.get("requires")),
            "trains": _sanitize_weight_map(skill_model.get("trains")),
            "tests": _sanitize_weight_map(skill_model.get("tests")),
        },
        "failure_model": {
            "common_failures": _sanitize_common_failures(failure_model.get("common_failures")),
        },
        "learning_role": {
            "first_exposure": clamp01(learning_role.get("first_exposure", 0.0)),
            "practice": clamp01(learning_role.get("practice", 0.0)),
            "review": clamp01(learning_role.get("review", 0.0)),
            "assessment": clamp01(learning_role.get("assessment", 0.0)),
            "recovery": clamp01(learning_role.get("recovery", 0.0)),
            "challenge": clamp01(learning_role.get("challenge", 0.0)),
        },
        "pattern_cluster": normalize_label(metadata.get("pattern_cluster", "")),
        "estimated_time_minutes": clamp_int(metadata.get("estimated_time_minutes", 0), default=0, minimum=0, maximum=240),
        "solution_dna_summary": str(metadata.get("solution_dna_summary", "") or "").strip(),
        "metadata_confidence": clamp01(metadata.get("metadata_confidence", 0.0)),
    }

    if include_proposed:
        proposed = metadata.get("proposed_new_labels") if isinstance(metadata.get("proposed_new_labels"), dict) else {}
        cleaned["proposed_new_labels"] = {
            "domains": [normalize_label(x) for x in proposed.get("domains", []) if normalize_label(x)],
            "algorithms": [normalize_label(x) for x in proposed.get("algorithms", []) if normalize_label(x)],
            "patterns": [normalize_label(x) for x in proposed.get("patterns", []) if normalize_label(x)],
            "micro_skills": [normalize_label(x) for x in proposed.get("micro_skills", []) if normalize_label(x)],
            "failure_types": [normalize_label(x) for x in proposed.get("failure_types", []) if normalize_label(x)],
            "pattern_clusters": [normalize_label(x) for x in proposed.get("pattern_clusters", []) if normalize_label(x)],
        }

    missing = [key for key in PASS3_REQUIRED_KEYS if key not in metadata] if include_proposed else [key for key in PASS1_REQUIRED_KEYS if key not in metadata]
    if missing:
        issues.append("missing_keys:" + ",".join(sorted(missing)))
    return cleaned, issues


def sanitize_allowed_labels(labels: Iterable[str]) -> List[str]:
    clean = []
    seen = set()
    for label in labels:
        normalized = normalize_label(label)
        if normalized and normalized not in seen:
            seen.add(normalized)
            clean.append(normalized)
    return clean


def filter_to_allowed_weight_map(weight_map: Mapping[str, float], allowed: Sequence[str]) -> tuple[Dict[str, float], List[str]]:
    allowed_set = set(sanitize_allowed_labels(allowed))
    kept: Dict[str, float] = {}
    proposed: List[str] = []
    for label, weight in weight_map.items():
        normalized = normalize_label(label)
        if not normalized:
            continue
        if normalized in allowed_set:
            kept[normalized] = clamp01(weight)
        else:
            proposed.append(normalized)
    return kept, sorted(set(proposed))


def validate_pass3_metadata(
    metadata: Dict[str, Any],
    *,
    raw_difficulty: str | None = None,
    allowed_domains: Sequence[str] = (),
    allowed_algorithms: Sequence[str] = (),
    allowed_patterns: Sequence[str] = (),
    allowed_micro_skills: Sequence[str] = (),
    allowed_failure_types: Sequence[str] = (),
    allowed_pattern_clusters: Sequence[str] = (),
) -> List[str]:
    issues: List[str] = []

    if not isinstance(metadata, dict):
        return ["metadata_not_object"]

    for key in PASS3_REQUIRED_KEYS:
        if key not in metadata:
            issues.append(f"missing_{key}")

    taxonomy = metadata.get("taxonomy", {})
    skill_model = metadata.get("skill_model", {})
    failure_model = metadata.get("failure_model", {})
    learning_role = metadata.get("learning_role", {})
    difficulty_vector = metadata.get("difficulty_vector", {})
    proposed_new_labels = metadata.get("proposed_new_labels", {})

    if not isinstance(proposed_new_labels, dict):
        issues.append("proposed_new_labels_not_object")
        proposed_new_labels = {}

    for label_type, allowed in (
        ("domains", allowed_domains),
        ("algorithms", allowed_algorithms),
        ("patterns", allowed_patterns),
        ("micro_skills", allowed_micro_skills),
    ):
        labels = taxonomy.get(label_type, {})
        if not isinstance(labels, dict):
            issues.append(f"{label_type}_not_object")
            continue
        allowed_set = set(sanitize_allowed_labels(allowed))
        for label, weight in labels.items():
            if normalize_label(label) not in allowed_set:
                issues.append(f"unknown_{label_type[:-1]}:{normalize_label(label)}")
            if not isinstance(weight, (int, float)) or not (0.0 <= float(weight) <= 1.0):
                issues.append(f"bad_weight:{label_type}:{label}")

    if not isinstance(difficulty_vector, dict):
        issues.append("difficulty_vector_not_object")
    else:
        for key in ("overall", "conceptual", "implementation", "edge_cases"):
            value = difficulty_vector.get(key)
            if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
                issues.append(f"bad_difficulty:{key}")

    if not isinstance(skill_model, dict) or not isinstance(failure_model, dict) or not isinstance(learning_role, dict):
        issues.append("malformed_sections")

    requires = skill_model.get("requires", {}) if isinstance(skill_model, dict) else {}
    trains = skill_model.get("trains", {}) if isinstance(skill_model, dict) else {}
    tests = skill_model.get("tests", {}) if isinstance(skill_model, dict) else {}
    if not requires and not trains and not tests:
        issues.append("empty_skill_model")
    if not trains:
        issues.append("no_trainable_skill")

    patterns = taxonomy.get("patterns", {}) if isinstance(taxonomy, dict) else {}
    micro_skills = taxonomy.get("micro_skills", {}) if isinstance(taxonomy, dict) else {}
    if not patterns:
        issues.append("no_patterns")
    if len(micro_skills) < 3 and str(raw_difficulty or "").lower() != "easy":
        issues.append("too_few_micro_skills")

    if isinstance(proposed_new_labels, dict):
        if any(proposed_new_labels.get(k) for k in proposed_new_labels):
            issues.append("proposed_new_labels_present")

    confidence = metadata.get("metadata_confidence", 0.0)
    if isinstance(confidence, (int, float)) and float(confidence) < 0.6:
        issues.append("low_confidence")

    if str(raw_difficulty or "").lower() == "hard":
        overall = difficulty_vector.get("overall", 0.0) if isinstance(difficulty_vector, dict) else 0.0
        if isinstance(overall, (int, float)) and float(overall) < 0.35:
            issues.append("hard_but_low_overall")
    if str(raw_difficulty or "").lower() == "easy":
        overall = difficulty_vector.get("overall", 0.0) if isinstance(difficulty_vector, dict) else 0.0
        if isinstance(overall, (int, float)) and float(overall) > 0.75:
            issues.append("easy_but_high_overall")

    if isinstance(allowed_pattern_clusters, Sequence) and allowed_pattern_clusters:
        cluster = normalize_label(metadata.get("pattern_cluster", ""))
        if cluster and cluster not in set(sanitize_allowed_labels(allowed_pattern_clusters)):
            issues.append(f"unknown_pattern_cluster:{cluster}")

    return issues


def status_from_issues(issues: Sequence[str]) -> str:
    if not issues:
        return "valid"
    severe_markers = {
        "metadata_not_object",
        "missing_taxonomy",
        "missing_difficulty_vector",
        "missing_skill_model",
        "missing_failure_model",
        "missing_learning_role",
        "missing_pattern_cluster",
        "missing_estimated_time_minutes",
        "missing_solution_dna_summary",
        "missing_metadata_confidence",
        "empty_skill_model",
        "no_patterns",
        "too_few_micro_skills",
        "low_confidence",
        "no_trainable_skill",
        "hard_but_low_overall",
        "easy_but_high_overall",
        "proposed_new_labels_present",
    }
    if any(issue in severe_markers or issue.startswith("unknown_") or issue.startswith("bad_") for issue in issues):
        return "needs_review"
    return "failed"

