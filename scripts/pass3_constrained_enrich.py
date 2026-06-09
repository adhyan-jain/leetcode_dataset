from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.io_utils import append_jsonl, ensure_parent_dir, extract_text_value, infer_problem_id, load_input_records, read_existing_ids, unwrap_raw_and_metadata
from utils.ollama_client import generate
from utils.text_utils import best_match, clamp01, clamp_int, normalize_label, normalize_whitespace, snake_case


logger = logging.getLogger("pass3_constrained_enrich")

CATEGORY_KEYS = (
    "domains",
    "algorithms",
    "patterns",
    "micro_skills",
    "failure_types",
    "pattern_clusters",
    "learning_targets",
    "problem_roles",
    "problem_traits",
)

CORE_MAIN_KEYS = ("domains", "algorithms", "patterns", "micro_skills")
WEIGHT_MAP_KEYS = ("taxonomy", "skill_model", "problem_roles", "problem_traits")

MAX_CAPS = {
    "domains": 5,
    "algorithms": 15,
    "patterns": 25,
    "micro_skills": 50,
    "failure_types": 30,
    "pattern_clusters": 20,
    "learning_targets": 30,
    "problem_roles": 10,
    "problem_traits": 20,
    "topic_paths": 15,
}

TRIVIAL_EASY_THRESHOLD = 0.40

STRICT_REVIEW_ISSUES = {
    "invalid_json",
    "missing_",
    "malformed_",
    "metadata_not_object",
    "unable_to_repair",
    "no_pattern",
    "empty_trains",
    "requires_gt_trains",
    "easy_high_overall",
    "hard_low_overall",
    "easy_high_requires",
    "pattern_cluster_not_allowed",
    "topic_path_not_allowed",
}

SOFT_REVIEW_ISSUES = {
    "low_confidence",
}

VECTOR_MATCH_THRESHOLDS = {
    "domains": 0.34,
    "algorithms": 0.52,
    "patterns": 0.48,
    "micro_skills": 0.43,
    "failure_types": 0.42,
    "learning_targets": 0.42,
    "pattern_clusters": 0.40,
    "problem_roles": 0.35,
    "problem_traits": 0.35,
}

BROAD_MICRO_SKILLS = {
    "dynamic_programming",
    "graph",
    "tree",
    "array",
    "string",
    "hash_table",
    "math",
    "binary_search",
    "backtracking",
    "greedy",
    "simulation",
    "design",
    "sql",
    "matrix",
    "heap",
    "trie",
    "stack_queue",
    "linked_list",
}

ROUTING_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "be",
    "because",
    "by",
    "can",
    "constraint",
    "constraints",
    "data",
    "design",
    "description",
    "each",
    "example",
    "examples",
    "find",
    "given",
    "have",
    "input",
    "is",
    "length",
    "not",
    "notice",
    "of",
    "one",
    "output",
    "problem",
    "return",
    "same",
    "solution",
    "such",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "use",
    "valid",
    "we",
    "would",
    "you",
    "your",
    "with",
    "without",
}

WRONG_CATEGORY_ALIASES = {
    "dynamic_programming": {"domains", "algorithms"},
    "binary_search": {"domains", "algorithms"},
    "shortest_path": {"algorithms", "patterns"},
    "union_find": {"algorithms"},
    "graph": {"domains"},
    "tree": {"domains"},
    "string": {"domains"},
    "array": {"domains"},
    "matrix": {"domains"},
    "heap": {"domains"},
    "trie": {"domains"},
    "linked_list": {"domains"},
    "stack_queue": {"domains"},
}


FAMILY_RULES = [
    {
        "name": "dynamic_programming",
        "triggers": {
            "dynamic_programming",
            "dp",
            "memoization",
            "recurrence",
            "fibonacci",
            "climbing_stairs",
            "edit_distance",
            "knapsack",
            "sequence_alignment",
            "house_robber",
            "palindrome_subsequence",
            "tabulation",
            "coin_change",
            "word_break",
        },
        "seed": {
            "domains": {"dynamic_programming": 1.0, "combinatorics": 0.6, "matrix": 0.2},
            "algorithms": {
                "dynamic_programming_top_down": 1.0,
                "dynamic_programming_bottom_up": 1.0,
                "memoization": 0.9,
                "bitmask_dp": 0.5,
                "digit_dp": 0.5,
                "tree_dp": 0.4,
            },
            "patterns": {
                "dp_linear_sequence": 1.0,
                "dp_grid": 0.9,
                "dp_interval": 0.8,
                "word_break": 0.7,
                "subset_sum": 0.7,
                "combination_sum": 0.6,
            },
            "micro_skills": {
                "dp_transition_design": 1.0,
                "memoization_table_design": 0.9,
                "tabulation_ordering": 0.9,
                "base_case_selection": 0.8,
                "state_encoding": 0.7,
                "index_management": 0.7,
            },
            "failure_types": {
                "missing_base_case": 1.0,
                "incorrect_dp_transition": 1.0,
                "incorrect_recursive_transition": 0.9,
                "missing_memoization": 0.8,
                "incorrect_state_transition": 0.8,
            },
            "pattern_clusters": {"dynamic_programming_core": 1.0},
            "topic_paths": {"dsa_foundations > dynamic_programming_and_backtracking": 1.0},
            "learning_targets": {"strengthen_dp_transitions": 1.0},
        },
    },
    {
        "name": "binary_search",
        "triggers": {
            "binary_search",
            "rotated",
            "answer_space",
            "parametric",
            "monotonic",
            "search_a_2d_matrix",
            "matrix_search",
            "threshold",
            "lower_bound",
            "upper_bound",
        },
        "seed": {
            "domains": {"binary_search": 1.0, "array": 0.8, "matrix": 0.4, "interval": 0.2},
            "algorithms": {
                "binary_search": 1.0,
                "binary_search_on_answer": 1.0,
                "ordered_map": 0.5,
                "quickselect": 0.3,
            },
            "patterns": {
                "binary_search_rotated_array": 1.0,
                "merge_intervals": 0.2,
                "meeting_rooms": 0.2,
                "cyclic_sort": 0.2,
                "buy_and_sell_stock": 0.2,
                "jump_game": 0.2,
            },
            "micro_skills": {
                "binary_search_bounds": 1.0,
                "boundary_handling": 1.0,
                "index_management": 0.9,
                "ordering_constraints": 0.7,
                "partitioning_logic": 0.5,
                "matrix_coordinate_mapping": 0.6,
            },
            "failure_types": {
                "incorrect_binary_search_bound": 1.0,
                "off_by_one": 0.9,
                "edge_case_omission": 0.8,
                "wrong_partitioning": 0.5,
            },
            "pattern_clusters": {"searching_sorting_ranges": 1.0},
            "topic_paths": {"dsa_foundations > searching_sorting_ranges": 1.0},
            "learning_targets": {"practice_two_pointers": 0.4, "review_boundary_handling": 1.0},
        },
    },
    {
        "name": "graph_paths",
        "triggers": {
            "graph",
            "shortest_path",
            "dijkstra",
            "bellman",
            "floyd",
            "warshall",
            "mst",
            "topological",
            "course_schedule",
            "network",
            "connected",
            "component",
            "island",
            "cycle",
        },
        "seed": {
            "domains": {"graph": 1.0, "tree": 0.1, "heap": 0.4},
            "algorithms": {
                "depth_first_search": 0.8,
                "breadth_first_search": 0.8,
                "topological_sort": 0.8,
                "union_find": 0.8,
                "shortest_path": 1.0,
                "dijkstra_shortest_path": 0.9,
                "bellman_ford_shortest_path": 0.9,
                "floyd_warshall_all_pairs_shortest_path": 0.9,
                "minimum_spanning_tree": 0.7,
                "prim_mst": 0.5,
                "kruskal_mst": 0.5,
            },
            "patterns": {
                "graph_connected_components": 1.0,
                "shortest_path": 1.0,
                "topological_ordering": 0.9,
                "graph_union_find_components": 0.9,
                "tree_level_order": 0.2,
            },
            "micro_skills": {
                "graph_adjacency_traversal": 1.0,
                "visited_marking": 0.9,
                "graph_cycle_detection": 0.8,
                "topological_indegree_tracking": 0.8,
                "graph_weight_handling": 0.8,
                "edge_relaxation": 0.8,
                "distance_initialization": 0.7,
            },
            "failure_types": {
                "graph_traversal_miss": 1.0,
                "missed_visited_mark": 0.9,
                "invalid_topological_order": 0.9,
                "wrong_relaxation_update": 0.9,
                "wrong_distance_initialization": 0.8,
                "negative_cycle_missed": 0.8,
                "wrong_mst_edge_choice": 0.6,
            },
            "pattern_clusters": {"graph_connectivity_and_paths": 1.0},
            "topic_paths": {"dsa_foundations > graphs_and_search": 1.0},
            "learning_targets": {"practice_graph_connectivity": 1.0},
        },
    },
    {
        "name": "tree",
        "triggers": {"tree", "bst", "binary_tree", "ancestor", "lca", "traversal", "root"},
        "seed": {
            "domains": {"tree": 1.0, "binary_tree": 0.8, "binary_search_tree": 0.7, "heap": 0.2},
            "algorithms": {
                "tree_traversal": 1.0,
                "depth_first_search": 0.8,
                "breadth_first_search": 0.8,
                "binary_tree_construction": 0.7,
                "binary_lifting": 0.4,
            },
            "patterns": {
                "tree_level_order": 1.0,
                "tree_dfs": 1.0,
                "bst_validation": 0.9,
                "lowest_common_ancestor": 0.9,
                "tree_construction": 0.9,
                "trie_prefix_search": 0.3,
            },
            "micro_skills": {
                "tree_node_access": 1.0,
                "recursion_depth_control": 0.8,
                "queue_usage": 0.8,
                "base_case_selection": 0.8,
                "subtree_aggregation": 0.6,
            },
            "failure_types": {
                "wrong_parent_reconstruction": 0.9,
                "incorrect_recursive_transition": 0.8,
                "missing_base_case": 0.8,
                "null_pointer_access": 0.8,
            },
            "pattern_clusters": {"tree_traversal_and_construction": 1.0},
            "topic_paths": {"dsa_foundations > trees_and_heaps": 1.0},
            "learning_targets": {"practice_tree_traversal": 1.0},
        },
    },
    {
        "name": "linked_list",
        "triggers": {"linked_list", "reverse", "cycle", "fast_slow", "merge_k_sorted_lists", "reverse_k_group"},
        "seed": {
            "domains": {"linked_list": 1.0, "stack_queue": 0.4, "design": 0.1},
            "algorithms": {"linked_list_manipulation": 1.0, "depth_first_search": 0.1, "simulation": 0.4},
            "patterns": {
                "linked_list_reversal": 1.0,
                "linked_list_cycle": 0.9,
                "fast_slow_pointer": 0.9,
                "merge_k_sorted_lists": 0.5,
                "reverse_k_group": 0.5,
            },
            "micro_skills": {
                "linked_list_pointer_update": 1.0,
                "pointer_update": 0.9,
                "sentinel_dummy_node": 0.8,
                "null_handling": 0.7,
                "cycle_entry_detection": 0.8,
            },
            "failure_types": {
                "pointer_mismanagement": 1.0,
                "null_pointer_access": 0.9,
                "infinite_loop": 0.8,
                "wrong_k_way_merge_order": 0.6,
            },
            "pattern_clusters": {"linked_list_pointer_work": 1.0},
            "topic_paths": {"dsa_foundations > linked_structures": 1.0},
            "learning_targets": {"linked_list_basics": 1.0},
        },
    },
    {
        "name": "sliding_window",
        "triggers": {"window", "substring", "subarray", "anagram", "minimum_window", "longest_substring", "sliding"},
        "seed": {
            "domains": {"string": 0.9, "array": 0.8, "hash_table": 0.4},
            "algorithms": {"sliding_window_variable": 1.0, "prefix_sum": 0.4, "hash_map_lookup": 0.6},
            "patterns": {
                "sliding_window_variable": 1.0,
                "longest_substring_without_repeating": 1.0,
                "minimum_window_substring": 1.0,
                "subarray_sum": 0.7,
                "anagram_detection": 0.7,
            },
            "micro_skills": {
                "window_shrink_expand": 1.0,
                "string_indexing": 0.9,
                "character_frequency_count": 0.8,
                "state_tracking": 0.8,
            },
            "failure_types": {
                "wrong_sliding_window_update": 1.0,
                "off_by_one": 0.8,
                "duplicate_handling_error": 0.7,
                "edge_case_omission": 0.7,
            },
            "pattern_clusters": {"sliding_window_strings": 1.0},
            "topic_paths": {"dsa_foundations > arrays_and_strings": 1.0},
            "learning_targets": {"practice_sliding_window": 1.0},
        },
    },
    {
        "name": "prefix_sum",
        "triggers": {"prefix", "range_sum", "subarray_sum", "difference", "running_sum"},
        "seed": {
            "domains": {"array": 1.0, "math": 0.2},
            "algorithms": {"prefix_sum": 1.0, "hash_map_lookup": 0.5, "simulation": 0.3},
            "patterns": {"subarray_sum": 1.0, "trapping_rain_water": 0.7, "dp_linear_sequence": 0.3},
            "micro_skills": {"prefix_sum_construction": 1.0, "overflow_guarding": 0.7, "rolling_accumulator": 0.7},
            "failure_types": {"wrong_prefix_sum_usage": 1.0, "integer_overflow": 0.7, "off_by_one": 0.6},
            "pattern_clusters": {"pair_and_sum": 0.7},
            "topic_paths": {"dsa_foundations > arrays_and_strings": 1.0},
            "learning_targets": {"learn_prefix_sum": 1.0},
        },
    },
    {
        "name": "interval",
        "triggers": {"interval", "meeting", "calendar", "merge", "overlap"},
        "seed": {
            "domains": {"interval": 1.0, "array": 0.3, "heap": 0.3},
            "algorithms": {"interval_merge": 1.0, "sorting": 0.7, "heap_priority_queue": 0.4},
            "patterns": {"merge_intervals": 1.0, "meeting_rooms": 1.0, "interval_scheduling": 0.6},
            "micro_skills": {"interval_comparison": 1.0, "ordering_constraints": 0.8, "boundary_handling": 0.7},
            "failure_types": {"ordering_error": 1.0, "off_by_one": 0.8, "wrong_greedy_choice": 0.6},
            "pattern_clusters": {"interval_scheduling_and_merging": 1.0},
            "topic_paths": {"dsa_foundations > intervals_and_ranges": 1.0},
            "learning_targets": {"learn_interval_merging": 1.0},
        },
    },
    {
        "name": "backtracking",
        "triggers": {"backtracking", "subset", "permutation", "combination", "queen", "sudoku", "search"},
        "seed": {
            "domains": {"combinatorics": 1.0, "matrix": 0.2, "graph": 0.1},
            "algorithms": {"backtracking": 1.0, "divide_and_conquer": 0.3},
            "patterns": {
                "backtracking_combinations": 1.0,
                "combination_sum": 1.0,
                "subset_sum": 0.9,
                "n_queen": 1.0,
                "sudoku_validation": 0.8,
            },
            "micro_skills": {
                "pruning": 1.0,
                "state_reset": 0.9,
                "base_case_selection": 0.8,
                "recursion_depth_control": 0.8,
            },
            "failure_types": {
                "failed_pruning": 1.0,
                "incorrect_backtracking_restore": 0.9,
                "missing_base_case": 0.8,
                "edge_case_omission": 0.7,
            },
            "pattern_clusters": {"backtracking_and_combinatorics": 1.0},
            "topic_paths": {"dsa_foundations > dynamic_programming_and_backtracking": 1.0},
            "learning_targets": {"practice_backtracking_pruning": 1.0},
        },
    },
    {
        "name": "greedy",
        "triggers": {"greedy", "jump_game", "stock", "profit", "schedule"},
        "seed": {
            "domains": {"greedy": 1.0, "array": 0.5, "interval": 0.4, "math": 0.1},
            "algorithms": {"greedy_choice": 1.0, "sorting": 0.4, "heap_priority_queue": 0.2},
            "patterns": {"buy_and_sell_stock": 1.0, "jump_game": 1.0, "meeting_rooms": 0.6},
            "micro_skills": {"state_tracking": 0.9, "ordering_constraints": 0.8, "rolling_accumulator": 0.6},
            "failure_types": {"wrong_greedy_choice": 1.0, "incorrect_state_transition": 0.7},
            "pattern_clusters": {"design_and_state": 0.3},
            "topic_paths": {"dsa_foundations > searching_sorting_ranges": 0.8},
            "learning_targets": {"practice_graph_connectivity": 0.0},
        },
    },
    {
        "name": "string",
        "triggers": {"string", "anagram", "palindrome", "regex", "match", "kmp", "trie"},
        "seed": {
            "domains": {"string": 1.0, "trie": 0.2, "hash_table": 0.2},
            "algorithms": {"hash_map_lookup": 0.6, "trie_search": 0.8, "string_matching": 1.0},
            "patterns": {"palindrome_check": 1.0, "string_matching": 1.0, "valid_parentheses": 0.6},
            "micro_skills": {"string_indexing": 1.0, "substring_extraction": 0.9, "character_frequency_count": 0.8},
            "failure_types": {"string_parsing_error": 1.0, "off_by_one": 0.7, "duplicate_handling_error": 0.7},
            "pattern_clusters": {"string_and_stack_patterns": 1.0},
            "topic_paths": {"dsa_foundations > arrays_and_strings": 1.0},
            "learning_targets": {"string_basics": 1.0},
        },
    },
    {
        "name": "matrix",
        "triggers": {"matrix", "grid", "spiral", "rotation", "coordinate"},
        "seed": {
            "domains": {"matrix": 1.0, "array": 0.3, "graph": 0.1},
            "algorithms": {"matrix_traversal": 1.0, "simulation": 0.4, "depth_first_search": 0.2},
            "patterns": {"matrix_spiral_traversal": 1.0, "matrix_rotation": 1.0, "sudoku_validation": 0.5},
            "micro_skills": {"matrix_coordinate_mapping": 1.0, "direction_iteration": 0.8, "boundary_handling": 0.8},
            "failure_types": {"incorrect_indexing": 0.9, "off_by_one": 0.8, "wrong_iteration_order": 0.7},
            "pattern_clusters": {"array_rearrangement": 0.8},
            "topic_paths": {"dsa_foundations > arrays_and_strings": 0.8},
            "learning_targets": {"matrix_traversal_and_simulation": 1.0},
        },
    },
    {
        "name": "heap",
        "triggers": {"heap", "priority_queue", "median", "top_k"},
        "seed": {
            "domains": {"heap": 1.0, "array": 0.2, "graph": 0.1},
            "algorithms": {"heap_priority_queue": 1.0, "quickselect": 0.5, "ordered_map": 0.2},
            "patterns": {"heap_top_k": 1.0, "design_median_finder": 0.9, "meeting_rooms": 0.7},
            "micro_skills": {"heap_usage": 1.0, "state_tracking": 0.7, "ordering_constraints": 0.6},
            "failure_types": {"heap_selection_error": 1.0, "wrong_k_way_merge_order": 0.7, "ordering_error": 0.6},
            "pattern_clusters": {"array_rearrangement": 0.6},
            "topic_paths": {"dsa_foundations > trees_and_heaps": 1.0},
            "learning_targets": {"learn_heap_selection": 1.0},
        },
    },
    {
        "name": "design",
        "triggers": {"design", "cache", "lru", "median_finder", "data structure"},
        "seed": {
            "domains": {"design": 1.0, "hash_table": 0.5, "linked_list": 0.3},
            "algorithms": {"ordered_map": 0.7, "hash_map_lookup": 0.8, "heap_priority_queue": 0.3},
            "patterns": {"design_lru_cache": 1.0, "design_median_finder": 1.0},
            "micro_skills": {"state_tracking": 1.0, "hash_map_usage": 0.8, "pointer_update": 0.4},
            "failure_types": {"state_reset_error": 0.8, "null_pointer_access": 0.5},
            "pattern_clusters": {"design_and_state": 1.0},
            "topic_paths": {"dsa_foundations > math_bit_sql_design": 1.0},
            "learning_targets": {"design_data_structures": 1.0},
        },
    },
    {
        "name": "sql",
        "triggers": {"sql", "query", "join", "group", "window", "rank"},
        "seed": {
            "domains": {"sql": 1.0},
            "algorithms": {"ordered_map": 0.4},
            "patterns": {"sql_group_by": 1.0, "sql_window_function": 1.0},
            "micro_skills": {"sql_grouping_reasoning": 1.0, "sql_window_reasoning": 1.0},
            "failure_types": {"sql_join_error": 0.9, "sql_grouping_error": 0.9, "ordering_error": 0.4},
            "pattern_clusters": {"design_and_state": 0.1},
            "topic_paths": {"dsa_foundations > math_bit_sql_design": 1.0},
            "learning_targets": {"learn_sql_windows": 1.0},
        },
    },
    {
        "name": "bit",
        "triggers": {"bit", "xor", "mask", "hamming", "binary"},
        "seed": {
            "domains": {"bit_manipulation": 1.0, "math": 0.4},
            "algorithms": {"bit_manipulation": 1.0, "dynamic_programming_bottom_up": 0.2},
            "patterns": {"bitmask_dp": 1.0, "n_queen": 0.2},
            "micro_skills": {"bitmask_operations": 1.0, "bitset_usage": 0.9},
            "failure_types": {"wrong_bitset_state": 1.0, "integer_overflow": 0.4},
            "pattern_clusters": {"bitmask_and_dp": 1.0},
            "topic_paths": {"dsa_foundations > math_bit_sql_design": 1.0},
            "learning_targets": {"bit_manipulation_basics": 1.0},
        },
    },
    {
        "name": "segment_tree",
        "triggers": {"segment_tree", "fenwick", "binary indexed tree", "range query"},
        "seed": {
            "domains": {"segment_tree": 1.0, "binary_indexed_tree": 1.0, "array": 0.2},
            "algorithms": {"segment_tree_query_update": 1.0, "binary_indexed_tree_query_update": 1.0, "coordinate_compression": 0.7},
            "patterns": {"segment_tree_range_query": 1.0, "fenwick_prefix_query": 1.0},
            "micro_skills": {"lazy_propagation": 1.0, "range_update_logic": 0.9, "coordinate_compression": 0.8},
            "failure_types": {"wrong_segment_tree_merge": 1.0, "incorrect_coordinate_compression": 0.9},
            "pattern_clusters": {"range_query_structures": 1.0},
            "topic_paths": {"dsa_foundations > searching_sorting_ranges": 0.8},
            "learning_targets": {"practice_prefix_sum": 0.0},
        },
    },
    {
        "name": "randomized",
        "triggers": {"random", "reservoir", "sampling", "probability"},
        "seed": {
            "domains": {"probability": 1.0, "randomized": 1.0, "math": 0.4},
            "algorithms": {"reservoir_sampling": 1.0, "quickselect": 0.6, "meet_in_the_middle": 0.3},
            "patterns": {"reservoir_sampling": 1.0, "meet_in_the_middle": 0.7, "heap_top_k": 0.3},
            "micro_skills": {"probabilistic_reasoning": 1.0, "randomized_pivoting": 0.9, "reservoir_update": 0.8},
            "failure_types": {"wrong_reservoir_update": 1.0, "wrong_randomized_partition": 0.9},
            "pattern_clusters": {"randomized_and_sampling": 1.0},
            "topic_paths": {"dsa_foundations > math_bit_sql_design": 0.7},
            "learning_targets": {"probability_and_randomized": 1.0},
        },
    },
]

DIFFICULTY_BUCKETS = (
    (0.2, "0.0-0.2"),
    (0.4, "0.2-0.4"),
    (0.6, "0.4-0.6"),
    (0.8, "0.6-0.8"),
    (1.1, "0.8-1.0"),
)


@dataclass
class LoadedTaxonomy:
    raw: Dict[str, Any]
    items_by_category: Dict[str, List[Dict[str, Any]]]
    item_by_label: Dict[str, Dict[str, Any]]
    alias_to_label: Dict[str, str]
    parent_to_children: Dict[str, List[str]]
    vector_by_category: Dict[str, List[Dict[str, Any]]]


def setup_global_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_alias_list(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, list):
        for item in value:
            label = normalize_taxonomy_label(item)
            if label:
                out.append(label)
    return dedupe(out)


def dedupe(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        label = normalize_taxonomy_label(item)
        if label and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def merge_unique(*lists: Iterable[str]) -> List[str]:
    out: List[str] = []
    for lst in lists:
        out.extend(lst)
    return dedupe(out)


def normalize_taxonomy_label(value: Any) -> str:
    return normalize_whitespace(snake_case(value))


@lru_cache(maxsize=20000)
def build_text_signature(text: str) -> tuple[tuple[str, float], ...]:
    normalized = normalize_taxonomy_label(text)
    if not normalized:
        return ()
    weights: Counter[str] = Counter()
    for token in extract_tokens(text):
        if token:
            weights[f"tok:{token}"] += 1.0
    padded = f"_{normalized}_"
    if len(padded) >= 3:
        for idx in range(len(padded) - 2):
            gram = padded[idx : idx + 3]
            if gram.strip("_"):
                weights[f"tri:{gram}"] += 0.35
    return tuple(sorted(weights.items()))


def signature_to_counter(signature: tuple[tuple[str, float], ...]) -> Counter[str]:
    return Counter({key: float(value) for key, value in signature})


def cosine_from_signatures(left: tuple[tuple[str, float], ...], right: tuple[tuple[str, float], ...]) -> float:
    if not left or not right:
        return 0.0
    a = signature_to_counter(left)
    b = signature_to_counter(right)
    dot = sum(weight * b.get(key, 0.0) for key, weight in a.items())
    if dot <= 0:
        return 0.0
    na = math.sqrt(sum(weight * weight for weight in a.values()))
    nb = math.sqrt(sum(weight * weight for weight in b.values()))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def vector_category_signature(item: Dict[str, Any]) -> tuple[tuple[str, float], ...]:
    parts = [item.get("label", "")]
    parts.extend(item.get("aliases", []) if isinstance(item.get("aliases"), list) else [])
    description = item.get("description", "")
    if description:
        parts.append(description)
    parent = item.get("parent")
    if parent:
        parts.append(parent)
    return build_text_signature(" ".join(str(part) for part in parts if part))


def vector_match_category_label(raw_label: str, category: str, active_taxonomy: LoadedTaxonomy) -> str:
    raw_label = normalize_taxonomy_label(raw_label)
    if not raw_label:
        return ""
    query = build_text_signature(raw_label)
    if not query:
        return ""
    allowed = active_taxonomy.items_by_category.get(category, [])
    threshold = VECTOR_MATCH_THRESHOLDS.get(category, 0.45)
    best_label = ""
    best_score = 0.0
    second_score = 0.0
    for item in allowed:
        candidate = item.get("vector_signature")
        if not candidate:
            candidate = vector_category_signature(item)
        score = cosine_from_signatures(query, candidate)
        label = item.get("label", "")
        if not label:
            continue
        if score > best_score:
            second_score = best_score
            best_score = score
            best_label = label
        elif score > second_score:
            second_score = score
    if not best_label:
        return ""
    if best_score < threshold:
        return ""
    if best_score < 0.75 and (best_score - second_score) < 0.03:
        return ""
    return best_label


def resolve_label_in_allowed_subset(
    label: str,
    category: str,
    active_taxonomy: LoadedTaxonomy,
    label_mapping: Mapping[str, str],
    allowed_labels: set[str],
) -> str:
    normalized = normalize_taxonomy_label(label)
    if not normalized:
        return ""
    mapped = label_mapping.get(normalized, normalized)
    if mapped in allowed_labels:
        return mapped
    alias = active_taxonomy.alias_to_label.get(mapped, mapped)
    if alias in allowed_labels:
        return alias
    query = build_text_signature(normalized)
    if not query:
        return ""
    best_label = ""
    best_score = 0.0
    second_score = 0.0
    threshold = VECTOR_MATCH_THRESHOLDS.get(category, 0.45)
    for item in active_taxonomy.items_by_category.get(category, []):
        candidate_label = item.get("label", "")
        if candidate_label not in allowed_labels:
            continue
        candidate = item.get("vector_signature")
        if not candidate:
            candidate = vector_category_signature(item)
        score = cosine_from_signatures(query, candidate)
        if score > best_score:
            second_score = best_score
            best_score = score
            best_label = candidate_label
        elif score > second_score:
            second_score = score
    if not best_label:
        return ""
    if best_score < threshold:
        return ""
    if best_score < 0.75 and (best_score - second_score) < 0.03:
        return ""
    return best_label


def canonicalize_label(label: str, active_taxonomy: LoadedTaxonomy, label_mapping: Mapping[str, str]) -> str:
    normalized = normalize_taxonomy_label(label)
    if not normalized:
        return ""
    mapped = label_mapping.get(normalized, normalized)
    return active_taxonomy.alias_to_label.get(mapped, mapped)


def canonicalize_label_for_category(label: str, category: str, active_taxonomy: LoadedTaxonomy, label_mapping: Mapping[str, str]) -> str:
    normalized = normalize_taxonomy_label(label)
    if not normalized:
        return ""
    allowed = {item["label"] for item in active_taxonomy.items_by_category.get(category, [])}
    mapped = label_mapping.get(normalized, normalized)
    if mapped in allowed:
        return mapped
    alias = active_taxonomy.alias_to_label.get(mapped, mapped)
    if alias in allowed:
        return alias
    vector_match = vector_match_category_label(normalized, category, active_taxonomy)
    if vector_match in allowed:
        return vector_match
    return ""


def canonicalize_topic_path(value: Any, allowed_paths: Iterable[str]) -> str:
    text = normalize_whitespace(str(value))
    if not text:
        return ""
    allowed_list = [normalize_whitespace(str(path)) for path in allowed_paths if normalize_whitespace(str(path))]
    if text in allowed_list:
        return text

    def collapse_path_key(path: str) -> str:
        normalized = normalize_taxonomy_label(path)
        return normalized.replace("_", "")

    text_key = collapse_path_key(text)
    if not text_key:
        return ""
    for path in allowed_list:
        if collapse_path_key(path) == text_key:
            return path
    return ""


def load_taxonomy_file(path: Path) -> LoadedTaxonomy:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"taxonomy file must be a JSON object: {path}")

    items_by_category: Dict[str, List[Dict[str, Any]]] = {key: [] for key in CATEGORY_KEYS}
    item_by_label: Dict[str, Dict[str, Any]] = {}
    alias_to_label: Dict[str, str] = {}
    parent_to_children: Dict[str, List[str]] = defaultdict(list)
    vector_by_category: Dict[str, List[Dict[str, Any]]] = {key: [] for key in CATEGORY_KEYS}

    for category in CATEGORY_KEYS:
        raw_items = data.get(category, [])
        if not isinstance(raw_items, list):
            continue
        for raw_item in raw_items:
            if isinstance(raw_item, str):
                item = {"label": normalize_taxonomy_label(raw_item), "description": "", "aliases": [], "parent": None}
            elif isinstance(raw_item, dict):
                item = {
                    "label": normalize_taxonomy_label(raw_item.get("label") or raw_item.get("name")),
                    "description": normalize_whitespace(str(raw_item.get("description", "") or "")),
                    "aliases": normalize_alias_list(raw_item.get("aliases", [])),
                    "parent": normalize_taxonomy_label(raw_item.get("parent")) or None,
                }
            else:
                continue
            if not item["label"]:
                continue
            item["category"] = category
            item["vector_signature"] = vector_category_signature(item)
            items_by_category[category].append(item)
            vector_by_category[category].append(item)
            item_by_label[item["label"]] = item
            alias_to_label[item["label"]] = item["label"]
            for alias in item["aliases"]:
                alias_to_label[alias] = item["label"]
            if item.get("parent"):
                parent_to_children[item["parent"]].append(item["label"])

    return LoadedTaxonomy(raw=data, items_by_category=items_by_category, item_by_label=item_by_label, alias_to_label=alias_to_label, parent_to_children=parent_to_children, vector_by_category=vector_by_category)


def load_flat_mapping(path: Path) -> Dict[str, str]:
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    if all(isinstance(v, str) for v in data.values()):
        for k, v in data.items():
            nk, nv = normalize_taxonomy_label(k), normalize_taxonomy_label(v)
            if nk and nv:
                out[nk] = nv
        return out
    for value in data.values():
        if not isinstance(value, dict):
            continue
        for k, v in value.items():
            nk, nv = normalize_taxonomy_label(k), normalize_taxonomy_label(v)
            if nk and nv:
                out[nk] = nv
    return out


def load_topic_hierarchy(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "topic": normalize_taxonomy_label(item.get("topic")),
                "description": normalize_whitespace(str(item.get("description", "") or "")),
                "parent": normalize_taxonomy_label(item.get("parent")) or None,
                "children": dedupe(item.get("children", []) if isinstance(item.get("children"), list) else []),
                "contains_domains": dedupe(item.get("contains_domains", []) if isinstance(item.get("contains_domains"), list) else []),
                "contains_algorithms": dedupe(item.get("contains_algorithms", []) if isinstance(item.get("contains_algorithms"), list) else []),
                "contains_patterns": dedupe(item.get("contains_patterns", []) if isinstance(item.get("contains_patterns"), list) else []),
                "contains_micro_skills": dedupe(item.get("contains_micro_skills", []) if isinstance(item.get("contains_micro_skills"), list) else []),
            }
        )
    return out


def load_allowed_subset(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    tax = load_taxonomy_file(path)
    return tax.items_by_category


def build_problem_fields(raw: Dict[str, Any]) -> Dict[str, str]:
    return {
        "id": str(raw.get("id") or raw.get("problem_id") or raw.get("question_id") or raw.get("slug") or ""),
        "title": extract_text_value(raw, ("title", "name", "question_title", "problem_title")),
        "difficulty": extract_text_value(raw, ("difficulty", "level", "difficulty_level")),
        "tags": extract_text_value(raw, ("tags", "tag", "topic_tags", "category")),
        "description": extract_text_value(raw, ("description", "statement", "problem", "content", "question", "body")),
        "similar_questions": extract_text_value(raw, ("similar_questions", "similar", "related_questions")),
        "companies": extract_text_value(raw, ("companies", "company", "company_tags")),
        "acceptance_rate": extract_text_value(raw, ("acceptance_rate", "acceptance", "acceptanceRate")),
    }


def extract_tokens(text: str) -> List[str]:
    if not text:
        return []
    raw_parts = re.split(r"[^a-zA-Z0-9]+", normalize_whitespace(text))
    tokens: List[str] = []
    for part in raw_parts:
        normalized = normalize_label(part)
        if normalized:
            tokens.append(normalized)
    return tokens


def extract_signal_tokens(text: str) -> List[str]:
    tokens = []
    for token in extract_tokens(text):
        if len(token) == 1 and token not in {"k", "n", "m"}:
            continue
        if token in ROUTING_STOPWORDS:
            continue
        tokens.append(token)
    return dedupe(tokens)


def raw_to_list(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, dict):
                out.extend(extract_tokens(json.dumps(item, ensure_ascii=False)))
            else:
                out.extend(extract_tokens(str(item)))
        return dedupe(out)
    if isinstance(value, dict):
        return extract_tokens(json.dumps(value, ensure_ascii=False))
    return extract_tokens(str(value or ""))


def build_routing_signals(raw: Dict[str, Any], pass1_metadata: Dict[str, Any]) -> Dict[str, Any]:
    fields = build_problem_fields(raw)
    metadata_taxonomy = pass1_metadata.get("taxonomy") if isinstance(pass1_metadata.get("taxonomy"), dict) else {}
    skill_model = pass1_metadata.get("skill_model") if isinstance(pass1_metadata.get("skill_model"), dict) else {}
    failure_model = pass1_metadata.get("failure_model") if isinstance(pass1_metadata.get("failure_model"), dict) else {}

    keywords = merge_unique(
        extract_signal_tokens(fields["title"]),
        extract_signal_tokens(fields["description"]),
        extract_signal_tokens(fields["tags"]),
        extract_signal_tokens(fields["similar_questions"]),
        extract_signal_tokens(fields["companies"]),
        extract_signal_tokens(fields["acceptance_rate"]),
        extract_signal_tokens(str(pass1_metadata.get("pattern_cluster", ""))),
        extract_signal_tokens(str(pass1_metadata.get("solution_dna_summary", ""))),
    )

    raw_topics = raw_to_list(raw.get("related_topics"))
    similar_topics = raw_to_list(raw.get("similar_questions"))
    keywords = merge_unique(keywords, raw_topics, similar_topics)

    pass1_domains = dedupe(metadata_taxonomy.get("domains", {}).keys() if isinstance(metadata_taxonomy.get("domains"), dict) else [])
    pass1_algorithms = dedupe(metadata_taxonomy.get("algorithms", {}).keys() if isinstance(metadata_taxonomy.get("algorithms"), dict) else [])
    pass1_patterns = dedupe(metadata_taxonomy.get("patterns", {}).keys() if isinstance(metadata_taxonomy.get("patterns"), dict) else [])
    pass1_micro_skills = dedupe(metadata_taxonomy.get("micro_skills", {}).keys() if isinstance(metadata_taxonomy.get("micro_skills"), dict) else [])
    pass1_failures = []
    common_failures = failure_model.get("common_failures", {})
    if isinstance(common_failures, dict):
        pass1_failures = dedupe(common_failures.keys())
    pass1_pattern_cluster = normalize_taxonomy_label(pass1_metadata.get("pattern_cluster", ""))
    difficulty_label = normalize_label(fields["difficulty"])
    if difficulty_label not in {"easy", "medium", "hard"}:
        difficulty_label = normalize_label(raw.get("difficulty", "")) or difficulty_label

    return {
        "keywords": keywords,
        "raw_topics": raw_topics,
        "pass1_domains": pass1_domains,
        "pass1_algorithms": pass1_algorithms,
        "pass1_patterns": pass1_patterns,
        "pass1_micro_skills": pass1_micro_skills,
        "pass1_failures": pass1_failures,
        "pass1_pattern_cluster": pass1_pattern_cluster,
        "difficulty_label": difficulty_label,
    }


def route_families(signals: Dict[str, Any]) -> List[str]:
    haystack = set(signals["keywords"]) | set(signals["raw_topics"]) | set(signals["pass1_domains"]) | set(signals["pass1_algorithms"]) | set(signals["pass1_patterns"]) | set(signals["pass1_micro_skills"]) | set(signals["pass1_failures"]) | {signals["pass1_pattern_cluster"]}
    matched: List[str] = []
    for family in FAMILY_RULES:
        if family["triggers"] & haystack:
            matched.append(family["name"])
    if not matched and ("dynamic_programming" in haystack or "memoization" in haystack):
        matched.append("dynamic_programming")
    if not matched and ("binary_search" in haystack or "search" in haystack):
        matched.append("binary_search")
    if not matched and ("graph" in haystack or "shortest_path" in haystack):
        matched.append("graph_paths")
    return dedupe(matched)


def score_label(item: Dict[str, Any], signals: Dict[str, Any], family_seeds: Mapping[str, float], category: str, label_to_item: Mapping[str, Dict[str, Any]], mapping: Mapping[str, str]) -> float:
    label = item["label"]
    aliases = set(item.get("aliases", []))
    haystack = set(signals["keywords"]) | set(signals["raw_topics"]) | set(signals["pass1_domains"]) | set(signals["pass1_algorithms"]) | set(signals["pass1_patterns"]) | set(signals["pass1_micro_skills"]) | set(signals["pass1_failures"]) | {signals["pass1_pattern_cluster"]}
    score = 0.0

    if label in haystack:
        score += 5.0
    if aliases & haystack:
        score += 4.0
    if label in signals["keywords"]:
        score += 3.0
    if any(alias in signals["keywords"] for alias in aliases):
        score += 2.5
    if label in signals.get(f"pass1_{category}", []):
        score += 6.0
    mapped_pass1 = mapping.get(label)
    if mapped_pass1 and mapped_pass1 in signals.get(f"pass1_{category}", []):
        score += 5.0

    for family_name, family_score in family_seeds.items():
        family = next((f for f in FAMILY_RULES if f["name"] == family_name), None)
        if not family:
            continue
        seed = family["seed"].get(category, {})
        if label in seed:
            score += 4.0 * family_score * float(seed[label])
        parent = item.get("parent")
        if parent and parent in seed:
            score += 1.0 * family_score * float(seed[parent])

    if item.get("parent") and item["parent"] in label_to_item:
        score += 0.3
    return score


def family_score_map(families: Sequence[str]) -> Dict[str, float]:
    return {family: 1.0 - (idx * 0.1) for idx, family in enumerate(families)}


def build_topic_paths(selected: Dict[str, List[str]], topic_hierarchy: List[Dict[str, Any]]) -> List[str]:
    selected_set = set(selected.get("domains", [])) | set(selected.get("algorithms", [])) | set(selected.get("patterns", [])) | set(selected.get("micro_skills", []))
    scored: List[tuple[float, str]] = []
    for node in topic_hierarchy:
        score = 0.0
        score += sum(1.0 for x in node.get("contains_domains", []) if x in selected_set)
        score += sum(0.8 for x in node.get("contains_algorithms", []) if x in selected_set)
        score += sum(0.8 for x in node.get("contains_patterns", []) if x in selected_set)
        score += sum(0.4 for x in node.get("contains_micro_skills", []) if x in selected_set)
        if node.get("topic"):
            if node["topic"] in selected_set:
                score += 0.8
            path = node["topic"] if not node.get("parent") else f"{node['parent']} > {node['topic']}"
            scored.append((score, path))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return dedupe([path for score, path in scored if score > 0][:MAX_CAPS["topic_paths"]])


def expand_category_with_parents(labels: List[str], taxonomy: LoadedTaxonomy, category: str) -> List[str]:
    out = list(labels)
    for label in labels:
        item = taxonomy.item_by_label.get(label)
        if not item:
            continue
        parent = item.get("parent")
        if parent and parent in taxonomy.item_by_label:
            out.append(parent)
            for child in taxonomy.parent_to_children.get(parent, []):
                if child != label:
                    out.append(child)
    return dedupe(out)


def select_taxonomy_subset(routing_signals: Dict[str, Any], active_taxonomy: LoadedTaxonomy, topic_hierarchy: List[Dict[str, Any]], label_mapping: Mapping[str, str]) -> Dict[str, List[str]]:
    families = route_families(routing_signals)
    family_scores = family_score_map(families)
    selected: Dict[str, List[str]] = {key: [] for key in CATEGORY_KEYS}

    # Pre-seed from pass1 labels and family rules.
    for family_name in families:
        family = next((f for f in FAMILY_RULES if f["name"] == family_name), None)
        if not family:
            continue
        for category in ("domains", "algorithms", "patterns", "micro_skills", "failure_types", "pattern_clusters", "learning_targets"):
            selected[category].extend(list(family["seed"].get(category, {}).keys()))

    # Pull in pass1 labels explicitly.
    selected["domains"].extend(routing_signals["pass1_domains"])
    selected["algorithms"].extend(routing_signals["pass1_algorithms"])
    selected["patterns"].extend(routing_signals["pass1_patterns"])
    selected["micro_skills"].extend(routing_signals["pass1_micro_skills"])
    selected["failure_types"].extend(routing_signals["pass1_failures"])
    if routing_signals["pass1_pattern_cluster"]:
        selected["pattern_clusters"].append(routing_signals["pass1_pattern_cluster"])

    # Generic scoring per category.
    for category in ("domains", "algorithms", "patterns", "micro_skills", "failure_types", "pattern_clusters", "learning_targets", "problem_roles", "problem_traits"):
        if category not in active_taxonomy.items_by_category:
            continue
        scored: List[tuple[float, str]] = []
        for item in active_taxonomy.items_by_category[category]:
            score = score_label(item, routing_signals, family_scores, category, active_taxonomy.item_by_label, label_mapping)
            if category == "micro_skills" and item["label"] in BROAD_MICRO_SKILLS:
                score -= 0.4
            if category == "patterns" and item["label"] in selected["micro_skills"]:
                score += 0.7
            if category == "micro_skills" and item["label"] in selected["patterns"]:
                score += 0.5
            scored.append((score, item["label"]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [label for score, label in scored if score > 0][: MAX_CAPS.get(category, 20)]
        selected[category] = merge_unique(selected.get(category, []), top)

    # Additional family-specific must-have labels.
    if "dynamic_programming" in families:
        selected["patterns"].extend(["dp_linear_sequence", "dp_grid", "dp_interval", "word_break", "subset_sum"])
        selected["micro_skills"].extend(["dp_transition_design", "memoization_table_design", "tabulation_ordering", "base_case_selection"])
        selected["failure_types"].extend(["missing_base_case", "incorrect_dp_transition", "missing_memoization"])
    if "binary_search" in families:
        selected["patterns"].extend(["binary_search_rotated_array", "merge_intervals", "meeting_rooms"])
        selected["micro_skills"].extend(["binary_search_bounds", "boundary_handling", "index_management"])
        selected["failure_types"].extend(["incorrect_binary_search_bound", "off_by_one", "edge_case_omission"])
    if "graph_paths" in families:
        selected["algorithms"].extend(["dijkstra_shortest_path", "bellman_ford_shortest_path", "floyd_warshall_all_pairs_shortest_path", "minimum_spanning_tree"])
        selected["patterns"].extend(["shortest_path", "graph_connected_components", "topological_ordering", "graph_union_find_components"])
        selected["micro_skills"].extend(["graph_adjacency_traversal", "visited_marking", "graph_weight_handling", "edge_relaxation"])
        selected["failure_types"].extend(["wrong_relaxation_update", "wrong_distance_initialization", "graph_traversal_miss"])
    if "tree" in families:
        selected["patterns"].extend(["tree_level_order", "tree_dfs", "bst_validation", "tree_construction", "lowest_common_ancestor"])
        selected["micro_skills"].extend(["tree_node_access", "recursion_depth_control", "queue_usage", "base_case_selection"])
        selected["failure_types"].extend(["wrong_parent_reconstruction", "null_pointer_access", "missing_base_case"])
    if "linked_list" in families:
        selected["patterns"].extend(["linked_list_reversal", "linked_list_cycle", "fast_slow_pointer", "reverse_k_group"])
        selected["micro_skills"].extend(["linked_list_pointer_update", "pointer_update", "sentinel_dummy_node"])
        selected["failure_types"].extend(["pointer_mismanagement", "infinite_loop", "null_pointer_access"])

    # Canonicalize all routed labels before prompt construction.
    for category in ("domains", "algorithms", "patterns", "micro_skills", "failure_types", "pattern_clusters", "learning_targets", "problem_roles", "problem_traits"):
        selected[category] = dedupe(
            canonicalize_label_for_category(label, category, active_taxonomy, label_mapping)
            for label in selected.get(category, [])
            if canonicalize_label_for_category(label, category, active_taxonomy, label_mapping)
        )

    # Topic paths from hierarchy.
    selected["topic_paths"] = build_topic_paths(selected, topic_hierarchy)

    # Learning targets / roles / traits.
    selected["learning_targets"] = build_learning_targets(selected, routing_signals, active_taxonomy)
    selected["problem_roles"] = build_problem_roles(routing_signals, selected)
    selected["problem_traits"] = build_problem_traits(routing_signals, selected)

    # Final cap enforcement.
    for category in ("domains", "algorithms", "patterns", "micro_skills", "failure_types", "pattern_clusters", "learning_targets", "problem_roles", "problem_traits"):
        selected[category] = dedupe(selected.get(category, []))[: MAX_CAPS.get(category, 20)]
    selected["topic_paths"] = dedupe(selected.get("topic_paths", []))[: MAX_CAPS["topic_paths"]]
    return selected


def build_learning_targets(selected: Dict[str, List[str]], routing_signals: Dict[str, Any], active_taxonomy: LoadedTaxonomy) -> List[str]:
    patterns = set(selected.get("patterns", []))
    algorithms = set(selected.get("algorithms", []))
    micro = set(selected.get("micro_skills", []))
    out: List[str] = []
    if "dp_linear_sequence" in patterns or "dynamic_programming_bottom_up" in algorithms:
        out.extend(["learn_prefix_sum", "strengthen_dp_transitions", "recover_from_state_errors"])
    if "binary_search" in algorithms or "binary_search_on_answer" in algorithms:
        out.extend(["review_boundary_handling", "practice_two_pointers"])
    if "graph_connected_components" in patterns or "shortest_path" in patterns:
        out.extend(["practice_graph_connectivity", "learn_heap_selection"])
    if "merge_intervals" in patterns or "meeting_rooms" in patterns:
        out.extend(["learn_interval_merging", "review_boundary_handling"])
    if "tree_level_order" in patterns or "tree_dfs" in patterns:
        out.extend(["practice_tree_traversal", "strengthen_dp_transitions"])
    if "linked_list_reversal" in patterns:
        out.extend(["linked_list_basics", "recover_from_state_errors"])
    if "sliding_window_variable" in patterns or "minimum_window_substring" in patterns:
        out.extend(["practice_sliding_window", "review_boundary_handling"])
    if "bitmask_dp" in algorithms or "bit_manipulation" in algorithms:
        out.extend(["bit_manipulation_basics", "practice_backtracking_pruning"])
    if "sql_window_function" in patterns or "sql_group_by" in patterns:
        out.extend(["learn_sql_windows", "review_boundary_handling"])
    if not out:
        out.extend(["first_exposure", "guided_practice"])
    return dedupe(out)


def build_problem_roles(routing_signals: Dict[str, Any], selected: Dict[str, List[str]]) -> Dict[str, float]:
    difficulty = routing_signals.get("difficulty_label", "")
    overall = estimate_overall_difficulty(difficulty, selected)
    roles = {
        "first_exposure": 0.0,
        "guided_practice": 0.0,
        "reinforcement": 0.0,
        "review": 0.0,
        "recovery": 0.0,
        "challenge": 0.0,
        "assessment": 0.0,
    }
    if difficulty == "easy":
        roles.update({"first_exposure": 0.85, "guided_practice": 0.95, "reinforcement": 0.7, "review": 0.55, "assessment": 0.2, "recovery": 0.25, "challenge": 0.1})
    elif difficulty == "medium":
        roles.update({"first_exposure": 0.3, "guided_practice": 0.8, "reinforcement": 0.85, "review": 0.65, "assessment": 0.55, "recovery": 0.45, "challenge": 0.45})
    else:
        roles.update({"first_exposure": 0.1, "guided_practice": 0.35, "reinforcement": 0.65, "review": 0.75, "assessment": 0.8, "recovery": 0.85, "challenge": 0.95})
    if overall > 0.75:
        roles["challenge"] = max(roles["challenge"], 0.9)
    if overall < 0.35:
        roles["first_exposure"] = max(roles["first_exposure"], 0.8)
    return {k: clamp01(v) for k, v in roles.items()}


def build_problem_traits(routing_signals: Dict[str, Any], selected: Dict[str, List[str]]) -> Dict[str, float]:
    difficulty = routing_signals.get("difficulty_label", "")
    overall = estimate_overall_difficulty(difficulty, selected)
    traits = {
        "difficulty": overall,
        "acceptance_rate": 0.65,
        "frequency": 0.7,
        "implementation_length": 0.55,
        "edge_case_density": 0.5,
        "prerequisite_load": 0.4,
        "novelty": 0.45,
        "repetition_risk": 0.35,
        "solution_uniqueness": 0.5,
    }
    if "shortest_path" in selected.get("patterns", []):
        traits["implementation_length"] = 0.7
        traits["prerequisite_load"] = 0.6
    if "dp_linear_sequence" in selected.get("patterns", []):
        traits["prerequisite_load"] = max(traits["prerequisite_load"], 0.65)
        traits["edge_case_density"] = max(traits["edge_case_density"], 0.6)
    if "merge_intervals" in selected.get("patterns", []):
        traits["edge_case_density"] = max(traits["edge_case_density"], 0.7)
    if "design_lru_cache" in selected.get("patterns", []):
        traits["implementation_length"] = 0.75
        traits["solution_uniqueness"] = 0.65
    return {k: clamp01(v) for k, v in traits.items()}


def estimate_overall_difficulty(difficulty_label: str, selected: Dict[str, List[str]]) -> float:
    if difficulty_label == "easy":
        base = 0.25
    elif difficulty_label == "medium":
        base = 0.55
    elif difficulty_label == "hard":
        base = 0.82
    else:
        base = 0.5
    base += 0.04 * max(0, len(selected.get("patterns", [])) - 1)
    base += 0.02 * max(0, len(selected.get("micro_skills", [])) - 5)
    return clamp01(base)


def format_allowed_items(items: Sequence[Dict[str, Any]]) -> str:
    rendered: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            rendered.append(
                {
                    "label": item.get("label", ""),
                    "description": item.get("description", ""),
                    "aliases": item.get("aliases", []),
                    "parent": item.get("parent"),
                }
            )
        else:
            rendered.append({"label": str(item), "description": "", "aliases": [], "parent": None})
    return json.dumps(rendered, ensure_ascii=False, indent=2)


def format_allowed_paths(paths: Sequence[str]) -> str:
    return json.dumps(list(paths), ensure_ascii=False, indent=2)


def extract_json(text: str) -> Dict[str, Any]:
    text = normalize_whitespace(text)
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_weight_map(value: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(value, dict):
        for label, raw_weight in value.items():
            normalized = normalize_taxonomy_label(label)
            if not normalized:
                continue
            weight = clamp01(raw_weight)
            if weight > 0:
                out[normalized] = max(out.get(normalized, 0.0), weight)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                label = normalize_taxonomy_label(item)
                if label:
                    out[label] = 1.0
            elif isinstance(item, dict):
                label = normalize_taxonomy_label(item.get("label") or item.get("name"))
                if label:
                    out[label] = max(out.get(label, 0.0), clamp01(item.get("weight", 1.0)))
    return out


def normalize_label_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return dedupe([normalize_taxonomy_label(x) for x in value if normalize_taxonomy_label(x)])
    if isinstance(value, str):
        return dedupe([normalize_taxonomy_label(value)])
    return []


def normalize_failure_model(value: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(value, dict):
        return out
    for label, payload in value.items():
        key = normalize_taxonomy_label(label)
        if not key:
            continue
        affected = []
        severity = 0.0
        if isinstance(payload, dict):
            affected = normalize_label_list(payload.get("affected_skills", []))
            severity = clamp01(payload.get("severity", 0.0))
        out[key] = {"affected_skills": affected, "severity": severity}
    return out


def normalize_problem_roles(value: Any) -> Dict[str, float]:
    return normalize_weight_map(value)


def normalize_problem_traits(value: Any) -> Dict[str, float]:
    return normalize_weight_map(value)


def sanitize_final_metadata(candidate: Any, selected: Dict[str, List[str]], active_taxonomy: LoadedTaxonomy, label_mapping: Mapping[str, str], routing_signals: Dict[str, Any]) -> tuple[Dict[str, Any], List[str], Dict[str, List[str]]]:
    issues: List[str] = []
    proposed: Dict[str, List[str]] = {key: [] for key in CATEGORY_KEYS}

    if not isinstance(candidate, dict):
        return {}, ["invalid_json"], proposed

    expected = {
        "taxonomy": {},
        "topic_path": [],
        "difficulty_vector": {},
        "skill_model": {},
        "failure_model": {},
        "learning_targets": [],
        "problem_roles": {},
        "problem_traits": {},
        "pattern_cluster": "",
        "estimated_time_minutes": 0,
        "solution_dna_summary": "",
        "metadata_confidence": 0.0,
        "proposed_new_labels": {key: [] for key in CATEGORY_KEYS},
        "validation": {"needs_human_review": False, "review_reasons": []},
    }
    merged = dict(expected)
    merged.update(candidate)

    taxonomy = merged.get("taxonomy") if isinstance(merged.get("taxonomy"), dict) else {}
    skill_model = merged.get("skill_model") if isinstance(merged.get("skill_model"), dict) else {}
    failure_model = merged.get("failure_model") if isinstance(merged.get("failure_model"), dict) else {}
    difficulty_vector = merged.get("difficulty_vector") if isinstance(merged.get("difficulty_vector"), dict) else {}
    validation = merged.get("validation") if isinstance(merged.get("validation"), dict) else {}

    cleaned = {
        "taxonomy": {key: {} for key in CORE_MAIN_KEYS},
        "topic_path": [],
        "difficulty_vector": {
            "overall": clamp01(difficulty_vector.get("overall", 0.0)),
            "conceptual": clamp01(difficulty_vector.get("conceptual", 0.0)),
            "implementation": clamp01(difficulty_vector.get("implementation", 0.0)),
            "edge_cases": clamp01(difficulty_vector.get("edge_cases", 0.0)),
            "debugging": clamp01(difficulty_vector.get("debugging", 0.0)),
        },
        "skill_model": {"requires": {}, "trains": {}, "tests": {}},
        "failure_model": {"common_failures": {}},
        "learning_targets": [],
        "problem_roles": {},
        "problem_traits": {},
        "pattern_cluster": normalize_taxonomy_label(merged.get("pattern_cluster", "")),
        "estimated_time_minutes": clamp_int(merged.get("estimated_time_minutes", 0), default=0, minimum=0, maximum=240),
        "solution_dna_summary": normalize_whitespace(str(merged.get("solution_dna_summary", "") or "")),
        "metadata_confidence": clamp01(merged.get("metadata_confidence", 0.0)),
        "proposed_new_labels": {key: [] for key in CATEGORY_KEYS},
        "validation": {
            "needs_human_review": bool(validation.get("needs_human_review", False)),
            "review_reasons": normalize_label_list(validation.get("review_reasons", [])),
        },
    }

    # Taxonomy maps.
    for category in CORE_MAIN_KEYS:
        raw_map = normalize_weight_map(taxonomy.get(category, {}))
        allowed = set(selected.get(category, []))
        remapped: Dict[str, float] = {}
        for label, weight in raw_map.items():
            mapped = resolve_label_in_allowed_subset(label, category, active_taxonomy, label_mapping, allowed)
            if mapped:
                remapped[mapped] = max(remapped.get(mapped, 0.0), clamp01(weight))
            else:
                proposed[category].append(label_mapping.get(normalize_taxonomy_label(label), normalize_taxonomy_label(label)))
        cleaned["taxonomy"][category] = remapped

    # Topic path.
    topic_path = merged.get("topic_path", [])
    if isinstance(topic_path, str):
        topic_path = [topic_path]
    if not isinstance(topic_path, list):
        topic_path = []
    allowed_paths = set(selected.get("topic_paths", []))
    cleaned["topic_path"] = dedupe([canonicalize_topic_path(x, allowed_paths) for x in topic_path if canonicalize_topic_path(x, allowed_paths)])
    for path in topic_path:
        npath = normalize_whitespace(str(path))
        if npath and not canonicalize_topic_path(npath, allowed_paths):
            continue

    # Skill model.
    for field in ("requires", "trains", "tests"):
        raw_map = normalize_weight_map(skill_model.get(field, {}))
        allowed = set(selected.get("micro_skills", [])) | set(selected.get("patterns", []))
        remapped: Dict[str, float] = {}
        for label, weight in raw_map.items():
            mapped = resolve_label_in_allowed_subset(label, "micro_skills", active_taxonomy, label_mapping, allowed)
            if not mapped:
                mapped = resolve_label_in_allowed_subset(label, "patterns", active_taxonomy, label_mapping, allowed)
            if mapped:
                remapped[mapped] = max(remapped.get(mapped, 0.0), clamp01(weight))
            else:
                proposed["micro_skills"].append(label_mapping.get(normalize_taxonomy_label(label), normalize_taxonomy_label(label)))
        cleaned["skill_model"][field] = remapped

    # Failure model.
    common_failures = normalize_failure_model(failure_model.get("common_failures", {}))
    allowed_failures = set(selected.get("failure_types", []))
    allowed_affected = set(selected.get("micro_skills", [])) | set(selected.get("patterns", []))
    remapped_failures: Dict[str, Dict[str, Any]] = {}
    for failure, payload in common_failures.items():
        mapped_failure = resolve_label_in_allowed_subset(failure, "failure_types", active_taxonomy, label_mapping, allowed_failures)
        if not mapped_failure:
            proposed["failure_types"].append(label_mapping.get(normalize_taxonomy_label(failure), normalize_taxonomy_label(failure)))
            continue
        affected = []
        for label in payload.get("affected_skills", []):
            mapped = resolve_label_in_allowed_subset(label, "micro_skills", active_taxonomy, label_mapping, allowed_affected)
            if not mapped:
                mapped = resolve_label_in_allowed_subset(label, "patterns", active_taxonomy, label_mapping, allowed_affected)
            if mapped:
                affected.append(mapped)
            else:
                proposed["micro_skills"].append(label_mapping.get(normalize_taxonomy_label(label), normalize_taxonomy_label(label)))
        severity = clamp01(payload.get("severity", 0.0))
        remapped_failures[mapped_failure] = {"affected_skills": dedupe(affected), "severity": severity}
    cleaned["failure_model"]["common_failures"] = remapped_failures

    # Learning targets.
    raw_targets = merged.get("learning_targets", [])
    if isinstance(raw_targets, list):
        for item in raw_targets:
            if isinstance(item, dict):
                label = normalize_taxonomy_label(item.get("label") or item.get("name"))
            else:
                label = normalize_taxonomy_label(item)
            if not label:
                continue
            mapped = resolve_label_in_allowed_subset(label, "learning_targets", active_taxonomy, label_mapping, set(selected.get("learning_targets", [])))
            if mapped:
                cleaned["learning_targets"].append(mapped)
            else:
                proposed["learning_targets"].append(label_mapping.get(label, label))
    cleaned["learning_targets"] = dedupe(cleaned["learning_targets"])

    # Problem roles / traits.
    cleaned["problem_roles"] = normalize_problem_roles(merged.get("problem_roles", {}))
    cleaned["problem_traits"] = normalize_problem_traits(merged.get("problem_traits", {}))
    for label in list(cleaned["problem_roles"].keys()):
        mapped = resolve_label_in_allowed_subset(label, "problem_roles", active_taxonomy, label_mapping, set(selected.get("problem_roles", [])))
        if not mapped:
            proposed["problem_roles"].append(label_mapping.get(normalize_taxonomy_label(label), normalize_taxonomy_label(label)))
            cleaned["problem_roles"].pop(label, None)
    for label in list(cleaned["problem_traits"].keys()):
        mapped = resolve_label_in_allowed_subset(label, "problem_traits", active_taxonomy, label_mapping, set(selected.get("problem_traits", [])))
        if not mapped:
            proposed["problem_traits"].append(label_mapping.get(normalize_taxonomy_label(label), normalize_taxonomy_label(label)))
            cleaned["problem_traits"].pop(label, None)

    # Pattern cluster.
    if cleaned["pattern_cluster"] and cleaned["pattern_cluster"] not in set(selected.get("pattern_clusters", [])):
        mapped_cluster = resolve_label_in_allowed_subset(cleaned["pattern_cluster"], "pattern_clusters", active_taxonomy, label_mapping, set(selected.get("pattern_clusters", [])))
        if mapped_cluster:
            cleaned["pattern_cluster"] = mapped_cluster
        else:
            proposed["pattern_clusters"].append(cleaned["pattern_cluster"])

    # Proposed new labels.
    merged_proposed = merged.get("proposed_new_labels") if isinstance(merged.get("proposed_new_labels"), dict) else {}
    for category in CATEGORY_KEYS:
        existing = normalize_label_list(merged_proposed.get(category, []))
        proposed[category] = dedupe(list(proposed.get(category, [])) + existing)
    cleaned["proposed_new_labels"] = proposed
    cleaned["validation"]["needs_human_review"] = False
    cleaned["validation"]["review_reasons"] = []

    # Global validation checks.
    raw_diff = normalize_label(routing_signals.get("difficulty_label", ""))
    overall = cleaned["difficulty_vector"]["overall"]
    if raw_diff == "easy" and overall > 0.75:
        issues.append("easy_high_overall")
    if raw_diff == "hard" and overall < 0.35:
        issues.append("hard_low_overall")
    if raw_diff == "easy" and max(cleaned["skill_model"]["requires"].values(), default=0.0) > 0.75:
        issues.append("easy_high_requires")
    if max(cleaned["skill_model"]["requires"].values(), default=0.0) > max(cleaned["skill_model"]["trains"].values(), default=0.0) + 0.05:
        issues.append("requires_gt_trains")
    if not cleaned["taxonomy"]["patterns"]:
        issues.append("no_pattern")
    if len(cleaned["taxonomy"]["micro_skills"]) < 2 and not (raw_diff == "easy" and overall <= TRIVIAL_EASY_THRESHOLD and len(cleaned["taxonomy"]["patterns"]) <= 1):
        issues.append("too_few_micro_skills")
    if not cleaned["skill_model"]["trains"]:
        issues.append("empty_trains")
    non_trivial = raw_diff != "easy" or overall > 0.35 or len(cleaned["taxonomy"]["patterns"]) > 1
    if non_trivial and raw_diff == "hard" and not cleaned["failure_model"]["common_failures"]:
        issues.append("empty_failure_model")
    if cleaned["metadata_confidence"] < 0.45:
        issues.append("low_confidence")

    for label in cleaned["taxonomy"]["micro_skills"]:
        if label in BROAD_MICRO_SKILLS:
            issues.append(f"broad_micro_skill:{label}")
    for label in cleaned["taxonomy"]["micro_skills"]:
        if label in active_taxonomy.items_by_category["domains"] or label in active_taxonomy.items_by_category["algorithms"] or label in active_taxonomy.items_by_category["patterns"]:
            issues.append(f"category_confusion:{label}")

    # Missing or malformed top-level keys.
    required = ["taxonomy", "topic_path", "difficulty_vector", "skill_model", "failure_model", "learning_targets", "problem_roles", "problem_traits", "pattern_cluster", "estimated_time_minutes", "solution_dna_summary", "metadata_confidence", "proposed_new_labels", "validation"]
    for key in required:
        if key not in cleaned:
            issues.append(f"missing_{key}")

    # Final validation state.
    review_reasons = []
    for issue in issues:
        if issue in STRICT_REVIEW_ISSUES or issue in SOFT_REVIEW_ISSUES:
            review_reasons.append(issue)
    cleaned["validation"]["needs_human_review"] = bool(review_reasons)
    cleaned["validation"]["review_reasons"] = dedupe(review_reasons)

    return cleaned, dedupe(issues), proposed


def classify_status(issues: Sequence[str]) -> str:
    severe_prefixes = ("invalid_json", "missing_", "malformed_", "metadata_not_object", "unable_to_repair")
    if any(issue.startswith(severe_prefixes) for issue in issues):
        return "failed"
    if issues:
        return "needs_review"
    return "valid"


def prompt_template(subset: Dict[str, List[Dict[str, Any]]], topic_paths: List[str], fields: Dict[str, str], previous_metadata: Dict[str, Any]) -> str:
    return f"""You are normalizing DSA problem metadata for a recommendation engine.

You MUST use only the allowed taxonomy labels in the main fields.
If a needed label is missing, put it in proposed_new_labels, but do not use it in main metadata.

The recommender uses this metadata to:

* recommend the next best question
* estimate user skill mastery
* detect weak areas
* suggest what to learn or revise
* recover after failure
* avoid repetitive practice

Allowed taxonomy subset:

Domains:
{format_allowed_items(subset["domains"])}

Algorithms:
{format_allowed_items(subset["algorithms"])}

Patterns:
{format_allowed_items(subset["patterns"])}

Micro-skills:
{format_allowed_items(subset["micro_skills"])}

Failure types:
{format_allowed_items(subset["failure_types"])}

Pattern clusters:
{format_allowed_items(subset["pattern_clusters"])}

Learning targets:
{format_allowed_items(subset["learning_targets"])}

Problem roles:
{format_allowed_items(subset["problem_roles"])}

Problem traits:
{format_allowed_items(subset["problem_traits"])}

Topic hierarchy paths:
{format_allowed_paths(topic_paths)}

Problem:
ID: {fields['id']}
Title: {fields['title']}
Difficulty: {fields['difficulty']}
Tags: {fields['tags']}
Description: {fields['description']}
Similar questions: {fields['similar_questions']}

Previous noisy metadata:
{json.dumps(previous_metadata, ensure_ascii=False, indent=2)}

Return ONLY valid JSON in this exact shape:

{{
"taxonomy": {{
"domains": {{}},
"algorithms": {{}},
"patterns": {{}},
"micro_skills": {{}}
}},
"topic_path": [],
"difficulty_vector": {{
"overall": 0.0,
"conceptual": 0.0,
"implementation": 0.0,
"edge_cases": 0.0,
"debugging": 0.0
}},
"skill_model": {{
"requires": {{}},
"trains": {{}},
"tests": {{}}
}},
"failure_model": {{
"common_failures": {{}}
}},
"learning_targets": [],
"problem_roles": {{}},
"problem_traits": {{}},
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
"pattern_clusters": [],
"learning_targets": [],
"problem_roles": [],
"problem_traits": []
}}
}}

Rules:

* Use only allowed labels in main fields.
* Use snake_case labels.
* All weights must be between 0 and 1.
* Use 1 to 3 domains.
* Use 1 to 3 algorithms.
* Use 1 to 4 patterns.
* Use 3 to 10 micro_skills unless the problem is extremely trivial.
* requires means prerequisite mastery before recommending this problem.
* trains means skills improved by solving this problem.
* tests means skills this problem can diagnose.
* requires should usually be lower than trains.
* Beginner-friendly Easy problems should not have high requires.
* Every common_failure must have affected_skills or affected_patterns.
* Failure severity must be between 0.1 and 1.0.
* Do not use broad topics as micro_skills.
* Do not use domains as algorithms.
* Do not use algorithms as micro_skills.
* Do not invent labels in main fields.
* If unsure, lower metadata_confidence.
* Do not output markdown or commentary.
"""


def repair_prompt(errors: Sequence[str], bad_output: str, subset: Dict[str, Any], topic_paths: List[str]) -> str:
    safe_subset = {k: subset[k] for k in subset if k != "topic_paths" and isinstance(k, str)}
    subset_json = json.dumps(safe_subset, ensure_ascii=False, indent=2)
    return f"""Your previous JSON failed validation.

Validation errors:
{json.dumps(list(errors), ensure_ascii=False, indent=2)}

Previous output:
{bad_output}

Allowed taxonomy subset:
{subset_json}

Allowed topic paths:
{format_allowed_paths(topic_paths)}

Fix the JSON.
Return ONLY valid JSON.
Do not add explanations.
Use only allowed labels in main fields.
If a label is missing, place it in proposed_new_labels.
Every common_failure must have affected_skills or affected_patterns.
All weights must be between 0 and 1.
"""


def call_llm(prompt: str, model: str, ollama_url: str, timeout: int, retries: int, temperature: float, sleep_between_requests: float) -> tuple[Dict[str, Any] | None, str | None, str | None]:
    last_error = None
    last_raw = None
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
            last_raw = response.response_text
            parsed = extract_json(response.response_text)
            return parsed, response.response_text, None
        except Exception as exc:
            last_error = str(exc)
            logger.warning("LLM attempt %s/%s failed: %s", attempt, retries, exc)
    return None, last_raw, last_error


def process_row(
    raw: Dict[str, Any],
    pass1_metadata: Dict[str, Any],
    *,
    active_taxonomy: LoadedTaxonomy,
    master_taxonomy: LoadedTaxonomy,
    label_mapping: Mapping[str, str],
    topic_hierarchy: List[Dict[str, Any]],
    model: str,
    ollama_url: str,
    timeout: int,
    retries: int,
    temperature: float,
    sleep_between_requests: float,
    dry_run: bool = False,
    write_dry_run: bool = False,
    debug: bool = False,
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None, List[str], str | None, str | None]:
    fields = build_problem_fields(raw)
    routing = build_routing_signals(raw, pass1_metadata)
    subset = select_taxonomy_subset(routing, active_taxonomy, topic_hierarchy, label_mapping)
    prompt = prompt_template(subset, subset["topic_paths"], fields, pass1_metadata)

    if dry_run or debug:
        logger.info("Routing signals for %s: %s", fields["id"], json.dumps(routing, ensure_ascii=False, indent=2))
        logger.info("Selected subset for %s: %s", fields["id"], json.dumps({k: subset[k] for k in CATEGORY_KEYS}, ensure_ascii=False, indent=2))
        logger.info("Prompt preview for %s:\n%s", fields["id"], prompt[:2000])

    if dry_run and not write_dry_run:
        return None, None, [], None, None

    parsed, raw_response_text, error = call_llm(prompt, model, ollama_url, timeout, retries, temperature, sleep_between_requests)
    if parsed is None:
        return None, None, [f"invalid_json:{error or 'unknown_error'}"], raw_response_text, error

    candidate, issues, proposed = sanitize_final_metadata(parsed, subset, active_taxonomy, label_mapping, routing)
    status = classify_status(issues)

    if status == "valid":
        return candidate, {"status": status, "issues": issues}, issues, raw_response_text, None

    repair = repair_prompt(issues, raw_response_text or json.dumps(parsed, ensure_ascii=False), subset, subset["topic_paths"])
    repaired, repaired_raw, repair_error = call_llm(repair, model, ollama_url, timeout, 1, temperature, sleep_between_requests)
    if repaired is None:
        return candidate if candidate else None, {"status": "failed", "issues": issues}, issues + [f"unable_to_repair:{repair_error or 'unknown_error'}"], raw_response_text, repair_error

    candidate2, issues2, _ = sanitize_final_metadata(repaired, subset, active_taxonomy, label_mapping, routing)
    if not candidate2:
        return None, {"status": "failed", "issues": issues2}, issues2, repaired_raw, repair_error

    status2 = classify_status(issues2)
    if status2 == "failed":
        return None, {"status": "failed", "issues": issues2}, issues2, repaired_raw, repair_error
    return candidate2, {"status": status2, "issues": issues2}, issues2, repaired_raw, repair_error


def bucket_difficulty(value: float) -> str:
    for threshold, label in DIFFICULTY_BUCKETS:
        if value < threshold:
            return label
    return "0.8-1.0"


def bucket_confidence(value: float) -> str:
    return bucket_difficulty(value)


def update_summary(summary: Dict[str, Any], metadata: Dict[str, Any], issues: Sequence[str], row: Dict[str, Any]) -> None:
    status = classify_status(issues)
    summary["total"] += 1
    summary[f"{status}_count"] += 1
    summary["issue_counter"].update(issues)

    if any(metadata.get("proposed_new_labels", {}).get(k) for k in metadata.get("proposed_new_labels", {})):
        summary["proposed_counter"].update(
            label for labels in metadata.get("proposed_new_labels", {}).values() for label in labels
        )

    diff = metadata.get("difficulty_vector", {})
    overall = float(diff.get("overall", 0.0) or 0.0) if isinstance(diff, dict) else 0.0
    summary["difficulty_counter"][bucket_difficulty(overall)] += 1

    confidence = float(metadata.get("metadata_confidence", 0.0) or 0.0)
    summary["confidence_counter"][bucket_confidence(confidence)] += 1

    for label in metadata.get("taxonomy", {}).get("domains", {}):
        summary["domain_counter"][label] += 1
    for label in metadata.get("taxonomy", {}).get("algorithms", {}):
        summary["algorithm_counter"][label] += 1
    for label in metadata.get("taxonomy", {}).get("patterns", {}):
        summary["pattern_counter"][label] += 1
    for label in metadata.get("taxonomy", {}).get("micro_skills", {}):
        summary["micro_skill_counter"][label] += 1

    req = metadata.get("skill_model", {}).get("requires", {})
    trains = metadata.get("skill_model", {}).get("trains", {})
    fail = metadata.get("failure_model", {}).get("common_failures", {})
    if max(req.values(), default=0.0) > max(trains.values(), default=0.0) + 0.05:
        summary["suspicious_requires"].append(
            {"problem_id": row["problem_id"], "reason": "requires_gt_trains", "title": row["raw"].get("title", "")}
        )
    if not fail or all(not payload.get("affected_skills") and not payload.get("affected_patterns") if isinstance(payload, dict) else True for payload in fail.values()):
        summary["weak_failure_models"].append(
            {"problem_id": row["problem_id"], "reason": "empty_or_weak_failure_model", "title": row["raw"].get("title", "")}
        )
    if status == "needs_review":
        summary["needs_review_examples"].append(
            {
                "problem_id": row["problem_id"],
                "title": row["raw"].get("title", ""),
                "issues": list(issues)[:10],
            }
        )
    if any(issue.startswith("unknown_") for issue in issues):
        summary["outside_taxonomy"].append(
            {"problem_id": row["problem_id"], "issues": [i for i in issues if i.startswith("unknown_")]}
        )


def build_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Pass 3 Validation Report",
        "",
        f"- Total rows: {summary['total']}",
        f"- Valid count: {summary['valid_count']}",
        f"- Needs review count: {summary['needs_review_count']}",
        f"- Failed count: {summary['failed_count']}",
        "",
        "## Top Validation Issues",
    ]
    if summary["issue_counter"]:
        for label, freq in summary["issue_counter"].most_common(50):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")

    lines.extend(["", "## Top Proposed New Labels"])
    if summary["proposed_counter"]:
        for label, freq in summary["proposed_counter"].most_common(50):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")

    lines.extend(["", "## Metadata Confidence Distribution"])
    for bucket, freq in summary["confidence_counter"].items():
        lines.append(f"- `{bucket}`: {freq}")

    lines.extend(["", "## Difficulty Distribution"])
    for bucket, freq in summary["difficulty_counter"].items():
        lines.append(f"- `{bucket}`: {freq}")

    lines.extend(["", "## Top Domains"])
    if summary["domain_counter"]:
        for label, freq in summary["domain_counter"].most_common(30):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")

    lines.extend(["", "## Top Algorithms"])
    if summary["algorithm_counter"]:
        for label, freq in summary["algorithm_counter"].most_common(30):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")

    lines.extend(["", "## Top Patterns"])
    if summary["pattern_counter"]:
        for label, freq in summary["pattern_counter"].most_common(30):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")

    lines.extend(["", "## Top Micro-Skills"])
    if summary["micro_skill_counter"]:
        for label, freq in summary["micro_skill_counter"].most_common(40):
            lines.append(f"- `{label}`: {freq}")
    else:
        lines.append("- None")

    lines.extend(["", "## Suspicious Requires"])
    if summary["suspicious_requires"]:
        for item in summary["suspicious_requires"][:20]:
            lines.append(f"- `{item['problem_id']}` ({item['title']}): {item['reason']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Weak Failure Models"])
    if summary["weak_failure_models"]:
        for item in summary["weak_failure_models"][:20]:
            lines.append(f"- `{item['problem_id']}` ({item['title']}): {item['reason']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Labels Outside Allowed Taxonomy"])
    if summary["outside_taxonomy"]:
        for item in summary["outside_taxonomy"][:20]:
            lines.append(f"- `{item['problem_id']}`: {item['issues']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Needs Review Examples"])
    if summary["needs_review_examples"]:
        for item in summary["needs_review_examples"][:20]:
            lines.append(f"- `{item['problem_id']}` ({item['title']}): {item['issues']}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> int:
    setup_global_logger()

    parser = argparse.ArgumentParser(description="Run constrained pass 3 DSA metadata enrichment.")
    parser.add_argument("--input", default="data/intermediate/pass1_enriched.jsonl")
    parser.add_argument("--master-taxonomy", default="data/processed/master_taxonomy.json")
    parser.add_argument("--active-taxonomy", default="data/processed/active_taxonomy.json")
    parser.add_argument("--label-mapping", default="data/processed/label_mapping.json")
    parser.add_argument("--topic-hierarchy", default="data/processed/topic_hierarchy.json")
    parser.add_argument("--output", default="data/processed/problems_enriched_final.jsonl")
    parser.add_argument("--review-output", default="data/processed/pass3_needs_review.jsonl")
    parser.add_argument("--failures", default="data/failures/pass3_failures.jsonl")
    parser.add_argument("--report", default="data/processed/pass3_validation_report.md")
    parser.add_argument("--model", default="qwen2.5-coder:14b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-dry-run", action="store_true")
    parser.add_argument("--debug-problem-id", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    master_taxonomy = load_taxonomy_file(Path(args.master_taxonomy))
    active_taxonomy = load_taxonomy_file(Path(args.active_taxonomy))
    label_mapping = load_flat_mapping(Path(args.label_mapping))
    topic_hierarchy = load_topic_hierarchy(Path(args.topic_hierarchy))

    output_path = ensure_parent_dir(args.output)
    review_path = ensure_parent_dir(args.review_output)
    failures_path = ensure_parent_dir(args.failures)
    report_path = ensure_parent_dir(args.report)

    if args.force:
        for path in (output_path, review_path, failures_path, report_path):
            if path.exists():
                path.unlink()

    records = load_input_records(input_path)
    existing_ids = read_existing_ids(output_path) if args.resume and not args.force else set()
    logger.info("Loaded %s input rows from %s", len(records), input_path)

    summary = {
        "total": 0,
        "valid_count": 0,
        "needs_review_count": 0,
        "failed_count": 0,
        "issue_counter": Counter(),
        "proposed_counter": Counter(),
        "confidence_counter": Counter(),
        "difficulty_counter": Counter(),
        "domain_counter": Counter(),
        "algorithm_counter": Counter(),
        "pattern_counter": Counter(),
        "micro_skill_counter": Counter(),
        "suspicious_requires": [],
        "weak_failure_models": [],
        "outside_taxonomy": [],
        "needs_review_examples": [],
    }

    to_process = []
    for index, record in enumerate(records):
        if index < args.start_index:
            continue
        raw, previous_metadata = unwrap_raw_and_metadata(record)
        problem_id = infer_problem_id(raw, fallback=f"row_{index}")
        if args.debug_problem_id and str(problem_id) != str(args.debug_problem_id):
            continue
        if not args.force and problem_id in existing_ids:
            continue
        to_process.append((problem_id, raw, previous_metadata))
    if args.limit is not None:
        to_process = to_process[: args.limit]

    if args.dry_run and not args.write_dry_run:
        logger.info("Dry run enabled; no output files will be written.")

    for problem_id, raw, previous_metadata in to_process:
        row = {"problem_id": problem_id, "raw": raw, "pass1_metadata": previous_metadata}
        try:
            metadata, validation, issues, raw_response, error = process_row(
                raw,
                previous_metadata,
                active_taxonomy=active_taxonomy,
                master_taxonomy=master_taxonomy,
                label_mapping=label_mapping,
                topic_hierarchy=topic_hierarchy,
                model=args.model,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                retries=args.retries,
                temperature=args.temperature,
                sleep_between_requests=args.sleep_between_requests,
                dry_run=args.dry_run,
                write_dry_run=args.write_dry_run,
                debug=bool(args.debug_problem_id),
            )

            if args.dry_run and not args.write_dry_run:
                continue

            if metadata is None or validation is None:
                summary["failed_count"] += 1
                failure_row = {
                    "problem_id": problem_id,
                    "raw": raw,
                    "pass1_metadata": previous_metadata,
                    "error": error or "unable_to_repair",
                    "raw_response": raw_response,
                    "validation_issues": issues,
                }
                append_jsonl(failures_path, [failure_row])
                continue

            status = validation["status"]
            final_row = {"problem_id": problem_id, "raw": raw, "metadata": metadata, "validation": validation}
            update_summary(summary, metadata, issues, row)
            if status == "valid":
                append_jsonl(output_path, [final_row])
            elif status == "needs_review":
                append_jsonl(output_path, [final_row])
                append_jsonl(review_path, [final_row])
            else:
                summary["failed_count"] += 1
                append_jsonl(failures_path, [{
                    "problem_id": problem_id,
                    "raw": raw,
                    "pass1_metadata": previous_metadata,
                    "error": error or "validation_failed",
                    "raw_response": raw_response,
                    "validation_issues": issues,
                }])

            logger.info("Processed %s status=%s issues=%s", problem_id, status, issues[:6])
            if args.debug_problem_id:
                logger.info("LLM response for %s:\n%s", problem_id, raw_response or "")
                logger.info("Validation for %s:\n%s", problem_id, json.dumps(validation, ensure_ascii=False, indent=2))
        except Exception as exc:
            summary["failed_count"] += 1
            logger.exception("Unexpected error while processing %s", problem_id)
            append_jsonl(failures_path, [{
                "problem_id": problem_id,
                "raw": raw,
                "pass1_metadata": previous_metadata,
                "error": str(exc),
                "raw_response": None,
                "validation_issues": [str(exc)],
            }])

    if not (args.dry_run and not args.write_dry_run):
        report = build_report(summary)
        report_path.write_text(report, encoding="utf-8")
        logger.info("Wrote report to %s", report_path)

    logger.info(
        "Finished. total=%s valid=%s needs_review=%s failed=%s",
        summary["total"],
        summary["valid_count"],
        summary["needs_review_count"],
        summary["failed_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
