from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\Users\AdhyanJ\Desktop\datasets\leetcode_dataset")
CANONICAL = ROOT / "data" / "processed" / "canonical_taxonomy.json"
PROCESSED = ROOT / "data" / "processed"


def add_item(section: str, item: dict[str, Any], key: str = "label") -> None:
    items = data.setdefault(section, [])
    for existing in items:
        if isinstance(existing, dict) and existing.get(key) == item.get(key):
            return
    items.append(item)


def update_topic(topic: str, **updates: Any) -> None:
    for item in data.setdefault("topic_hierarchy", []):
        if isinstance(item, dict) and item.get("topic") == topic:
            for key, value in updates.items():
                if key.startswith("contains_") or key == "children":
                    current = item.get(key, []) if isinstance(item.get(key), list) else []
                    merged = list(current) + list(value)
                    item[key] = list(dict.fromkeys(merged))
                else:
                    item[key] = value
            return
    new_item = {"topic": topic}
    new_item.update(updates)
    data.setdefault("topic_hierarchy", []).append(new_item)


data = json.loads(CANONICAL.read_text(encoding="utf-8"))

for item in [
    {"label": "kmp_string_matching", "description": "Prefix-function based linear-time string matching.", "aliases": ["kmp", "knuth_morris_pratt"], "parent": None},
    {"label": "z_algorithm_string_matching", "description": "Z-function based linear-time string matching.", "aliases": ["z_algorithm", "z_algo"], "parent": None},
    {"label": "manacher_palindrome_radius", "description": "Linear-time palindrome radius expansion.", "aliases": ["manacher"], "parent": None},
    {"label": "rolling_hash_string_matching", "description": "Hash-based substring matching with rolling hashes.", "aliases": ["rolling_hash", "rabin_karp", "string_hashing"], "parent": None},
    {"label": "suffix_array", "description": "Suffix array construction for substring ordering and queries.", "aliases": [], "parent": None},
    {"label": "suffix_automaton", "description": "Automaton over substrings for advanced string queries.", "aliases": [], "parent": None},
    {"label": "tarjan_scc", "description": "Tarjan low-link algorithm for strongly connected components.", "aliases": ["tarjan", "tarjan_strongly_connected_components"], "parent": None},
    {"label": "kosaraju_scc", "description": "Kosaraju two-pass algorithm for strongly connected components.", "aliases": ["kosaraju", "kosaraju_strongly_connected_components"], "parent": None},
    {"label": "articulation_points", "description": "Low-link articulation point detection in graphs.", "aliases": [], "parent": None},
    {"label": "bridges", "description": "Low-link bridge detection in graphs.", "aliases": [], "parent": None},
    {"label": "eulerian_path", "description": "Eulerian path construction with degree balance and edge consumption.", "aliases": ["hierholzer", "hierholzer_algorithm", "eulerian"], "parent": None},
    {"label": "sieve_of_eratosthenes", "description": "Prime sieve for generating primes and marking composites.", "aliases": ["sieve", "eratosthenes"], "parent": None},
    {"label": "prime_factorization", "description": "Prime factor decomposition for number theory tasks.", "aliases": [], "parent": None},
    {"label": "matrix_exponentiation", "description": "Matrix powers for recurrence acceleration and walk counting.", "aliases": ["matrix_power"], "parent": None},
    {"label": "merge_sort", "description": "Divide-and-conquer stable merge sorting.", "aliases": [], "parent": None},
    {"label": "counting_sort", "description": "Counting-based sorting over bounded ranges.", "aliases": [], "parent": None},
    {"label": "bucket_sort", "description": "Distribution-based sorting using buckets.", "aliases": [], "parent": None},
    {"label": "deque", "description": "Double-ended queue as a core tool for window and order maintenance.", "aliases": [], "parent": None},
    {"label": "monotonic_queue", "description": "Deque-based monotonic queue for sliding-window extrema.", "aliases": [], "parent": None},
    {"label": "lfu_cache", "description": "Least-frequently-used cache design with frequency buckets.", "aliases": ["lfu_cache_design"], "parent": None},
    {"label": "min_stack", "description": "Stack supporting constant-time minimum queries.", "aliases": [], "parent": None},
    {"label": "max_stack", "description": "Stack supporting constant-time maximum queries.", "aliases": [], "parent": None},
    {"label": "median_of_data_stream", "description": "Online median maintenance with balanced heaps.", "aliases": ["median_finder"], "parent": None},
    {"label": "ordered_set", "description": "Ordered set or tree-backed set supporting order statistics.", "aliases": [], "parent": None},
    {"label": "line_sweep", "description": "Event-based line sweep for intervals and geometry.", "aliases": ["sweep_line"], "parent": None},
    {"label": "convex_hull", "description": "Computational geometry convex hull construction.", "aliases": [], "parent": None},
    {"label": "kadane_algorithm", "description": "Running maximum subarray accumulation with reset logic.", "aliases": ["kadane"], "parent": None},
    {"label": "zero_one_bfs", "description": "Shortest path optimization for 0-1 weighted graphs using a deque.", "aliases": ["zero_one_bfs_shortest_path"], "parent": None},
]:
    add_item("algorithms", item)

for item in [
    {"label": "string_matching", "description": "General pattern of matching a substring or pattern against a larger string.", "aliases": [], "parent": None, "related_algorithms": ["kmp_string_matching", "z_algorithm_string_matching", "rolling_hash_string_matching"], "common_micro_skills": ["failure_function_construction", "rolling_hash_collision_control", "string_indexing"]},
    {"label": "palindrome_manacher", "description": "Palindrome detection with center expansion or linear radius propagation.", "aliases": ["manacher_palindrome"], "parent": None, "related_algorithms": ["manacher_palindrome_radius"], "common_micro_skills": ["palindrome_radius_expansion", "boundary_handling", "index_management"]},
    {"label": "rolling_hash_search", "description": "Substring search and comparison using rolling hash fingerprints.", "aliases": [], "parent": None, "related_algorithms": ["rolling_hash_string_matching"], "common_micro_skills": ["rolling_hash_collision_control", "modular_arithmetic_guarding"]},
    {"label": "suffix_structure_processing", "description": "Substring analytics using suffix arrays or suffix automata.", "aliases": [], "parent": None, "related_algorithms": ["suffix_array", "suffix_automaton"], "common_micro_skills": ["suffix_structure_building", "state_encoding", "index_management"]},
    {"label": "strongly_connected_components", "description": "Decompose directed graphs into strongly connected components.", "aliases": ["scc"], "parent": None, "related_algorithms": ["tarjan_scc", "kosaraju_scc"], "common_micro_skills": ["graph_adjacency_traversal", "visited_marking", "scc_lowlink_reasoning"]},
    {"label": "eulerian_path_construction", "description": "Build an ordering that uses each edge exactly once when possible.", "aliases": ["hierholzer_path"], "parent": None, "related_algorithms": ["eulerian_path"], "common_micro_skills": ["eulerian_edge_traversal_tracking", "graph_adjacency_traversal"]},
    {"label": "prime_sieve_factorization", "description": "Prime generation and factor decomposition for number theory tasks.", "aliases": [], "parent": None, "related_algorithms": ["sieve_of_eratosthenes", "prime_factorization"], "common_micro_skills": ["sieve_marking", "prime_factor_extraction"]},
    {"label": "matrix_exponentiation_dp", "description": "Recurrence acceleration via matrix powers and transition matrices.", "aliases": [], "parent": None, "related_algorithms": ["matrix_exponentiation"], "common_micro_skills": ["matrix_power_decomposition", "state_encoding"]},
    {"label": "sorting_and_order_statistics", "description": "Sorting, ranking, and selection families.", "aliases": [], "parent": None, "related_algorithms": ["merge_sort", "counting_sort", "bucket_sort", "ordered_set"], "common_micro_skills": ["bucket_bound_assignment", "order_statistic_querying"]},
    {"label": "cache_design", "description": "Data structure design for bounded caches with eviction policies.", "aliases": ["lru_lfu_cache"], "parent": None, "related_algorithms": ["lru_cache", "lfu_cache"], "common_micro_skills": ["cache_eviction_policy", "pointer_update", "hash_map_usage"]},
    {"label": "data_stream_median", "description": "Online maintenance of statistics from a stream.", "aliases": ["median_finder"], "parent": None, "related_algorithms": ["median_of_data_stream"], "common_micro_skills": ["heap_usage", "heap_rebalancing", "state_tracking"]},
    {"label": "deque_window_processing", "description": "Use a deque to maintain a moving window or monotonic frontier.", "aliases": [], "parent": None, "related_algorithms": ["deque", "monotonic_queue"], "common_micro_skills": ["deque_window_management", "monotonic_queue_invariant"]},
    {"label": "line_sweep_intervals", "description": "Sort events and sweep across intervals or geometric events.", "aliases": ["sweep_line"], "parent": None, "related_algorithms": ["line_sweep"], "common_micro_skills": ["line_sweep_event_sorting", "boundary_handling"]},
    {"label": "convex_hull_geometry", "description": "Geometric hull construction and orientation reasoning.", "aliases": [], "parent": None, "related_algorithms": ["convex_hull"], "common_micro_skills": ["cross_product_orientation", "geometry_point_ordering"]},
    {"label": "kadane_max_subarray", "description": "Track the best subarray ending at each position to maximize total sum.", "aliases": ["maximum_subarray"], "parent": None, "related_algorithms": ["kadane_algorithm"], "common_micro_skills": ["kadane_running_max_tracking", "state_tracking"]},
    {"label": "monotonic_queue_window", "description": "Maintain monotonic extrema within a moving sliding window.", "aliases": [], "parent": None, "related_algorithms": ["monotonic_queue", "deque"], "common_micro_skills": ["monotonic_queue_invariant", "deque_window_management"]},
    {"label": "prefix_suffix_combination", "description": "Combine prefix and suffix aggregates to answer local or global constraints.", "aliases": [], "parent": None, "related_algorithms": ["prefix_sum"], "common_micro_skills": ["prefix_sum_construction", "state_tracking"]},
    {"label": "advanced_bit_tricks", "description": "Apply bitwise identities, masks, and binary reasoning.", "aliases": [], "parent": None, "related_algorithms": ["bit_manipulation", "bitmask_dp"], "common_micro_skills": ["bitmask_operations", "state_encoding"]},
]:
    add_item("patterns", item)

for item in [
    {"label": "failure_function_construction", "description": "Build prefix-function style failure links for string matching.", "aliases": [], "parent": None, "related_patterns": ["string_matching"]},
    {"label": "palindrome_radius_expansion", "description": "Expand palindrome radii correctly across mirrored centers.", "aliases": [], "parent": None, "related_patterns": ["palindrome_manacher"]},
    {"label": "rolling_hash_collision_control", "description": "Choose hash bases and moduli that keep collision risk acceptable.", "aliases": [], "parent": None, "related_patterns": ["rolling_hash_search", "string_matching"]},
    {"label": "suffix_structure_building", "description": "Construct suffix arrays or automata without corrupting ordering invariants.", "aliases": [], "parent": None, "related_patterns": ["suffix_structure_processing"]},
    {"label": "scc_lowlink_reasoning", "description": "Track discovery and low-link values for SCC-style graph decompositions.", "aliases": [], "parent": None, "related_patterns": ["strongly_connected_components"]},
    {"label": "eulerian_edge_traversal_tracking", "description": "Consume edges exactly once while preserving Eulerian path validity.", "aliases": [], "parent": None, "related_patterns": ["eulerian_path_construction"]},
    {"label": "sieve_marking", "description": "Mark composite numbers correctly while traversing prime candidates.", "aliases": [], "parent": None, "related_patterns": ["prime_sieve_factorization"]},
    {"label": "prime_factor_extraction", "description": "Repeatedly factor out primes and manage multiplicities correctly.", "aliases": [], "parent": None, "related_patterns": ["prime_sieve_factorization"]},
    {"label": "matrix_power_decomposition", "description": "Break exponentiation into binary powers and combine transition matrices.", "aliases": [], "parent": None, "related_patterns": ["matrix_exponentiation_dp"]},
    {"label": "merge_sort_split_merge", "description": "Split recursively and merge sorted halves without losing stability or order.", "aliases": [], "parent": None, "related_patterns": ["sorting_and_order_statistics"]},
    {"label": "bucket_bound_assignment", "description": "Choose bucket ranges and placement rules correctly.", "aliases": [], "parent": None, "related_patterns": ["sorting_and_order_statistics"]},
    {"label": "deque_window_management", "description": "Maintain a deque that tracks window candidates and expirations.", "aliases": [], "parent": None, "related_patterns": ["deque_window_processing", "monotonic_queue_window"]},
    {"label": "line_sweep_event_sorting", "description": "Sort and process sweep events in the correct order.", "aliases": [], "parent": None, "related_patterns": ["line_sweep_intervals"]},
    {"label": "cross_product_orientation", "description": "Use vector cross products to determine geometric orientation and turns.", "aliases": [], "parent": None, "related_patterns": ["convex_hull_geometry"]},
    {"label": "cache_eviction_policy", "description": "Maintain the correct eviction rule for cache state updates.", "aliases": [], "parent": None, "related_patterns": ["cache_design"]},
    {"label": "heap_rebalancing", "description": "Keep heaps balanced so the chosen statistic stays correct.", "aliases": [], "parent": None, "related_patterns": ["data_stream_median"]},
    {"label": "order_statistic_querying", "description": "Track rank or order statistics under insertions and deletions.", "aliases": [], "parent": None, "related_patterns": ["sorting_and_order_statistics"]},
    {"label": "modular_arithmetic_guarding", "description": "Prevent overflow and preserve congruence when working mod integers.", "aliases": [], "parent": None, "related_patterns": ["rolling_hash_search", "prime_sieve_factorization", "matrix_exponentiation_dp"]},
    {"label": "kadane_running_max_tracking", "description": "Maintain the best running subarray sum and reset correctly.", "aliases": [], "parent": None, "related_patterns": ["kadane_max_subarray"]},
    {"label": "monotonic_queue_invariant", "description": "Preserve monotonic order while expiring stale window elements.", "aliases": [], "parent": None, "related_patterns": ["monotonic_queue_window"]},
    {"label": "geometry_point_ordering", "description": "Order points consistently for hull or sweep computations.", "aliases": [], "parent": None, "related_patterns": ["convex_hull_geometry"]},
    {"label": "array_indexing", "description": "Address array elements safely with correct index math.", "aliases": [], "parent": None, "related_patterns": ["two_pointers", "subarray_sum"]},
    {"label": "hash_map_lookup", "description": "Retrieve and update hash-map entries without losing key semantics.", "aliases": ["hashmap_lookup"], "parent": None, "related_patterns": ["string_matching", "cache_design", "two_sum"]},
    {"label": "conditional_statement", "description": "Apply branching logic to encode base cases and control flow.", "aliases": [], "parent": None, "related_patterns": ["kadane_max_subarray", "dp_linear_sequence"]},
    {"label": "two_pointer_technique", "description": "Coordinate left and right pointers across ordered data.", "aliases": ["two_pointer"], "parent": None, "related_patterns": ["two_pointers", "sliding_window_variable"]},
    {"label": "ordering_constraints", "description": "Preserve ordering rules while choosing among candidate states.", "aliases": ["ordering_constraint"], "parent": None, "related_patterns": ["binary_search_rotated_array", "sorting_and_order_statistics"]},
    {"label": "board_representation", "description": "Encode chessboard-like or grid-like states cleanly.", "aliases": [], "parent": None, "related_patterns": ["backtracking_combinations"]},
    {"label": "constraint_checking", "description": "Validate whether a partial assignment respects all rules.", "aliases": [], "parent": None, "related_patterns": ["backtracking_combinations", "strongly_connected_components"]},
    {"label": "node_comparison", "description": "Compare node values or structure while traversing trees or lists.", "aliases": [], "parent": None, "related_patterns": ["tree_dfs", "linked_list_reversal"]},
]:
    add_item("micro_skills", item)

for item in [
    {"label": "incorrect_failure_function", "description": "Prefix-function or failure-link computation is wrong.", "aliases": [], "affected_skills": ["failure_function_construction"], "affected_patterns": ["string_matching"]},
    {"label": "incorrect_palindrome_radius", "description": "Palindrome radius expansion stops too early or overshoots.", "aliases": [], "affected_skills": ["palindrome_radius_expansion"], "affected_patterns": ["palindrome_manacher"]},
    {"label": "wrong_rolling_hash_collision_handling", "description": "Rolling hash collisions or modulus choices are mishandled.", "aliases": [], "affected_skills": ["rolling_hash_collision_control"], "affected_patterns": ["rolling_hash_search"]},
    {"label": "suffix_structure_construction_error", "description": "Suffix array or automaton construction breaks invariants.", "aliases": [], "affected_skills": ["suffix_structure_building"], "affected_patterns": ["suffix_structure_processing"]},
    {"label": "scc_lowlink_error", "description": "Low-link or component bookkeeping is incorrect.", "aliases": [], "affected_skills": ["scc_lowlink_reasoning"], "affected_patterns": ["strongly_connected_components"]},
    {"label": "eulerian_edge_reuse_error", "description": "An edge is reused, skipped, or consumed in the wrong order.", "aliases": [], "affected_skills": ["eulerian_edge_traversal_tracking"], "affected_patterns": ["eulerian_path_construction"]},
    {"label": "wrong_sieve_marking", "description": "Composite marking or prime iteration is incorrect.", "aliases": [], "affected_skills": ["sieve_marking"], "affected_patterns": ["prime_sieve_factorization"]},
    {"label": "prime_factor_missed", "description": "A prime factor or multiplicity is omitted during factorization.", "aliases": [], "affected_skills": ["prime_factor_extraction"], "affected_patterns": ["prime_sieve_factorization"]},
    {"label": "wrong_matrix_exponent_order", "description": "Matrix multiplication order or power decomposition is incorrect.", "aliases": [], "affected_skills": ["matrix_power_decomposition"], "affected_patterns": ["matrix_exponentiation_dp"]},
    {"label": "merge_sort_split_error", "description": "Recursive split or merge step loses ordering or elements.", "aliases": [], "affected_skills": ["merge_sort_split_merge"], "affected_patterns": ["sorting_and_order_statistics"]},
    {"label": "bucket_assignment_error", "description": "Bucket ranges or insertion logic are incorrect.", "aliases": [], "affected_skills": ["bucket_bound_assignment"], "affected_patterns": ["sorting_and_order_statistics"]},
    {"label": "deque_window_mismanagement", "description": "Stale elements are not removed or candidates are misplaced in a deque window.", "aliases": [], "affected_skills": ["deque_window_management"], "affected_patterns": ["deque_window_processing", "monotonic_queue_window"]},
    {"label": "line_sweep_event_order_error", "description": "Events are processed in the wrong sweep order.", "aliases": [], "affected_skills": ["line_sweep_event_sorting"], "affected_patterns": ["line_sweep_intervals"]},
    {"label": "wrong_orientation_check", "description": "Cross-product or orientation sign is computed incorrectly.", "aliases": [], "affected_skills": ["cross_product_orientation"], "affected_patterns": ["convex_hull_geometry"]},
    {"label": "wrong_cache_eviction", "description": "The cache evicts the wrong entry or updates recency or frequency incorrectly.", "aliases": [], "affected_skills": ["cache_eviction_policy"], "affected_patterns": ["cache_design"]},
    {"label": "heap_rebalance_error", "description": "Heap sizes or ordering are not restored after updates.", "aliases": [], "affected_skills": ["heap_rebalancing"], "affected_patterns": ["data_stream_median"]},
    {"label": "order_statistic_misalignment", "description": "Rank or order statistic queries are returned from the wrong position.", "aliases": [], "affected_skills": ["order_statistic_querying"], "affected_patterns": ["sorting_and_order_statistics"]},
    {"label": "wrong_modular_arithmetic", "description": "Modulo arithmetic or overflow handling is incorrect.", "aliases": [], "affected_skills": ["modular_arithmetic_guarding"], "affected_patterns": ["rolling_hash_search", "prime_sieve_factorization", "matrix_exponentiation_dp"]},
    {"label": "kadane_state_reset_error", "description": "Running subarray state is reset at the wrong time or not at all.", "aliases": [], "affected_skills": ["kadane_running_max_tracking"], "affected_patterns": ["kadane_max_subarray"]},
    {"label": "monotonic_queue_invariant_break", "description": "The monotonic order or stale expiration invariant is violated.", "aliases": [], "affected_skills": ["monotonic_queue_invariant"], "affected_patterns": ["monotonic_queue_window"]},
]:
    add_item("failure_types", item)

for item in [
    {"label": "string_algorithms_and_palindromes", "description": "Pattern family for matching, palindromes, and suffix-based string work.", "patterns": ["string_matching", "palindrome_manacher", "rolling_hash_search", "suffix_structure_processing"]},
    {"label": "advanced_graph_algorithms", "description": "Directed and weighted graph decompositions beyond simple traversal.", "patterns": ["strongly_connected_components", "eulerian_path_construction", "graph_connected_components", "shortest_path"]},
    {"label": "advanced_number_theory_and_math", "description": "Prime, modular, recurrence, and geometry-adjacent math routines.", "patterns": ["prime_sieve_factorization", "matrix_exponentiation_dp", "advanced_bit_tricks", "convex_hull_geometry"]},
    {"label": "streaming_and_cache_design", "description": "Online processing with bounded memory and eviction policies.", "patterns": ["cache_design", "data_stream_median", "deque_window_processing", "monotonic_queue_window"]},
    {"label": "sorting_and_order_statistics", "description": "Sorting, ranking, and selection families.", "patterns": ["sorting_and_order_statistics", "kadane_max_subarray"]},
    {"label": "geometry_and_line_sweep", "description": "Sweep-line and computational geometry problem families.", "patterns": ["line_sweep_intervals", "convex_hull_geometry"]},
    {"label": "subarray_and_prefix_variants", "description": "Prefix, suffix, and subarray accumulation families.", "patterns": ["prefix_suffix_combination", "kadane_max_subarray"]},
    {"label": "advanced_data_structures", "description": "Specialized structures such as ordered sets, heaps, and monotonic queues.", "patterns": ["deque_window_processing", "monotonic_queue_window", "data_stream_median", "cache_design"]},
]:
    add_item("pattern_clusters", item)

for item in [
    {"label": "practice_string_matching", "description": "Learn how to align and compare pattern strings efficiently.", "target_type": "pattern", "target_label": "string_matching"},
    {"label": "practice_palindromes", "description": "Strengthen palindrome reasoning, center expansion, and symmetry handling.", "target_type": "pattern", "target_label": "palindrome_manacher"},
    {"label": "practice_scc", "description": "Review strongly connected component decomposition and low-link reasoning.", "target_type": "pattern", "target_label": "strongly_connected_components"},
    {"label": "practice_eulerian_paths", "description": "Learn to build paths that consume every edge exactly once.", "target_type": "pattern", "target_label": "eulerian_path_construction"},
    {"label": "practice_number_theory", "description": "Review primes, factors, modular arithmetic, and recurrence acceleration.", "target_type": "pattern", "target_label": "prime_sieve_factorization"},
    {"label": "practice_cache_design", "description": "Reinforce cache state, eviction, and update logic.", "target_type": "pattern", "target_label": "cache_design"},
    {"label": "practice_data_streams", "description": "Practice online maintenance of statistics from a stream.", "target_type": "pattern", "target_label": "data_stream_median"},
    {"label": "practice_order_statistics", "description": "Strengthen rank, selection, and sorted-container reasoning.", "target_type": "pattern", "target_label": "sorting_and_order_statistics"},
    {"label": "practice_geometry", "description": "Reinforce geometric orientation and sweep reasoning.", "target_type": "pattern", "target_label": "convex_hull_geometry"},
    {"label": "practice_monotonic_queue", "description": "Learn monotonic deque maintenance for sliding windows.", "target_type": "pattern", "target_label": "monotonic_queue_window"},
    {"label": "linked_list_basics", "description": "Rebuild the basics of pointer manipulation and node rewiring.", "target_type": "domain", "target_label": "linked_list"},
    {"label": "bit_manipulation_basics", "description": "Review masks, shifts, and bitwise reasoning from first principles.", "target_type": "algorithm", "target_label": "bit_manipulation"},
    {"label": "practice_sorting", "description": "Strengthen comparison, counting, and bucket-based sorting.", "target_type": "algorithm", "target_label": "merge_sort"},
    {"label": "practice_zero_one_bfs", "description": "Review shortest paths with deque-based 0-1 BFS.", "target_type": "algorithm", "target_label": "zero_one_bfs"},
]:
    add_item("learning_targets", item)

for item in [
    {"label": "diagnostic", "description": "Use the problem to diagnose a specific weakness."},
    {"label": "warmup", "description": "Use the problem as a light warmup or re-entry task."},
    {"label": "benchmark", "description": "Use the problem as a calibration or baseline test."},
]:
    add_item("problem_roles", item)

for item in [
    {"label": "branching_factor", "description": "How many distinct decision branches the solution explores."},
    {"label": "state_space_size", "description": "How large the logical or DP state space becomes."},
    {"label": "implementation_depth", "description": "How much code or pointer state the solution needs."},
    {"label": "runtime_criticality", "description": "How sensitive the problem is to time complexity improvements."},
    {"label": "data_structure_pressure", "description": "How much the solution depends on maintaining a specialized structure."},
]:
    add_item("problem_traits", item)

for topic, children in {
    "dsa_foundations": ["arrays_and_strings", "linked_structures", "trees_and_heaps", "graphs_and_search", "searching_sorting_ranges", "dynamic_programming_and_backtracking", "math_bit_sql_design"],
    "arrays_and_strings": ["pair_and_sum", "sliding_window_strings", "string_algorithms", "prefix_suffix_variants"],
    "linked_structures": ["linked_list_pointer_work", "design_and_state"],
    "trees_and_heaps": ["tree_traversal_and_construction", "heap_and_priority_queue", "trie_and_prefix"],
    "graphs_and_search": ["graph_connectivity_and_paths", "advanced_graph_algorithms"],
    "searching_sorting_ranges": ["binary_search_patterns", "interval_scheduling_and_merging", "sorting_and_order_statistics", "geometry_and_line_sweep"],
    "dynamic_programming_and_backtracking": ["dynamic_programming_core", "backtracking_and_combinatorics", "state_compression_dp"],
    "math_bit_sql_design": ["advanced_number_theory_and_math", "bit_manipulation_patterns", "sql_patterns", "streaming_and_cache_design"],
}.items():
    update_topic(topic, children=children)

for topic, updates in {
    "arrays_and_strings": {
        "contains_domains": ["array", "string", "hash_table", "interval", "matrix"],
        "contains_algorithms": ["two_pointers", "sliding_window_variable", "prefix_sum", "binary_search", "sorting", "kmp_string_matching", "rolling_hash_string_matching", "manacher_palindrome_radius"],
        "contains_patterns": ["two_sum", "three_sum", "subarray_sum", "longest_substring_without_repeating", "minimum_window_substring", "anagram_detection", "string_matching", "palindrome_manacher", "rolling_hash_search", "prefix_suffix_combination"],
        "contains_micro_skills": ["string_indexing", "window_shrink_expand", "prefix_sum_construction", "failure_function_construction", "palindrome_radius_expansion", "rolling_hash_collision_control"],
    },
    "linked_structures": {
        "contains_domains": ["linked_list", "stack_queue", "design"],
        "contains_algorithms": ["linked_list_manipulation", "deque", "lru_cache", "lfu_cache", "min_stack", "max_stack"],
        "contains_patterns": ["linked_list_reversal", "linked_list_cycle", "fast_slow_pointer", "cache_design", "deque_window_processing"],
        "contains_micro_skills": ["linked_list_pointer_update", "pointer_update", "cache_eviction_policy", "deque_window_management"],
    },
    "trees_and_heaps": {
        "contains_domains": ["tree", "binary_tree", "binary_search_tree", "heap", "trie"],
        "contains_algorithms": ["tree_traversal", "breadth_first_search", "depth_first_search", "heap_priority_queue", "median_of_data_stream", "ordered_set"],
        "contains_patterns": ["tree_level_order", "tree_dfs", "bst_validation", "tree_construction", "lowest_common_ancestor", "heap_top_k", "data_stream_median"],
        "contains_micro_skills": ["tree_node_access", "recursion_depth_control", "heap_usage", "heap_rebalancing", "order_statistic_querying"],
    },
    "graphs_and_search": {
        "contains_domains": ["graph", "disjoint_set", "probability", "randomized"],
        "contains_algorithms": ["breadth_first_search", "depth_first_search", "topological_sort", "union_find", "dijkstra_shortest_path", "bellman_ford_shortest_path", "floyd_warshall_all_pairs_shortest_path", "tarjan_scc", "kosaraju_scc", "eulerian_path"],
        "contains_patterns": ["graph_connected_components", "shortest_path", "topological_ordering", "graph_union_find_components", "strongly_connected_components", "eulerian_path_construction"],
        "contains_micro_skills": ["graph_adjacency_traversal", "visited_marking", "graph_cycle_detection", "graph_weight_handling", "edge_relaxation", "scc_lowlink_reasoning", "eulerian_edge_traversal_tracking"],
    },
    "searching_sorting_ranges": {
        "contains_domains": ["array", "interval", "ordered_map", "segment_tree", "binary_indexed_tree", "ordered_set"],
        "contains_algorithms": ["binary_search", "binary_search_on_answer", "merge_sort", "counting_sort", "bucket_sort", "line_sweep", "kadane_algorithm"],
        "contains_patterns": ["binary_search_rotated_array", "merge_intervals", "meeting_rooms", "sorting_and_order_statistics", "line_sweep_intervals", "kadane_max_subarray"],
        "contains_micro_skills": ["binary_search_bounds", "boundary_handling", "index_management", "line_sweep_event_sorting", "bucket_bound_assignment"],
    },
    "dynamic_programming_and_backtracking": {
        "contains_domains": ["combinatorics", "bit_manipulation", "matrix"],
        "contains_algorithms": ["dynamic_programming_top_down", "dynamic_programming_bottom_up", "backtracking", "bitmask_dp", "digit_dp", "matrix_exponentiation", "zero_one_bfs"],
        "contains_patterns": ["dp_linear_sequence", "dp_grid", "dp_interval", "backtracking_combinations", "subset_sum", "combination_sum", "bitmask_dp", "matrix_exponentiation_dp"],
        "contains_micro_skills": ["dp_transition_design", "memoization_table_design", "tabulation_ordering", "state_encoding", "subset_enumeration", "modular_arithmetic_guarding"],
    },
    "math_bit_sql_design": {
        "contains_domains": ["math", "bit_manipulation", "number_theory", "sql", "design", "geometry", "probability", "randomized"],
        "contains_algorithms": ["bit_manipulation", "ordered_map", "sieve_of_eratosthenes", "prime_factorization", "convex_hull", "median_of_data_stream", "lru_cache", "lfu_cache"],
        "contains_patterns": ["sql_group_by", "sql_window_function", "advanced_bit_tricks", "prime_sieve_factorization", "convex_hull_geometry", "cache_design", "data_stream_median"],
        "contains_micro_skills": ["sql_grouping_reasoning", "bitmask_operations", "sieve_marking", "prime_factor_extraction", "cross_product_orientation", "cache_eviction_policy"],
    },
}.items():
    update_topic(topic, **updates)

for item in [
    ("pair_and_sum", "dsa_foundations > arrays_and_strings", "Pair and Sum", "Problems that rely on fixed-size or complement-based pairing.", ["array", "hash_table"], ["two_pointers", "hash_map_lookup"], ["two_sum", "subarray_sum"], ["index_management", "hash_map_usage"]),
    ("sliding_window_strings", "dsa_foundations > arrays_and_strings", "Sliding Window Strings", "String windows, anagrams, and substring constraints.", ["string", "hash_table"], ["sliding_window_variable", "prefix_sum"], ["longest_substring_without_repeating", "minimum_window_substring", "anagram_detection"], ["window_shrink_expand", "character_frequency_count"]),
    ("string_algorithms", "dsa_foundations > arrays_and_strings", "String Algorithms", "Pattern matching, palindromes, and suffix structures.", ["string"], ["kmp_string_matching", "z_algorithm_string_matching", "manacher_palindrome_radius", "rolling_hash_string_matching"], ["string_matching", "palindrome_manacher", "rolling_hash_search"], ["failure_function_construction", "palindrome_radius_expansion", "rolling_hash_collision_control"]),
    ("prefix_suffix_variants", "dsa_foundations > arrays_and_strings", "Prefix Suffix Variants", "Prefix and suffix accumulation or combination problems.", ["array", "string"], ["prefix_sum", "kadane_algorithm"], ["prefix_suffix_combination", "kadane_max_subarray"], ["prefix_sum_construction", "kadane_running_max_tracking"]),
    ("graph_connectivity_and_paths", "dsa_foundations > graphs_and_search", "Graph Connectivity And Paths", "Connectivity, traversal, and shortest path problems.", ["graph", "disjoint_set"], ["breadth_first_search", "depth_first_search", "union_find", "dijkstra_shortest_path", "bellman_ford_shortest_path"], ["graph_connected_components", "shortest_path", "graph_union_find_components"], ["graph_adjacency_traversal", "visited_marking", "edge_relaxation"]),
    ("advanced_graph_algorithms", "dsa_foundations > graphs_and_search", "Advanced Graph Algorithms", "SCC, Eulerian, and advanced directed-graph families.", ["graph"], ["tarjan_scc", "kosaraju_scc", "eulerian_path", "floyd_warshall_all_pairs_shortest_path"], ["strongly_connected_components", "eulerian_path_construction"], ["scc_lowlink_reasoning", "eulerian_edge_traversal_tracking"]),
    ("binary_search_patterns", "dsa_foundations > searching_sorting_ranges", "Binary Search Patterns", "Classic and answer-space binary search variants.", ["array", "matrix", "interval"], ["binary_search", "binary_search_on_answer"], ["binary_search_rotated_array"], ["binary_search_bounds", "boundary_handling"]),
    ("interval_scheduling_and_merging", "dsa_foundations > searching_sorting_ranges", "Interval Scheduling And Merging", "Merging, scheduling, and overlap reasoning.", ["interval"], ["merge_intervals", "line_sweep"], ["merge_intervals", "meeting_rooms", "line_sweep_intervals"], ["boundary_handling", "line_sweep_event_sorting"]),
    ("sorting_and_order_statistics", "dsa_foundations > searching_sorting_ranges", "Sorting And Order Statistics", "Sorting, ranking, and selection families.", ["array", "ordered_set"], ["merge_sort", "counting_sort", "bucket_sort", "ordered_set"], ["sorting_and_order_statistics"], ["bucket_bound_assignment", "order_statistic_querying"]),
    ("geometry_and_line_sweep", "dsa_foundations > searching_sorting_ranges", "Geometry And Line Sweep", "Sweep-line and computational geometry families.", ["geometry", "interval"], ["line_sweep", "convex_hull"], ["line_sweep_intervals", "convex_hull_geometry"], ["line_sweep_event_sorting", "cross_product_orientation"]),
    ("dynamic_programming_core", "dsa_foundations > dynamic_programming_and_backtracking", "Dynamic Programming Core", "State transitions, memoization, and tabulation.", ["array", "matrix"], ["dynamic_programming_top_down", "dynamic_programming_bottom_up"], ["dp_linear_sequence", "dp_grid", "dp_interval"], ["dp_transition_design", "memoization_table_design"]),
    ("backtracking_and_combinatorics", "dsa_foundations > dynamic_programming_and_backtracking", "Backtracking And Combinatorics", "Search, prune, and enumerate combinations and placements.", ["combinatorics"], ["backtracking"], ["backtracking_combinations"], ["pruning", "duplicate_handling"]),
    ("state_compression_dp", "dsa_foundations > dynamic_programming_and_backtracking", "State Compression DP", "Bitmask and compact-state dynamic programming.", ["bit_manipulation"], ["bitmask_dp"], ["bitmask_dp"], ["state_encoding", "bitmask_operations"]),
    ("advanced_number_theory_and_math", "dsa_foundations > math_bit_sql_design", "Advanced Number Theory And Math", "Primes, modular arithmetic, and recurrence acceleration.", ["math", "number_theory"], ["sieve_of_eratosthenes", "prime_factorization", "matrix_exponentiation"], ["prime_sieve_factorization", "matrix_exponentiation_dp"], ["sieve_marking", "modular_arithmetic_guarding"]),
    ("bit_manipulation_patterns", "dsa_foundations > math_bit_sql_design", "Bit Manipulation Patterns", "Bitwise masks, subsets, and binary reasoning.", ["bit_manipulation"], ["bit_manipulation", "bitmask_dp"], ["advanced_bit_tricks"], ["bitmask_operations", "state_encoding"]),
    ("sql_patterns", "dsa_foundations > math_bit_sql_design", "Sql Patterns", "Group-by and window-function query reasoning.", ["sql"], ["sql_group_by", "sql_window_function"], ["sql_group_by", "sql_window_function"], ["sql_grouping_reasoning"]),
    ("streaming_and_cache_design", "dsa_foundations > math_bit_sql_design", "Streaming And Cache Design", "Online maintenance, eviction, and bounded-memory design.", ["design", "stack_queue"], ["lru_cache", "lfu_cache", "median_of_data_stream", "deque", "monotonic_queue"], ["cache_design", "data_stream_median", "deque_window_processing"], ["cache_eviction_policy", "heap_rebalancing", "deque_window_management"]),
]:
    topic, parent, title, desc, domains, algos, patterns_list, skills = item
    add_item("topic_hierarchy", {"topic": topic, "description": desc, "parent": parent, "children": [], "contains_domains": domains, "contains_algorithms": algos, "contains_patterns": patterns_list, "contains_micro_skills": skills}, key="topic")

label_mapping = data.setdefault("label_mapping", {})
label_mapping.update({
    "kmp": "kmp_string_matching",
    "knuth_morris_pratt": "kmp_string_matching",
    "z_algo": "z_algorithm_string_matching",
    "manacher": "manacher_palindrome_radius",
    "rabin_karp": "rolling_hash_string_matching",
    "rolling_hash": "rolling_hash_string_matching",
    "tarjan": "tarjan_scc",
    "kosaraju": "kosaraju_scc",
    "hierholzer": "eulerian_path",
    "eulerian": "eulerian_path",
    "sieve": "sieve_of_eratosthenes",
    "eratosthenes": "sieve_of_eratosthenes",
    "matrix_power": "matrix_exponentiation",
    "median_finder": "median_of_data_stream",
    "sweep_line": "line_sweep",
    "kadane": "kadane_algorithm",
    "zero_one_bfs": "zero_one_bfs",
    "merge_interval": "merge_intervals",
    "meeting_room": "meeting_rooms",
    "binary_search_bound": "binary_search_bounds",
    "ordering_constraint": "ordering_constraints",
    "bitmask_operation": "bitmask_operations",
    "hashmap_lookup": "hash_map_lookup",
    "two_pointer": "two_pointer_technique",
    "conditional_statement": "conditional_statement",
    "array_indexing": "array_indexing",
})

qn = data.setdefault("quality_notes", {})
qn.setdefault("labels_to_merge", [])
qn.setdefault("labels_to_remove", [])
qn.setdefault("labels_needing_human_review", [])
for label in [
    "kmp -> kmp_string_matching",
    "z_algorithm -> z_algorithm_string_matching",
    "manacher -> manacher_palindrome_radius",
    "rolling_hash -> rolling_hash_string_matching",
]:
    if label not in qn["labels_to_merge"]:
        qn["labels_to_merge"].append(label)
for label in ["diagnostic", "warmup", "benchmark"]:
    if label not in qn["labels_needing_human_review"]:
        qn["labels_needing_human_review"].append(label)

for section in ["domains", "algorithms", "patterns", "micro_skills", "failure_types", "pattern_clusters", "learning_targets", "problem_roles", "problem_traits"]:
    seen = set()
    cleaned = []
    for item in data.get(section, []):
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if label and label not in seen:
            seen.add(label)
            cleaned.append(item)
    data[section] = cleaned

seen_topics = set()
cleaned_topics = []
for item in data.get("topic_hierarchy", []):
    if not isinstance(item, dict):
        continue
    topic = item.get("topic")
    if topic and topic not in seen_topics:
        seen_topics.add(topic)
        cleaned_topics.append(item)
data["topic_hierarchy"] = cleaned_topics

CANONICAL.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
MASTER = PROCESSED / "master_taxonomy.json"
ACTIVE = PROCESSED / "active_taxonomy.json"
LABEL = PROCESSED / "label_mapping.json"
TOPIC = PROCESSED / "topic_hierarchy.json"
MASTER.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
ACTIVE.write_text(json.dumps({k: v for k, v in data.items() if k not in {"label_mapping", "quality_notes"}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
LABEL.write_text(json.dumps(data["label_mapping"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
TOPIC.write_text(json.dumps(data["topic_hierarchy"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("updated taxonomy assets")
