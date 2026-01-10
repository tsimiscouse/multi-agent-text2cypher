"""
Output Similarity Metrics

Compares execution outputs of generated vs ground truth Cypher queries.
Implements Jaccard similarity and Pass@1 for query outputs.
"""

import logging
from typing import List, Dict, Any, Set, Tuple, Hashable, Optional
from neo4j.graph import Node, Relationship

logger = logging.getLogger(__name__)


def floatify(v: Any) -> Any:
    """
    Convert values to float if possible, recursively for lists/dicts.

    Args:
        v: Value to convert

    Returns:
        Converted value (float if numeric string, otherwise unchanged)
    """
    # String: return as-is
    if isinstance(v, str):
        return v

    # Try converting to float
    try:
        return float(v)
    except:
        pass

    # Recursively handle lists
    if isinstance(v, list):
        return [floatify(x) for x in v]

    # Recursively handle dicts
    if isinstance(v, dict):
        return {k: floatify(u) for k, u in v.items()}

    # Return unchanged
    return v


def make_hashable(v: Any) -> Hashable:
    """
    Convert value to hashable type for set operations.

    Args:
        v: Value to convert

    Returns:
        Hashable version of value
    """
    float_v = floatify(v)

    if not isinstance(float_v, Hashable):
        return str(float_v)
    else:
        return float_v


def jaccard_formula(set_a: Set, set_b: Set) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Args:
        set_a: First set
        set_b: Second set

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def make_alignment(
    list_a: List[Dict],
    list_b: List[Dict]
) -> Tuple[List[Set], List[Set]]:
    """
    Align rows from two lists of dicts based on similarity.

    This is used when queries don't have ORDER BY (order doesn't matter).

    Args:
        list_a: First list of dictionaries
        list_b: Second list of dictionaries

    Returns:
        Tuple of aligned set views
    """
    # Ensure list_a is the smaller one
    swap = len(list_a) > len(list_b)

    # Convert dicts to sets of hashable values
    sets_a = [{make_hashable(v) for k, v in row.items()} for row in list_a]
    sets_b = [{make_hashable(v) for k, v in row.items()} for row in list_b]

    if swap:
        sets_a, sets_b = sets_b, sets_a

    # Align rows based on maximum similarity
    for i in range(len(sets_a)):
        max_sim = -1
        max_j = -1

        # Find best match for row i
        for j in range(i, len(sets_b)):
            sim = jaccard_formula(sets_a[i], sets_b[j])
            if sim > max_sim:
                max_j = j
                max_sim = sim

        # Swap to align
        sets_b[i], sets_b[max_j] = sets_b[max_j], sets_b[i]

    # Restore original order if swapped
    if swap:
        sets_a, sets_b = sets_b, sets_a

    return sets_a, sets_b


def compute_output_similarity(
    output_gt: List[Dict],
    output_gen: List[Dict],
    has_order_by: bool
) -> float:
    """
    Compute Jaccard similarity between query outputs.

    Args:
        output_gt: Ground truth output (list of dicts)
        output_gen: Generated query output (list of dicts)
        has_order_by: Whether query has ORDER BY clause

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if has_order_by:
        # Preserve original row order
        view_a = [row.values() for row in output_gt]
        view_b = [row.values() for row in output_gen]
    else:
        # Align rows based on similarity
        view_a, view_b = make_alignment(output_gt, output_gen)

    # Create sets with (row_index, value) pairs
    total_set_a = set()
    for i, s in enumerate(view_a):
        for elem in s:
            total_set_a.add((i, make_hashable(elem)))

    total_set_b = set()
    for i, s in enumerate(view_b):
        for elem in s:
            total_set_b.add((i, make_hashable(elem)))

    # Compute Jaccard similarity
    intersection = total_set_a & total_set_b
    union = total_set_a | total_set_b

    if len(union) == 0 and len(intersection) == 0:
        return 1.0
    elif len(union) == 0:
        return 0.0

    similarity = len(intersection) / len(union)

    return round(similarity, 2)


def extract_properties(record: Dict) -> Dict:
    """
    Extract properties from Neo4j nodes/relationships.

    If record contains Node or Relationship objects, extract their properties.

    Args:
        record: Query result record

    Returns:
        Dict with extracted properties
    """
    for key, value in record.items():
        if isinstance(value, (Node, Relationship)):
            return dict(value)

    return record


def execute_query(executor, query: str) -> Optional[List[Dict]]:
    """
    Execute query and return results.

    Args:
        executor: GraphExecutor instance
        query: Cypher query to execute

    Returns:
        List of result dicts, or None if execution failed
    """
    try:
        result = executor.execute(query)

        # Extract properties from nodes/relationships
        processed = [extract_properties(record) for record in result]

        return processed
    except Exception as e:
        logger.debug(f"Query execution failed: {e}")
        return None


def compute_jaccard_output(
    executor,
    ground_truth_query: str,
    generated_query: str
) -> Optional[float]:
    """
    Compute Jaccard similarity between outputs of two queries.

    Args:
        executor: GraphExecutor instance
        ground_truth_query: Ground truth Cypher query
        generated_query: Generated Cypher query

    Returns:
        Jaccard similarity (0.0 to 1.0), or None if execution failed
    """
    # Execute both queries
    output_gt = execute_query(executor, ground_truth_query)
    output_gen = execute_query(executor, generated_query)

    # If either failed, return None
    if output_gt is None or output_gen is None:
        return None

    # Check if query has ORDER BY
    has_order_by = "order by" in ground_truth_query.lower()

    # Compute similarity
    similarity = compute_output_similarity(output_gt, output_gen, has_order_by)

    return similarity


def compute_pass_at_1_output(
    executor,
    ground_truth_query: str,
    generated_query: str
) -> bool:
    """
    Compute Pass@1 for query output (similarity == 1.0).

    Args:
        executor: GraphExecutor instance
        ground_truth_query: Ground truth Cypher query
        generated_query: Generated Cypher query

    Returns:
        True if outputs are identical (similarity == 1.0)
    """
    similarity = compute_jaccard_output(executor, ground_truth_query, generated_query)

    if similarity is None:
        return False

    return similarity == 1.0


if __name__ == "__main__":
    # Test output similarity metrics
    print("Testing Output Similarity Metrics...")
    print("=" * 60)

    # Mock outputs
    output_gt = [
        {"n.nama": "Basis Data", "n.sks": 3},
        {"n.nama": "Aljabar Linear", "n.sks": 3},
    ]

    output_gen_perfect = [
        {"n.nama": "Basis Data", "n.sks": 3},
        {"n.nama": "Aljabar Linear", "n.sks": 3},
    ]

    output_gen_partial = [
        {"n.nama": "Basis Data", "n.sks": 3},
        {"n.nama": "Kalkulus", "n.sks": 4},
    ]

    output_gen_wrong = [
        {"n.nama": "Statistika", "n.sks": 3},
    ]

    # Test perfect match
    print("\nTest 1: Perfect match")
    sim = compute_output_similarity(output_gt, output_gen_perfect, False)
    print(f"  Jaccard Output: {sim}")
    print(f"  Pass@1 Output: {sim == 1.0}")

    # Test partial match
    print("\nTest 2: Partial match")
    sim = compute_output_similarity(output_gt, output_gen_partial, False)
    print(f"  Jaccard Output: {sim}")
    print(f"  Pass@1 Output: {sim == 1.0}")

    # Test wrong output
    print("\nTest 3: Wrong output")
    sim = compute_output_similarity(output_gt, output_gen_wrong, False)
    print(f"  Jaccard Output: {sim}")
    print(f"  Pass@1 Output: {sim == 1.0}")

    # Test with ORDER BY
    print("\nTest 4: With ORDER BY (order matters)")
    output_gt_ordered = [
        {"n.nama": "Aljabar Linear", "n.sks": 3},
        {"n.nama": "Basis Data", "n.sks": 3},
    ]
    output_gen_reversed = [
        {"n.nama": "Basis Data", "n.sks": 3},
        {"n.nama": "Aljabar Linear", "n.sks": 3},
    ]
    sim = compute_output_similarity(output_gt_ordered, output_gen_reversed, True)
    print(f"  Jaccard Output: {sim}")
    print(f"  Pass@1 Output: {sim == 1.0}")

    print("\n" + "=" * 60)
    print("Output similarity metrics test complete!")
