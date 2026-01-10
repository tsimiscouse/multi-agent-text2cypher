"""
Evaluation Metrics

Comprehensive metrics for comparing baseline vs multi-agent system.
Includes Pass@k, KG Valid@k, execution metrics, and cost analysis.
"""

import json
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict


logger = logging.getLogger(__name__)


def compare_outputs(gt_result: Any, generated_result: Any) -> bool:
    """
    Compare ground truth execution result with generated query result.

    Args:
        gt_result: Ground truth execution result
        generated_result: Generated query execution result

    Returns:
        True if outputs match (same records, order-agnostic)
    """
    # Handle None cases
    if gt_result is None and generated_result is None:
        return True
    if gt_result is None or generated_result is None:
        return False

    # Handle empty cases
    if not gt_result and not generated_result:
        return True
    if not gt_result or not generated_result:
        return False

    # Parse JSON strings if needed
    if isinstance(gt_result, str):
        try:
            gt_result = json.loads(gt_result)
        except:
            pass

    if isinstance(generated_result, str):
        try:
            generated_result = json.loads(generated_result)
        except:
            pass

    # Convert to normalized format for comparison
    def normalize(result):
        if isinstance(result, list):
            # Sort list of dicts by converting to sorted tuples
            return sorted([
                tuple(sorted(r.items())) if isinstance(r, dict) else r
                for r in result
            ])
        return result

    try:
        return normalize(gt_result) == normalize(generated_result)
    except:
        # Fallback to string comparison if normalization fails
        return str(gt_result) == str(generated_result)


def compute_pass_at_k(
    baseline_results: List[Dict[str, Any]],
    multiagent_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute Pass@k metrics.

    Args:
        baseline_results: List of baseline result dicts
        multiagent_results: List of multi-agent result dicts

    Returns:
        Dictionary with pass_at_1 and pass_at_k rates
    """
    total = len(baseline_results)
    baseline_pass = 0
    multiagent_pass = 0

    for baseline, multiagent in zip(baseline_results, multiagent_results):
        # Baseline Pass@1
        if baseline.get('pass_at_1', False):
            baseline_pass += 1

        # Multi-agent Pass@k
        if multiagent.get('pass_at_k', False):
            multiagent_pass += 1

    return {
        'baseline_pass_at_1': baseline_pass / total if total > 0 else 0.0,
        'multiagent_pass_at_k': multiagent_pass / total if total > 0 else 0.0,
        'improvement': (multiagent_pass - baseline_pass) / total if total > 0 else 0.0,
        'relative_improvement': ((multiagent_pass / baseline_pass) - 1) * 100 if baseline_pass > 0 else 0.0
    }


def compute_kg_valid_at_k(
    baseline_results: List[Dict[str, Any]],
    multiagent_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute KG Valid@k metrics (queries that execute successfully).

    Args:
        baseline_results: List of baseline result dicts
        multiagent_results: List of multi-agent result dicts

    Returns:
        Dictionary with kg_valid rates
    """
    total = len(baseline_results)
    baseline_valid = sum(1 for r in baseline_results if r.get('execution_success', False))
    multiagent_valid = sum(1 for r in multiagent_results if r.get('execution_success', False))

    return {
        'baseline_kg_valid': baseline_valid / total if total > 0 else 0.0,
        'multiagent_kg_valid': multiagent_valid / total if total > 0 else 0.0,
        'improvement': (multiagent_valid - baseline_valid) / total if total > 0 else 0.0,
        'relative_improvement': ((multiagent_valid / baseline_valid) - 1) * 100 if baseline_valid > 0 else 0.0
    }


def compute_recovery_rate(multiagent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute recovery rate (queries that failed initially but succeeded after refinement).

    Args:
        multiagent_results: List of multi-agent result dicts with iteration history

    Returns:
        Dictionary with recovery statistics
    """
    total_refined = 0  # Questions that required refinement
    recovered = 0      # Questions that were recovered

    for result in multiagent_results:
        iterations = result.get('total_iterations', 1)

        # Questions that required refinement (more than 1 iteration)
        if iterations > 1:
            total_refined += 1

            # Check if it succeeded in the end
            if result.get('execution_success', False):
                # Check if first attempt failed
                all_iterations = result.get('all_iterations', [])
                if all_iterations and len(all_iterations) > 0:
                    first_attempt = all_iterations[0]
                    first_failed = first_attempt.get('evaluation') in ['incorrect', 'error']

                    if first_failed:
                        recovered += 1

    return {
        'total_refined': total_refined,
        'recovered': recovered,
        'recovery_rate': recovered / total_refined if total_refined > 0 else 0.0,
        'refinement_rate': total_refined / len(multiagent_results) if len(multiagent_results) > 0 else 0.0
    }


def compute_iteration_statistics(multiagent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics about refinement iterations.

    Args:
        multiagent_results: List of multi-agent result dicts

    Returns:
        Dictionary with iteration statistics
    """
    iterations = [r.get('total_iterations', 1) for r in multiagent_results]

    # Distribution
    from collections import Counter
    iteration_dist = Counter(iterations)

    # First-attempt success
    first_attempt_success = sum(1 for r in multiagent_results if r.get('total_iterations', 1) == 1)

    return {
        'avg_iterations': sum(iterations) / len(iterations) if iterations else 0.0,
        'median_iterations': sorted(iterations)[len(iterations) // 2] if iterations else 0,
        'max_iterations_used': max(iterations) if iterations else 0,
        'iteration_distribution': dict(iteration_dist),
        'first_attempt_success': first_attempt_success,
        'first_attempt_success_rate': first_attempt_success / len(multiagent_results) if multiagent_results else 0.0
    }


def compute_cost_metrics(
    baseline_results: List[Dict[str, Any]],
    multiagent_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute cost metrics (tokens and latency).

    Args:
        baseline_results: List of baseline result dicts
        multiagent_results: List of multi-agent result dicts

    Returns:
        Dictionary with cost statistics
    """
    # Token usage
    baseline_total_tokens = sum(r.get('total_tokens', 0) for r in baseline_results)
    multiagent_total_tokens = sum(r.get('total_tokens', 0) for r in multiagent_results)

    baseline_avg_tokens = baseline_total_tokens / len(baseline_results) if baseline_results else 0
    multiagent_avg_tokens = multiagent_total_tokens / len(multiagent_results) if multiagent_results else 0

    # Latency
    baseline_total_time = sum(r.get('elapsed_time', 0) for r in baseline_results)
    multiagent_total_time = sum(r.get('elapsed_time', 0) for r in multiagent_results)

    baseline_avg_time = baseline_total_time / len(baseline_results) if baseline_results else 0
    multiagent_avg_time = multiagent_total_time / len(multiagent_results) if multiagent_results else 0

    return {
        'tokens': {
            'baseline_total': baseline_total_tokens,
            'multiagent_total': multiagent_total_tokens,
            'baseline_avg': baseline_avg_tokens,
            'multiagent_avg': multiagent_avg_tokens,
            'token_overhead': multiagent_total_tokens - baseline_total_tokens,
            'token_overhead_pct': ((multiagent_total_tokens / baseline_total_tokens) - 1) * 100 if baseline_total_tokens > 0 else 0.0
        },
        'latency': {
            'baseline_total': baseline_total_time,
            'multiagent_total': multiagent_total_time,
            'baseline_avg': baseline_avg_time,
            'multiagent_avg': multiagent_avg_time,
            'latency_overhead': multiagent_total_time - baseline_total_time,
            'latency_overhead_pct': ((multiagent_total_time / baseline_total_time) - 1) * 100 if baseline_total_time > 0 else 0.0
        }
    }


def compute_stratified_metrics(
    baseline_results: List[Dict[str, Any]],
    multiagent_results: List[Dict[str, Any]],
    stratify_by: str = 'complexity'
) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics stratified by a category (complexity, reasoning_level, sublevel).

    Args:
        baseline_results: List of baseline result dicts
        multiagent_results: List of multi-agent result dicts
        stratify_by: Field to stratify by

    Returns:
        Dictionary mapping category values to metrics
    """
    stratified = defaultdict(lambda: {'baseline': [], 'multiagent': []})

    for baseline, multiagent in zip(baseline_results, multiagent_results):
        category = baseline.get(stratify_by, 'Unknown')
        stratified[category]['baseline'].append(baseline)
        stratified[category]['multiagent'].append(multiagent)

    results = {}
    for category, data in stratified.items():
        baseline_data = data['baseline']
        multiagent_data = data['multiagent']

        results[category] = {
            'count': len(baseline_data),
            'pass_at_k': compute_pass_at_k(baseline_data, multiagent_data),
            'kg_valid': compute_kg_valid_at_k(baseline_data, multiagent_data),
            'iterations': compute_iteration_statistics(multiagent_data),
            'cost': compute_cost_metrics(baseline_data, multiagent_data)
        }

    return results


def compute_all_metrics(
    baseline_results: List[Dict[str, Any]],
    multiagent_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.

    Args:
        baseline_results: List of baseline result dicts
        multiagent_results: List of multi-agent result dicts

    Returns:
        Dictionary with all metrics
    """
    logger.info("Computing comprehensive metrics...")

    metrics = {
        'dataset_size': len(baseline_results),
        'pass_at_k': compute_pass_at_k(baseline_results, multiagent_results),
        'kg_valid': compute_kg_valid_at_k(baseline_results, multiagent_results),
        'recovery': compute_recovery_rate(multiagent_results),
        'iterations': compute_iteration_statistics(multiagent_results),
        'cost': compute_cost_metrics(baseline_results, multiagent_results),
        'stratified': {
            'by_complexity': compute_stratified_metrics(baseline_results, multiagent_results, 'complexity'),
            'by_reasoning': compute_stratified_metrics(baseline_results, multiagent_results, 'reasoning_level'),
            'by_sublevel': compute_stratified_metrics(baseline_results, multiagent_results, 'sublevel')
        }
    }

    logger.info("Metrics computation complete")

    return metrics


if __name__ == "__main__":
    # Test metrics computation
    print("Testing evaluation metrics...")

    # Mock data
    baseline_mock = [
        {'pass_at_1': True, 'execution_success': True, 'total_tokens': 250, 'elapsed_time': 2.5, 'complexity': 'Easy'},
        {'pass_at_1': False, 'execution_success': False, 'total_tokens': 260, 'elapsed_time': 2.8, 'complexity': 'Medium'},
        {'pass_at_1': True, 'execution_success': True, 'total_tokens': 240, 'elapsed_time': 2.3, 'complexity': 'Hard'},
    ]

    multiagent_mock = [
        {'pass_at_k': True, 'execution_success': True, 'total_tokens': 300, 'elapsed_time': 3.2, 'total_iterations': 1, 'complexity': 'Easy'},
        {'pass_at_k': True, 'execution_success': True, 'total_tokens': 800, 'elapsed_time': 8.5, 'total_iterations': 3, 'complexity': 'Medium',
         'all_iterations': [{'evaluation': 'error'}, {'evaluation': 'incorrect'}, {'evaluation': 'accept'}]},
        {'pass_at_k': True, 'execution_success': True, 'total_tokens': 600, 'elapsed_time': 6.1, 'total_iterations': 2, 'complexity': 'Hard',
         'all_iterations': [{'evaluation': 'error'}, {'evaluation': 'accept'}]},
    ]

    # Test Pass@k
    print("\nPass@k:")
    pass_metrics = compute_pass_at_k(baseline_mock, multiagent_mock)
    for key, value in pass_metrics.items():
        print(f"  {key}: {value}")

    # Test KG Valid
    print("\nKG Valid:")
    kg_metrics = compute_kg_valid_at_k(baseline_mock, multiagent_mock)
    for key, value in kg_metrics.items():
        print(f"  {key}: {value}")

    # Test Recovery Rate
    print("\nRecovery Rate:")
    recovery_metrics = compute_recovery_rate(multiagent_mock)
    for key, value in recovery_metrics.items():
        print(f"  {key}: {value}")

    # Test Iteration Statistics
    print("\nIteration Statistics:")
    iter_metrics = compute_iteration_statistics(multiagent_mock)
    for key, value in iter_metrics.items():
        print(f"  {key}: {value}")

    # Test Cost Metrics
    print("\nCost Metrics:")
    cost_metrics = compute_cost_metrics(baseline_mock, multiagent_mock)
    print(f"  Tokens: {cost_metrics['tokens']}")
    print(f"  Latency: {cost_metrics['latency']}")

    # Test All Metrics
    print("\n" + "="*60)
    print("All Metrics:")
    all_metrics = compute_all_metrics(baseline_mock, multiagent_mock)
    print(f"Dataset size: {all_metrics['dataset_size']}")
    print(f"Pass@k improvement: {all_metrics['pass_at_k']['improvement']:.1%}")
    print(f"Recovery rate: {all_metrics['recovery']['recovery_rate']:.1%}")
    print(f"Avg iterations: {all_metrics['iterations']['avg_iterations']:.2f}")
    print(f"Token overhead: {all_metrics['cost']['tokens']['token_overhead_pct']:.1f}%")

    print("\nEvaluation metrics test complete!")
