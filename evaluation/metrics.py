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


def compute_reasoning_quality(multiagent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute ReAct reasoning quality metrics.

    Args:
        multiagent_results: List of multi-agent result dicts with reasoning traces

    Returns:
        Dictionary with reasoning quality statistics
    """
    total_questions = len(multiagent_results)
    questions_with_reasoning = 0
    total_reasoning_steps = 0
    quality_scores = []

    for result in multiagent_results:
        if result.get('reasoning_enabled', False):
            questions_with_reasoning += 1
            total_reasoning_steps += result.get('total_reasoning_steps', 0)

            avg_quality = result.get('avg_reasoning_quality', 0.0)
            if avg_quality > 0:
                quality_scores.append(avg_quality)

    return {
        'questions_with_reasoning': questions_with_reasoning,
        'reasoning_coverage': questions_with_reasoning / total_questions if total_questions > 0 else 0.0,
        'total_reasoning_steps': total_reasoning_steps,
        'avg_reasoning_steps_per_question': total_reasoning_steps / questions_with_reasoning if questions_with_reasoning > 0 else 0.0,
        'avg_reasoning_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        'min_quality': min(quality_scores) if quality_scores else 0.0,
        'max_quality': max(quality_scores) if quality_scores else 0.0
    }


def compute_reasoning_statistics(multiagent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute detailed ReAct reasoning statistics per iteration.

    Args:
        multiagent_results: List of multi-agent result dicts with iteration history

    Returns:
        Dictionary with detailed reasoning statistics
    """
    total_iterations = 0
    iterations_with_reasoning = 0
    reasoning_steps_by_iteration = defaultdict(list)
    reasoning_quality_by_iteration = defaultdict(list)

    # Agent-specific reasoning stats
    agent_reasoning_steps = defaultdict(int)
    agent_reasoning_count = defaultdict(int)

    for result in multiagent_results:
        iterations = result.get('iterations', [])

        for iteration in iterations:
            total_iterations += 1
            iteration_has_reasoning = False
            iteration_steps = 0

            # Check each agent's reasoning in this iteration
            for agent_name, reasoning_field in [
                ('generator', iteration.get('generator_reasoning')),
                ('evaluator', iteration.get('evaluator_reasoning')),
                ('instructions', iteration.get('instructions_reasoning')),
                ('aggregator', iteration.get('aggregator_reasoning'))
            ]:
                if reasoning_field and isinstance(reasoning_field, dict):
                    iteration_has_reasoning = True
                    steps = reasoning_field.get('num_steps', 0)
                    quality = reasoning_field.get('quality_score', 0.0)

                    iteration_steps += steps
                    agent_reasoning_steps[agent_name] += steps
                    agent_reasoning_count[agent_name] += 1

                    if quality > 0:
                        reasoning_quality_by_iteration[iteration.get('iteration_number', 0)].append(quality)

            if iteration_has_reasoning:
                iterations_with_reasoning += 1
                reasoning_steps_by_iteration[iteration.get('iteration_number', 0)].append(iteration_steps)

    # Compute average reasoning steps by iteration number
    avg_steps_by_iteration = {}
    for iter_num, steps_list in reasoning_steps_by_iteration.items():
        avg_steps_by_iteration[iter_num] = sum(steps_list) / len(steps_list) if steps_list else 0.0

    # Compute average quality by iteration number
    avg_quality_by_iteration = {}
    for iter_num, quality_list in reasoning_quality_by_iteration.items():
        avg_quality_by_iteration[iter_num] = sum(quality_list) / len(quality_list) if quality_list else 0.0

    # Compute agent-specific averages
    agent_avg_steps = {}
    for agent, total_steps in agent_reasoning_steps.items():
        count = agent_reasoning_count[agent]
        agent_avg_steps[agent] = total_steps / count if count > 0 else 0.0

    return {
        'total_iterations': total_iterations,
        'iterations_with_reasoning': iterations_with_reasoning,
        'reasoning_coverage_iterations': iterations_with_reasoning / total_iterations if total_iterations > 0 else 0.0,
        'avg_steps_by_iteration': avg_steps_by_iteration,
        'avg_quality_by_iteration': avg_quality_by_iteration,
        'agent_avg_reasoning_steps': agent_avg_steps,
        'agent_reasoning_usage': {agent: count for agent, count in agent_reasoning_count.items()}
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
    Compute all evaluation metrics including ReAct reasoning metrics.

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
        'reasoning_quality': compute_reasoning_quality(multiagent_results),
        'reasoning_statistics': compute_reasoning_statistics(multiagent_results),
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
        {
            'pass_at_k': True, 'execution_success': True, 'total_tokens': 300, 'elapsed_time': 3.2,
            'total_iterations': 1, 'complexity': 'Easy', 'reasoning_enabled': True,
            'total_reasoning_steps': 5, 'avg_reasoning_quality': 0.85,
            'iterations': [
                {
                    'iteration_number': 1, 'evaluation': 'accept',
                    'generator_reasoning': {'num_steps': 3, 'quality_score': 0.9},
                    'evaluator_reasoning': {'num_steps': 2, 'quality_score': 0.8}
                }
            ]
        },
        {
            'pass_at_k': True, 'execution_success': True, 'total_tokens': 800, 'elapsed_time': 8.5,
            'total_iterations': 3, 'complexity': 'Medium', 'reasoning_enabled': True,
            'total_reasoning_steps': 18, 'avg_reasoning_quality': 0.75,
            'all_iterations': [{'evaluation': 'error'}, {'evaluation': 'incorrect'}, {'evaluation': 'accept'}],
            'iterations': [
                {
                    'iteration_number': 1, 'evaluation': 'error',
                    'generator_reasoning': {'num_steps': 3, 'quality_score': 0.7},
                    'evaluator_reasoning': {'num_steps': 2, 'quality_score': 0.6}
                },
                {
                    'iteration_number': 2, 'evaluation': 'incorrect',
                    'generator_reasoning': {'num_steps': 3, 'quality_score': 0.8},
                    'evaluator_reasoning': {'num_steps': 2, 'quality_score': 0.7},
                    'instructions_reasoning': {'num_steps': 2, 'quality_score': 0.75},
                    'aggregator_reasoning': {'num_steps': 2, 'quality_score': 0.8}
                },
                {
                    'iteration_number': 3, 'evaluation': 'accept',
                    'generator_reasoning': {'num_steps': 3, 'quality_score': 0.85},
                    'evaluator_reasoning': {'num_steps': 1, 'quality_score': 0.9}
                }
            ]
        },
        {
            'pass_at_k': True, 'execution_success': True, 'total_tokens': 600, 'elapsed_time': 6.1,
            'total_iterations': 2, 'complexity': 'Hard', 'reasoning_enabled': True,
            'total_reasoning_steps': 12, 'avg_reasoning_quality': 0.8,
            'all_iterations': [{'evaluation': 'error'}, {'evaluation': 'accept'}],
            'iterations': [
                {
                    'iteration_number': 1, 'evaluation': 'error',
                    'generator_reasoning': {'num_steps': 3, 'quality_score': 0.75},
                    'evaluator_reasoning': {'num_steps': 2, 'quality_score': 0.7},
                    'instructions_reasoning': {'num_steps': 2, 'quality_score': 0.8}
                },
                {
                    'iteration_number': 2, 'evaluation': 'accept',
                    'generator_reasoning': {'num_steps': 3, 'quality_score': 0.9},
                    'evaluator_reasoning': {'num_steps': 2, 'quality_score': 0.85}
                }
            ]
        },
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

    # Test Reasoning Quality
    print("\nReAct Reasoning Quality:")
    reasoning_quality = compute_reasoning_quality(multiagent_mock)
    for key, value in reasoning_quality.items():
        print(f"  {key}: {value}")

    # Test Reasoning Statistics
    print("\nReAct Reasoning Statistics:")
    reasoning_stats = compute_reasoning_statistics(multiagent_mock)
    print(f"  Total iterations: {reasoning_stats['total_iterations']}")
    print(f"  Iterations with reasoning: {reasoning_stats['iterations_with_reasoning']}")
    print(f"  Reasoning coverage: {reasoning_stats['reasoning_coverage_iterations']:.1%}")
    print(f"  Agent avg steps: {reasoning_stats['agent_avg_reasoning_steps']}")
    print(f"  Avg steps by iteration: {reasoning_stats['avg_steps_by_iteration']}")

    # Test All Metrics
    print("\n" + "="*60)
    print("All Metrics (including ReAct):")
    all_metrics = compute_all_metrics(baseline_mock, multiagent_mock)
    print(f"Dataset size: {all_metrics['dataset_size']}")
    print(f"Pass@k improvement: {all_metrics['pass_at_k']['improvement']:.1%}")
    print(f"Recovery rate: {all_metrics['recovery']['recovery_rate']:.1%}")
    print(f"Avg iterations: {all_metrics['iterations']['avg_iterations']:.2f}")
    print(f"Token overhead: {all_metrics['cost']['tokens']['token_overhead_pct']:.1f}%")
    print(f"\nReAct Reasoning:")
    print(f"  Coverage: {all_metrics['reasoning_quality']['reasoning_coverage']:.1%}")
    print(f"  Avg quality: {all_metrics['reasoning_quality']['avg_reasoning_quality']:.2f}")
    print(f"  Avg steps/question: {all_metrics['reasoning_quality']['avg_reasoning_steps_per_question']:.1f}")

    print("\n" + "="*60)
    print("Evaluation metrics test complete!")
