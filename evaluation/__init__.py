"""
Evaluation Module - Metrics Computation

This package contains evaluation metrics for comparing
baseline vs multi-agent system performance.
"""

from .metrics import (
    compare_outputs,
    compute_pass_at_k,
    compute_kg_valid_at_k,
    compute_recovery_rate,
    compute_iteration_statistics,
    compute_cost_metrics,
    compute_stratified_metrics,
    compute_all_metrics
)

__all__ = [
    "compare_outputs",
    "compute_pass_at_k",
    "compute_kg_valid_at_k",
    "compute_recovery_rate",
    "compute_iteration_statistics",
    "compute_cost_metrics",
    "compute_stratified_metrics",
    "compute_all_metrics",
]
