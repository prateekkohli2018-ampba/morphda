"""
Task-clustered bootstrap for confidence intervals.

From paper Section 18.3:
  1. Sample BASE TASKS with replacement (not individual programs/mutants).
  2. Retain all models, seeds, mutants, and method outcomes for sampled tasks.
  3. Compute the metric or paired difference.
  4. Repeat 10,000 times.
  5. Report 95th-percentile confidence intervals.

This prevents pseudo-replication from many mutants per task inflating
apparent statistical significance.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Sequence


def task_clustered_bootstrap(
    task_ids: Sequence[str],
    values: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_iterations: int = 10_000,
    ci_level: float = 0.95,
    rng_seed: int = 0,
) -> tuple[float, float, float]:
    """
    Task-clustered bootstrap confidence interval.

    Args:
        task_ids: Task ID for each observation (determines cluster membership).
        values: Per-observation metric values (same length as task_ids).
        stat_fn: Aggregate statistic to compute on each bootstrap sample.
        n_iterations: Number of bootstrap iterations.
        ci_level: Confidence level (0.95 = 95%).
        rng_seed: Random seed.

    Returns:
        (point_estimate, lower_ci, upper_ci)
    """
    task_ids_arr = np.array(task_ids)
    values_arr   = np.array(values, dtype=float)
    unique_tasks = np.unique(task_ids_arr)
    n_tasks = len(unique_tasks)

    rng = np.random.default_rng(rng_seed)
    boot_stats = np.empty(n_iterations)

    for i in range(n_iterations):
        sampled_tasks = rng.choice(unique_tasks, size=n_tasks, replace=True)
        boot_vals = np.concatenate([
            values_arr[task_ids_arr == t]
            for t in sampled_tasks
        ])
        boot_stats[i] = stat_fn(boot_vals)

    point = stat_fn(values_arr)
    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return float(point), lower, upper


def paired_bootstrap_test(
    task_ids: Sequence[str],
    method_a: Sequence[float],
    method_b: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_iterations: int = 10_000,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """
    Task-clustered paired bootstrap test for difference stat(A) - stat(B).

    Returns:
        (observed_diff, two_sided_p_value)
    """
    task_ids_arr = np.array(task_ids)
    a_arr = np.array(method_a, dtype=float)
    b_arr = np.array(method_b, dtype=float)
    unique_tasks = np.unique(task_ids_arr)
    n_tasks = len(unique_tasks)

    observed_diff = stat_fn(a_arr) - stat_fn(b_arr)

    rng = np.random.default_rng(rng_seed)
    boot_diffs = np.empty(n_iterations)

    for i in range(n_iterations):
        sampled_tasks = rng.choice(unique_tasks, size=n_tasks, replace=True)
        boot_a = np.concatenate([a_arr[task_ids_arr == t] for t in sampled_tasks])
        boot_b = np.concatenate([b_arr[task_ids_arr == t] for t in sampled_tasks])
        boot_diffs[i] = stat_fn(boot_a) - stat_fn(boot_b)

    # Two-sided p-value: fraction of bootstrap diffs more extreme than observed
    # under the null H0: diff = 0 (shift bootstrap diffs to center at 0)
    centered = boot_diffs - np.mean(boot_diffs)
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed_diff)))

    return float(observed_diff), p_value
