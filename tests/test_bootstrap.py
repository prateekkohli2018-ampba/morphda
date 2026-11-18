"""Tests for task-clustered bootstrap statistics."""

import numpy as np
import pytest

from morphda.evaluation.bootstrap import task_clustered_bootstrap, paired_bootstrap_test


def test_bootstrap_ci_contains_true_mean():
    # 10 tasks (more clusters → tighter CI), each with 50 Bernoulli(0.7) observations
    rng = np.random.default_rng(0)
    task_ids = []
    values = []
    for t in range(10):
        n = 50
        task_ids.extend([f"task_{t}"] * n)
        values.extend(rng.binomial(1, 0.7, n).tolist())

    point, lo, hi = task_clustered_bootstrap(task_ids, values, n_iterations=2000, rng_seed=42)
    # Point estimate is within CI
    assert lo <= point <= hi
    # CI is informative (not degenerate)
    assert hi > lo
    # True mean is 0.7; with 500 observations the estimate should be close
    assert abs(point - 0.7) < 0.1


def test_bootstrap_single_task():
    task_ids = ["t1"] * 50
    values = [1.0] * 25 + [0.0] * 25
    point, lo, hi = task_clustered_bootstrap(task_ids, values, n_iterations=500, rng_seed=0)
    assert abs(point - 0.5) < 0.01


def test_paired_bootstrap_detects_significant_difference():
    rng = np.random.default_rng(1)
    n_tasks = 20
    task_ids = []
    a_vals, b_vals = [], []
    for t in range(n_tasks):
        n = 50
        task_ids.extend([f"task_{t}"] * n)
        a_vals.extend(rng.binomial(1, 0.8, n).tolist())  # method A: 80%
        b_vals.extend(rng.binomial(1, 0.5, n).tolist())  # method B: 50%

    diff, p = paired_bootstrap_test(task_ids, a_vals, b_vals, n_iterations=2000, rng_seed=0)
    assert diff > 0.2    # A substantially better than B
    assert p < 0.05      # significant


def test_paired_bootstrap_no_difference():
    rng = np.random.default_rng(2)
    n_tasks = 10
    task_ids = []
    a_vals, b_vals = [], []
    for t in range(n_tasks):
        n = 50
        task_ids.extend([f"t{t}"] * n)
        obs = rng.binomial(1, 0.6, n).tolist()
        a_vals.extend(obs)
        b_vals.extend(obs)  # identical

    diff, p = paired_bootstrap_test(task_ids, a_vals, b_vals, n_iterations=1000, rng_seed=0)
    assert abs(diff) < 0.01
    assert p > 0.3  # should not be significant
