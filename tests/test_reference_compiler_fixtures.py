"""
Hand-verified fixture tests for the reference compiler.

Each test uses a TINY DataFrame where the correct answer can be computed
by hand or with a simple one-liner, then asserts the compiler matches.

This is the independent second implementation required by Section 8.6 and 16.
"""

from __future__ import annotations

import pandas as pd
import pytest

from morphda.reference.compiler import compile_task, run_reference
from morphda.tasks.schema import (
    AggregationSpec,
    ComparisonSpec,
    DateScope,
    FilterSpec,
    MetricSpec,
    PostFilterSpec,
    RankingSpec,
    TaskSpec,
)


def _run(task: TaskSpec, tables: dict) -> object:
    return run_reference(task, tables)


# ─── Level 1: scalar aggregations ────────────────────────────────────────────

def test_scalar_sum():
    """Sum of revenue: 10+20+30 = 60."""
    tables = {"t": pd.DataFrame({"revenue": [10.0, 20.0, 30.0]})}
    task = TaskSpec(
        task_id="fix_l1_sum", scenario_id="test",
        question_family="scalar_sum", difficulty_level=1,
        inputs=["t"],
        metric=MetricSpec(name="total", operation="sum", column="revenue"),
        output_type="scalar",
    )
    assert abs(_run(task, tables) - 60.0) < 1e-9


def test_scalar_mean():
    """Mean of [10, 20, 30] = 20."""
    tables = {"t": pd.DataFrame({"revenue": [10.0, 20.0, 30.0]})}
    task = TaskSpec(
        task_id="fix_l1_mean", scenario_id="test",
        question_family="scalar_mean", difficulty_level=1,
        inputs=["t"],
        metric=MetricSpec(name="avg", operation="mean", column="revenue"),
        output_type="scalar",
    )
    assert abs(_run(task, tables) - 20.0) < 1e-9


def test_scalar_median_odd():
    """Median of [10, 30, 20] = 20 (middle when sorted)."""
    tables = {"t": pd.DataFrame({"revenue": [10.0, 30.0, 20.0]})}
    task = TaskSpec(
        task_id="fix_l1_median", scenario_id="test",
        question_family="scalar_median", difficulty_level=1,
        inputs=["t"],
        metric=MetricSpec(name="med", operation="median", column="revenue"),
        output_type="scalar",
    )
    assert abs(_run(task, tables) - 20.0) < 1e-9


def test_scalar_count_distinct():
    """Distinct customer_ids: {1, 2, 3} → 3."""
    tables = {"t": pd.DataFrame({"customer_id": [1, 2, 2, 3, 1]})}
    task = TaskSpec(
        task_id="fix_l1_nunique", scenario_id="test",
        question_family="scalar_count_distinct", difficulty_level=1,
        inputs=["t"],
        metric=MetricSpec(name="dc", operation="count_distinct", column="customer_id"),
        output_type="scalar",
    )
    assert _run(task, tables) == 3


def test_scalar_filtered_sum():
    """Sum revenue where status != 'cancelled': 10+30 = 40."""
    tables = {
        "t": pd.DataFrame({
            "revenue": [10.0, 20.0, 30.0],
            "status": ["ok", "cancelled", "ok"],
        })
    }
    task = TaskSpec(
        task_id="fix_l1_fsum", scenario_id="test",
        question_family="filtered_agg", difficulty_level=1,
        inputs=["t"],
        filters=[FilterSpec(column="status", operator="not_equal", value="cancelled")],
        metric=MetricSpec(name="total", operation="sum", column="revenue"),
        output_type="scalar",
    )
    assert abs(_run(task, tables) - 40.0) < 1e-9


# ─── Level 2: grouped ranking ─────────────────────────────────────────────────

def test_grouped_sum_rank_winner():
    """
    category A: 10+20 = 30
    category B: 5+5   = 10
    Winner by descending sum: A
    """
    tables = {
        "t": pd.DataFrame({
            "category": ["A", "A", "B", "B"],
            "revenue":  [10.0, 20.0, 5.0, 5.0],
        })
    }
    task = TaskSpec(
        task_id="fix_l2_rank", scenario_id="test",
        question_family="grouped_rank", difficulty_level=2,
        inputs=["t"],
        metric=MetricSpec(name="total", operation="sum", column="revenue"),
        group_by=["category"],
        ranking=RankingSpec(direction="descending", k=1),
        output_type="label",
    )
    assert _run(task, tables) == "A"


def test_grouped_sum_rank_ascending():
    """Winner by ascending sum (smallest): B."""
    tables = {
        "t": pd.DataFrame({
            "category": ["A", "A", "B", "B"],
            "revenue":  [10.0, 20.0, 5.0, 5.0],
        })
    }
    task = TaskSpec(
        task_id="fix_l2_rank_asc", scenario_id="test",
        question_family="grouped_rank_asc", difficulty_level=2,
        inputs=["t"],
        metric=MetricSpec(name="total", operation="sum", column="revenue"),
        group_by=["category"],
        ranking=RankingSpec(direction="ascending", k=1),
        output_type="label",
    )
    assert _run(task, tables) == "B"


def test_grouped_top_k():
    """Top-2 categories by sum: ['A', 'B'] in that order (C is worst)."""
    tables = {
        "t": pd.DataFrame({
            "category": ["A", "B", "C"],
            "revenue":  [30.0, 20.0, 5.0],
        })
    }
    task = TaskSpec(
        task_id="fix_l2_topk", scenario_id="test",
        question_family="grouped_rank", difficulty_level=2,
        inputs=["t"],
        metric=MetricSpec(name="total", operation="sum", column="revenue"),
        group_by=["category"],
        ranking=RankingSpec(direction="descending", k=2),
        output_type="ranked_list",
    )
    result = _run(task, tables)
    assert list(result)[:2] == ["A", "B"]


def test_date_filtered_grouped_rank():
    """
    Only 2025 rows eligible. A=100, B=50. Winner: A.
    2024 row for B (value=9999) must be excluded.
    """
    tables = {
        "t": pd.DataFrame({
            "category":  ["A",          "B",          "B"],
            "revenue":   [100.0,         50.0,         9999.0],
            "event_date":["2025-03-15", "2025-06-01", "2024-11-01"],
        })
    }
    task = TaskSpec(
        task_id="fix_l2_date_rank", scenario_id="test",
        question_family="date_filtered_grouped_rank", difficulty_level=2,
        inputs=["t"],
        date=DateScope(
            column="event_date",
            current_start="2025-01-01", current_end="2025-12-31",
            inclusive_bounds=True,
        ),
        metric=MetricSpec(name="total", operation="sum", column="revenue"),
        group_by=["category"],
        ranking=RankingSpec(direction="descending", k=1),
        output_type="label",
    )
    assert _run(task, tables) == "A"


# ─── Level 3: grouped ratio ───────────────────────────────────────────────────

def test_grouped_ratio_rank():
    """
    session_id, converting (1=yes/0=no):
      A: sessions [1,2,3], converting=[1,2,None]  → rate = 2/3 ≈ 0.667
      B: sessions [4,5,6], converting=[4,None,None] → rate = 1/3 ≈ 0.333
    Winner by conversion rate (descending): A
    (using count_distinct on session_id as denominator)
    """
    tables = {
        "t": pd.DataFrame({
            "category":    ["A", "A", "A", "B", "B", "B"],
            "session_id":  [1,   2,   3,   4,   5,   6],
            "converting":  [1,   2,   None, 4,  None, None],
        })
    }
    task = TaskSpec(
        task_id="fix_l3_ratio", scenario_id="test",
        question_family="grouped_ratio_rank", difficulty_level=3,
        inputs=["t"],
        metric=MetricSpec(
            name="conversion_rate",
            operation="ratio",
            numerator=AggregationSpec(operation="count_distinct", column="converting"),
            denominator=AggregationSpec(operation="count_distinct", column="session_id"),
        ),
        group_by=["category"],
        ranking=RankingSpec(direction="descending", k=1),
        output_type="label",
    )
    assert _run(task, tables) == "A"


def test_grouped_ratio_minimum_denominator_filter():
    """
    A: 5 sessions, 4 converting → rate = 0.8, eligible (>= 3)
    B: 2 sessions, 2 converting → rate = 1.0, but INELIGIBLE (< 3)
    C: 4 sessions, 1 converting → rate = 0.25, eligible
    Winner must be A (B excluded despite higher rate).
    """
    rows = []
    for i in range(5):
        rows.append({"cat": "A", "sid": 10+i, "conv": 10+i if i < 4 else None})
    for i in range(2):
        rows.append({"cat": "B", "sid": 20+i, "conv": 20+i})
    for i in range(4):
        rows.append({"cat": "C", "sid": 30+i, "conv": 30 if i == 0 else None})

    tables = {"t": pd.DataFrame(rows)}
    task = TaskSpec(
        task_id="fix_l3_ratio_thresh", scenario_id="test",
        question_family="grouped_ratio_rank", difficulty_level=3,
        inputs=["t"],
        metric=MetricSpec(
            name="rate",
            operation="ratio",
            numerator=AggregationSpec(operation="count_distinct", column="conv"),
            denominator=AggregationSpec(operation="count_distinct", column="sid"),
        ),
        group_by=["cat"],
        post_filter=PostFilterSpec(minimum_denominator=3),
        ranking=RankingSpec(direction="descending", k=1),
        output_type="label",
    )
    result = _run(task, tables)
    assert result == "A", f"Expected 'A' (B excluded by threshold), got {result!r}"


# ─── Level 4: period comparison ───────────────────────────────────────────────

def test_period_comparison_simple_sum():
    """
    A: current=100, prior=50  → pct_change = (100-50)/50 = +100%
    B: current=30,  prior=20  → pct_change = (30-20)/20  = +50%
    Winner by largest pct increase: A
    """
    tables = {
        "t": pd.DataFrame({
            "category":  ["A", "A", "B", "B"],
            "revenue":   [100.0, 50.0, 30.0, 20.0],
            "dt":        ["2025-06-01", "2024-06-01", "2025-06-01", "2024-06-01"],
        })
    }
    task = TaskSpec(
        task_id="fix_l4_pct", scenario_id="test",
        question_family="period_comparison_rank", difficulty_level=4,
        inputs=["t"],
        date=DateScope(
            column="dt",
            current_start="2025-01-01", current_end="2025-12-31",
            previous_start="2024-01-01", previous_end="2024-12-31",
            inclusive_bounds=True,
        ),
        metric=MetricSpec(name="revenue", operation="sum", column="revenue"),
        comparison=ComparisonSpec(operation="percentage_change"),
        group_by=["category"],
        ranking=RankingSpec(direction="descending", k=1),
        output_type="label",
    )
    assert _run(task, tables) == "A"


def test_period_comparison_handles_missing_prior():
    """
    A: has both periods → included
    B: only current period (no prior) → excluded (NaN after reindex)
    Winner: A
    """
    tables = {
        "t": pd.DataFrame({
            "category": ["A", "A", "B"],
            "revenue":  [100.0, 40.0, 90.0],
            "dt":       ["2025-06-01", "2024-06-01", "2025-06-01"],
        })
    }
    task = TaskSpec(
        task_id="fix_l4_missing_prior", scenario_id="test",
        question_family="period_comparison_rank", difficulty_level=4,
        inputs=["t"],
        date=DateScope(
            column="dt",
            current_start="2025-01-01", current_end="2025-12-31",
            previous_start="2024-01-01", previous_end="2024-12-31",
            inclusive_bounds=True,
        ),
        metric=MetricSpec(name="revenue", operation="sum", column="revenue"),
        comparison=ComparisonSpec(operation="percentage_change"),
        group_by=["category"],
        ranking=RankingSpec(direction="descending", k=1),
        output_type="label",
    )
    result = _run(task, tables)
    assert result == "A", f"B (no prior period) must be excluded; got {result!r}"


# ─── Compiler output sanity ───────────────────────────────────────────────────

def test_compiled_source_has_analyze_function():
    task = TaskSpec(
        task_id="fix_src", scenario_id="test",
        question_family="scalar_sum", difficulty_level=1,
        inputs=["t"],
        metric=MetricSpec(name="total", operation="sum", column="revenue"),
        output_type="scalar",
    )
    src = compile_task(task)
    assert "def analyze(tables" in src
    assert "return result" in src
    assert "import pandas as pd" in src
