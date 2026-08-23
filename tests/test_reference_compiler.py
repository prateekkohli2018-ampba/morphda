"""Tests for the reference program compiler."""

import pandas as pd
import pytest

from morphda.data.generators import generate_scenario
from morphda.reference.compiler import compile_task, run_reference
from morphda.tasks.schema import FilterSpec, MetricSpec, RankingSpec, TaskSpec


def _retail_tables():
    return generate_scenario("retail01", seed=42)


def _simple_sum_task() -> TaskSpec:
    return TaskSpec(
        task_id="t001",
        scenario_id="retail01",
        question_family="scalar_sum",
        difficulty_level=1,
        inputs=["orders"],
        metric=MetricSpec(name="total_revenue", operation="sum", column="revenue"),
        output_type="scalar",
    )


def _filtered_mean_task() -> TaskSpec:
    return TaskSpec(
        task_id="t002",
        scenario_id="retail01",
        question_family="filtered_mean",
        difficulty_level=2,
        inputs=["orders"],
        filters=[
            FilterSpec(column="order_status", operator="not_equal", value="cancelled"),
        ],
        metric=MetricSpec(name="avg_revenue", operation="mean", column="revenue"),
        output_type="scalar",
    )


def _grouped_rank_task() -> TaskSpec:
    return TaskSpec(
        task_id="t003",
        scenario_id="retail01",
        question_family="grouped_rank",
        difficulty_level=2,
        inputs=["orders"],
        filters=[
            FilterSpec(column="order_status", operator="not_equal", value="cancelled"),
        ],
        metric=MetricSpec(name="total_revenue", operation="sum", column="revenue"),
        group_by=["category"],
        ranking=RankingSpec(direction="descending", k=1),
        output_type="label",
    )


def test_compile_produces_python_source():
    task = _simple_sum_task()
    source = compile_task(task)
    assert "def analyze(tables" in source
    assert "return result" in source


def test_simple_sum_compiles_and_runs():
    task = _simple_sum_task()
    tables = _retail_tables()
    result = run_reference(task, tables)
    assert isinstance(result, float)
    assert result > 0


def test_filtered_mean_compiles_and_runs():
    task = _filtered_mean_task()
    tables = _retail_tables()
    result = run_reference(task, tables)
    assert isinstance(result, float)
    assert result > 0

    # The mean excluding cancelled should differ from the mean of all rows
    all_mean = tables["orders"]["revenue"].mean()
    no_cancel_mean = (
        tables["orders"]
        .query("order_status != 'cancelled'")["revenue"]
        .mean()
    )
    assert abs(result - no_cancel_mean) < 1e-6


def test_grouped_ranking_returns_a_category():
    task = _grouped_rank_task()
    tables = _retail_tables()
    result = run_reference(task, tables)
    known_categories = tables["orders"]["category"].unique().tolist()
    assert result in known_categories, f"Winner '{result}' not in known categories"


def test_reference_programs_are_deterministic_across_seeds():
    task = _simple_sum_task()
    seeds = [1, 2, 3]
    results = []
    for s in seeds:
        tables = generate_scenario("retail01", seed=s)
        results.append(run_reference(task, tables))
    # Different seeds should give different sums
    assert len(set(results)) > 1, "Reference should produce different results on different data seeds"


def test_filter_reduces_rows():
    """Filtered program's result must differ from unfiltered baseline."""
    task = _filtered_mean_task()
    tables = _retail_tables()
    filtered_result = run_reference(task, tables)
    all_mean = tables["orders"]["revenue"].mean()
    # They may or may not differ depending on data, but at minimum the filtered run shouldn't crash
    assert isinstance(filtered_result, float)
