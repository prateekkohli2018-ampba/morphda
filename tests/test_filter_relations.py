"""
Tests for filter/scope metamorphic relations (MR-F1 through MR-F4).

Tests verify behavior:
  - Correct reference programs must pass all applicable relations.
  - Programs with missing filters must be detected.
"""

import pandas as pd
import numpy as np
import pytest

from morphda.data.generators import generate_scenario
from morphda.execution.sandbox import execute_program
from morphda.relations.filters import (
    OutOfScopeExtremeRowInvariance,
    InScopeSentinelSensitivity,
    BoundaryQuartet,
    ConjunctIsolationTest,
)
from morphda.tasks.schema import FilterSpec, DateScope, MetricSpec, RankingSpec, TaskSpec


def _make_task_with_date() -> TaskSpec:
    return TaskSpec(
        task_id="test_f001",
        scenario_id="retail01",
        question_family="date_filtered_rank",
        difficulty_level=3,
        inputs=["orders"],
        filters=[FilterSpec(column="order_status", operator="not_equal", value="cancelled")],
        date=DateScope(
            column="order_date",
            current_start="2025-06-01",
            current_end="2025-09-30",
            inclusive_bounds=True,
        ),
        metric=MetricSpec(name="total_revenue", operation="sum", column="revenue"),
        group_by=["category"],
        ranking=RankingSpec(direction="descending", k=1),
        output_type="label",
    )


# Reference program: correct date + status filter + group rank
CORRECT_PROGRAM = """
import pandas as pd

def analyze(tables):
    df = tables["orders"].copy()
    df = df[df["order_status"] != "cancelled"]
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")
    df = df[(df["order_date"] >= pd.Timestamp("2025-06-01")) &
            (df["order_date"] <= pd.Timestamp("2025-09-30"))]
    grouped = df.groupby("category", observed=True)["revenue"].sum()
    grouped = grouped.sort_values(ascending=False)
    return grouped.index[0]
"""

# Buggy: missing date filter
MISSING_DATE_FILTER = """
import pandas as pd

def analyze(tables):
    df = tables["orders"].copy()
    df = df[df["order_status"] != "cancelled"]
    grouped = df.groupby("category", observed=True)["revenue"].sum()
    return grouped.idxmax()
"""

# Buggy: missing status filter
MISSING_STATUS_FILTER = """
import pandas as pd

def analyze(tables):
    df = tables["orders"].copy()
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed")
    df = df[(df["order_date"] >= pd.Timestamp("2025-06-01")) &
            (df["order_date"] <= pd.Timestamp("2025-09-30"))]
    grouped = df.groupby("category", observed=True)["revenue"].sum()
    return grouped.idxmax()
"""


def _get_tables():
    return generate_scenario("retail01", seed=42)


def _run(program, tables):
    r = execute_program(program, tables)
    assert r.success, f"Execution failed: {r.exception}"
    return r.output


# ─── MR-F1 ───────────────────────────────────────────────────────────────────

class TestOutOfScopeExtremeRowInvariance:
    relation = OutOfScopeExtremeRowInvariance()

    def test_correct_program_passes(self):
        task = _make_task_with_date()
        tables = _get_tables()
        cases = self.relation.generate_cases(tables, task)
        assert len(cases) > 0
        src = _run(CORRECT_PROGRAM, tables)
        for case in cases:
            fu = _run(CORRECT_PROGRAM, case.tables)
            passed, witness = self.relation.check(src, fu, case, task)
            assert passed, f"Correct program failed MR-F1: {witness}"

    def test_missing_date_filter_detected(self):
        task = _make_task_with_date()
        tables = _get_tables()
        cases = self.relation.generate_cases(tables, task)
        src = _run(MISSING_DATE_FILTER, tables)
        any_violation = False
        for case in cases:
            fu = _run(MISSING_DATE_FILTER, case.tables)
            passed, witness = self.relation.check(src, fu, case, task)
            if not passed:
                any_violation = True
                assert witness is not None
                assert "filter" in witness.likely_issue
        assert any_violation, "MR-F1 must detect missing date filter"

    def test_applicable_when_task_has_filters(self):
        assert self.relation.is_applicable(_make_task_with_date())

    def test_not_applicable_when_no_filters(self):
        task = TaskSpec(
            task_id="t_nofilter",
            scenario_id="retail01",
            question_family="scalar_sum",
            difficulty_level=1,
            inputs=["orders"],
            metric=MetricSpec(name="total", operation="sum", column="revenue"),
            output_type="scalar",
        )
        assert not self.relation.is_applicable(task)


# ─── MR-F2 ───────────────────────────────────────────────────────────────────

class TestInScopeSentinelSensitivity:
    relation = InScopeSentinelSensitivity()

    def test_correct_program_detects_sentinel_winner(self):
        task = _make_task_with_date()
        tables = _get_tables()
        cases = self.relation.generate_cases(tables, task)
        assert len(cases) > 0
        src = _run(CORRECT_PROGRAM, tables)
        case = cases[0]
        fu = _run(CORRECT_PROGRAM, case.tables)
        passed, _ = self.relation.check(src, fu, case, task)
        assert passed, "Correct program must return sentinel as winner when dominant rows added"

    def test_applicable_for_grouped_label_tasks(self):
        assert self.relation.is_applicable(_make_task_with_date())


# ─── MR-F3 ───────────────────────────────────────────────────────────────────

class TestBoundaryQuartet:
    relation = BoundaryQuartet()

    def test_correct_program_passes_out_of_scope_boundaries(self):
        task = _make_task_with_date()
        tables = _get_tables()
        cases = self.relation.generate_cases(tables, task)
        src = _run(CORRECT_PROGRAM, tables)
        for case in cases:
            if case.scope_status != "out_of_scope":
                continue
            fu = _run(CORRECT_PROGRAM, case.tables)
            passed, witness = self.relation.check(src, fu, case, task)
            assert passed, f"Correct program failed boundary check {case.case_id}: {witness}"

    def test_generates_four_boundary_cases(self):
        task = _make_task_with_date()
        tables = _get_tables()
        cases = self.relation.generate_cases(tables, task)
        assert len(cases) == 4

    def test_applicable_when_date_scope_present(self):
        assert self.relation.is_applicable(_make_task_with_date())

    def test_not_applicable_without_date(self):
        task = TaskSpec(
            task_id="t_nodate",
            scenario_id="retail01",
            question_family="scalar_sum",
            difficulty_level=1,
            inputs=["orders"],
            metric=MetricSpec(name="total", operation="sum", column="revenue"),
            output_type="scalar",
        )
        assert not self.relation.is_applicable(task)


# ─── MR-F4 ───────────────────────────────────────────────────────────────────

class TestConjunctIsolationTest:
    relation = ConjunctIsolationTest()

    def _two_filter_task(self) -> TaskSpec:
        return TaskSpec(
            task_id="test_f004",
            scenario_id="retail01",
            question_family="multi_filter_rank",
            difficulty_level=3,
            inputs=["orders"],
            filters=[
                FilterSpec(column="order_status", operator="not_equal", value="cancelled"),
                FilterSpec(column="customer_type", operator="equal", value="new"),
            ],
            metric=MetricSpec(name="total_revenue", operation="sum", column="revenue"),
            group_by=["category"],
            ranking=RankingSpec(direction="descending", k=1),
            output_type="label",
        )

    def test_generates_three_cases(self):
        task = self._two_filter_task()
        tables = _get_tables()
        cases = self.relation.generate_cases(tables, task)
        assert len(cases) == 3
        labels = {c.case_id.split("_")[-2] for c in cases}
        # should have A_only, B_only, A_and_B probes
        case_types = [c.scope_status for c in cases]
        assert case_types.count("out_of_scope") == 2
        assert case_types.count("in_scope") == 1

    def test_correct_program_passes_out_of_scope_probes(self):
        task = self._two_filter_task()
        tables = _get_tables()
        CORRECT_MULTI = """
import pandas as pd

def analyze(tables):
    df = tables["orders"].copy()
    df = df[(df["order_status"] != "cancelled") & (df["customer_type"] == "new")]
    grouped = df.groupby("category", observed=True)["revenue"].sum()
    return grouped.idxmax()
"""
        cases = self.relation.generate_cases(tables, task)
        src = _run(CORRECT_MULTI, tables)
        for case in cases:
            if case.scope_status != "out_of_scope":
                continue
            fu = _run(CORRECT_MULTI, case.tables)
            passed, witness = self.relation.check(src, fu, case, task)
            assert passed, f"Correct program failed conjunct probe {case.case_id}: {witness}"

    def test_applicable_with_two_or_more_filters(self):
        assert self.relation.is_applicable(self._two_filter_task())

    def test_not_applicable_with_one_filter(self):
        task = TaskSpec(
            task_id="t_onefilter",
            scenario_id="retail01",
            question_family="filtered_sum",
            difficulty_level=1,
            inputs=["orders"],
            filters=[FilterSpec(column="order_status", operator="not_equal", value="cancelled")],
            metric=MetricSpec(name="total", operation="sum", column="revenue"),
            output_type="scalar",
        )
        assert not self.relation.is_applicable(task)
