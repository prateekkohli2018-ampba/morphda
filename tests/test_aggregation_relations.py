"""
Tests for aggregation metamorphic relations (MR-A1 through MR-A6).
"""

import pandas as pd
import numpy as np
import pytest

from morphda.execution.sandbox import execute_program
from morphda.relations.aggregation import (
    FullRowDuplicationAlgebra,
    SingleValuePerturbation,
    MeanVsMedianOutlierTest,
    CountVsDistinctCountTest,
    GlobalAdditiveTranslation,
)
from morphda.tasks.schema import MetricSpec, TaskSpec


def _make_tables(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "sales": pd.DataFrame({
            "sale_id":  range(n),
            "category": rng.choice(["A", "B", "C"], size=n),
            "revenue":  rng.uniform(10, 500, size=n),
        })
    }


def _task(operation: str, output_type: str = "scalar") -> TaskSpec:
    return TaskSpec(
        task_id=f"test_agg_{operation}",
        scenario_id="test",
        question_family=f"scalar_{operation}",
        difficulty_level=1,
        inputs=["sales"],
        metric=MetricSpec(name="metric", operation=operation, column="revenue"),
        output_type=output_type,
    )


def _run(program, tables):
    r = execute_program(program, tables)
    assert r.success, f"Execution failed: {r.exception}"
    return r.output


SUM_PROGRAM  = "def analyze(t): return t['sales']['revenue'].sum()"
MEAN_PROGRAM = "def analyze(t): return t['sales']['revenue'].mean()"
MEDIAN_PROGRAM = "def analyze(t): return t['sales']['revenue'].median()"
COUNT_PROGRAM  = "def analyze(t): return len(t['sales'])"
NUNIQUE_PROGRAM = "def analyze(t): return t['sales']['sale_id'].nunique()"


# ─── MR-A1 ───────────────────────────────────────────────────────────────────

class TestFullRowDuplicationAlgebra:
    r = FullRowDuplicationAlgebra()

    def test_sum_doubles_after_duplication(self):
        task = _task("sum")
        tables = _make_tables()
        cases = self.r.generate_cases(tables, task)
        src = _run(SUM_PROGRAM, tables)
        fu = _run(SUM_PROGRAM, cases[0].tables)
        assert abs(fu - 2 * src) < 1e-6

    def test_mean_unchanged_after_duplication(self):
        task = _task("mean")
        tables = _make_tables()
        cases = self.r.generate_cases(tables, task)
        src = _run(MEAN_PROGRAM, tables)
        fu = _run(MEAN_PROGRAM, cases[0].tables)
        passed, _ = self.r.check(src, fu, cases[0], task)
        assert passed

    def test_sum_used_instead_of_mean_detected(self):
        task = _task("mean")  # expects mean
        tables = _make_tables()
        cases = self.r.generate_cases(tables, task)
        src = _run(SUM_PROGRAM, tables)   # but program uses sum
        fu = _run(SUM_PROGRAM, cases[0].tables)
        passed, witness = self.r.check(src, fu, cases[0], task)
        assert not passed
        assert witness is not None
        assert "sum" in witness.likely_issue


# ─── MR-A2 ───────────────────────────────────────────────────────────────────

class TestSingleValuePerturbation:
    r = SingleValuePerturbation()

    def test_sum_increases_by_delta(self):
        task = _task("sum")
        tables = _make_tables()
        cases = self.r.generate_cases(tables, task)
        assert len(cases) == 1
        case = cases[0]
        delta = case.expected_delta
        src = _run(SUM_PROGRAM, tables)
        fu = _run(SUM_PROGRAM, case.tables)
        assert abs(fu - (src + delta)) < 1.0  # allow small float error

    def test_count_unchanged_after_value_perturbation(self):
        task = _task("count")
        tables = _make_tables()
        cases = self.r.generate_cases(tables, task)
        src = _run(COUNT_PROGRAM, tables)
        fu = _run(COUNT_PROGRAM, cases[0].tables)
        passed, _ = self.r.check(src, fu, cases[0], task)
        assert passed

    def test_sum_used_as_count_detected(self):
        task = _task("count")
        tables = _make_tables()
        cases = self.r.generate_cases(tables, task)
        src = _run(SUM_PROGRAM, tables)   # sum used instead of count
        fu = _run(SUM_PROGRAM, cases[0].tables)
        passed, witness = self.r.check(src, fu, cases[0], task)
        assert not passed
        assert witness is not None


# ─── MR-A5 ───────────────────────────────────────────────────────────────────

class TestMeanVsMedianOutlierTest:
    r = MeanVsMedianOutlierTest()

    def test_median_unchanged_by_outlier_perturbation(self):
        task = _task("median")
        tables = _make_tables(n=101)
        cases = self.r.generate_cases(tables, task)
        assert len(cases) == 1
        src = _run(MEDIAN_PROGRAM, tables)
        fu = _run(MEDIAN_PROGRAM, cases[0].tables)
        passed, _ = self.r.check(src, fu, cases[0], task)
        assert passed

    def test_mean_used_instead_of_median_detected(self):
        """When task expects median but program uses mean, the outlier perturbation causes a change."""
        task = _task("median")
        tables = _make_tables(n=101)
        cases = self.r.generate_cases(tables, task)
        src = _run(MEAN_PROGRAM, tables)
        fu = _run(MEAN_PROGRAM, cases[0].tables)
        passed, witness = self.r.check(src, fu, cases[0], task)
        # Mean CHANGES after outlier perturbation → should be detected as violation for median task
        assert not passed
        assert witness is not None
        assert "mean" in witness.likely_issue


# ─── MR-A6 ───────────────────────────────────────────────────────────────────

class TestCountVsDistinctCountTest:
    r = CountVsDistinctCountTest()

    def test_distinct_count_unchanged_after_duplication(self):
        task = _task("count_distinct")
        tables = _make_tables()
        cases = self.r.generate_cases(tables, task)
        src = _run(NUNIQUE_PROGRAM, tables)
        fu = _run(NUNIQUE_PROGRAM, cases[0].tables)
        passed, _ = self.r.check(src, fu, cases[0], task)
        assert passed

    def test_count_used_instead_of_distinct_detected(self):
        task = _task("count_distinct")
        tables = _make_tables()
        cases = self.r.generate_cases(tables, task)
        src = _run(COUNT_PROGRAM, tables)   # count instead of nunique
        fu = _run(COUNT_PROGRAM, cases[0].tables)
        passed, witness = self.r.check(src, fu, cases[0], task)
        # After duplication, count increases but distinct doesn't → mismatch
        assert not passed
        assert witness is not None
        assert "distinct" in witness.likely_issue
