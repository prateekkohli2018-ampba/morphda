"""
Tests for universal metamorphic relations (MR-U1 through MR-U4).

Tests verify BEHAVIOR: that correct programs pass, and that
programs with positional/index/column dependence fail.
"""

import numpy as np
import pandas as pd
import pytest

from morphda.execution.sandbox import execute_program
from morphda.relations.universal import (
    ColumnOrderInvariance,
    IndexRelabelingInvariance,
    IrrelevantColumnAdditionInvariance,
    RowPermutationInvariance,
)
from morphda.tasks.schema import MetricSpec, TaskSpec


def _make_task() -> TaskSpec:
    return TaskSpec(
        task_id="test_task_001",
        scenario_id="test",
        question_family="grouped_sum",
        difficulty_level=1,
        inputs=["orders"],
        metric=MetricSpec(name="total_revenue", operation="sum", column="revenue"),
        group_by=["category"],
        output_type="label",
    )


def _make_tables() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    return {
        "orders": pd.DataFrame({
            "category": rng.choice(["A", "B", "C"], size=100),
            "revenue": rng.uniform(10, 500, size=100),
            "order_id": range(100),
        })
    }


def _make_tables_iloc_trap() -> dict[str, pd.DataFrame]:
    """
    Data crafted so that iloc[0] returns "Z" (a rare label),
    but the actual group winner by sum is "A".
    After row shuffling, P(iloc[0]="Z") ≈ 1%, so violation is reliably detected.
    """
    df = pd.DataFrame({
        "category": ["Z"] + ["A"] * 70 + ["B"] * 29,
        "revenue": [1.0] + [100.0] * 70 + [50.0] * 29,
        "order_id": range(100),
    })
    return {"orders": df}


def _make_tables_column_trap() -> dict[str, pd.DataFrame]:
    """
    Data with 3 clearly distinct columns.
    After column reordering, positional access (`iloc[:, 0]`) will hit the wrong column.
    """
    return {
        "orders": pd.DataFrame({
            "category": ["A"] * 50 + ["B"] * 50,
            "revenue": list(range(1, 101)),
            "order_id": list(range(100, 200)),
        })
    }


# --- Correct program (should pass all relations) ---

CORRECT_PROGRAM = """
import pandas as pd

def analyze(tables):
    df = tables["orders"]
    grouped = df.groupby("category")["revenue"].sum()
    return grouped.idxmax()
"""

# --- Buggy programs ---

ILOC_PROGRAM = """
import pandas as pd

def analyze(tables):
    df = tables["orders"]
    # Bug: uses iloc[0] without sorting — positional dependence
    return df["category"].iloc[0]
"""

INDEX_DEPENDENT_PROGRAM = """
import pandas as pd

def analyze(tables):
    df = tables["orders"]
    # Bug: treats index as a key
    return df.loc[0, "category"]
"""

COLUMN_POSITIONAL_PROGRAM = """
import pandas as pd

def analyze(tables):
    df = tables["orders"]
    # Bug: uses column position instead of name
    grouped = df.groupby(df.iloc[:, 0])[df.columns[1]].sum()
    return grouped.idxmax()
"""


def _run_on_tables(program: str, tables: dict) -> object:
    result = execute_program(program, tables)
    assert result.success, f"Execution failed: {result.exception}"
    return result.output


# --- MR-U1: Row permutation ---

class TestRowPermutationInvariance:
    relation = RowPermutationInvariance()

    def test_correct_program_passes(self):
        task = _make_task()
        tables = _make_tables()
        cases = self.relation.generate_cases(tables, task)
        source_out = _run_on_tables(CORRECT_PROGRAM, tables)
        for case in cases:
            fu_out = _run_on_tables(CORRECT_PROGRAM, case.tables)
            passed, witness = self.relation.check(source_out, fu_out, case, task)
            assert passed, f"Correct program failed MR-U1: {witness}"

    def test_iloc_program_fails(self):
        task = _make_task()
        tables = _make_tables_iloc_trap()  # "Z" at row 0, but "A" is winner
        source_out = _run_on_tables(ILOC_PROGRAM, tables)
        assert source_out == "Z", "Pre-condition: iloc trap should return 'Z'"

        any_violation = False
        # Try multiple seeds to guarantee we find a shuffle where iloc[0] != "Z"
        for seed in range(20):
            cases = self.relation.generate_cases(tables, task, rng_seed=seed)
            for case in cases:
                fu_out = _run_on_tables(ILOC_PROGRAM, case.tables)
                if fu_out != source_out:
                    passed, witness = self.relation.check(source_out, fu_out, case, task)
                    assert not passed
                    assert witness is not None
                    assert "positional" in witness.likely_issue or "sort" in witness.likely_issue
                    any_violation = True
                    break
            if any_violation:
                break
        assert any_violation, "MR-U1 should detect iloc[0] positional dependence across seeds"

    def test_applicable_to_all_tasks(self):
        assert self.relation.is_applicable(_make_task())


# --- MR-U2: Index relabeling ---

class TestIndexRelabelingInvariance:
    relation = IndexRelabelingInvariance()

    def test_correct_program_passes(self):
        task = _make_task()
        tables = _make_tables()
        cases = self.relation.generate_cases(tables, task)
        source_out = _run_on_tables(CORRECT_PROGRAM, tables)
        for case in cases:
            fu_out = _run_on_tables(CORRECT_PROGRAM, case.tables)
            passed, witness = self.relation.check(source_out, fu_out, case, task)
            assert passed, f"Correct program failed MR-U2: {witness}"

    def test_index_dependent_program_fails(self):
        task = _make_task()
        tables = _make_tables()
        cases = self.relation.generate_cases(tables, task)
        source_out = _run_on_tables(INDEX_DEPENDENT_PROGRAM, tables)
        any_violation = False
        for case in cases:
            fu_result = execute_program(INDEX_DEPENDENT_PROGRAM, case.tables)
            if not fu_result.success:
                any_violation = True  # KeyError on relabeled index also counts
                continue
            passed, _ = self.relation.check(source_out, fu_result.output, case, task)
            if not passed:
                any_violation = True
        assert any_violation, "MR-U2 should detect .loc[0] index dependence"


# --- MR-U3: Column order ---

class TestColumnOrderInvariance:
    relation = ColumnOrderInvariance()

    def test_correct_program_passes(self):
        task = _make_task()
        tables = _make_tables()
        cases = self.relation.generate_cases(tables, task)
        source_out = _run_on_tables(CORRECT_PROGRAM, tables)
        for case in cases:
            fu_out = _run_on_tables(CORRECT_PROGRAM, case.tables)
            passed, witness = self.relation.check(source_out, fu_out, case, task)
            assert passed, f"Correct program failed MR-U3: {witness}"

    def test_column_positional_program_fails(self):
        task = _make_task()
        tables = _make_tables_column_trap()
        source_out = _run_on_tables(COLUMN_POSITIONAL_PROGRAM, tables)

        any_violation = False
        for seed in range(20):
            cases = self.relation.generate_cases(tables, task, rng_seed=seed)
            for case in cases:
                # Only test when column order actually changed
                orig_cols = list(tables["orders"].columns)
                new_cols = list(case.tables["orders"].columns)
                if new_cols == orig_cols:
                    continue
                fu_result = execute_program(COLUMN_POSITIONAL_PROGRAM, case.tables)
                if not fu_result.success:
                    any_violation = True  # groupby on numeric column → runtime error
                    break
                passed, _ = self.relation.check(source_out, fu_result.output, case, task)
                if not passed:
                    any_violation = True
                    break
            if any_violation:
                break
        assert any_violation, "MR-U3 should detect positional column access when columns reordered"


# --- MR-U4: Irrelevant column addition ---

class TestIrrelevantColumnAdditionInvariance:
    relation = IrrelevantColumnAdditionInvariance()

    def test_correct_program_passes(self):
        task = _make_task()
        tables = _make_tables()
        cases = self.relation.generate_cases(tables, task)
        source_out = _run_on_tables(CORRECT_PROGRAM, tables)
        for case in cases:
            fu_out = _run_on_tables(CORRECT_PROGRAM, case.tables)
            passed, witness = self.relation.check(source_out, fu_out, case, task)
            assert passed, f"Correct program failed MR-U4: {witness}"

    def test_distractor_columns_added(self):
        task = _make_task()
        tables = _make_tables()
        cases = self.relation.generate_cases(tables, task)
        assert len(cases) == 1
        augmented = cases[0].tables["orders"]
        original_cols = set(tables["orders"].columns)
        added_cols = set(augmented.columns) - original_cols
        assert len(added_cols) == 5, f"Expected 5 distractor columns, got {len(added_cols)}"
        assert all("_distractor_" in c for c in added_cols)
