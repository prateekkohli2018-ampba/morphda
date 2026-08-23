"""
Filter and scope metamorphic relations: MR-F1 through MR-F4.

These relations verify that programs correctly apply the filters and
date boundaries declared in the task specification.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from morphda.execution.normalization import outputs_equal
from morphda.relations.base import MetamorphicRelation, TransformedCase, ViolationWitness
from morphda.tasks.schema import TaskSpec


def _parse_date(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _sample_row(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    idx = int(rng.integers(0, len(df)))
    return df.iloc[idx].to_dict()


class OutOfScopeExtremeRowInvariance(MetamorphicRelation):
    """
    MR-F1: Add rows that violate an explicit filter and carry large metric values.
    The output must remain unchanged.
    """

    relation_id = "MR-F1"
    relation_family = "filter_scope"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return bool(task_spec.filters or task_spec.date)

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        cases = []
        rng = np.random.default_rng(rng_seed)

        for table_name in task_spec.inputs:
            if table_name not in tables:
                continue
            df = tables[table_name]
            extra_rows: list[dict] = []

            # Date-based out-of-scope rows
            if task_spec.date and task_spec.date.column in df.columns:
                date_col = task_spec.date.column
                # For period_comparison tasks, go 2 years back to avoid landing in the prior window
                years_back = 2 if (task_spec.comparison and task_spec.date.previous_start) else 1
                out_date = (_parse_date(task_spec.date.current_start) - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")
                n_extra = min(20, max(5, len(df) // 10))
                for _ in range(n_extra):
                    row = _sample_row(df, rng)
                    row[date_col] = out_date
                    for col in df.select_dtypes(include="number").columns:
                        row[col] = float(df[col].abs().max() * 100 + 9999)
                    extra_rows.append(row)

            # Status / equality filter violations
            # For period_comparison tasks: filter-violation rows MUST also be placed
            # outside the date window to prevent in-period contamination.
            is_period_task = (task_spec.comparison is not None
                              and task_spec.date is not None
                              and task_spec.date.previous_start is not None)

            for flt in task_spec.filters:
                if flt.operator == "not_equal" and flt.column in df.columns:
                    for _ in range(5):
                        row = _sample_row(df, rng)
                        row[flt.column] = flt.value
                        for col in df.select_dtypes(include="number").columns:
                            row[col] = float(df[col].abs().max() * 100 + 9999)
                        # For period tasks: ensure filter-violation rows are also date-OOS
                        if is_period_task and task_spec.date and task_spec.date.column in row:
                            years_back = 2
                            row[task_spec.date.column] = (
                                _parse_date(task_spec.date.current_start)
                                - timedelta(days=365 * years_back)
                            ).strftime("%Y-%m-%d")
                        extra_rows.append(row)
                elif flt.operator == "equal" and flt.column in df.columns:
                    excluded_val = f"_excluded_{int(rng.integers(9999))}"
                    for _ in range(5):
                        row = _sample_row(df, rng)
                        row[flt.column] = excluded_val
                        for col in df.select_dtypes(include="number").columns:
                            row[col] = float(df[col].abs().max() * 100 + 9999)
                        if is_period_task and task_spec.date and task_spec.date.column in row:
                            years_back = 2
                            row[task_spec.date.column] = (
                                _parse_date(task_spec.date.current_start)
                                - timedelta(days=365 * years_back)
                            ).strftime("%Y-%m-%d")
                        extra_rows.append(row)

            if not extra_rows:
                continue

            extra_df = pd.DataFrame(extra_rows, columns=df.columns)
            augmented = {
                name: (pd.concat([t, extra_df], ignore_index=True) if name == table_name else t.copy())
                for name, t in tables.items()
            }
            cases.append(TransformedCase(
                case_id=f"{self.relation_id}_{table_name}_seed{rng_seed}",
                tables=augmented,
                description=(
                    f"Added {len(extra_rows)} out-of-scope rows to '{table_name}' "
                    "with extreme metric values. Output must be unchanged."
                ),
                expected_relation_type="equal",
                scope_status="out_of_scope",
            ))
        return cases

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        if outputs_equal(source_output, follow_up_output, task_spec.output_type, tolerance):
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation="equal",
            likely_issue="missing_filter_wrong_column_or_filter_applied_after_aggregation",
        )


class InScopeSentinelSensitivity(MetamorphicRelation):
    """
    MR-F2: Add eligible dominant rows with a sentinel group label.
    The winner must switch to the sentinel.
    """

    relation_id = "MR-F2"
    relation_family = "filter_scope"
    SENTINEL_LABEL = "__SENTINEL_GROUP__"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        # period_comparison tasks need sentinel rows in BOTH periods — not supported here
        is_period = (task_spec.comparison is not None and
                     task_spec.comparison.operation == "percentage_change")
        return bool(task_spec.group_by) and task_spec.output_type in ("label", "ranked_list") and not is_period

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        rng = np.random.default_rng(rng_seed)
        primary = task_spec.inputs[0]
        if primary not in tables:
            return []
        df = tables[primary]
        group_col = task_spec.group_by[0]
        if group_col not in df.columns:
            return []

        # Determine whether sentinel needs extreme HIGH or extreme LOW metric
        ranking = task_spec.ranking
        wants_high = (ranking is None) or (ranking.direction == "descending")

        n = max(200, len(df) // 2)
        sentinel_rows = []
        num_cols = list(df.select_dtypes(include="number").columns)

        # For ratio tasks: identify numerator and denominator columns
        metric = task_spec.metric
        num_col = (metric.numerator.column if metric.numerator else None)
        den_col = (metric.denominator.column if metric.denominator else None)

        for i in range(n):
            row = _sample_row(df, rng)
            row[group_col] = self.SENTINEL_LABEL

            if wants_high:
                # Set all numerics to extreme high values
                for col in num_cols:
                    row[col] = float(df[col].abs().max() * 50 + 9999 + i)
                # For ratio tasks: make numerator = denominator (100% conversion)
                if num_col and den_col and num_col in row and den_col in row:
                    # Assign unique IDs so count_distinct works correctly
                    uid = 900000 + i
                    row[den_col] = uid           # denominator: each sentinel has unique session
                    row[num_col] = uid           # numerator = denominator → 100% rate
            else:
                # Set all numerics to extreme low values (for ascending sort)
                for col in num_cols:
                    row[col] = float(df[col].min()) - abs(df[col].std()) * 10 - i - 1.0

            # Comply with date filter (current period)
            if task_spec.date and task_spec.date.column in row:
                row[task_spec.date.column] = task_spec.date.current_start

            # Comply with equality/not-equality filters
            for flt in task_spec.filters:
                if flt.operator == "equal" and flt.column in row:
                    row[flt.column] = flt.value
                elif flt.operator == "not_equal" and flt.column in row:
                    candidates = [v for v in df[flt.column].dropna().unique() if v != flt.value]
                    if candidates:
                        row[flt.column] = candidates[0]

            sentinel_rows.append(row)

        extra_df = pd.DataFrame(sentinel_rows, columns=df.columns)
        augmented = {
            name: (pd.concat([t, extra_df], ignore_index=True) if name == primary else t.copy())
            for name, t in tables.items()
        }
        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=augmented,
            description=(
                f"Added {n} eligible dominant rows (group='{self.SENTINEL_LABEL}'). "
                "Winner must switch to sentinel."
            ),
            expected_relation_type="winner_switch",
            expected_delta=self.SENTINEL_LABEL,
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        fu_str = str(follow_up_output).strip() if follow_up_output is not None else ""
        if self.SENTINEL_LABEL in fu_str:
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation=f"output_must_contain_sentinel={self.SENTINEL_LABEL}",
            likely_issue="hardcoded_output_over_filtering_wrong_metric_or_wrong_filter",
        )


class BoundaryQuartet(MetamorphicRelation):
    """
    MR-F3: Probe all four date boundary positions (before start, at start,
    at end, after end). Out-of-scope probes must not change the output.
    """

    relation_id = "MR-F3"
    relation_family = "filter_scope"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return task_spec.date is not None

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        rng = np.random.default_rng(rng_seed)
        date = task_spec.date  # type: ignore[union-attr]
        primary = task_spec.inputs[0]
        if primary not in tables:
            return []
        df = tables[primary]
        if date.column not in df.columns:
            return []

        start = _parse_date(date.current_start)
        end = _parse_date(date.current_end)
        one_day = timedelta(days=1)

        fmt = "%Y-%m-%d"
        boundary_cases = [
            ("before_start", (start - one_day).strftime(fmt), "out_of_scope"),
            ("at_start",     start.strftime(fmt),             "in_scope"),
            ("at_end",       end.strftime(fmt),               "in_scope"),
            ("after_end",    (end + one_day).strftime(fmt),   "out_of_scope"),
        ]

        cases = []
        for label, bdate, scope in boundary_cases:
            row = _sample_row(df, rng)
            row[date.column] = bdate
            for col in df.select_dtypes(include="number").columns:
                if col != date.column:
                    row[col] = float(df[col].abs().max() * 10 + 999)
            extra_df = pd.DataFrame([row], columns=df.columns)
            augmented = {
                name: (pd.concat([t, extra_df], ignore_index=True) if name == primary else t.copy())
                for name, t in tables.items()
            }
            cases.append(TransformedCase(
                case_id=f"{self.relation_id}_{label}_seed{rng_seed}",
                tables=augmented,
                description=f"Boundary probe at {bdate!r} ({label}). Scope: {scope}.",
                expected_relation_type="equal" if scope == "out_of_scope" else "may_change",
                scope_status=scope,
            ))
        return cases

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        if case.scope_status != "out_of_scope":
            return True, None  # in-scope boundary probes may change result
        if outputs_equal(source_output, follow_up_output, task_spec.output_type, tolerance):
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation="equal (out-of-scope boundary must not affect result)",
            likely_issue="boundary_gt_vs_gte_or_lt_vs_lte_date_truncation",
        )


class ConjunctIsolationTest(MetamorphicRelation):
    """
    MR-F4: For filters A AND B, probe rows satisfying only A, only B, or both.
    Rows satisfying only one conjunct must not affect the result.
    """

    relation_id = "MR-F4"
    relation_family = "filter_scope"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return len(task_spec.filters) >= 2

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        rng = np.random.default_rng(rng_seed)
        primary = task_spec.inputs[0]
        if primary not in tables:
            return []
        df = tables[primary]
        flt_a = task_spec.filters[0]
        flt_b = task_spec.filters[1]
        if flt_a.column not in df.columns or flt_b.column not in df.columns:
            return []

        def _apply(row: dict, flt: object, satisfy: bool) -> dict:
            row = row.copy()
            if flt.operator == "equal":
                row[flt.column] = flt.value if satisfy else f"_excl_{int(rng.integers(9999))}"
            elif flt.operator == "not_equal":
                if satisfy:
                    cands = [v for v in df[flt.column].dropna().unique() if v != flt.value]
                    row[flt.column] = cands[0] if cands else "_valid"
                else:
                    row[flt.column] = flt.value
            return row

        base = _sample_row(df, rng)
        cases = []
        for label, (sat_a, sat_b), scope in [
            ("A_only",  (True, False),  "out_of_scope"),
            ("B_only",  (False, True),  "out_of_scope"),
            ("A_and_B", (True,  True),  "in_scope"),
        ]:
            row = _apply(_apply(base.copy(), flt_a, sat_a), flt_b, sat_b)
            extra_df = pd.DataFrame([row], columns=df.columns)
            augmented = {
                name: (pd.concat([t, extra_df], ignore_index=True) if name == primary else t.copy())
                for name, t in tables.items()
            }
            cases.append(TransformedCase(
                case_id=f"{self.relation_id}_{label}_seed{rng_seed}",
                tables=augmented,
                description=f"Conjunct probe: {label}. Scope: {scope}.",
                expected_relation_type="equal" if scope == "out_of_scope" else "may_change",
                scope_status=scope,
            ))
        return cases

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        if case.scope_status != "out_of_scope":
            return True, None
        if outputs_equal(source_output, follow_up_output, task_spec.output_type, tolerance):
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation="equal (single-conjunct probe must not affect result)",
            likely_issue="dropped_conjunct_or_AND_converted_to_OR",
        )


FILTER_RELATIONS: list[MetamorphicRelation] = [
    OutOfScopeExtremeRowInvariance(),
    InScopeSentinelSensitivity(),
    BoundaryQuartet(),
    ConjunctIsolationTest(),
]
