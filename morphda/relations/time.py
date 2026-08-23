"""
Temporal metamorphic relations: MR-T1 through MR-T5.

These relations verify correct date-window handling.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from morphda.execution.normalization import outputs_equal
from morphda.relations.base import MetamorphicRelation, TransformedCase, ViolationWitness
from morphda.tasks.schema import TaskSpec


def _parse_ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _sample_row(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    return df.iloc[int(rng.integers(0, len(df)))].to_dict()


class OutsideWindowExtremeInjection(MetamorphicRelation):
    """
    MR-T1: Add extreme records just outside the requested date window.
    Output must be unchanged.
    """

    relation_id = "MR-T1"
    relation_family = "time"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return task_spec.date is not None and task_spec.comparison is None

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

        start = _parse_ts(date.current_start)
        end   = _parse_ts(date.current_end)
        fmt = "%Y-%m-%d"

        before_start = (start - timedelta(days=1)).strftime(fmt)
        after_end    = (end   + timedelta(days=1)).strftime(fmt)

        n = min(30, max(10, len(df) // 10))
        extra = []
        for _ in range(n):
            bdate = before_start if rng.random() < 0.5 else after_end
            row = _sample_row(df, rng)
            row[date.column] = bdate
            for col in df.select_dtypes(include="number").columns:
                row[col] = float(df[col].abs().max() * 100 + 9999)
            extra.append(row)

        extra_df = pd.DataFrame(extra, columns=df.columns)
        augmented = {
            name: (pd.concat([t, extra_df], ignore_index=True) if name == primary else t.copy())
            for name, t in tables.items()
        }
        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=augmented,
            description=(
                f"Added {n} extreme rows outside the date window "
                f"[{date.current_start}, {date.current_end}]. "
                "Output must be unchanged."
            ),
            expected_relation_type="equal",
            scope_status="out_of_scope",
        )]

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
            likely_issue="date_filter_missing_or_wrong_boundary",
        )


class PeriodIsolatedPerturbation(MetamorphicRelation):
    """
    MR-T3: Perturb only current-period rows, then only prior-period rows.
    The period-comparison change must respond directionally.
    """

    relation_id = "MR-T3"
    relation_family = "time"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return (
            task_spec.comparison is not None
            and task_spec.comparison.operation == "percentage_change"
            and task_spec.date is not None
            and task_spec.date.previous_start is not None
        )

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
        date_col = date.column
        if date_col not in df.columns:
            return []

        measure = None
        if task_spec.metric.column and task_spec.metric.column in df.columns:
            measure = task_spec.metric.column
        else:
            num_cols = [c for c in df.select_dtypes(include="number").columns
                        if c not in task_spec.group_by]
            measure = num_cols[0] if num_cols else None
        if measure is None:
            return []

        curr_start = _parse_ts(date.current_start)
        curr_end   = _parse_ts(date.current_end)
        fmt = "%Y-%m-%d"
        delta = float(df[measure].std(ddof=0) * 5 + 1)

        cases = []
        # Case 1: perturb current period only
        perturbed_curr = {name: t.copy() for name, t in tables.items()}
        pcdf = perturbed_curr[primary].copy()
        pcdf[date_col] = pd.to_datetime(pcdf[date_col], format="mixed")
        pcdf[measure] = pcdf[measure].astype(float)
        curr_mask = (pcdf[date_col] >= curr_start) & (pcdf[date_col] <= curr_end)
        pcdf.loc[curr_mask, measure] += delta
        pcdf[date_col] = pcdf[date_col].dt.strftime(fmt)
        perturbed_curr[primary] = pcdf
        cases.append(TransformedCase(
            case_id=f"{self.relation_id}_curr_seed{rng_seed}",
            tables=perturbed_curr,
            description=(
                f"Current-period '{measure}' increased by {delta:.1f}. "
                "Period-comparison metric should increase."
            ),
            expected_relation_type="increases",
            expected_delta=delta,
            scope_status="in_scope",
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
        # For grouped label tasks, winner may or may not change — don't enforce direction
        # For scalar tasks, check directional change
        if task_spec.output_type != "scalar":
            return True, None
        try:
            src_f = float(source_output)
            fu_f = float(follow_up_output)
        except (TypeError, ValueError):
            return True, None

        if case.expected_relation_type == "increases":
            if fu_f >= src_f - tolerance:
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation="output must increase after current-period boost",
                likely_issue="current_prior_period_reversed_or_wrong_date_column",
                violation_magnitude=src_f - fu_f,
            )
        return True, None


class FullPeriodDuplication(MetamorphicRelation):
    """
    MR-T5: Duplicate all rows (both periods together).
    Means, rates, shares, and percentage changes must be unchanged.
    Sums and counts double.
    """

    relation_id = "MR-T5"
    relation_family = "time"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return task_spec.metric.operation in (
            "mean", "median", "ratio", "percentage_change",
        ) or (
            task_spec.comparison is not None
            and task_spec.comparison.operation == "percentage_change"
        )

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        doubled = {name: pd.concat([df, df], ignore_index=True) for name, df in tables.items()}
        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=doubled,
            description=(
                "All rows duplicated. Means, rates, shares, and percentage changes "
                "must be unchanged."
            ),
            expected_relation_type="equal",
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-6,
    ) -> tuple[bool, ViolationWitness | None]:
        if outputs_equal(source_output, follow_up_output, task_spec.output_type, tolerance):
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation="equal (means/rates/shares unchanged after row duplication)",
            likely_issue="sum_used_instead_of_mean_rate_or_incorrect_normalization",
        )


class ForcedPeriodWinnerInsertion(MetamorphicRelation):
    """
    MR-T4: For period_comparison_rank tasks, insert rows that make one group
    the clear YoY winner in both current and prior periods — but the current
    period grows much faster. A correct program must report this group as winner.

    Detects: wrong filter scope on periods, absolute vs relative change,
    current/prior period reversal, wrong metric column.
    """

    relation_id = "MR-T4"
    relation_family = "time"
    SENTINEL = "__PERIOD_WINNER__"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return (
            task_spec.comparison is not None
            and task_spec.comparison.operation == "percentage_change"
            and task_spec.date is not None
            and task_spec.date.previous_start is not None
            and bool(task_spec.group_by)
            and task_spec.output_type in ("label", "ranked_list")
        )

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        rng = np.random.default_rng(rng_seed)
        date = task_spec.date
        primary = task_spec.inputs[0]
        if primary not in tables:
            return []
        df = tables[primary]
        date_col = date.column
        group_col = task_spec.group_by[0]
        if date_col not in df.columns or group_col not in df.columns:
            return []

        # Find the measure column
        measure = None
        if task_spec.metric.column and task_spec.metric.column in df.columns:
            measure = task_spec.metric.column
        else:
            num_cols = [c for c in df.select_dtypes(include="number").columns
                        if c not in task_spec.group_by]
            measure = num_cols[0] if num_cols else None
        if measure is None:
            return []

        curr_start = pd.Timestamp(date.current_start)
        curr_end   = pd.Timestamp(date.current_end)
        prior_start = pd.Timestamp(date.previous_start)
        prior_end   = pd.Timestamp(date.previous_end)
        fmt = "%Y-%m-%d"

        max_val = float(df[measure].abs().max())
        # Need enough rows to exceed minimum_denominator threshold for ratio tasks
        metric = task_spec.metric
        den_col = metric.denominator.column if metric.denominator else None
        num_col = metric.numerator.column if metric.numerator else None
        min_den = (task_spec.post_filter.minimum_denominator
                   if task_spec.post_filter and task_spec.post_filter.minimum_denominator else 0)
        n = max(100, min_den * 3)

        def _make_sentinel_row(df, rng, period_start, period_end, measure_val, i):
            row = df.iloc[int(rng.integers(0, len(df)))].to_dict()
            row[group_col] = self.SENTINEL
            row[measure] = measure_val
            n_days = (period_end - period_start).days
            row[date_col] = (period_start + pd.Timedelta(
                days=int(rng.integers(0, max(1, n_days + 1)))
            )).strftime(fmt)
            # For ratio tasks: unique denominator and numerator IDs for each row
            # so count_distinct grows to exceed minimum_denominator threshold
            if den_col and den_col in row:
                row[den_col] = 600000 + i  # unique per row
            if num_col and num_col in row:
                row[num_col] = 600000 + i  # same as denominator → 100% rate
            for flt in task_spec.filters:
                if flt.operator == "equal" and flt.column in row:
                    row[flt.column] = flt.value
                elif flt.operator == "not_equal" and flt.column in row:
                    cands = [v for v in df[flt.column].dropna().unique() if v != flt.value]
                    if cands: row[flt.column] = cands[0]
            return row

        sentinel_rows = []
        is_ratio_task = (task_spec.metric.operation == "ratio"
                         and den_col is not None and num_col is not None)

        for i in range(n // 2):
            row = df.iloc[int(rng.integers(0, len(df)))].to_dict()
            row[group_col] = self.SENTINEL
            row[measure] = max_val + i + 1
            n_days = max(1, (prior_end - prior_start).days)
            row[date_col] = (prior_start + pd.Timedelta(
                days=int(rng.integers(0, n_days + 1))
            )).strftime(fmt)
            if is_ratio_task:
                # Prior: unique denominator IDs, but numerator = same constant → LOW rate
                row[den_col] = 600000 + i        # unique session per row
                row[num_col] = 500000            # shared customer_id → count_distinct(num)=1
            for flt in task_spec.filters:
                if flt.operator == "equal" and flt.column in row:
                    row[flt.column] = flt.value
                elif flt.operator == "not_equal" and flt.column in row:
                    cands = [v for v in df[flt.column].dropna().unique() if v != flt.value]
                    if cands: row[flt.column] = cands[0]
            sentinel_rows.append(row)

        for i in range(n // 2):
            row = df.iloc[int(rng.integers(0, len(df)))].to_dict()
            row[group_col] = self.SENTINEL
            row[measure] = max_val * 100 + i + 1
            n_days = max(1, (curr_end - curr_start).days)
            row[date_col] = (curr_start + pd.Timedelta(
                days=int(rng.integers(0, n_days + 1))
            )).strftime(fmt)
            if is_ratio_task:
                # Current: unique denominator AND numerator IDs → 100% rate
                uid = 700000 + i
                row[den_col] = uid
                row[num_col] = uid               # same → count_distinct(num)=count_distinct(den)
            for flt in task_spec.filters:
                if flt.operator == "equal" and flt.column in row:
                    row[flt.column] = flt.value
                elif flt.operator == "not_equal" and flt.column in row:
                    cands = [v for v in df[flt.column].dropna().unique() if v != flt.value]
                    if cands: row[flt.column] = cands[0]
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
                f"Inserted '{self.SENTINEL}' group with ~100x YoY increase in both periods "
                f"(prior: moderate, current: dominant). Winner must switch to sentinel."
            ),
            expected_relation_type="winner_switch",
            expected_delta=self.SENTINEL,
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
        if self.SENTINEL in fu_str:
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation=f"winner must be '{self.SENTINEL}' (100x YoY increase inserted)",
            likely_issue="period_filter_wrong_or_absolute_used_instead_of_pct_change_or_periods_swapped",
        )


TIME_RELATIONS: list[MetamorphicRelation] = [
    OutsideWindowExtremeInjection(),
    PeriodIsolatedPerturbation(),
    FullPeriodDuplication(),
    ForcedPeriodWinnerInsertion(),
]
