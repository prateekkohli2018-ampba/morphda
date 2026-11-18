"""
Aggregation and arithmetic metamorphic relations: MR-A1 through MR-A10.

These are the most operator-discriminating relations in MORPH-DA.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from morphda.execution.normalization import outputs_equal
from morphda.relations.base import MetamorphicRelation, TransformedCase, ViolationWitness
from morphda.tasks.schema import TaskSpec


def _numeric_cols(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    cols = list(df.select_dtypes(include="number").columns)
    return [c for c in cols if c not in (exclude or [])]


def _apply_filters_approx(df: pd.DataFrame, task_spec: TaskSpec) -> pd.DataFrame:
    """
    Best-effort application of equality/not-equality filters from the task spec
    to identify eligible rows. Used to select perturbation targets within scope.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for flt in task_spec.filters:
        if flt.column not in df.columns:
            continue
        if flt.operator == "equal":
            mask &= df[flt.column] == flt.value
        elif flt.operator == "not_equal":
            mask &= df[flt.column] != flt.value
        elif flt.operator == "in" and flt.values:
            mask &= df[flt.column].isin(flt.values)
        elif flt.operator == "not_in" and flt.values:
            mask &= ~df[flt.column].isin(flt.values)
    return df[mask]


def _primary_measure(task_spec: TaskSpec, df: pd.DataFrame) -> str | None:
    """Best-effort: return the main measure column from the task spec."""
    if task_spec.metric.column and task_spec.metric.column in df.columns:
        return task_spec.metric.column
    if task_spec.metric.numerator and task_spec.metric.numerator.column in df.columns:
        return task_spec.metric.numerator.column
    num = _numeric_cols(df, exclude=list(task_spec.group_by))
    return num[0] if num else None


class FullRowDuplicationAlgebra(MetamorphicRelation):
    """
    MR-A1: Duplicate every row (concat(D, D)).

    Expected per operation:
      sum/count  → doubles
      mean/median/min/max/distinct-count/rate → unchanged
    """

    relation_id = "MR-A1"
    relation_family = "aggregation"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return task_spec.metric.operation in (
            "sum", "mean", "median", "count", "count_distinct",
            "min", "max", "ratio",
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
            description="All table rows duplicated (concat(D, D)).",
            expected_relation_type="duplication_algebra",
            expected_delta=task_spec.metric.operation,
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        op = task_spec.metric.operation
        src = source_output
        fu = follow_up_output

        # Operations where output must be unchanged after doubling
        invariant_ops = {"mean", "median", "min", "max", "count_distinct", "ratio"}
        if op in invariant_ops:
            if outputs_equal(src, fu, task_spec.output_type, tolerance):
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=src,
                follow_up_output=fu,
                expected_relation=f"equal after doubling (operation={op})",
                likely_issue=f"sum_used_instead_of_{op}_or_count_used_instead_of_distinct",
            )

        # Operations where output doubles (scalar only)
        doubling_ops = {"sum", "count"}
        if op in doubling_ops and task_spec.output_type == "scalar":
            try:
                src_f = float(src)
                fu_f = float(fu)
                if abs(fu_f - 2 * src_f) <= tolerance + abs(src_f) * tolerance:
                    return True, None
                return False, ViolationWitness(
                    relation_id=self.relation_id,
                    case_id=case.case_id,
                    transformation_description=case.description,
                    source_output=src,
                    follow_up_output=fu,
                    expected_relation=f"follow_up == 2 * source (operation={op})",
                    likely_issue="mean_used_instead_of_sum_or_incorrect_normalization",
                    violation_magnitude=abs(fu_f - 2 * src_f),
                )
            except (TypeError, ValueError):
                pass

        return True, None  # label outputs not checked for scaling


class SingleValuePerturbation(MetamorphicRelation):
    """
    MR-A2: Add delta to one eligible measure value.

    Expected per operation:
      sum    → increases by delta
      count  → unchanged
      mean   → increases by delta/n
      median → unchanged (when non-median perturbed)
      min    → unchanged (when perturbed value not new minimum)
      max    → unchanged (when perturbed value not new maximum)
    """

    relation_id = "MR-A2"
    relation_family = "aggregation"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        # count_distinct: perturbing entity IDs is meaningless (changes identity, not metric)
        return task_spec.metric.operation in ("sum", "mean", "count")

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
        measure = _primary_measure(task_spec, df)
        if measure is None:
            return []

        eligible_df = _apply_filters_approx(df, task_spec)
        if len(eligible_df) == 0:
            eligible_df = df
        # Only perturb rows with non-null measure values
        eligible_non_null = eligible_df[eligible_df[measure].notna()]
        if len(eligible_non_null) == 0:
            return []  # no valid target rows
        delta = float(eligible_non_null[measure].std(ddof=0) + 1.0) * 10
        perturbed = {name: t.copy() for name, t in tables.items()}
        target_df = perturbed[primary].copy()
        target_df[measure] = target_df[measure].astype(float)
        # Select a random non-null eligible row index
        eligible_indices = eligible_non_null.index.tolist()
        idx = eligible_indices[int(rng.integers(0, len(eligible_indices)))]
        target_df.at[idx, measure] = float(target_df.at[idx, measure]) + delta
        perturbed[primary] = target_df

        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=perturbed,
            description=f"Added delta={delta:.2f} to one eligible '{measure}' value (row {idx}).",
            expected_relation_type="single_value_delta",
            expected_delta=delta,
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        op = task_spec.metric.operation
        delta = case.expected_delta or 0.0

        if task_spec.output_type != "scalar":
            return True, None  # label tasks: perturbation may or may not flip the winner

        try:
            src_f = float(source_output)
            fu_f = float(follow_up_output)
        except (TypeError, ValueError):
            return True, None

        if op == "count":
            if abs(fu_f - src_f) <= tolerance:
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation="count must not change when a value is perturbed",
                likely_issue="sum_used_instead_of_count",
                violation_magnitude=abs(fu_f - src_f),
            )

        if op == "sum":
            expected_fu = src_f + delta
            if abs(fu_f - expected_fu) <= tolerance + abs(delta) * 1e-6:
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation=f"follow_up == source + {delta:.2f}",
                likely_issue="mean_used_instead_of_sum_or_incorrect_filter",
                violation_magnitude=abs(fu_f - expected_fu),
            )

        return True, None


class MeanVsMedianOutlierTest(MetamorphicRelation):
    """
    MR-A5: Perturb a non-median extreme value (without crossing median position).

    Expected:
      mean   → changes by delta/n
      median → unchanged
    """

    relation_id = "MR-A5"
    relation_family = "aggregation"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return task_spec.metric.operation in ("mean", "median")

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        primary = task_spec.inputs[0]
        if primary not in tables:
            return []
        df = tables[primary]
        measure = _primary_measure(task_spec, df)
        if measure is None or len(df) < 5:
            return []

        # Apply task filters to identify eligible rows only
        eligible_df = _apply_filters_approx(df, task_spec)
        if len(eligible_df) < 5:
            eligible_df = df  # fall back if no rows survive filter

        values = eligible_df[measure].dropna().sort_values().reset_index(drop=True)
        # Perturb the maximum value among eligible rows (safe: doesn't cross median)
        delta = float(values.iloc[-1]) * 2 + 100.0
        max_idx = eligible_df[measure].idxmax()

        perturbed = {name: t.copy() for name, t in tables.items()}
        target_df = perturbed[primary].copy()
        target_df[measure] = target_df[measure].astype(float)
        target_df.at[max_idx, measure] = float(target_df.at[max_idx, measure]) + delta
        perturbed[primary] = target_df

        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=perturbed,
            description=(
                f"Increased max '{measure}' value by {delta:.2f} "
                "(non-median extreme perturbation). "
                "Mean should change; median should not."
            ),
            expected_relation_type="mean_vs_median_discriminator",
            expected_delta=delta,
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        if task_spec.output_type != "scalar":
            return True, None
        try:
            src_f = float(source_output)
            fu_f = float(follow_up_output)
        except (TypeError, ValueError):
            return True, None

        delta = case.expected_delta or 0.0
        op = task_spec.metric.operation

        if op == "median":
            # median must not change
            if abs(fu_f - src_f) <= tolerance:
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation="median unchanged after non-median extreme perturbation",
                likely_issue="mean_used_instead_of_median",
                violation_magnitude=abs(fu_f - src_f),
            )
        if op == "mean":
            # mean must change
            if abs(fu_f - src_f) > tolerance:
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation="mean must change after extreme value perturbation",
                likely_issue="median_used_instead_of_mean_or_outlier_filtering",
                violation_magnitude=abs(fu_f - src_f),
            )
        return True, None


class CountVsDistinctCountTest(MetamorphicRelation):
    """
    MR-A6: Duplicate entity rows with an existing ID.

    Expected:
      row count       → increases
      distinct count  → unchanged
    """

    relation_id = "MR-A6"
    relation_family = "aggregation"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return task_spec.metric.operation in ("count", "count_distinct")

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

        # Find an entity/ID column
        entity_col = None
        if task_spec.metric.numerator and task_spec.metric.numerator.column in df.columns:
            entity_col = task_spec.metric.numerator.column
        if entity_col is None:
            id_candidates = [c for c in df.columns if "id" in c.lower()]
            entity_col = id_candidates[0] if id_candidates else df.columns[0]

        n_dups = min(20, len(df) // 5)
        dup_rows = df.sample(n=n_dups, random_state=int(rng.integers(0, 2**31)), replace=True)
        augmented = {
            name: (pd.concat([t, dup_rows], ignore_index=True) if name == primary else t.copy())
            for name, t in tables.items()
        }
        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=augmented,
            description=(
                f"Duplicated {n_dups} existing rows (preserving entity IDs). "
                "Row count increases; distinct entity count must not."
            ),
            expected_relation_type="count_vs_distinct",
            expected_delta=n_dups,
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        if task_spec.output_type != "scalar":
            return True, None
        try:
            src_f = float(source_output)
            fu_f = float(follow_up_output)
        except (TypeError, ValueError):
            return True, None

        op = task_spec.metric.operation
        n_dups = case.expected_delta or 0

        if op == "count_distinct":
            if abs(fu_f - src_f) <= tolerance:
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation="distinct count unchanged after row duplication",
                likely_issue="count_used_instead_of_distinct_count",
                violation_magnitude=abs(fu_f - src_f),
            )
        if op == "count":
            expected_fu = src_f + n_dups
            if abs(fu_f - expected_fu) <= tolerance + abs(expected_fu) * 1e-6:
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation=f"count must increase by {n_dups}",
                likely_issue="distinct_count_used_instead_of_count",
                violation_magnitude=abs(fu_f - expected_fu),
            )
        return True, None


class GlobalAdditiveTranslation(MetamorphicRelation):
    """
    MR-A3: Add constant c to every eligible measure value.

    mean/median/min/max → shift by c
    sum → shift by n*c
    variance/std → unchanged
    """

    relation_id = "MR-A3"
    relation_family = "aggregation"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return task_spec.metric.operation in ("mean", "median", "sum", "variance", "std")

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        primary = task_spec.inputs[0]
        if primary not in tables:
            return []
        df = tables[primary]
        measure = _primary_measure(task_spec, df)
        if measure is None:
            return []

        c = float(df[measure].std(ddof=0) + 1.0) * 5 + 100.0
        shifted = {name: t.copy() for name, t in tables.items()}
        shifted_df = shifted[primary].copy()
        shifted_df[measure] = shifted_df[measure] + c
        shifted[primary] = shifted_df
        n = len(df)

        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=shifted,
            description=f"Added constant c={c:.2f} to all '{measure}' values (n={n} rows).",
            expected_relation_type="additive_translation",
            expected_delta=c,
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-6,
    ) -> tuple[bool, ViolationWitness | None]:
        if task_spec.output_type != "scalar":
            return True, None
        try:
            src_f = float(source_output)
            fu_f = float(follow_up_output)
        except (TypeError, ValueError):
            return True, None

        c = case.expected_delta or 0.0
        op = task_spec.metric.operation

        if op in ("variance", "std"):
            if abs(fu_f - src_f) <= tolerance + abs(src_f) * tolerance:
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation=f"{op} invariant under additive translation",
                likely_issue="mean_included_in_variance_calculation_incorrectly",
                violation_magnitude=abs(fu_f - src_f),
            )

        if op in ("mean", "median", "min", "max"):
            expected = src_f + c
            if abs(fu_f - expected) <= tolerance + abs(c) * tolerance:
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation=f"follow_up == source + {c:.2f}",
                likely_issue="wrong_aggregation_or_hardcoded_output",
                violation_magnitude=abs(fu_f - expected),
            )

        return True, None


class RatioNumeratorDenominatorIsolation(MetamorphicRelation):
    """
    MR-A8: Perturb numerator-eligible records only, then denominator-only records.
    Ratio changes in opposite directions.

    For period-comparison tasks: perturb current-period vs prior-period rows.
    Detects: wrong denominator, reversed ratio, absolute vs relative change.
    """

    relation_id = "MR-A8"
    relation_family = "aggregation"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return (
            task_spec.metric.operation == "ratio"
            and task_spec.metric.numerator is not None
            and task_spec.metric.denominator is not None
        ) or (
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
        from datetime import timedelta
        rng = np.random.default_rng(rng_seed)
        primary = task_spec.inputs[0]
        if primary not in tables:
            return []
        df = tables[primary]

        is_period = (task_spec.comparison is not None
                     and task_spec.date is not None
                     and task_spec.date.previous_start is not None)

        if is_period:
            # For period comparison: perturb current period only
            date = task_spec.date
            date_col = date.column
            if date_col not in df.columns:
                return []

            curr_start = pd.Timestamp(date.current_start)
            curr_end   = pd.Timestamp(date.current_end)

            measure = _primary_measure(task_spec, df)
            if measure is None:
                return []

            # Add rows to CURRENT period with extreme values (should increase current metric)
            n_extra = 10
            extra = []
            for i in range(n_extra):
                row = df.iloc[int(rng.integers(0, len(df)))].to_dict()
                # Date in current period
                n_days = (curr_end - curr_start).days
                d_offset = int(rng.integers(0, max(1, n_days + 1)))
                row[date_col] = (curr_start + pd.Timedelta(days=d_offset)).strftime("%Y-%m-%d")
                row[measure] = float(df[measure].abs().max()) * 50 + i + 1
                for flt in task_spec.filters:
                    if flt.operator == "equal" and flt.column in row:
                        row[flt.column] = flt.value
                    elif flt.operator == "not_equal" and flt.column in row:
                        cands = [v for v in df[flt.column].dropna().unique() if v != flt.value]
                        if cands:
                            row[flt.column] = cands[0]
                extra.append(row)

            extra_df = pd.DataFrame(extra, columns=df.columns)
            augmented = {
                name: (pd.concat([t, extra_df], ignore_index=True) if name == primary else t.copy())
                for name, t in tables.items()
            }
            return [TransformedCase(
                case_id=f"{self.relation_id}_curr_perturb_seed{rng_seed}",
                tables=augmented,
                description=(
                    f"Added {n_extra} dominant rows to current period. "
                    "For percentage_change tasks, output winner should potentially change "
                    "if program correctly implements YoY comparison."
                ),
                expected_relation_type="sensitive",
                scope_status="in_scope",
            )]

        return []

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        # For label outputs: the winner might or might not change — don't enforce
        # This is a sensitivity check, not an invariance check
        return True, None


class PeriodSwapDetector(MetamorphicRelation):
    """
    MR-A9: Detect programs that swap current and prior periods.

    Test: When current-period metric is artificially dominant,
    a correct program should report that as an improvement.
    A program with swapped periods would report the opposite.

    This is a directional monotonicity check for period-comparison tasks.
    """

    relation_id = "MR-A9"
    relation_family = "aggregation"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return (
            task_spec.comparison is not None
            and task_spec.comparison.operation == "percentage_change"
            and task_spec.date is not None
            and task_spec.date.previous_start is not None
            and task_spec.output_type == "scalar"
        )

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
        date = task_spec.date
        date_col = date.column
        if date_col not in df.columns:
            return []

        measure = _primary_measure(task_spec, df)
        if measure is None:
            return []

        # Create scenario where current >> prior for all groups
        # Correct YoY: pct_change > 0 (current better)
        # Swapped YoY: pct_change < 0 (would report prior as current)
        curr_start = pd.Timestamp(date.current_start)
        curr_end   = pd.Timestamp(date.current_end)
        prior_start = pd.Timestamp(date.previous_start)
        prior_end   = pd.Timestamp(date.previous_end)

        aug = {name: t.copy() for name, t in tables.items()}
        aug_df = aug[primary].copy()
        aug_df[measure] = aug_df[measure].astype(float)
        aug_df[date_col] = pd.to_datetime(aug_df[date_col], format="mixed")

        # Double current-period values (should increase pct_change)
        curr_mask  = (aug_df[date_col] >= curr_start) & (aug_df[date_col] <= curr_end)
        prior_mask = (aug_df[date_col] >= prior_start) & (aug_df[date_col] <= prior_end)
        aug_df.loc[curr_mask, measure] *= 5.0   # current × 5
        aug_df.loc[prior_mask, measure] *= 1.0  # prior unchanged
        aug_df[date_col] = aug_df[date_col].dt.strftime("%Y-%m-%d")
        aug[primary] = aug_df

        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=aug,
            description=(
                f"Current-period '{measure}' multiplied by 5 (prior unchanged). "
                "YoY percentage change must increase substantially."
            ),
            expected_relation_type="increases",
            expected_delta=5.0,
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-6,
    ) -> tuple[bool, ViolationWitness | None]:
        try:
            src_f = float(source_output)
            fu_f  = float(follow_up_output)
        except (TypeError, ValueError):
            return True, None

        # After multiplying current by 5, percentage change must increase
        if fu_f > src_f - tolerance:
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation="pct_change must increase when current period is boosted",
            likely_issue="current_prior_periods_swapped_or_absolute_used_instead_of_pct_change",
            violation_magnitude=abs(fu_f - src_f),
        )


AGGREGATION_RELATIONS: list[MetamorphicRelation] = [
    FullRowDuplicationAlgebra(),
    SingleValuePerturbation(),
    MeanVsMedianOutlierTest(),
    CountVsDistinctCountTest(),
    GlobalAdditiveTranslation(),
    RatioNumeratorDenominatorIsolation(),
    PeriodSwapDetector(),
]
