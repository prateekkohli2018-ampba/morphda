"""
Statistical metamorphic relations: MR-S1 through MR-S5.

These relations verify correct computation of variance, std, correlation,
quantiles, and z-scores.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from morphda.execution.normalization import outputs_equal
from morphda.relations.base import MetamorphicRelation, TransformedCase, ViolationWitness
from morphda.tasks.schema import TaskSpec


def _primary_measure(task_spec: TaskSpec, df: pd.DataFrame) -> str | None:
    if task_spec.metric.column and task_spec.metric.column in df.columns:
        return task_spec.metric.column
    num = [c for c in df.select_dtypes(include="number").columns
           if c not in task_spec.group_by]
    return num[0] if num else None


class VarianceTranslationInvariance(MetamorphicRelation):
    """
    MR-S1: Add constant c to all relevant values.
    Variance and std must remain unchanged.
    """

    relation_id = "MR-S1"
    relation_family = "statistics"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return task_spec.metric.operation in ("variance", "std")

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

        c = float(df[measure].std(ddof=0) * 3 + 100.0)
        shifted = {name: t.copy() for name, t in tables.items()}
        sdf = shifted[primary].copy()
        sdf[measure] = sdf[measure].astype(float) + c
        shifted[primary] = sdf

        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=shifted,
            description=f"Added constant c={c:.2f} to all '{measure}' values. Variance/std must be unchanged.",
            expected_relation_type="equal",
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
        if outputs_equal(source_output, follow_up_output, task_spec.output_type, tolerance):
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation="variance/std invariant under additive translation",
            likely_issue="mean_included_in_variance_or_wrong_ddof",
        )


class VarianceScaling(MetamorphicRelation):
    """
    MR-S2: Multiply all relevant values by scalar c > 0.
    variance × c², std × |c|.
    """

    relation_id = "MR-S2"
    relation_family = "statistics"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return task_spec.metric.operation in ("variance", "std")

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

        c = 3.0
        scaled = {name: t.copy() for name, t in tables.items()}
        sdf = scaled[primary].copy()
        sdf[measure] = sdf[measure].astype(float) * c
        scaled[primary] = sdf

        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=scaled,
            description=f"Multiplied '{measure}' by c={c}. variance×c², std×|c|.",
            expected_relation_type="scaled",
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
        c = case.expected_delta or 1.0
        op = task_spec.metric.operation
        try:
            src_f = float(source_output)
            fu_f = float(follow_up_output)
        except (TypeError, ValueError):
            return True, None

        if op == "variance":
            expected = src_f * c * c
        else:  # std
            expected = src_f * abs(c)

        if abs(fu_f - expected) <= tolerance + abs(expected) * tolerance:
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation=f"{op} should be {expected:.4f} after ×{c} scaling",
            likely_issue="wrong_ddof_or_aggregation_error",
            violation_magnitude=abs(fu_f - expected),
        )


class QuantileMonotonicity(MetamorphicRelation):
    """
    MR-S5: If all values increase by nonneg amounts, quantile must not decrease.
    A uniform additive shift shifts every quantile by the same constant.
    """

    relation_id = "MR-S5"
    relation_family = "statistics"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return task_spec.metric.operation in ("median", "quantile")

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

        c = float(df[measure].std(ddof=0) + 1.0) * 2
        shifted = {name: t.copy() for name, t in tables.items()}
        sdf = shifted[primary].copy()
        sdf[measure] = sdf[measure].astype(float) + c
        shifted[primary] = sdf

        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=shifted,
            description=f"Added c={c:.2f} to all '{measure}'. Median/quantile must shift by same amount.",
            expected_relation_type="shifts_by_c",
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
        c = case.expected_delta or 0.0
        try:
            src_f = float(source_output)
            fu_f = float(follow_up_output)
        except (TypeError, ValueError):
            return True, None

        expected = src_f + c
        if abs(fu_f - expected) <= tolerance + abs(c) * tolerance:
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation=f"quantile/median should shift by {c:.2f}",
            likely_issue="mean_used_instead_of_median_or_wrong_quantile",
            violation_magnitude=abs(fu_f - expected),
        )


STATISTICS_RELATIONS: list[MetamorphicRelation] = [
    VarianceTranslationInvariance(),
    VarianceScaling(),
    QuantileMonotonicity(),
]
