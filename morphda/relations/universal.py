"""
Universal metamorphic relations — applicable to all analytical tasks.

MR-U1: Row-permutation invariance
MR-U2: Index-relabeling invariance
MR-U3: Column-order invariance
MR-U4: Irrelevant-column addition invariance
"""

from __future__ import annotations

import string
import uuid
from typing import Any

import numpy as np
import pandas as pd

from morphda.execution.normalization import outputs_equal
from morphda.relations.base import (
    MetamorphicRelation,
    RelationResult,
    TransformedCase,
    ViolationWitness,
)
from morphda.tasks.schema import TaskSpec


class RowPermutationInvariance(MetamorphicRelation):
    """MR-U1: Shuffling rows must not change the result."""

    relation_id = "MR-U1"
    relation_family = "universal"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        # Exclude explicitly sequential tasks (none in current DSL)
        return True

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        rng = np.random.default_rng(rng_seed)
        shuffled = {
            name: df.sample(frac=1, random_state=int(rng.integers(0, 2**31)))
            .reset_index(drop=True)
            for name, df in tables.items()
        }
        return [
            TransformedCase(
                case_id=f"{self.relation_id}_seed{rng_seed}",
                tables=shuffled,
                description="All table rows shuffled; row-independent computations must be unchanged.",
                expected_relation_type="equal",
            )
        ]

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
            likely_issue="positional_row_dependence_or_missing_sort",
        )


class IndexRelabelingInvariance(MetamorphicRelation):
    """MR-U2: Replacing the DataFrame index with random labels must not change the result."""

    relation_id = "MR-U2"
    relation_family = "universal"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return True

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        relabeled = {}
        rng = np.random.default_rng(rng_seed)
        for name, df in tables.items():
            new_idx = rng.integers(10_000, 99_999, size=len(df))
            relabeled[name] = df.set_index(pd.Index(new_idx, name=df.index.name))
        return [
            TransformedCase(
                case_id=f"{self.relation_id}_seed{rng_seed}",
                tables=relabeled,
                description="DataFrame index replaced with random integers; result must be unchanged.",
                expected_relation_type="equal",
            )
        ]

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
            likely_issue="index_used_as_business_key_or_accidental_index_alignment",
        )


class ColumnOrderInvariance(MetamorphicRelation):
    """MR-U3: Randomly reordering columns must not change the result."""

    relation_id = "MR-U3"
    relation_family = "universal"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return True

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        rng = np.random.default_rng(rng_seed)
        reordered = {
            name: df[rng.permutation(df.columns).tolist()]
            for name, df in tables.items()
        }
        return [
            TransformedCase(
                case_id=f"{self.relation_id}_seed{rng_seed}",
                tables=reordered,
                description="Column order randomized; result must be unchanged.",
                expected_relation_type="equal",
            )
        ]

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
            likely_issue="positional_column_access_iloc_or_schema_order_assumption",
        )


class IrrelevantColumnAdditionInvariance(MetamorphicRelation):
    """MR-U4: Adding random columns not referenced in the task must not change the result."""

    relation_id = "MR-U4"
    relation_family = "universal"

    _DISTRACTOR_PREFIX = "_distractor_"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return True

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        rng = np.random.default_rng(rng_seed)
        augmented = {}
        for name, df in tables.items():
            extra = pd.DataFrame({
                f"{self._DISTRACTOR_PREFIX}numeric_{i}": rng.uniform(0, 1000, size=len(df))
                for i in range(3)
            } | {
                f"{self._DISTRACTOR_PREFIX}cat_{i}": rng.choice(
                    list(string.ascii_uppercase[:8]), size=len(df)
                )
                for i in range(2)
            })
            augmented[name] = pd.concat([df, extra], axis=1)
        return [
            TransformedCase(
                case_id=f"{self.relation_id}_seed{rng_seed}",
                tables=augmented,
                description=(
                    "Five irrelevant columns added with names prefixed "
                    f"'{self._DISTRACTOR_PREFIX}'; result must be unchanged."
                ),
                expected_relation_type="equal",
            )
        ]

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
            likely_issue="positional_column_access_or_automatic_numeric_selection",
        )


# Registry
UNIVERSAL_RELATIONS: list[MetamorphicRelation] = [
    RowPermutationInvariance(),
    IndexRelabelingInvariance(),
    ColumnOrderInvariance(),
    IrrelevantColumnAdditionInvariance(),
]
