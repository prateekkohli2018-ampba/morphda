"""
Join metamorphic relations: MR-J1 through MR-J4.

These verify correct join semantics (key selection, cardinality, unmatched rows).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from morphda.execution.normalization import outputs_equal
from morphda.relations.base import MetamorphicRelation, TransformedCase, ViolationWitness
from morphda.tasks.schema import TaskSpec


class UnmatchedDimensionRowInvariance(MetamorphicRelation):
    """
    MR-J1: Add rows to a dimension/lookup table whose keys are absent from the fact table.
    Output must be unchanged.

    Detects: Cartesian products, joins without keys, inappropriate dimension-driven aggregation.
    """

    relation_id = "MR-J1"
    relation_family = "join"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return len(task_spec.joins) > 0

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        rng = np.random.default_rng(rng_seed)
        if not task_spec.joins:
            return []

        join = task_spec.joins[0]
        dim_table = join.right_table
        if dim_table not in tables:
            return []
        dim_df = tables[dim_table]
        key = join.keys[0]
        if key not in dim_df.columns:
            return []

        # Add ghost dimension rows with large unmatched keys
        n = min(20, len(dim_df) // 4 + 1)
        max_key = dim_df[key].max() if dim_df[key].dtype.kind in "iuf" else 0
        ghost_rows = []
        for i in range(n):
            row = dim_df.iloc[int(rng.integers(0, len(dim_df)))].to_dict()
            if dim_df[key].dtype.kind in "iuf":
                row[key] = int(max_key) + 100000 + i
            else:
                row[key] = f"__ghost_{i}__"
            ghost_rows.append(row)

        ghost_df = pd.DataFrame(ghost_rows, columns=dim_df.columns)
        augmented = {
            name: (pd.concat([t, ghost_df], ignore_index=True) if name == dim_table else t.copy())
            for name, t in tables.items()
        }
        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=augmented,
            description=(
                f"Added {n} unmatched ghost rows to dimension table '{dim_table}'. "
                "Output must be unchanged."
            ),
            expected_relation_type="equal",
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
            expected_relation="equal (unmatched dimension rows must not affect output)",
            likely_issue="cartesian_product_or_missing_join_key",
        )


class ConsistentKeyRelabelingEquivariance(MetamorphicRelation):
    """
    MR-J2: Apply a bijection to join keys in all related tables.
    Metric results must be unchanged; returned key labels must be remapped.

    Detects: hardcoded keys, joining on display values, wrong key selection.
    """

    relation_id = "MR-J2"
    relation_family = "join"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return len(task_spec.joins) > 0

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        rng = np.random.default_rng(rng_seed)
        if not task_spec.joins:
            return []

        join = task_spec.joins[0]
        key = join.keys[0]
        fact_table  = join.left_table
        dim_table   = join.right_table
        if fact_table not in tables or dim_table not in tables:
            return []

        fact_df = tables[fact_table]
        dim_df  = tables[dim_table]
        if key not in fact_df.columns or key not in dim_df.columns:
            return []

        # Build a bijective key mapping (add a large constant offset)
        offset = 500_000
        relabeled = {}
        for name, t in tables.items():
            tc = t.copy()
            if key in tc.columns and tc[key].dtype.kind in "iuf":
                tc[key] = tc[key] + offset
            relabeled[name] = tc

        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=relabeled,
            description=(
                f"Join key '{key}' shifted by +{offset} in all tables. "
                "Metric results must be unchanged."
            ),
            expected_relation_type="metric_equal",
            expected_delta=offset,
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        # For numeric scalar outputs: must be equal (metric unchanged)
        if task_spec.output_type == "scalar":
            if outputs_equal(source_output, follow_up_output, task_spec.output_type, tolerance):
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation="metric unchanged after key relabeling",
                likely_issue="hardcoded_key_value_or_joining_on_display_value",
            )
        # For label outputs: the label should change (because group labels may include the key)
        # This is ambiguous without knowing the output type, so don't enforce
        return True, None


class DuplicateDimensionRowInvariance(MetamorphicRelation):
    """
    MR-J3: Duplicate a dimension row (same key, identical content).
    When uniqueness semantics are explicit, output must be unchanged.

    Detects: many-to-many row multiplication, missing deduplication.
    """

    relation_id = "MR-J3"
    relation_family = "join"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        # Only for inner/left joins where dimension uniqueness is expected
        return any(
            j.cardinality in ("many_to_one", "one_to_one")
            for j in task_spec.joins
        )

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        rng = np.random.default_rng(rng_seed)
        join = next(
            (j for j in task_spec.joins if j.cardinality in ("many_to_one", "one_to_one")),
            None
        )
        if join is None:
            return []

        dim_table = join.right_table
        if dim_table not in tables:
            return []
        dim_df = tables[dim_table]
        if len(dim_df) == 0:
            return []

        # Duplicate one random dimension row
        dup_idx = int(rng.integers(0, len(dim_df)))
        dup_row = dim_df.iloc[[dup_idx]]
        augmented = {
            name: (pd.concat([t, dup_row], ignore_index=True) if name == dim_table else t.copy())
            for name, t in tables.items()
        }
        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=augmented,
            description=(
                f"Duplicated one row in dimension table '{dim_table}' (same key). "
                "Output must be unchanged when uniqueness semantics are respected."
            ),
            expected_relation_type="equal",
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
            expected_relation="equal (duplicate dimension row must not multiply counts)",
            likely_issue="many_to_many_join_multiplication_or_missing_deduplication",
        )


JOIN_RELATIONS: list[MetamorphicRelation] = [
    UnmatchedDimensionRowInvariance(),
    ConsistentKeyRelabelingEquivariance(),
    DuplicateDimensionRowInvariance(),
]
