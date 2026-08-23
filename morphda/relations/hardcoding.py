"""
Hardcoding and memorization metamorphic relations: MR-H1 through MR-H3.

These relations detect programs that return hardcoded answers instead of
computing from the supplied data.
"""

from __future__ import annotations

import random
from datetime import timedelta
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


class CounterfactualAnswerFlip(MetamorphicRelation):
    """
    MR-H1: Modify relevant values so the correct answer MUST change.

    If the program output stays the same after the data changes dramatically,
    it almost certainly hardcoded the original answer.

    Strategy:
      - For label/grouped tasks: insert a dominant group that cannot lose.
      - For scalar tasks: multiply all measure values by a large factor.
    """

    relation_id = "MR-H1"
    relation_family = "hardcoding"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        is_period = (task_spec.comparison is not None and
                     task_spec.comparison.operation == "percentage_change")
        return not is_period

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

        if task_spec.output_type == "scalar":
            op = task_spec.metric.operation
            # count_distinct is invariant to value scaling — skip this case
            if op in ("count_distinct", "count"):
                return []
            measure = _primary_measure(task_spec, df)
            if measure is None:
                return []
            # Multiply all measure values by 1000; output should scale accordingly
            scaled = {name: t.copy() for name, t in tables.items()}
            scaled_df = scaled[primary].copy()
            scaled_df[measure] = scaled_df[measure].astype(float) * 1000.0
            scaled[primary] = scaled_df
            return [TransformedCase(
                case_id=f"{self.relation_id}_scale_seed{rng_seed}",
                tables=scaled,
                description=f"Multiplied '{measure}' by 1000. Scalar output must change.",
                expected_relation_type="changes",
            )]

        elif task_spec.group_by and task_spec.output_type in ("label", "ranked_list"):
            # Insert a sentinel group with values that guarantee it wins
            SENTINEL = "__HARDCODE_TEST__"
            group_col = task_spec.group_by[0]
            if group_col not in df.columns:
                return []
            measure = _primary_measure(task_spec, df)
            if measure is None:
                return []

            # Direction-aware: for ascending sort, sentinel needs the LOWEST values
            ranking = task_spec.ranking
            wants_high = (ranking is None) or (ranking.direction == "descending")

            # For ratio tasks: need enough unique denominator IDs to exceed minimum_denominator
            metric = task_spec.metric
            num_col = metric.numerator.column if metric.numerator else None
            den_col = metric.denominator.column if metric.denominator else None
            min_den = (task_spec.post_filter.minimum_denominator
                       if task_spec.post_filter and task_spec.post_filter.minimum_denominator else 0)
            n = max(max(200, min_den * 3), len(df))
            rows = []
            for i in range(n):
                row = df.iloc[int(rng.integers(0, len(df)))].to_dict()
                row[group_col] = SENTINEL
                if wants_high:
                    row[measure] = float(df[measure].abs().max()) * 500 + i + 1
                else:
                    row[measure] = float(df[measure].min()) - abs(df[measure].std()) * 100 - i - 1

                # For ratio tasks: assign unique denominator/numerator IDs
                if den_col and den_col in row:
                    row[den_col] = 700000 + i       # unique per row → count_distinct grows
                if num_col and num_col in row and wants_high:
                    row[num_col] = 700000 + i       # same as denominator → 100% rate

                if task_spec.date and task_spec.date.column in row:
                    row[task_spec.date.column] = task_spec.date.current_start
                for flt in task_spec.filters:
                    if flt.operator == "equal" and flt.column in row:
                        row[flt.column] = flt.value
                    elif flt.operator == "not_equal" and flt.column in row:
                        cands = [v for v in df[flt.column].dropna().unique() if v != flt.value]
                        if cands:
                            row[flt.column] = cands[0]
                rows.append(row)

            extra = pd.DataFrame(rows, columns=df.columns)
            augmented = {
                name: (pd.concat([t, extra], ignore_index=True) if name == primary else t.copy())
                for name, t in tables.items()
            }
            return [TransformedCase(
                case_id=f"{self.relation_id}_dominant_seed{rng_seed}",
                tables=augmented,
                description=(
                    f"Inserted dominant group '{SENTINEL}'. "
                    "Winner must switch; output that stays unchanged is likely hardcoded."
                ),
                expected_relation_type="changes",
                expected_delta=SENTINEL,
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
        # For label tasks with dominant sentinel: output must contain the sentinel
        if case.expected_delta == "__HARDCODE_TEST__":
            fu_str = str(follow_up_output)
            if "__HARDCODE_TEST__" in fu_str:
                return True, None
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation="output must change to sentinel after dominant group inserted",
                likely_issue="hardcoded_label_or_answer_not_computed_from_data",
            )

        # For scalar tasks: output must change significantly
        if not outputs_equal(source_output, follow_up_output, task_spec.output_type, tolerance):
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation="output must change after 1000x value scaling",
            likely_issue="hardcoded_numeric_answer_or_data_ignored",
        )


class LabelPermutationEquivariance(MetamorphicRelation):
    """
    MR-H2: Apply a bijection to all group label values.
    Numeric results must be unchanged; output labels must be remapped.

    Detects hardcoded label returns.
    """

    relation_id = "MR-H2"
    relation_family = "hardcoding"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return bool(task_spec.group_by) and task_spec.output_type in ("label", "ranked_list")

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

        original_labels = df[group_col].dropna().unique().tolist()
        suffix = f"_remap_{int(rng.integers(1000, 9999))}"
        mapping = {lbl: f"{lbl}{suffix}" for lbl in original_labels}
        inverse = {v: k for k, v in mapping.items()}

        relabeled = {name: t.copy() for name, t in tables.items()}
        rdf = relabeled[primary].copy()
        rdf[group_col] = rdf[group_col].map(mapping).fillna(rdf[group_col])
        relabeled[primary] = rdf

        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=relabeled,
            description=(
                f"Group labels in '{group_col}' remapped with suffix '{suffix}'. "
                "Output labels must change accordingly."
            ),
            expected_relation_type="label_equivariance",
            expected_delta={"mapping": mapping, "inverse": inverse},
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        info = case.expected_delta or {}
        mapping = info.get("mapping", {})
        src_str = str(source_output).strip() if source_output is not None else ""
        fu_str  = str(follow_up_output).strip() if follow_up_output is not None else ""
        expected_fu = mapping.get(src_str)

        if expected_fu is None:
            return True, None  # label not in mapping or multi-output

        if expected_fu in fu_str:
            return True, None

        # If the output is the ORIGINAL (unmapped) label, it's hardcoded
        if src_str in fu_str and expected_fu not in fu_str:
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation=f"output should be '{expected_fu}' after label remapping",
                likely_issue="hardcoded_label_or_join_on_display_value_not_key",
            )
        return True, None  # changed but not to a recognizable value


HARDCODING_RELATIONS: list[MetamorphicRelation] = [
    CounterfactualAnswerFlip(),
    LabelPermutationEquivariance(),
]
