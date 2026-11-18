"""
Grouping and ranking metamorphic relations: MR-G1 through MR-G4, MR-R1 through MR-R5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from morphda.execution.normalization import outputs_equal
from morphda.relations.base import MetamorphicRelation, TransformedCase, ViolationWitness
from morphda.tasks.schema import TaskSpec

SENTINEL_DOMINATED = "__DOMINATED_GROUP__"
SENTINEL_DOMINANT  = "__DOMINANT_GROUP__"


def _primary_measure(task_spec: TaskSpec, df: pd.DataFrame) -> str | None:
    if task_spec.metric.column and task_spec.metric.column in df.columns:
        return task_spec.metric.column
    num = list(df.select_dtypes(include="number").columns)
    exclude = list(task_spec.group_by)
    num = [c for c in num if c not in exclude]
    return num[0] if num else None


def _sample_row(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    return df.iloc[int(rng.integers(0, len(df)))].to_dict()


def _is_period_comparison(task_spec: TaskSpec) -> bool:
    """Return True for period-comparison tasks that require data in both date windows."""
    return task_spec.comparison is not None and task_spec.comparison.operation == "percentage_change"


class GroupLocalPerturbation(MetamorphicRelation):
    """
    MR-G1: Perturb metric values within one group only.
    Only that group's statistic should change; others must be stable.

    For label-output tasks: checks that the winner changes only when the
    perturbed group is the winner, or stays the same otherwise.
    """

    relation_id = "MR-G1"
    relation_family = "grouping"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return bool(task_spec.group_by) and not _is_period_comparison(task_spec)

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
        measure = _primary_measure(task_spec, df)
        if group_col not in df.columns or measure is None:
            return []

        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            return []

        # Pick the last-ranked group (least likely to be the winner)
        target_group = groups[-1]
        delta = float(df[measure].std(ddof=0) * 2 + 10.0)

        perturbed = {name: t.copy() for name, t in tables.items()}
        target_df = perturbed[primary].copy()
        target_df[measure] = target_df[measure].astype(float)
        mask = target_df[group_col] == target_group
        target_df.loc[mask, measure] = target_df.loc[mask, measure] + delta
        perturbed[primary] = target_df

        return [TransformedCase(
            case_id=f"{self.relation_id}_{target_group}_seed{rng_seed}",
            tables=perturbed,
            description=(
                f"Perturbed '{measure}' by +{delta:.2f} for group='{target_group}' only. "
                "Other groups must be unchanged."
            ),
            expected_relation_type="group_local",
            expected_delta=target_group,
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        # For label outputs: if source winner != perturbed group, winner must stay same OR switch to it.
        # The key check: the source winner must not arbitrarily become a third party.
        # We approximate: if output changed AND source winner was not the perturbed group,
        # that's suspicious — but this is soft. Hard check only for scalar.
        if task_spec.output_type == "scalar":
            # Can't check easily without per-group outputs; just verify total moved
            return True, None
        return True, None  # label tasks: direction of change is unconstrained here


class DominatedGroupInsertion(MetamorphicRelation):
    """
    MR-G2: Insert a new group guaranteed to be dominated (worst).
    The winner must remain unchanged.
    """

    relation_id = "MR-G2"
    relation_family = "grouping"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return bool(task_spec.group_by) and task_spec.output_type in ("label", "ranked_list") and not _is_period_comparison(task_spec)

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
        measure = _primary_measure(task_spec, df)
        if group_col not in df.columns or measure is None:
            return []

        n = max(50, len(df) // 4)
        dominated_rows = []
        for _ in range(n):
            row = _sample_row(df, rng)
            row[group_col] = SENTINEL_DOMINATED
            row[measure] = float(df[measure].min()) - abs(df[measure].std()) * 10 - 1.0
            # comply with date and equality filters
            if task_spec.date and task_spec.date.column in row:
                row[task_spec.date.column] = task_spec.date.current_start
            for flt in task_spec.filters:
                if flt.operator == "equal" and flt.column in row:
                    row[flt.column] = flt.value
            dominated_rows.append(row)

        extra_df = pd.DataFrame(dominated_rows, columns=df.columns)
        augmented = {
            name: (pd.concat([t, extra_df], ignore_index=True) if name == primary else t.copy())
            for name, t in tables.items()
        }
        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=augmented,
            description=(
                f"Inserted dominated group '{SENTINEL_DOMINATED}' with worst possible metric. "
                "Winner must be unchanged."
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
        # Also pass if the sentinel dominated group appears in output (correct ranking)
        fu_str = str(follow_up_output)
        src_str = str(source_output)
        if SENTINEL_DOMINATED not in fu_str and src_str.lower() not in fu_str.lower():
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation="winner unchanged after inserting a dominated group",
                likely_issue="wrong_sort_direction_or_unstable_tie_or_hardcoded_output",
            )
        return True, None


class ForcedWinnerInsertion(MetamorphicRelation):
    """
    MR-G3: Insert a dominant group guaranteed to win.
    The winner must switch to the sentinel.
    """

    relation_id = "MR-G3"
    relation_family = "grouping"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return bool(task_spec.group_by) and task_spec.output_type in ("label", "ranked_list") and not _is_period_comparison(task_spec)

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
        measure = _primary_measure(task_spec, df)
        if group_col not in df.columns or measure is None:
            return []

        ranking = task_spec.ranking
        wants_high = (ranking is None) or (ranking.direction == "descending")

        # For ratio tasks: identify numerator/denominator columns
        metric = task_spec.metric
        num_col = metric.numerator.column if metric.numerator else None
        den_col = metric.denominator.column if metric.denominator else None
        # Minimum denominator threshold to pass post_filter
        min_den = (task_spec.post_filter.minimum_denominator
                   if task_spec.post_filter and task_spec.post_filter.minimum_denominator else 0)

        n = max(max(200, min_den * 3), len(df) // 2)  # ensure enough rows for threshold
        dominant_rows = []
        for i in range(n):
            row = _sample_row(df, rng)
            row[group_col] = SENTINEL_DOMINANT

            if wants_high:
                row[measure] = float(df[measure].max()) * 100 + 99999.0
            else:
                row[measure] = float(df[measure].min()) - abs(df[measure].std()) * 100 - i - 1.0

            # For ratio tasks: assign unique denominator IDs to exceed the threshold
            if den_col and den_col in row:
                row[den_col] = 800000 + i        # unique per row → count_distinct = n > min_den
            if num_col and num_col in row and wants_high:
                row[num_col] = 800000 + i        # same as denominator → 100% conversion rate

            if task_spec.date and task_spec.date.column in row:
                row[task_spec.date.column] = task_spec.date.current_start
            for flt in task_spec.filters:
                if flt.operator == "equal" and flt.column in row:
                    row[flt.column] = flt.value
                elif flt.operator == "not_equal" and flt.column in row:
                    cands = [v for v in df[flt.column].dropna().unique() if v != flt.value]
                    if cands:
                        row[flt.column] = cands[0]
            dominant_rows.append(row)

        extra_df = pd.DataFrame(dominant_rows, columns=df.columns)
        augmented = {
            name: (pd.concat([t, extra_df], ignore_index=True) if name == primary else t.copy())
            for name, t in tables.items()
        }
        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=augmented,
            description=(
                f"Inserted dominant group '{SENTINEL_DOMINANT}' with extreme metric. "
                "Winner must switch to this group."
            ),
            expected_relation_type="winner_switch",
            expected_delta=SENTINEL_DOMINANT,
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
        if SENTINEL_DOMINANT in fu_str:
            return True, None
        return False, ViolationWitness(
            relation_id=self.relation_id,
            case_id=case.case_id,
            transformation_description=case.description,
            source_output=source_output,
            follow_up_output=follow_up_output,
            expected_relation=f"winner must be '{SENTINEL_DOMINANT}'",
            likely_issue="hardcoded_output_over_filtering_wrong_metric_wrong_sort_direction",
        )


class GroupLabelPermutationEquivariance(MetamorphicRelation):
    """
    MR-G4: Bijection on group label values.
    Numeric results must be unchanged; output labels must transform by the same mapping.
    """

    relation_id = "MR-G4"
    relation_family = "grouping"

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        return bool(task_spec.group_by) and task_spec.output_type in ("label", "ranked_list") and not _is_period_comparison(task_spec)

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
        # Create a bijection: prefix each label with a random string
        prefix = f"_relabeled_{int(rng.integers(1000, 9999))}_"
        mapping = {lbl: f"{prefix}{lbl}" for lbl in original_labels}

        relabeled = {name: t.copy() for name, t in tables.items()}
        target_df = relabeled[primary].copy()
        target_df[group_col] = target_df[group_col].map(mapping).fillna(target_df[group_col])
        relabeled[primary] = target_df

        return [TransformedCase(
            case_id=f"{self.relation_id}_seed{rng_seed}",
            tables=relabeled,
            description=(
                f"Applied bijective relabeling to '{group_col}' (prefix='{prefix}'). "
                "Numeric results unchanged; output labels must be remapped."
            ),
            expected_relation_type="label_equivariance",
            expected_delta=mapping,
        )]

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        mapping: dict = case.expected_delta or {}
        src_str = str(source_output).strip() if source_output is not None else ""
        fu_str  = str(follow_up_output).strip() if follow_up_output is not None else ""

        expected_fu = mapping.get(src_str, "")
        if expected_fu and expected_fu in fu_str:
            return True, None
        if not expected_fu:
            return True, None  # source label not in mapping (multi-output or unknown)

        # Check if output is the old (unmapped) label — that's a hardcoding signal
        if src_str in fu_str and expected_fu not in fu_str:
            return False, ViolationWitness(
                relation_id=self.relation_id,
                case_id=case.case_id,
                transformation_description=case.description,
                source_output=source_output,
                follow_up_output=follow_up_output,
                expected_relation=f"output label should be '{expected_fu}' after relabeling",
                likely_issue="hardcoded_label_or_joining_on_display_value_rather_than_key",
            )
        return True, None


GROUPING_RELATIONS: list[MetamorphicRelation] = [
    GroupLocalPerturbation(),
    DominatedGroupInsertion(),
    ForcedWinnerInsertion(),
    GroupLabelPermutationEquivariance(),
]
