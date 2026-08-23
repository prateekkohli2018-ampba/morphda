"""
Task specification schema and validation.

A task spec is the source of truth for:
  - gold answers (via the reference compiler)
  - relation applicability
  - mutation applicability
  - question generation
  - data generation constraints
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class FilterSpec(BaseModel):
    column: str
    operator: Literal[
        "equal", "not_equal", "greater", "greater_equal",
        "less", "less_equal", "in", "not_in", "is_null", "is_not_null",
    ]
    value: Any = None
    values: list[Any] | None = None

    @model_validator(mode="after")
    def check_value_xor_values(self) -> "FilterSpec":
        if self.operator in ("in", "not_in"):
            if not self.values:
                raise ValueError(f"operator '{self.operator}' requires 'values'")
        else:
            if self.value is None and self.operator not in ("is_null", "is_not_null"):
                raise ValueError(f"operator '{self.operator}' requires 'value'")
        return self


class DateScope(BaseModel):
    column: str
    current_start: str
    current_end: str
    previous_start: str | None = None
    previous_end: str | None = None
    inclusive_bounds: bool = True
    calendar_type: Literal["gregorian", "fiscal"] = "gregorian"


class JoinSpec(BaseModel):
    left_table: str
    right_table: str
    keys: list[str]
    cardinality: Literal["one_to_one", "many_to_one", "one_to_many", "many_to_many"]
    join_type: Literal["inner", "left", "right"] = "inner"
    as_of: bool = False  # slowly changing dimension join


class MetricSpec(BaseModel):
    name: str
    operation: Literal[
        "sum", "mean", "median", "count", "count_distinct",
        "min", "max", "std", "variance", "quantile",
        "ratio", "percentage_change", "weighted_mean", "correlation",
    ]
    column: str | None = None
    numerator: "AggregationSpec | None" = None
    denominator: "AggregationSpec | None" = None
    weight_column: str | None = None
    quantile_value: float | None = None


class AggregationSpec(BaseModel):
    operation: Literal["sum", "count", "count_distinct", "mean", "median", "min", "max"]
    column: str
    filters: list[FilterSpec] = Field(default_factory=list)


class PostFilterSpec(BaseModel):
    minimum_denominator: int | None = None
    minimum_count: int | None = None
    maximum_null_fraction: float | None = None


class RankingSpec(BaseModel):
    direction: Literal["ascending", "descending"]
    k: int = 1
    tie_break: list[dict[str, Literal["ascending", "descending"]]] = Field(
        default_factory=list
    )


class ComparisonSpec(BaseModel):
    operation: Literal[
        "percentage_change", "absolute_change", "ratio",
        "rank_by_current", "rank_by_change",
    ]


class TaskSpec(BaseModel):
    """Complete specification for one analytical task."""

    task_id: str
    scenario_id: str
    question_family: str
    difficulty_level: Literal[1, 2, 3, 4, 5]

    inputs: list[str]  # table names required

    scope: dict[str, Any] = Field(default_factory=dict)  # filters + date
    filters: list[FilterSpec] = Field(default_factory=list)
    date: DateScope | None = None
    joins: list[JoinSpec] = Field(default_factory=list)

    metric: MetricSpec
    group_by: list[str] = Field(default_factory=list)
    post_filter: PostFilterSpec | None = None
    comparison: ComparisonSpec | None = None
    ranking: RankingSpec | None = None

    output_type: Literal["scalar", "label", "label_value_pairs", "ranked_list"]
    missing_value_rule: Literal["skip", "zero_fill", "error"] = "skip"

    # Populated after construction
    canonical_question: str = ""
    relation_applicability: dict[str, bool] = Field(default_factory=dict)
    mutation_applicability: dict[str, bool] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}
