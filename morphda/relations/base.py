"""
Base class for all metamorphic relations.

Each relation:
  1. receives the source tables and a task spec
  2. generates one or more transformed table sets (follow-up cases)
  3. executes the candidate program on source and each follow-up
  4. checks the expected output relation
  5. returns a RelationResult with pass/fail and counterexample witness
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from morphda.tasks.schema import TaskSpec


@dataclass
class TransformedCase:
    """One follow-up test case produced by a transformation."""
    case_id: str
    tables: dict[str, pd.DataFrame]
    description: str
    expected_relation_type: str  # e.g. "equal", "increases", "unchanged"
    expected_delta: Any = None  # quantitative expectation when applicable
    scope_status: str = "in_scope"  # "in_scope" | "out_of_scope" | "boundary"


@dataclass
class ViolationWitness:
    """Machine- and human-readable counterexample."""
    relation_id: str
    case_id: str
    transformation_description: str
    source_output: Any
    follow_up_output: Any
    expected_relation: str
    likely_issue: str
    violation_magnitude: float | None = None

    def to_dict(self) -> dict:
        return {
            "relation_id": self.relation_id,
            "case_id": self.case_id,
            "transformation_description": self.transformation_description,
            "source_output": self.source_output,
            "follow_up_output": self.follow_up_output,
            "expected_relation": self.expected_relation,
            "likely_issue": self.likely_issue,
            "violation_magnitude": self.violation_magnitude,
        }


@dataclass
class RelationResult:
    relation_id: str
    relation_family: str
    applicable: bool
    passed: bool | None  # None when not applicable
    violations: list[ViolationWitness] = field(default_factory=list)
    cases_run: int = 0
    python_latency_ms: float = 0.0
    skip_reason: str | None = None

    @property
    def killed(self) -> bool:
        """True when the relation detected a fault (applicable and failed)."""
        return self.applicable and self.passed is False


class MetamorphicRelation(abc.ABC):
    """
    Abstract base for all MORPH-DA metamorphic relations.

    Subclasses must implement:
      - relation_id: str class attribute
      - relation_family: str class attribute
      - is_applicable(task_spec) -> bool
      - generate_cases(tables, task_spec) -> list[TransformedCase]
      - check(source_output, follow_up_output, case) -> tuple[bool, ViolationWitness | None]
    """

    relation_id: str = ""
    relation_family: str = ""

    def is_applicable(self, task_spec: TaskSpec) -> bool:
        """Return True if this relation can be meaningfully applied to this task."""
        raise NotImplementedError

    def generate_cases(
        self,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        rng_seed: int = 42,
    ) -> list[TransformedCase]:
        """Generate follow-up transformed table sets."""
        raise NotImplementedError

    def check(
        self,
        source_output: Any,
        follow_up_output: Any,
        case: TransformedCase,
        task_spec: TaskSpec,
        tolerance: float = 1e-9,
    ) -> tuple[bool, ViolationWitness | None]:
        """
        Check whether source and follow-up outputs satisfy the expected relation.

        Returns:
            (passed, witness) — witness is None when passed=True.
        """
        raise NotImplementedError

    def likely_issue(self, case: TransformedCase) -> str:
        """Human-readable guess at the likely fault causing a violation."""
        return "unknown_fault"
