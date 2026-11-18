"""
Base class for deterministic semantic mutation operators.

Each operator:
  - receives the trusted reference program source (as AST/CST)
  - injects exactly one semantic fault
  - returns a MutantRecord

Operators are independent of MORPH-DA relations.
The mutation engine never receives relation definitions.
"""

from __future__ import annotations

import abc
import ast
import hashlib
from dataclasses import dataclass, field


@dataclass
class MutantRecord:
    """Record produced by one mutation operator on one task."""

    task_id: str
    reference_hash: str
    generation_source: str  # "rulemut" | "llmmut"
    mutation_family: str
    mutation_operator: str
    mutated_program: str
    syntax_valid: bool = False
    execution_valid: bool = False
    contract_valid: bool = False
    non_equivalent_seeds: list[int] = field(default_factory=list)
    held_out_operator: bool = False
    manual_audit: dict | None = None

    @property
    def mutant_id(self) -> str:
        h = hashlib.sha256(self.mutated_program.encode()).hexdigest()[:8]
        return f"mut_{self.task_id}_{self.mutation_operator}_{h}"

    @property
    def is_valid(self) -> bool:
        return (
            self.syntax_valid
            and self.execution_valid
            and self.contract_valid
            and len(self.non_equivalent_seeds) > 0
        )

    def to_dict(self) -> dict:
        return {
            "mutant_id": self.mutant_id,
            "task_id": self.task_id,
            "reference_hash": self.reference_hash,
            "generation_source": self.generation_source,
            "mutation_family": self.mutation_family,
            "mutation_operator": self.mutation_operator,
            "mutated_program": self.mutated_program,
            "syntax_valid": self.syntax_valid,
            "execution_valid": self.execution_valid,
            "contract_valid": self.contract_valid,
            "non_equivalent_seeds": self.non_equivalent_seeds,
            "held_out_operator": self.held_out_operator,
        }


class MutationOperator(abc.ABC):
    """
    Abstract base for semantic mutation operators.

    Separation guarantee:
      - operators receive: task_spec, reference_program, input/output contract
      - operators do NOT receive: metamorphic transformations, relation checkers,
        previous MORPH-DA results, relation-specific failure examples
    """

    operator_id: str = ""
    mutation_family: str = ""
    held_out: bool = False  # reserved for final evaluation

    def is_applicable(self, reference_source: str, task_spec: object) -> bool:
        """Return True if this operator can produce a valid mutant for this task."""
        try:
            tree = ast.parse(reference_source)
            return self._has_target_pattern(tree, task_spec)
        except SyntaxError:
            return False

    def _has_target_pattern(self, tree: ast.AST, task_spec: object) -> bool:
        """Override to check for required AST patterns."""
        return True

    @abc.abstractmethod
    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        """
        Inject exactly one semantic fault into reference_source.

        Returns:
            Mutated source string, or None if mutation cannot be applied.
        """

    def generate(self, reference_source: str, task_spec: object, task_id: str) -> MutantRecord | None:
        """
        Apply mutation and return a MutantRecord (not yet validated).
        Validation (syntax, execution, equivalence) happens in the pipeline.
        """
        if not self.is_applicable(reference_source, task_spec):
            return None

        mutated = self.mutate(reference_source, task_spec)
        if mutated is None:
            return None

        ref_hash = hashlib.sha256(reference_source.encode()).hexdigest()

        record = MutantRecord(
            task_id=task_id,
            reference_hash=ref_hash,
            generation_source="rulemut",
            mutation_family=self.mutation_family,
            mutation_operator=self.operator_id,
            mutated_program=mutated,
            held_out_operator=self.held_out,
        )

        # Syntax check immediately
        try:
            ast.parse(mutated)
            record.syntax_valid = True
        except SyntaxError:
            record.syntax_valid = False

        return record
