"""
MORPH-DA Verification Engine.

Runs the candidate program on source data and all applicable
metamorphic follow-up cases, then aggregates violations into
a verification decision and counterexample witnesses.

The engine NEVER receives:
  - the gold answer
  - reference program source
  - mutation labels
  - relation-specific failure examples
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from morphda.execution.sandbox import execute_program, SandboxResult
from morphda.relations.base import MetamorphicRelation, RelationResult, ViolationWitness
from morphda.tasks.schema import TaskSpec


@dataclass
class VerificationReport:
    program_id: str
    task_id: str
    source_execution: SandboxResult
    relation_results: list[RelationResult] = field(default_factory=list)
    decision: str = "unknown"  # "pass" | "fail" | "error"
    witnesses: list[ViolationWitness] = field(default_factory=list)
    applicable_relations: int = 0
    violated_relations: int = 0
    total_python_runs: int = 0
    total_latency_ms: float = 0.0

    @property
    def violated_families(self) -> set[str]:
        """Unique relation families that detected a violation."""
        return {w.relation_id.rsplit('-', 1)[0] + '-' + w.relation_id.split('-')[1][:1]
                if '-' in w.relation_id else w.relation_id
                for w in self.witnesses}

    @property
    def n_violated_families(self) -> int:
        """Number of distinct relation families that fired."""
        families = set(
            rr.relation_family
            for rr in self.relation_results
            if rr.violations
        )
        return len(families)

    def to_dict(self) -> dict:
        return {
            "program_id": self.program_id,
            "task_id": self.task_id,
            "source_execution_success": self.source_execution.success,
            "decision": self.decision,
            "applicable_relations": self.applicable_relations,
            "violated_relations": self.violated_relations,
            "n_violated_families": self.n_violated_families,
            "total_python_runs": self.total_python_runs,
            "total_latency_ms": self.total_latency_ms,
            "witnesses": [w.to_dict() for w in self.witnesses],
        }


class VerificationEngine:
    """
    Execute metamorphic relations against a candidate program.

    Usage:
        engine = VerificationEngine(relations=UNIVERSAL_RELATIONS + FILTER_RELATIONS)
        report = engine.verify(
            program_source=candidate_code,
            tables=task_tables,
            task_spec=task_spec,
            program_id="modelA_task014_seed2",
        )
    """

    def __init__(
        self,
        relations: list[MetamorphicRelation],
        timeout_seconds: int = 30,
        rng_seed: int = 42,
        tolerance: float = 1e-9,
        min_violated_families: int = 1,
    ) -> None:
        """
        Args:
            min_violated_families: Minimum number of distinct relation families that must
                fire for the decision to be "fail". Default=1 (any violation).
                Set to 2 to reduce false positives at the cost of lower recall.
        """
        self.relations = relations
        self.timeout_seconds = timeout_seconds
        self.min_violated_families = min_violated_families
        self.rng_seed = rng_seed
        self.tolerance = tolerance

    def verify(
        self,
        program_source: str,
        tables: dict[str, pd.DataFrame],
        task_spec: TaskSpec,
        program_id: str = "",
    ) -> VerificationReport:
        report = VerificationReport(
            program_id=program_id,
            task_id=task_spec.task_id,
            source_execution=execute_program(
                program_source, tables, self.timeout_seconds
            ),
        )
        report.total_python_runs += 1

        if not report.source_execution.success:
            report.decision = "error"
            return report

        source_output = report.source_execution.output

        for relation in self.relations:
            if not relation.is_applicable(task_spec):
                report.relation_results.append(
                    RelationResult(
                        relation_id=relation.relation_id,
                        relation_family=relation.relation_family,
                        applicable=False,
                        passed=None,
                        skip_reason="not_applicable_to_task",
                    )
                )
                continue

            report.applicable_relations += 1
            cases = relation.generate_cases(tables, task_spec, self.rng_seed)

            rr = RelationResult(
                relation_id=relation.relation_id,
                relation_family=relation.relation_family,
                applicable=True,
                passed=True,
            )

            for case in cases:
                t0 = time.perf_counter()
                fu_result = execute_program(
                    program_source, case.tables, self.timeout_seconds
                )
                rr.python_latency_ms += (time.perf_counter() - t0) * 1000
                report.total_python_runs += 1
                rr.cases_run += 1

                if not fu_result.success:
                    # Execution failure on a transformed input is itself a signal,
                    # but we don't count it as a relation violation here.
                    continue

                passed, witness = relation.check(
                    source_output,
                    fu_result.output,
                    case,
                    task_spec,
                    self.tolerance,
                )
                if not passed and witness is not None:
                    rr.passed = False
                    rr.violations.append(witness)
                    report.witnesses.append(witness)

            if not rr.passed:
                report.violated_relations += 1

            report.relation_results.append(rr)
            report.total_latency_ms += rr.python_latency_ms

        # Decision: require min_violated_families distinct families to flag
        if self.min_violated_families <= 1:
            report.decision = "fail" if report.violated_relations > 0 else "pass"
        else:
            report.decision = "fail" if report.n_violated_families >= self.min_violated_families else "pass"
        return report
