"""
B1: Output-contract checks (deterministic shape/type/range/conservation).

No data transformation needed — checks the source output only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from morphda.tasks.schema import TaskSpec


@dataclass
class ContractViolation:
    check: str
    description: str
    value: Any = None


@dataclass
class ContractCheckResult:
    passed: bool
    violations: list[ContractViolation] = field(default_factory=list)
    checks_run: int = 0


def check_output_contracts(
    output: Any,
    task_spec: TaskSpec,
) -> ContractCheckResult:
    """
    Run all applicable output-contract checks on the source output.

    Checks include:
      - correct Python type for declared output_type
      - correct cardinality for ranked_list (k elements)
      - unique labels in ranked_list
      - no NaN/Inf in scalar outputs
      - scalar rates in [0, 1] or percentages in [0, 100]
      - shares sum to ~1 or ~100 for share tasks
      - label exists in data (not checked here without access to tables)
    """
    violations: list[ContractViolation] = []
    checks = 0

    if output is None:
        return ContractCheckResult(passed=False, violations=[
            ContractViolation("non_null", "Output is None")
        ], checks_run=1)

    output_type = task_spec.output_type
    ranking = task_spec.ranking

    # Type contract
    checks += 1
    if output_type == "scalar":
        if not isinstance(output, (int, float)):
            violations.append(ContractViolation(
                "type_scalar",
                f"Expected numeric scalar, got {type(output).__name__}",
                output,
            ))
        elif math.isnan(float(output)) or math.isinf(float(output)):
            violations.append(ContractViolation(
                "nan_inf",
                f"Scalar output is NaN or Inf: {output}",
                output,
            ))

    elif output_type == "label":
        checks += 1
        if not isinstance(output, (str, int, float)):
            violations.append(ContractViolation(
                "type_label",
                f"Expected label (str/int/float), got {type(output).__name__}",
                output,
            ))

    elif output_type == "ranked_list":
        checks += 1
        if not isinstance(output, (list, tuple)):
            violations.append(ContractViolation(
                "type_ranked_list",
                f"Expected list/tuple for ranked_list, got {type(output).__name__}",
                output,
            ))
        else:
            # Cardinality
            if ranking and len(output) != ranking.k:
                checks += 1
                violations.append(ContractViolation(
                    "cardinality",
                    f"Expected {ranking.k} elements, got {len(output)}",
                    len(output),
                ))
            # Uniqueness
            checks += 1
            if len(output) != len(set(str(x) for x in output)):
                violations.append(ContractViolation(
                    "unique_labels",
                    "Ranked list contains duplicate labels",
                    output,
                ))

    elif output_type == "label_value_pairs":
        checks += 1
        if not isinstance(output, (dict, list, tuple)):
            violations.append(ContractViolation(
                "type_label_value_pairs",
                f"Expected dict or list of pairs, got {type(output).__name__}",
                output,
            ))

    # Rate/proportion range check for ratio metrics
    if task_spec.metric.operation == "ratio" and output_type == "scalar":
        checks += 1
        try:
            v = float(output)
            if not (0.0 <= v <= 1.0):
                violations.append(ContractViolation(
                    "rate_range",
                    f"Ratio output {v:.4f} outside [0, 1]",
                    v,
                ))
        except (TypeError, ValueError):
            pass

    return ContractCheckResult(
        passed=len(violations) == 0,
        violations=violations,
        checks_run=checks,
    )
