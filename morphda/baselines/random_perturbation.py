"""
B6: Random data perturbation baseline (negative control).

Applies random row/value changes WITHOUT task-semantic relation design,
then flags any output instability. Demonstrates that arbitrary perturbation
≠ valid metamorphic testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from morphda.execution.normalization import outputs_equal
from morphda.execution.sandbox import execute_program, SandboxResult
from morphda.tasks.schema import TaskSpec


@dataclass
class RandomPerturbResult:
    flagged: bool
    n_perturbations: int
    n_unstable: int
    source_output: Any
    follow_up_outputs: list[Any] = field(default_factory=list)


def run_random_perturbation(
    program_source: str,
    tables: dict[str, pd.DataFrame],
    task_spec: TaskSpec,
    n_perturbations: int = 5,
    perturbation_fraction: float = 0.05,
    rng_seed: int = 42,
    timeout_seconds: int = 30,
) -> RandomPerturbResult:
    """
    Run the program on randomly perturbed table copies.

    Perturbations:
      - Randomly modify 5% of numeric cell values by ±10-50%.
      - Randomly delete 5% of rows.

    Flags the program if ANY perturbation changes the output.
    This is an overly sensitive approach (high FPR) used as a negative control.
    """
    rng = np.random.default_rng(rng_seed)

    source_result = execute_program(program_source, tables, timeout_seconds)
    if not source_result.success:
        return RandomPerturbResult(
            flagged=False,
            n_perturbations=0,
            n_unstable=0,
            source_output=None,
        )

    source_out = source_result.output
    fu_outputs = []
    n_unstable = 0

    for i in range(n_perturbations):
        perturbed = {}
        for name, df in tables.items():
            pdf = df.copy()
            # Random value perturbation
            for col in pdf.select_dtypes(include="number").columns:
                mask = rng.uniform(0, 1, size=len(pdf)) < perturbation_fraction
                factors = rng.uniform(0.5, 1.5, size=mask.sum())
                pdf.loc[mask, col] = pdf.loc[mask, col].astype(float) * factors
            # Random row deletion
            n_drop = int(len(pdf) * perturbation_fraction)
            drop_idx = rng.choice(pdf.index, size=n_drop, replace=False)
            pdf = pdf.drop(index=drop_idx).reset_index(drop=True)
            perturbed[name] = pdf

        fu_result = execute_program(program_source, perturbed, timeout_seconds)
        fu_out = fu_result.output if fu_result.success else None
        fu_outputs.append(fu_out)

        if not outputs_equal(source_out, fu_out, task_spec.output_type):
            n_unstable += 1

    return RandomPerturbResult(
        flagged=n_unstable > 0,
        n_perturbations=n_perturbations,
        n_unstable=n_unstable,
        source_output=source_out,
        follow_up_outputs=fu_outputs,
    )
