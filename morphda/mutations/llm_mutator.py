"""
LLM-generated hidden-fault mutant generator.

Implements the LLMMut track from paper Section 11.6.

Key design constraints:
  - Uses a DIFFERENT model from the primary analysis agent when possible
  - The mutation model does NOT see MORPH-DA relation descriptions
  - Outputs are oracle-filtered: must be non-equivalent on at least one seed
  - Manual audit of 20% of retained mutants is required
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable

from morphda.mutations.base import MutantRecord
from morphda.tasks.schema import TaskSpec


LLM_MUTATOR_SYSTEM = """You are creating a software-testing benchmark.
Given a correct data-analysis function and its natural-language task,
introduce exactly one subtle semantic error.

Requirements:
- Keep the code executable (no syntax errors, no import errors).
- Preserve the function signature: def analyze(tables: dict) -> object
- Preserve the output type (string label, number, or list).
- Do not create a syntax or runtime error.
- Do not hardcode the answer unless specifically requested.
- The error should resemble a realistic analyst mistake:
  examples: using mean instead of median, omitting a filter,
  using sum instead of count, wrong date boundary, wrong sort direction,
  count instead of distinct count, wrong join key, wrong denominator.
- Do not explain the error in comments.
- Do not change the function name.
- Return ONLY the modified Python function definition, no explanation."""


LLM_MUTATOR_USER_TEMPLATE = """Task:
{question}

Correct program:
```python
{program}
```

Introduce exactly one subtle semantic error. Return only the modified function."""


@dataclass
class LLMMutantCandidate:
    task_id: str
    reference_hash: str
    mutated_program: str
    syntax_valid: bool = False
    candidate_index: int = 0
    manual_audit: dict | None = None

    @property
    def mutant_id(self) -> str:
        h = hashlib.sha256(self.mutated_program.encode()).hexdigest()[:8]
        return f"llmmut_{self.task_id}_{h}"


def generate_llm_mutants(
    task_spec: TaskSpec,
    reference_source: str,
    llm_fn: Callable,
    n_candidates: int = 3,
) -> list[LLMMutantCandidate]:
    """
    Generate n_candidates LLM-produced mutants for one task.

    Args:
        task_spec:        The task being mutated.
        reference_source: The trusted reference program.
        llm_fn:           Callable (messages: list[dict]) → response with .content
        n_candidates:     Number of candidate mutants to generate.

    Returns:
        List of candidates (unvalidated; caller must filter for non-equivalence).
    """
    ref_hash = hashlib.sha256(reference_source.encode()).hexdigest()[:16]
    candidates = []

    for i in range(n_candidates):
        user_msg = LLM_MUTATOR_USER_TEMPLATE.format(
            question=task_spec.canonical_question,
            program=reference_source,
        )
        messages = [
            {"role": "system", "content": LLM_MUTATOR_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]

        try:
            response = llm_fn(messages)
        except Exception as exc:
            continue

        raw = response.content if hasattr(response, "content") else str(response)
        mutated = _extract_python(raw)
        if mutated is None:
            continue

        # Quick syntax check
        syntax_ok = False
        try:
            ast.parse(mutated)
            syntax_ok = True
        except SyntaxError:
            pass

        candidates.append(LLMMutantCandidate(
            task_id=task_spec.task_id,
            reference_hash=ref_hash,
            mutated_program=mutated,
            syntax_valid=syntax_ok,
            candidate_index=i,
        ))

    return candidates


def candidate_to_mutant_record(candidate: LLMMutantCandidate) -> MutantRecord:
    """Convert a validated LLMMutantCandidate to a MutantRecord for logging."""
    return MutantRecord(
        task_id=candidate.task_id,
        reference_hash=candidate.reference_hash,
        generation_source="llmmut",
        mutation_family="llm_hidden",
        mutation_operator=f"llmmut_{candidate.candidate_index}",
        mutated_program=candidate.mutated_program,
        syntax_valid=candidate.syntax_valid,
        execution_valid=False,  # caller sets after validation
        contract_valid=False,
        non_equivalent_seeds=[],
        held_out_operator=False,
        manual_audit=candidate.manual_audit,
    )


def _extract_python(raw: str) -> str | None:
    """Extract Python code from LLM response."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    stripped = raw.strip()
    if stripped.startswith(("def ", "import ", "# ")):
        return stripped
    return raw.strip() if len(raw.strip()) > 20 else None
