"""
Repair prompt templates for MORPH-DA.

Five repair strategies (from paper Section 14.2):
  R2 - always retry (generic)
  R4 - LLM-judge gated
  R5 - MORPH-gated, no witness
  R6 - MORPH relation-name feedback
  R7 - MORPH counterexample witness (primary experimental condition)
"""

from __future__ import annotations

from morphda.relations.base import ViolationWitness


def generic_retry_prompt(question: str, schema_summary: str, program: str) -> str:
    return f"""The following data-analysis program may contain an error.
Review it carefully and produce a corrected version.

Question:
{question}

Schema:
{schema_summary}

Current program:
```python
{program}
```

Return only the corrected analyze(tables) function. Do not include explanations."""


def relation_name_prompt(
    question: str,
    schema_summary: str,
    program: str,
    relation_id: str,
    relation_description: str,
) -> str:
    return f"""The following data-analysis program violated a behavioral property during verification.

Question:
{question}

Schema:
{schema_summary}

Violated property: {relation_id} — {relation_description}

Current program:
```python
{program}
```

Inspect the relevant logic and return a corrected analyze(tables) function.
Do not include explanations."""


def witness_guided_prompt(
    question: str,
    schema_summary: str,
    program: str,
    witness: ViolationWitness,
) -> str:
    """
    R7: Full counterexample witness prompt.
    The gold answer is NOT included.
    """
    return f"""The following data-analysis program executed successfully but violated
a necessary behavioral property.

Question:
{question}

Schema:
{schema_summary}

Behavioral property violated:
The program's output must satisfy: {witness.expected_relation}

Counterexample:
- Transformation applied: {witness.transformation_description}
- Output on original data: {witness.source_output!r}
- Output after transformation: {witness.follow_up_output!r}
- Likely issue: {witness.likely_issue.replace("_", " ")}

These outputs should be equal (or satisfy the stated relation), but they differ.
This suggests the program has a fault related to: {witness.likely_issue.replace("_", " ")}.

Inspect the filter, aggregation, join, or grouping logic and produce a corrected
analyze(tables) function. Recompute the result from the supplied tables.
Return only executable Python code."""
