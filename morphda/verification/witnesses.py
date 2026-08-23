"""
Counterexample witness builder and repair prompt generator.
"""

from __future__ import annotations

from morphda.relations.base import ViolationWitness
from morphda.repair.prompts import witness_guided_prompt, generic_retry_prompt, relation_name_prompt


def build_repair_prompt(
    question: str,
    schema_summary: str,
    original_program: str,
    witnesses: list[ViolationWitness],
    strategy: str = "witness",
) -> str:
    """
    Build a repair prompt for the given strategy.

    strategy: "generic" → R2, "relation_name" → R6, "witness" → R7
    """
    if strategy == "generic" or not witnesses:
        return generic_retry_prompt(question, schema_summary, original_program)
    if strategy == "relation_name":
        w = witnesses[0]
        return relation_name_prompt(
            question=question,
            schema_summary=schema_summary,
            program=original_program,
            relation_id=w.relation_id,
            relation_description=w.likely_issue.replace("_", " "),
        )
    return witness_guided_prompt(
        question=question,
        schema_summary=schema_summary,
        program=original_program,
        witness=witnesses[0],
    )
