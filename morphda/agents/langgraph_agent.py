"""
LangGraph data-analysis agent.

Implements the main agent graph from paper Section 10.1:

  START
    -> load_task
    -> inspect_schema
    -> plan_analysis      (LLM)
    -> generate_program   (LLM)
    -> safe_execute
    -> format_answer
    -> END

The agent outputs a reusable `analyze(tables)` function string.
LangGraph is the orchestration harness — NOT the research contribution.

Note: Requires langchain-core and a configured LLM client.
Install: see pyproject.toml [project.optional-dependencies] llm
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from morphda.agents.prompts import (
    SYSTEM_PROMPT,
    build_generation_prompt,
    build_schema_summary,
    build_sample_rows,
)
from morphda.execution.sandbox import execute_program, SandboxResult


@dataclass
class AgentResult:
    task_id: str
    question: str
    generated_program: str | None
    execution_result: SandboxResult | None
    source_output: Any
    input_tokens: int = 0
    output_tokens: int = 0
    model_latency_ms: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        return (
            self.generated_program is not None
            and self.execution_result is not None
            and self.execution_result.success
        )


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response (strip markdown fences)."""
    # Try ```python ... ``` blocks first
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to raw text if it starts with def or import
    stripped = text.strip()
    if stripped.startswith(("def ", "import ", "import\n")):
        return stripped
    return text.strip()


class MorphDaAgent:
    """
    Simple synchronous data-analysis agent.

    Can be used with any LLM that supports the messages API.
    For LangGraph integration, wrap nodes as LangGraph node functions.

    Args:
        llm: A callable(messages: list[dict]) -> str.
             Compatible with LangChain ChatModel.invoke() or raw API clients.
        timeout_seconds: Sandbox execution timeout.
        temperature: LLM sampling temperature (0 = deterministic).
    """

    def __init__(
        self,
        llm: Any,
        timeout_seconds: int = 30,
    ) -> None:
        self.llm = llm
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        question: str,
        tables: dict[str, pd.DataFrame],
        task_id: str = "",
    ) -> AgentResult:
        """Generate and execute an analysis program for the given question."""
        import time

        schema = build_schema_summary(tables)
        sample = build_sample_rows(tables, n=3)
        user_prompt = build_generation_prompt(question, schema, sample)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        t0 = time.perf_counter()
        try:
            response = self.llm(messages)
        except Exception as exc:
            return AgentResult(
                task_id=task_id,
                question=question,
                generated_program=None,
                execution_result=None,
                source_output=None,
                error=f"LLM call failed: {exc}",
            )
        model_latency_ms = (time.perf_counter() - t0) * 1000

        # Extract token counts if available (LangChain AIMessage has usage_metadata)
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            um = response.usage_metadata
            input_tokens  = getattr(um, "input_tokens",  0) or 0
            output_tokens = getattr(um, "output_tokens", 0) or 0

        raw_text = response.content if hasattr(response, "content") else str(response)
        program = _extract_code(raw_text)

        exec_result = execute_program(program, tables, self.timeout_seconds)

        return AgentResult(
            task_id=task_id,
            question=question,
            generated_program=program,
            execution_result=exec_result,
            source_output=exec_result.output if exec_result.success else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_latency_ms=model_latency_ms,
            error=exec_result.exception if not exec_result.success else None,
        )

    def repair(
        self,
        question: str,
        tables: dict[str, pd.DataFrame],
        original_program: str,
        feedback: str,
        task_id: str = "",
    ) -> AgentResult:
        """
        One-step repair given a witness or generic feedback.

        The gold answer is NOT included in the feedback.
        """
        import time

        schema = build_schema_summary(tables)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": feedback},
        ]

        t0 = time.perf_counter()
        try:
            response = self.llm(messages)
        except Exception as exc:
            return AgentResult(
                task_id=task_id,
                question=question,
                generated_program=None,
                execution_result=None,
                source_output=None,
                error=f"LLM repair call failed: {exc}",
            )
        model_latency_ms = (time.perf_counter() - t0) * 1000

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            um = response.usage_metadata
            input_tokens  = getattr(um, "input_tokens",  0) or 0
            output_tokens = getattr(um, "output_tokens", 0) or 0

        raw_text = response.content if hasattr(response, "content") else str(response)
        program = _extract_code(raw_text)
        exec_result = execute_program(program, tables, self.timeout_seconds)

        return AgentResult(
            task_id=task_id,
            question=question,
            generated_program=program,
            execution_result=exec_result,
            source_output=exec_result.output if exec_result.success else None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_latency_ms=model_latency_ms,
            error=exec_result.exception if not exec_result.success else None,
        )
