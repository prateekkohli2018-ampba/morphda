"""
B4: Independent LLM judge baseline.

A separate model receives the question, schema, generated program, and
source execution output. It predicts whether the program is correct.

The judge does NOT receive:
  - the gold answer
  - reference code
  - mutation labels
  - metamorphic-relation results

Usage:
    judge = LLMJudge(llm_fn)
    result = judge.evaluate(question, schema_summary, program, source_output)
    if result.flagged: ...

The llm_fn must be a callable: (messages: list[dict]) -> response_with_.content
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


JUDGE_SYSTEM = """You are a data analysis code reviewer.
Given a natural language question, a data schema, an analyze(tables) function,
and its output, determine whether the program correctly answers the question.

You do NOT have access to the gold answer. Use your understanding of the
question semantics and the code logic to assess correctness.

Respond ONLY with a JSON object (no markdown, no explanation):
{
  "probability_wrong": <float 0.0-1.0>,
  "predicted_error_type": "<none|missing_filter|wrong_aggregation|wrong_grouping|wrong_sort|wrong_denominator|hardcoded|other>",
  "short_evidence": "<one sentence max>"
}"""


JUDGE_USER_TEMPLATE = """Question:
{question}

Schema:
{schema}

Program:
```python
{program}
```

Execution output: {output!r}

Is this program correct?"""


@dataclass
class JudgeResult:
    flagged: bool            # True = judge thinks program is wrong
    probability_wrong: float
    predicted_error_type: str
    short_evidence: str
    raw_response: str
    model_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class LLMJudge:
    """
    B4: Independent LLM verification judge.

    Args:
        llm_fn:    Callable (messages: list[dict]) → response object with .content
        threshold: probability_wrong threshold for flagging. Default 0.5.
    """

    def __init__(self, llm_fn: Callable, threshold: float = 0.5) -> None:
        self.llm_fn = llm_fn
        self.threshold = threshold

    def evaluate(
        self,
        question: str,
        schema_summary: str,
        program: str,
        source_output: Any,
    ) -> JudgeResult:
        import time

        user_msg = JUDGE_USER_TEMPLATE.format(
            question=question,
            schema=schema_summary,
            program=program[:3000],  # truncate very long programs
            output=source_output,
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]

        t0 = time.perf_counter()
        try:
            response = self.llm_fn(messages)
        except Exception as exc:
            return JudgeResult(
                flagged=False,
                probability_wrong=0.0,
                predicted_error_type="error",
                short_evidence=f"LLM call failed: {exc}",
                raw_response="",
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        input_tokens = output_tokens = 0
        if hasattr(response, "usage_metadata"):
            um = response.usage_metadata
            input_tokens  = getattr(um, "input_tokens",  0) or 0
            output_tokens = getattr(um, "output_tokens", 0) or 0

        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_judge_response(raw)

        return JudgeResult(
            flagged=parsed["probability_wrong"] >= self.threshold,
            probability_wrong=parsed["probability_wrong"],
            predicted_error_type=parsed["predicted_error_type"],
            short_evidence=parsed["short_evidence"],
            raw_response=raw,
            model_latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _parse_judge_response(raw: str) -> dict:
    # Try direct JSON parse
    try:
        d = json.loads(raw.strip())
        return {
            "probability_wrong": float(d.get("probability_wrong", 0.0)),
            "predicted_error_type": str(d.get("predicted_error_type", "unknown")),
            "short_evidence": str(d.get("short_evidence", "")),
        }
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to extract JSON from response with surrounding text
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group())
            return {
                "probability_wrong": float(d.get("probability_wrong", 0.0)),
                "predicted_error_type": str(d.get("predicted_error_type", "unknown")),
                "short_evidence": str(d.get("short_evidence", "")),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: heuristic extraction
    pw = 0.0
    match = re.search(r"probability_wrong[\":\s]+([0-9.]+)", raw)
    if match:
        try:
            pw = float(match.group(1))
        except ValueError:
            pass

    return {
        "probability_wrong": pw,
        "predicted_error_type": "unknown",
        "short_evidence": raw[:200],
    }


class SameModelReview:
    """
    B3: Same-model self-review.

    The generator model reviews its own output.
    Uses the same LLM function but with a review prompt.
    """

    def __init__(self, llm_fn: Callable, threshold: float = 0.5) -> None:
        self.judge = LLMJudge(llm_fn, threshold)

    def evaluate(
        self,
        question: str,
        schema_summary: str,
        program: str,
        source_output: Any,
    ) -> JudgeResult:
        return self.judge.evaluate(question, schema_summary, program, source_output)
