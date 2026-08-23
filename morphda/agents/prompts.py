"""
Prompt templates for the LangGraph analysis agent.
"""

from __future__ import annotations


SYSTEM_PROMPT = """You are a data analysis expert. Given a natural-language question and
one or more pandas DataFrames, write a Python function that answers the question correctly.

Rules:
1. Define exactly one function: `def analyze(tables: dict) -> object`
2. `tables` is a dict mapping table name (str) to pandas DataFrame.
3. Return the answer directly — a string label, number, or list.
4. Do not print anything; do not call input().
5. Import only: pandas as pd, numpy as np, math, statistics, datetime, collections.
6. Do not hardcode answers; compute from the data.
7. Handle missing values as specified in the question; default: skip nulls.
8. For grouped ranking: sort explicitly before selecting top-k.
9. Return only executable Python code — no explanations, no markdown.
"""


def build_generation_prompt(
    question: str,
    schema_summary: str,
    sample_rows: str = "",
) -> str:
    parts = [
        f"Question:\n{question}",
        f"\nSchema:\n{schema_summary}",
    ]
    if sample_rows:
        parts.append(f"\nSample rows (first 3):\n{sample_rows}")
    parts.append(
        "\nWrite the `analyze(tables)` function. "
        "Return only the Python function definition, no explanation."
    )
    return "\n".join(parts)


def build_schema_summary(tables: dict) -> str:
    """Generate a compact schema summary for the prompt."""
    import pandas as pd
    lines = []
    for name, df in tables.items():
        lines.append(f"Table '{name}': {len(df)} rows")
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            n_null = int(df[col].isna().sum())
            sample = df[col].dropna().head(3).tolist()
            lines.append(f"  - {col} ({dtype}): {n_unique} unique, {n_null} nulls, sample={sample}")
    return "\n".join(lines)


def build_sample_rows(tables: dict, n: int = 3) -> str:
    """
    Return the first n rows of each table as a formatted string.

    WARNING: This sends raw row values to the configured LLM endpoint.
    Only use with synthetic benchmark data or after confirming that the
    data does not contain PII, proprietary metrics, or confidential values.
    For real corporate datasets, use build_schema_summary() alone.
    """
    lines = []
    for name, df in tables.items():
        lines.append(f"\n--- {name} (first {n} rows) ---")
        lines.append(df.head(n).to_string(index=False))
    return "\n".join(lines)
