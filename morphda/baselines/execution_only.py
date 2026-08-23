"""
B0: Execution-only baseline.

Rule: No exception AND parseable result → accept.
Detects: nothing deterministic; all wrong-but-executable programs pass.
"""

from __future__ import annotations

from typing import Any

from morphda.execution.sandbox import SandboxResult


def verify(result: SandboxResult) -> bool:
    """Return True (accept) when the program executed without error."""
    return result.success and result.output is not None
