"""
Safe Python execution sandbox for candidate analyze(tables) programs.

Design constraints:
  - programs must define analyze(tables: dict[str, pd.DataFrame]) -> Any
  - execution is time and memory bounded
  - stdout/stderr are captured but not trusted
  - the gold answer is NEVER passed to the sandbox
  - results are normalized before comparison
"""

from __future__ import annotations

import importlib.util
import io
import resource
import signal
import sys
import textwrap
import traceback
import types
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import pandas as pd


class SandboxResult:
    def __init__(
        self,
        success: bool,
        output: Any,
        exception: str | None,
        stdout: str,
        latency_ms: float,
    ) -> None:
        self.success = success
        self.output = output
        self.exception = exception
        self.stdout = stdout
        self.latency_ms = latency_ms

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "exception": self.exception,
            "stdout": self.stdout[:500] if self.stdout else None,
            "latency_ms": self.latency_ms,
        }


class TimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise TimeoutError("Sandbox execution timed out")


def execute_program(
    program_source: str,
    tables: dict[str, pd.DataFrame],
    timeout_seconds: int = 30,
    memory_mb: int = 512,
) -> SandboxResult:
    """
    Execute a candidate analyze(tables) program in a restricted environment.

    Args:
        program_source: Python source string; must define analyze(tables).
        tables: Input DataFrames; passed as deep copies to prevent mutation.
        timeout_seconds: Hard wall-clock limit.
        memory_mb: Soft memory ceiling (best-effort via resource limits).

    Returns:
        SandboxResult with success flag, output, and diagnostics.
    """
    import time

    # Deep-copy tables so programs cannot mutate benchmark data
    safe_tables = {k: v.copy(deep=True) for k, v in tables.items()}

    # Attempt to set memory limit (Unix only; ignored on macOS silently)
    try:
        mem_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except (ValueError, AttributeError):
        pass

    # Compile the program source
    try:
        code = compile(
            textwrap.dedent(program_source),
            filename="<candidate>",
            mode="exec",
        )
    except SyntaxError as exc:
        return SandboxResult(
            success=False,
            output=None,
            exception=f"SyntaxError: {exc}",
            stdout="",
            latency_ms=0.0,
        )

    # Build a restricted global namespace.
    # __import__ must live inside __builtins__ for `import X` statements to work.
    safe_globals: dict[str, Any] = {
        "__builtins__": {
            "__import__": _safe_import,
            "__build_class__": __builtins__["__build_class__"] if isinstance(__builtins__, dict) else getattr(__builtins__, "__build_class__"),  # noqa: E501
            "print": print,
            "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "str": str, "int": int, "float": float, "bool": bool,
            "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
            "sorted": sorted, "reversed": reversed,
            "isinstance": isinstance, "issubclass": issubclass, "type": type,
            "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
            "callable": callable, "iter": iter, "next": next,
            "any": any, "all": all,
            "ValueError": ValueError, "KeyError": KeyError,
            "TypeError": TypeError, "IndexError": IndexError,
            "AttributeError": AttributeError, "StopIteration": StopIteration,
            "NotImplementedError": NotImplementedError,
            "None": None, "True": True, "False": False,
            "object": object,
        },
        "pd": pd,
    }

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    # Use threading.Timer for timeout — works in any thread (unlike signal.alarm)
    import threading as _threading
    _timed_out = _threading.Event()
    _timer = None

    def _thread_timeout() -> None:
        _timed_out.set()

    # Only use signal.alarm in the main thread (faster); use threading elsewhere
    _in_main_thread = _threading.current_thread() is _threading.main_thread()
    if _in_main_thread:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    else:
        _timer = _threading.Timer(timeout_seconds, _thread_timeout)
        _timer.start()

    t0 = time.perf_counter()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, safe_globals)  # noqa: S102

        if _timed_out.is_set():
            raise TimeoutError(f"exceeded {timeout_seconds}s")

        analyze_fn = safe_globals.get("analyze")
        if analyze_fn is None or not callable(analyze_fn):
            return SandboxResult(
                success=False,
                output=None,
                exception="Program did not define callable 'analyze'",
                stdout=stdout_buf.getvalue(),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            result = analyze_fn(safe_tables)

        if _timed_out.is_set():
            raise TimeoutError(f"exceeded {timeout_seconds}s")

        latency_ms = (time.perf_counter() - t0) * 1000
        if _in_main_thread:
            signal.alarm(0)
        elif _timer:
            _timer.cancel()

        return SandboxResult(
            success=True,
            output=result,
            exception=None,
            stdout=stdout_buf.getvalue(),
            latency_ms=latency_ms,
        )

    except TimeoutError:
        return SandboxResult(
            success=False,
            output=None,
            exception=f"TimeoutError: exceeded {timeout_seconds}s",
            stdout=stdout_buf.getvalue(),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    except Exception as exc:
        if _in_main_thread:
            signal.alarm(0)
        elif _timer:
            _timer.cancel()
        return SandboxResult(
            success=False,
            output=None,
            exception=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            stdout=stdout_buf.getvalue(),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


_ALLOWED_IMPORTS = frozenset([
    "pandas", "numpy", "math", "statistics", "datetime",
    "collections", "itertools", "functools", "re",
])


def _safe_import(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
    top = name.split(".")[0]
    if top not in _ALLOWED_IMPORTS:
        raise ImportError(f"Import '{name}' is not allowed in the sandbox")
    return importlib.import_module(name)
