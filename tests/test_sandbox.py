"""Tests for the execution sandbox."""

import pandas as pd
import pytest

from morphda.execution.sandbox import execute_program


def _tables() -> dict:
    return {"df": pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})}


def test_valid_program_returns_result():
    prog = "def analyze(tables):\n    return tables['df']['x'].sum()"
    result = execute_program(prog, _tables())
    assert result.success
    assert result.output == 6


def test_missing_analyze_function():
    prog = "x = 1"
    result = execute_program(prog, _tables())
    assert not result.success
    assert "analyze" in (result.exception or "")


def test_syntax_error_returns_failure():
    prog = "def analyze(tables):\n    return ("
    result = execute_program(prog, _tables())
    assert not result.success
    assert "SyntaxError" in (result.exception or "")


def test_runtime_error_returns_failure():
    prog = "def analyze(tables):\n    return tables['nonexistent']['col'].sum()"
    result = execute_program(prog, _tables())
    assert not result.success
    assert result.exception is not None


def test_forbidden_import_blocked():
    prog = "import os\ndef analyze(tables):\n    return os.getcwd()"
    result = execute_program(prog, _tables())
    assert not result.success


def test_tables_are_deep_copied():
    """Program mutations to tables must not affect the caller's data."""
    prog = """
def analyze(tables):
    df = tables['df']
    df['x'] = 999
    return df['x'].sum()
"""
    tables = _tables()
    original_x = tables["df"]["x"].tolist()
    result = execute_program(prog, tables)
    assert result.success
    assert tables["df"]["x"].tolist() == original_x  # original untouched


def test_latency_ms_populated():
    prog = "def analyze(tables):\n    return 42"
    result = execute_program(prog, _tables())
    assert result.latency_ms > 0
