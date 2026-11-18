"""
Hardcoding and fragility mutation operators: HM-01 through HM-05.
"""

from __future__ import annotations

import ast

from morphda.mutations.base import MutationOperator


class HM01_HardcodedLabel(MutationOperator):
    """HM-01: Replace the final return value with a hardcoded string literal."""

    operator_id = "HM-01"
    mutation_family = "hardcoding"

    _HARDCODED_VALUE = "Electronics"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _HardcodeReturn(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_Return(self, node):
                if not self._done and node.value is not None:
                    self._done = True
                    node.value = ast.Constant(value=HM01_HardcodedLabel._HARDCODED_VALUE)
                return node

        transformer = _HardcodeReturn()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


class HM02_HardcodedNumeric(MutationOperator):
    """HM-02: Replace the final return with a hardcoded numeric literal (0.42)."""

    operator_id = "HM-02"
    mutation_family = "hardcoding"

    _HARDCODED_VALUE = 0.42

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _HardcodeReturn(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_Return(self, node):
                if not self._done and node.value is not None:
                    self._done = True
                    node.value = ast.Constant(value=HM02_HardcodedNumeric._HARDCODED_VALUE)
                return node

        transformer = _HardcodeReturn()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


class HM03_FirstRowDependence(MutationOperator):
    """HM-03: Replace the final selection with .iloc[0] (without sorting)."""

    operator_id = "HM-03"
    mutation_family = "hardcoding"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _ReplaceWithIloc0(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_Return(self, node):
                if not self._done and node.value is not None:
                    # Wrap with .iloc[0] if the value is a subscript/attribute
                    self._done = True
                    node.value = ast.Subscript(
                        value=ast.Attribute(
                            value=node.value,
                            attr="iloc",
                            ctx=ast.Load(),
                        ),
                        slice=ast.Constant(value=0),
                        ctx=ast.Load(),
                    )
                return node

        transformer = _ReplaceWithIloc0()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


class HM04_PositionalColumnDependence(MutationOperator):
    """HM-04: Replace df[named_column] with df.iloc[:, 0] (positional)."""

    operator_id = "HM-04"
    mutation_family = "hardcoding"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _ReplaceColumnWithPositional(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_Subscript(self, node):
                if (
                    not self._done
                    and isinstance(node.value, ast.Name)
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                ):
                    # Replace df["col"] with df.iloc[:, 0]
                    self._done = True
                    return ast.Subscript(
                        value=ast.Attribute(
                            value=node.value,
                            attr="iloc",
                            ctx=ast.Load(),
                        ),
                        slice=ast.Tuple(
                            elts=[
                                ast.Slice(),
                                ast.Constant(value=0),
                            ],
                            ctx=ast.Load(),
                        ),
                        ctx=ast.Load(),
                    )
                return self.generic_visit(node)

        transformer = _ReplaceColumnWithPositional()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


HARDCODING_OPERATORS: list[MutationOperator] = [
    HM01_HardcodedLabel(),
    HM02_HardcodedNumeric(),
    HM03_FirstRowDependence(),
    HM04_PositionalColumnDependence(),
]
