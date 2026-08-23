"""
Filter mutation operators: FM-01 through FM-10.

Each operator injects exactly one filter-related semantic fault
into a trusted reference program using Python AST transforms.
"""

from __future__ import annotations

import ast

from morphda.mutations.base import MutationOperator


class _CompareFlipTransformer(ast.NodeTransformer):
    """Flip one comparison operator; stops after first successful flip."""

    _FLIP = {
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
        ast.GtE: ast.Gt, ast.Gt: ast.GtE,
        ast.LtE: ast.Lt, ast.Lt: ast.LtE,
    }

    def __init__(self) -> None:
        self._flipped = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        if not self._flipped and node.ops:
            op_type = type(node.ops[0])
            if op_type in self._FLIP:
                node.ops[0] = self._FLIP[op_type]()
                self._flipped = True
        return self.generic_visit(node)


class FM01_RemoveAllFilters(MutationOperator):
    """FM-01: Remove all boolean mask / filter expressions."""

    operator_id = "FM-01"
    mutation_family = "filter"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _RemoveSubscripts(ast.NodeTransformer):
            """Replace df[boolean_series] with df."""
            def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
                # Heuristic: if the slice is a Compare or BoolOp, it's likely a filter
                if isinstance(node.slice, (ast.Compare, ast.BoolOp)):
                    return node.value  # return bare DataFrame reference
                return self.generic_visit(node)

        mutated_tree = _RemoveSubscripts().visit(tree)
        ast.fix_missing_locations(mutated_tree)
        try:
            return ast.unparse(mutated_tree)
        except Exception:
            return None


class FM04_InvertPredicate(MutationOperator):
    """FM-04: Flip one comparison operator (== -> !=, >= -> >, etc.)."""

    operator_id = "FM-04"
    mutation_family = "filter"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)
        transformer = _CompareFlipTransformer()
        mutated_tree = transformer.visit(tree)
        if not transformer._flipped:
            return None
        ast.fix_missing_locations(mutated_tree)
        return ast.unparse(mutated_tree)


class FM05_BoundaryFlip(MutationOperator):
    """FM-05: Change >= to > or <= to < (exclusive vs inclusive boundary)."""

    operator_id = "FM-05"
    mutation_family = "filter"

    _BOUNDARY_FLIP = {ast.GtE: ast.Gt, ast.LtE: ast.Lt}

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _BoundaryTransformer(ast.NodeTransformer):
            def __init__(self) -> None:
                self._flipped = False

            def visit_Compare(self, node: ast.Compare) -> ast.AST:
                for i, op in enumerate(node.ops):
                    if not self._flipped and type(op) in FM05_BoundaryFlip._BOUNDARY_FLIP:
                        node.ops[i] = FM05_BoundaryFlip._BOUNDARY_FLIP[type(op)]()
                        self._flipped = True
                        break
                return self.generic_visit(node)

        transformer = _BoundaryTransformer()
        mutated_tree = transformer.visit(tree)
        if not transformer._flipped:
            return None
        ast.fix_missing_locations(mutated_tree)
        return ast.unparse(mutated_tree)


class FM06_WrongLiteral(MutationOperator):
    """FM-06: Replace one year/period literal with the prior year (e.g. 2025 -> 2024)."""

    operator_id = "FM-06"
    mutation_family = "filter"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _LiteralShift(ast.NodeTransformer):
            def __init__(self) -> None:
                self._shifted = False

            def visit_Constant(self, node: ast.Constant) -> ast.AST:
                if not self._shifted and isinstance(node.value, int) and 2020 <= node.value <= 2030:
                    node.value -= 1
                    self._shifted = True
                return node

        transformer = _LiteralShift()
        mutated_tree = transformer.visit(tree)
        if not transformer._shifted:
            return None
        ast.fix_missing_locations(mutated_tree)
        return ast.unparse(mutated_tree)


class FM09_IncludeExcludedStatus(MutationOperator):
    """FM-09: Remove a != 'cancelled' / status exclusion filter."""

    operator_id = "FM-09"
    mutation_family = "filter"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        _EXCLUSION_VALUES = {"cancelled", "canceled", "refunded", "test", "excluded"}

        class _RemoveExclusion(ast.NodeTransformer):
            def __init__(self) -> None:
                self._removed = False

            def visit_Compare(self, node: ast.Compare) -> ast.AST:
                if (
                    not self._removed
                    and len(node.ops) == 1
                    and isinstance(node.ops[0], ast.NotEq)
                    and len(node.comparators) == 1
                    and isinstance(node.comparators[0], ast.Constant)
                    and isinstance(node.comparators[0].value, str)
                    and node.comparators[0].value.lower() in _EXCLUSION_VALUES
                ):
                    # Replace NotEq exclusion with True (always-pass)
                    self._removed = True
                    return ast.Constant(value=True)
                return self.generic_visit(node)

        transformer = _RemoveExclusion()
        mutated_tree = transformer.visit(tree)
        if not transformer._removed:
            return None
        ast.fix_missing_locations(mutated_tree)
        return ast.unparse(mutated_tree)


# Public registry for filter mutations
FILTER_OPERATORS: list[MutationOperator] = [
    FM01_RemoveAllFilters(),
    FM04_InvertPredicate(),
    FM05_BoundaryFlip(),
    FM06_WrongLiteral(),
    FM09_IncludeExcludedStatus(),
]
