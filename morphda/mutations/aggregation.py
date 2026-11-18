"""
Aggregation and statistical mutation operators: AM-01 through AM-14.
"""

from __future__ import annotations

import ast

from morphda.mutations.base import MutationOperator


class _AttributeSwapper(ast.NodeTransformer):
    """Swap one method call attribute name; stops after first match."""

    def __init__(self, from_name: str, to_name: str) -> None:
        self.from_name = from_name
        self.to_name = to_name
        self._swapped = False

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if not self._swapped and node.attr == self.from_name:
            node.attr = self.to_name
            self._swapped = True
        return self.generic_visit(node)


def _swap_method(source: str, from_name: str, to_name: str) -> str | None:
    tree = ast.parse(source)
    transformer = _AttributeSwapper(from_name, to_name)
    mutated = transformer.visit(tree)
    if not transformer._swapped:
        return None
    ast.fix_missing_locations(mutated)
    return ast.unparse(mutated)


class AM01_SumToMean(MutationOperator):
    """AM-01: Replace .mean() with .sum()"""
    operator_id = "AM-01"
    mutation_family = "aggregation"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method(reference_source, "mean", "sum")


class AM01b_MeanToSum(MutationOperator):
    """AM-01b: Replace .sum() with .mean()"""
    operator_id = "AM-01b"
    mutation_family = "aggregation"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method(reference_source, "sum", "mean")


class AM02_MedianToMean(MutationOperator):
    """AM-02: Replace .median() with .mean()"""
    operator_id = "AM-02"
    mutation_family = "aggregation"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method(reference_source, "median", "mean")


class AM02b_MeanToMedian(MutationOperator):
    """AM-02b: Replace .mean() with .median()"""
    operator_id = "AM-02b"
    mutation_family = "aggregation"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method(reference_source, "mean", "median")


class AM03_NuniqueToCount(MutationOperator):
    """AM-03: Replace .nunique() with .count() (distinct -> total count)"""
    operator_id = "AM-03"
    mutation_family = "aggregation"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method(reference_source, "nunique", "count")


class AM03b_CountToNunique(MutationOperator):
    """AM-03b: Replace .count() with .nunique()"""
    operator_id = "AM-03b"
    mutation_family = "aggregation"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method(reference_source, "count", "nunique")


class AM04_MaxToMin(MutationOperator):
    """AM-04: Replace .max() with .min()"""
    operator_id = "AM-04"
    mutation_family = "aggregation"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method(reference_source, "max", "min")


class AM04b_MinToMax(MutationOperator):
    """AM-04b: Replace .min() with .max()"""
    operator_id = "AM-04b"
    mutation_family = "aggregation"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method(reference_source, "min", "max")


class AM10_AbsoluteVsRelativeChange(MutationOperator):
    """
    AM-10: Convert percentage change to absolute change.
    Targets: (a - b) / b pattern -> (a - b)
    """
    operator_id = "AM-10"
    mutation_family = "aggregation"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _RemoveDivision(ast.NodeTransformer):
            def __init__(self) -> None:
                self._removed = False

            def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
                # Look for (a - b) / b pattern
                if (
                    not self._removed
                    and isinstance(node.op, ast.Div)
                    and isinstance(node.left, ast.BinOp)
                    and isinstance(node.left.op, ast.Sub)
                ):
                    self._removed = True
                    return node.left  # Drop the division
                return self.generic_visit(node)

        transformer = _RemoveDivision()
        mutated = transformer.visit(tree)
        if not transformer._removed:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


class AM14_NullRuleChange(MutationOperator):
    """AM-14: Change skipna=True to skipna=False (or vice versa)."""
    operator_id = "AM-14"
    mutation_family = "aggregation"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _FlipSkipna(ast.NodeTransformer):
            def __init__(self) -> None:
                self._flipped = False

            def visit_keyword(self, node: ast.keyword) -> ast.AST:
                if not self._flipped and node.arg == "skipna":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                        node.value.value = not node.value.value
                        self._flipped = True
                return node

        transformer = _FlipSkipna()
        mutated = transformer.visit(tree)
        if not transformer._flipped:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


# Public registry
AGGREGATION_OPERATORS: list[MutationOperator] = [
    AM01_SumToMean(),
    AM01b_MeanToSum(),
    AM02_MedianToMean(),
    AM02b_MeanToMedian(),
    AM03_NuniqueToCount(),
    AM03b_CountToNunique(),
    AM04_MaxToMin(),
    AM04b_MinToMax(),
    AM10_AbsoluteVsRelativeChange(),
    AM14_NullRuleChange(),
]
