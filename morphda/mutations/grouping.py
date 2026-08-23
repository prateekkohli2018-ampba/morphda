"""
Grouping and ranking mutation operators: GM-01 through GM-06, RM-01 through RM-08.
"""

from __future__ import annotations

import ast

from morphda.mutations.base import MutationOperator


def _swap_method(source: str, from_name: str, to_name: str) -> str | None:
    tree = ast.parse(source)

    class _Swapper(ast.NodeTransformer):
        def __init__(self):
            self._done = False

        def visit_Attribute(self, node):
            if not self._done and node.attr == from_name:
                node.attr = to_name
                self._done = True
            return self.generic_visit(node)

    t = _Swapper()
    mutated = t.visit(tree)
    if not t._done:
        return None
    ast.fix_missing_locations(mutated)
    return ast.unparse(mutated)


class RM01_SortDirectionReversal(MutationOperator):
    """RM-01: Flip sort direction (ascending=False → True or vice versa)."""

    operator_id = "RM-01"
    mutation_family = "ranking"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _FlipAscending(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_keyword(self, node):
                if not self._done and node.arg == "ascending":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                        node.value.value = not node.value.value
                        self._done = True
                return node

        transformer = _FlipAscending()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


class RM02_HeadToTail(MutationOperator):
    """RM-02: Replace .head(k) with .tail(k) (top-k → bottom-k)."""

    operator_id = "RM-02"
    mutation_family = "ranking"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method(reference_source, "head", "tail")


class RM02b_TailToHead(MutationOperator):
    """RM-02b: Replace .tail(k) with .head(k)."""

    operator_id = "RM-02b"
    mutation_family = "ranking"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method(reference_source, "tail", "head")


class RM04_MissingTieBreak(MutationOperator):
    """RM-04: Remove secondary sort key from multi-column sort_values call."""

    operator_id = "RM-04"
    mutation_family = "ranking"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _RemoveSecondSortKey(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_Call(self, node):
                if (
                    not self._done
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "sort_values"
                ):
                    # Look for a list of sort keys
                    if node.args and isinstance(node.args[0], ast.List):
                        elts = node.args[0].elts
                        if len(elts) >= 2:
                            node.args[0].elts = elts[:1]
                            self._done = True
                return self.generic_visit(node)

        transformer = _RemoveSecondSortKey()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


class GM01_WrongGroupingKey(MutationOperator):
    """GM-01: Swap one groupby column for an adjacent column in the DataFrame."""

    operator_id = "GM-01"
    mutation_family = "grouping"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _SwapGroupKey(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_Call(self, node):
                if (
                    not self._done
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "groupby"
                    and node.args
                ):
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        # Change to a confounding column name: append "_id" or prefix "sub_"
                        arg.value = arg.value + "_id"
                        self._done = True
                    elif isinstance(arg, ast.List) and arg.elts:
                        first = arg.elts[0]
                        if isinstance(first, ast.Constant) and isinstance(first.value, str):
                            first.value = first.value + "_id"
                            self._done = True
                return self.generic_visit(node)

        transformer = _SwapGroupKey()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


class GM02_OmitGrouping(MutationOperator):
    """GM-02: Remove groupby entirely (aggregate globally)."""

    operator_id = "GM-02"
    mutation_family = "grouping"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _RemoveGroupby(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_Call(self, node):
                if (
                    not self._done
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "groupby"
                ):
                    # Replace df.groupby(X) with df
                    self._done = True
                    return node.func.value
                return self.generic_visit(node)

        transformer = _RemoveGroupby()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


class GM03_ExtraGroupingKey(MutationOperator):
    """GM-03: Add an extra grouping key (finer granularity than intended)."""

    operator_id = "GM-03"
    mutation_family = "grouping"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _AddGroupKey(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_Call(self, node):
                if (
                    not self._done
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "groupby"
                    and node.args
                ):
                    arg = node.args[0]
                    extra = ast.Constant(value="region")
                    if isinstance(arg, ast.Constant):
                        node.args[0] = ast.List(
                            elts=[arg, extra], ctx=ast.Load()
                        )
                        self._done = True
                    elif isinstance(arg, ast.List):
                        arg.elts.append(extra)
                        self._done = True
                return self.generic_visit(node)

        transformer = _AddGroupKey()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


# Public registry
GROUPING_OPERATORS: list[MutationOperator] = [
    RM01_SortDirectionReversal(),
    RM02_HeadToTail(),
    RM02b_TailToHead(),
    RM04_MissingTieBreak(),
    GM01_WrongGroupingKey(),
    GM02_OmitGrouping(),
    GM03_ExtraGroupingKey(),
]
