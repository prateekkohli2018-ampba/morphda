"""
Join mutation operators: JM-01 through JM-08.
"""

from __future__ import annotations

import ast

from morphda.mutations.base import MutationOperator


def _swap_method_arg(source: str, method: str, from_val: str, to_val: str) -> str | None:
    """Swap a string keyword argument value in a specific method call."""
    tree = ast.parse(source)

    class _SwapKwarg(ast.NodeTransformer):
        def __init__(self):
            self._done = False

        def visit_Call(self, node):
            if (
                not self._done
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == method
            ):
                for kw in node.keywords:
                    if (
                        isinstance(kw.value, ast.Constant)
                        and kw.value.value == from_val
                    ):
                        kw.value.value = to_val
                        self._done = True
                        break
            return self.generic_visit(node)

    transformer = _SwapKwarg()
    mutated = transformer.visit(tree)
    if not transformer._done:
        return None
    ast.fix_missing_locations(mutated)
    return ast.unparse(mutated)


class JM01_InnerToLeftJoin(MutationOperator):
    """JM-01: Change inner join to left join (retains unmatched facts)."""

    operator_id = "JM-01"
    mutation_family = "join"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method_arg(reference_source, "merge", "inner", "left")


class JM02_LeftToInnerJoin(MutationOperator):
    """JM-02: Change left join to inner join (drops unmatched facts)."""

    operator_id = "JM-02"
    mutation_family = "join"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        return _swap_method_arg(reference_source, "merge", "left", "inner")


class JM07_JoinOmitted(MutationOperator):
    """JM-07: Remove the merge call entirely — analyze only the primary table."""

    operator_id = "JM-07"
    mutation_family = "join"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _RemoveMerge(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_Assign(self, node):
                # Look for: df = df.merge(...)
                if (
                    not self._done
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "merge"
                ):
                    self._done = True
                    # Remove this assignment by returning None (filtered out later)
                    return None  # type: ignore[return-value]
                return self.generic_visit(node)

            def generic_visit(self, node):
                if isinstance(node, (ast.Module, ast.FunctionDef)):
                    new_body = []
                    for child in node.body:
                        result = self.visit(child)
                        if result is not None:
                            new_body.append(result)
                    node.body = new_body
                    return node
                return super().generic_visit(node)

        transformer = _RemoveMerge()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


class JM08_PostJoinDuplicateRemovalOmitted(MutationOperator):
    """JM-08: Remove drop_duplicates call after a merge."""

    operator_id = "JM-08"
    mutation_family = "join"

    def mutate(self, reference_source: str, task_spec: object) -> str | None:
        tree = ast.parse(reference_source)

        class _RemoveDropDups(ast.NodeTransformer):
            def __init__(self):
                self._done = False

            def visit_Call(self, node):
                if (
                    not self._done
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "drop_duplicates"
                ):
                    self._done = True
                    return node.func.value  # Replace call with the DataFrame itself
                return self.generic_visit(node)

        transformer = _RemoveDropDups()
        mutated = transformer.visit(tree)
        if not transformer._done:
            return None
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)


JOIN_OPERATORS: list[MutationOperator] = [
    JM01_InnerToLeftJoin(),
    JM02_LeftToInnerJoin(),
    JM07_JoinOmitted(),
    JM08_PostJoinDuplicateRemovalOmitted(),
]
