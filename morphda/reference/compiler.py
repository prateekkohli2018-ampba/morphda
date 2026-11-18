"""
Reference program compiler.

Compiles a TaskSpec into a trusted, deterministic analyze(tables) function.
The compiled program is used:
  1. To produce gold answers on all data seeds.
  2. As input to the mutation engine.
  3. As the correctness oracle for LLM-generated candidates.

The compiler dispatches on question_family to emit the right Pandas idioms.
Correctness verified independently with hand-computed fixture tests.

Supported question families:
  Level 1: scalar_agg, filtered_agg
  Level 2: grouped_rank, filtered_grouped_rank, date_filtered_grouped_rank
  Level 3: grouped_ratio_rank, filtered_grouped_ratio_rank
  Level 4: period_comparison_rank (percentage_change across date windows)
  Level 5: cohort_ratio_rank (multi-filter + ratio + threshold + tie-break)
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from morphda.tasks.schema import AggregationSpec, FilterSpec, MetricSpec, TaskSpec


class ReferenceCompilerError(Exception):
    pass


# ─── Public API ───────────────────────────────────────────────────────────────

def compile_task(task_spec: TaskSpec) -> str:
    """
    Compile a TaskSpec into Python source defining analyze(tables).

    Returns:
        Python source string for the reference program.
    """
    body_lines = _compile_body(task_spec)
    parts = [
        "import pandas as pd",
        "import numpy as np",
        "",
        "def analyze(tables: dict) -> object:",
    ]
    for line in body_lines:
        parts.append(("    " + line) if line.strip() else "")
    return "\n".join(parts)


def run_reference(task_spec: TaskSpec, tables: dict[str, pd.DataFrame]) -> Any:
    """Compile and execute the reference program. Returns the gold answer."""
    from morphda.execution.sandbox import execute_program

    source = compile_task(task_spec)
    result = execute_program(source, tables)
    if not result.success:
        raise ReferenceCompilerError(
            f"Reference program failed on task {task_spec.task_id}: {result.exception}"
        )
    return result.output


# ─── Body compiler ────────────────────────────────────────────────────────────

def _compile_body(spec: TaskSpec) -> list[str]:
    """Return lines (no leading indent) for the function body."""
    lines: list[str] = []

    # 1. Load primary table
    primary = spec.inputs[0]
    lines.append(f"df = tables[{primary!r}].copy()")

    # 2. Joins
    for join in spec.joins:
        right = join.right_table
        how = join.join_type
        keys = join.keys
        lines.append(f"_right = tables[{right!r}].copy()")
        if len(keys) == 1:
            lines.append(f"df = df.merge(_right, on={keys[0]!r}, how={how!r})")
        else:
            lines.append(f"df = df.merge(_right, on={keys!r}, how={how!r})")

    # 3. Non-date filters
    for flt in spec.filters:
        lines.append(f"df = df[{_filter_expr(flt, 'df')}]")

    # 4. Date filter — emit BEFORE the date-range split for comparison tasks
    if spec.date and not spec.comparison:
        lines.extend(_emit_date_filter(spec, "df"))

    # 5. Dispatch on metric/comparison complexity
    if spec.comparison and spec.comparison.operation == "percentage_change":
        lines.extend(_compile_period_comparison(spec))
    elif spec.metric.operation == "ratio" and spec.group_by:
        lines.extend(_compile_grouped_ratio(spec))
    elif spec.group_by:
        lines.extend(_compile_grouped_simple(spec))
    else:
        lines.extend(_compile_scalar(spec))

    lines.append("return result")
    return lines


# ─── Compilation paths ────────────────────────────────────────────────────────

def _compile_scalar(spec: TaskSpec) -> list[str]:
    lines = []
    expr = _scalar_metric_expr(spec.metric, "df")
    lines.append(f"result = {expr}")
    return lines


def _compile_grouped_simple(spec: TaskSpec) -> list[str]:
    """Level 2: grouped aggregation + optional ranking (no ratio, no comparison)."""
    lines = []
    group_cols = spec.group_by
    metric = spec.metric
    measure = metric.column

    if not measure:
        raise ReferenceCompilerError(
            f"grouped_simple requires metric.column; got operation={metric.operation!r}"
        )

    agg_fn = _simple_agg_fn(metric.operation)
    lines.append(
        f"_agg = df.groupby({group_cols!r}, observed=True)[{measure!r}].{agg_fn}()"
    )

    # Post-filter (minimum count on the same column)
    if spec.post_filter and spec.post_filter.minimum_denominator is not None:
        thresh = spec.post_filter.minimum_denominator
        lines.append(
            f"_cnt = df.groupby({group_cols!r}, observed=True)[{measure!r}].count()"
        )
        lines.append(f"_agg = _agg[_cnt >= {thresh}]")

    lines.extend(_emit_ranking(spec, "_agg"))
    return lines


def _compile_grouped_ratio(spec: TaskSpec) -> list[str]:
    """Level 3: grouped ratio (count_distinct numerator / count_distinct denominator)."""
    lines = []
    group_cols = spec.group_by
    metric = spec.metric

    if not metric.numerator or not metric.denominator:
        raise ReferenceCompilerError("ratio metric requires numerator and denominator")

    num_col = metric.numerator.column
    den_col = metric.denominator.column
    num_fn  = "nunique" if metric.numerator.operation == "count_distinct" else "count"
    den_fn  = "nunique" if metric.denominator.operation == "count_distinct" else "count"

    lines.append(
        f"_num = df.groupby({group_cols!r}, observed=True)[{num_col!r}].{num_fn}()"
    )
    lines.append(
        f"_den = df.groupby({group_cols!r}, observed=True)[{den_col!r}].{den_fn}()"
    )
    lines.append("_agg = _num / _den")

    # Post-filter on denominator size
    if spec.post_filter and spec.post_filter.minimum_denominator is not None:
        thresh = spec.post_filter.minimum_denominator
        lines.append(f"_agg = _agg[_den >= {thresh}]")

    lines.extend(_emit_ranking(spec, "_agg"))
    return lines


def _compile_period_comparison(spec: TaskSpec) -> list[str]:
    """
    Level 4: percentage_change across two date windows.

    Requires spec.date to have both current_* and previous_* fields.
    Applies non-date filters to both periods, then computes:
      (_curr - _prior) / _prior  per group.
    """
    lines = []
    date = spec.date
    if date is None:
        raise ReferenceCompilerError("period_comparison requires a date spec")
    if not date.previous_start or not date.previous_end:
        raise ReferenceCompilerError(
            "period_comparison requires date.previous_start and previous_end"
        )

    col = date.column
    op_lo = ">=" if date.inclusive_bounds else ">"
    op_hi = "<=" if date.inclusive_bounds else "<"

    lines.append(f"df[{col!r}] = pd.to_datetime(df[{col!r}], format='mixed')")
    lines.append(
        f"_df_curr = df[(df[{col!r}] {op_lo} pd.Timestamp({date.current_start!r})) & "
        f"(df[{col!r}] {op_hi} pd.Timestamp({date.current_end!r}))]"
    )
    lines.append(
        f"_df_prior = df[(df[{col!r}] {op_lo} pd.Timestamp({date.previous_start!r})) & "
        f"(df[{col!r}] {op_hi} pd.Timestamp({date.previous_end!r}))]"
    )

    group_cols = spec.group_by
    metric = spec.metric

    if metric.operation == "ratio" and metric.numerator and metric.denominator:
        num_col = metric.numerator.column
        den_col = metric.denominator.column
        num_fn  = "nunique" if metric.numerator.operation == "count_distinct" else "count"
        den_fn  = "nunique" if metric.denominator.operation == "count_distinct" else "count"

        lines.append(
            f"_curr_num = _df_curr.groupby({group_cols!r}, observed=True)"
            f"[{num_col!r}].{num_fn}()"
        )
        lines.append(
            f"_curr_den = _df_curr.groupby({group_cols!r}, observed=True)"
            f"[{den_col!r}].{den_fn}()"
        )
        lines.append("_curr = _curr_num / _curr_den")
        lines.append(
            f"_prior_num = _df_prior.groupby({group_cols!r}, observed=True)"
            f"[{num_col!r}].{num_fn}()"
        )
        lines.append(
            f"_prior_den = _df_prior.groupby({group_cols!r}, observed=True)"
            f"[{den_col!r}].{den_fn}()"
        )
        lines.append("_prior = _prior_num / _prior_den")

        # Post-filter on current-period denominator support
        if spec.post_filter and spec.post_filter.minimum_denominator is not None:
            thresh = spec.post_filter.minimum_denominator
            lines.append(f"_curr_den = _curr_den[_curr_den >= {thresh}]")
            lines.append("_curr  = _curr[_curr_den.index]")
            lines.append("_prior = _prior.reindex(_curr_den.index)")

    else:
        # Simple metric per period
        if not metric.column:
            raise ReferenceCompilerError("period_comparison with simple metric requires metric.column")
        measure = metric.column
        agg_fn  = _simple_agg_fn(metric.operation)
        lines.append(
            f"_curr  = _df_curr.groupby({group_cols!r}, observed=True)"
            f"[{measure!r}].{agg_fn}()"
        )
        lines.append(
            f"_prior = _df_prior.groupby({group_cols!r}, observed=True)"
            f"[{measure!r}].{agg_fn}()"
        )

    # Percentage change: only for groups present in both periods
    lines.append("_prior = _prior.reindex(_curr.index)")
    lines.append("_agg = (_curr - _prior) / _prior.abs()")
    lines.append("_agg = _agg.dropna()")

    lines.extend(_emit_ranking(spec, "_agg"))
    return lines


# ─── Ranking emitter ──────────────────────────────────────────────────────────

def _emit_ranking(spec: TaskSpec, agg_var: str) -> list[str]:
    """Emit sort + selection lines for ranked output."""
    lines = []
    if not spec.ranking:
        lines.append(f"result = {agg_var}")
        return lines

    ranking = spec.ranking
    ascending = ranking.direction == "ascending"
    k = ranking.k

    if not ranking.tie_break:
        # Simple single-column sort
        lines.append(f"_agg = {agg_var}.sort_values(ascending={ascending})")
    else:
        # Multi-column sort: need a DataFrame
        # Build a DataFrame with the metric and each tie-break column
        group_col = spec.group_by[0] if spec.group_by else None
        lines.append(f"_rank_df = {agg_var}.reset_index()")
        lines.append(f"_rank_df = _rank_df.rename(columns={{_rank_df.columns[-1]: '_metric'}})")

        for tb_dict in ranking.tie_break:
            for tb_col, tb_dir in tb_dict.items():
                # Recompute the tie-break column from the (already filtered) df
                lines.append(
                    f"_rank_df[{tb_col!r}] = _rank_df.set_index("
                    f"_rank_df.columns[0]).index.map("
                    f"df.groupby({spec.group_by!r}, observed=True)"
                    f"[{tb_col!r}].sum())"
                )

        sort_cols   = ["_metric"]
        sort_ascend = [ascending]
        for tb_dict in ranking.tie_break:
            for tb_col, tb_dir in tb_dict.items():
                sort_cols.append(tb_col)
                sort_ascend.append(tb_dir == "ascending")

        lines.append(
            f"_rank_df = _rank_df.sort_values("
            f"by={sort_cols!r}, ascending={sort_ascend!r})"
        )
        group_col_name = spec.group_by[0] if spec.group_by else "_index"
        lines.append(f"_agg = _rank_df.set_index({group_col_name!r})['_metric']")

    if k == 1:
        lines.append("result = _agg.index[0]")
    else:
        lines.append(f"result = list(_agg.index[:{k}])")

    return lines


# ─── Date filter emitter ──────────────────────────────────────────────────────

def _emit_date_filter(spec: TaskSpec, df_var: str) -> list[str]:
    date = spec.date
    col = date.column
    start = date.current_start
    end   = date.current_end
    op_lo = ">=" if date.inclusive_bounds else ">"
    op_hi = "<=" if date.inclusive_bounds else "<"
    return [
        f"{df_var}[{col!r}] = pd.to_datetime({df_var}[{col!r}], format='mixed')",
        f"{df_var} = {df_var}[({df_var}[{col!r}] {op_lo} pd.Timestamp({start!r})) & "
        f"({df_var}[{col!r}] {op_hi} pd.Timestamp({end!r}))]",
    ]


# ─── Helper functions ─────────────────────────────────────────────────────────

def _filter_expr(flt: FilterSpec, df_var: str) -> str:
    col, op, val, vals = flt.column, flt.operator, flt.value, flt.values
    if op == "equal":        return f"{df_var}[{col!r}] == {val!r}"
    if op == "not_equal":    return f"{df_var}[{col!r}] != {val!r}"
    if op == "greater":      return f"{df_var}[{col!r}] > {val!r}"
    if op == "greater_equal":return f"{df_var}[{col!r}] >= {val!r}"
    if op == "less":         return f"{df_var}[{col!r}] < {val!r}"
    if op == "less_equal":   return f"{df_var}[{col!r}] <= {val!r}"
    if op == "in":           return f"{df_var}[{col!r}].isin({vals!r})"
    if op == "not_in":       return f"~{df_var}[{col!r}].isin({vals!r})"
    if op == "is_null":      return f"{df_var}[{col!r}].isna()"
    if op == "is_not_null":  return f"{df_var}[{col!r}].notna()"
    raise ReferenceCompilerError(f"Unknown filter operator: {op!r}")


def _simple_agg_fn(operation: str) -> str:
    mapping = {
        "sum": "sum", "mean": "mean", "median": "median",
        "count": "count", "count_distinct": "nunique",
        "min": "min", "max": "max", "std": "std", "variance": "var",
    }
    if operation not in mapping:
        raise ReferenceCompilerError(f"Unsupported simple aggregation: {operation!r}")
    return mapping[operation]


def _scalar_metric_expr(metric: MetricSpec, df_var: str) -> str:
    op  = metric.operation
    col = metric.column

    if col and op in ("sum", "mean", "median", "count", "min", "max", "std", "variance"):
        fn = _simple_agg_fn(op)
        return f"{df_var}[{col!r}].{fn}()"

    if op == "count_distinct" and col:
        return f"{df_var}[{col!r}].nunique()"

    if op == "ratio" and metric.numerator and metric.denominator:
        num_fn = "nunique" if metric.numerator.operation == "count_distinct" else "count"
        den_fn = "nunique" if metric.denominator.operation == "count_distinct" else "count"
        n = metric.numerator.column
        d = metric.denominator.column
        return f"({df_var}[{n!r}].{num_fn}() / {df_var}[{d!r}].{den_fn}())"

    raise ReferenceCompilerError(f"Cannot compile scalar metric: op={op!r}, col={col!r}")
