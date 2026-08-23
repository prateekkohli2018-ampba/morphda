"""
Task factory — generates structured TaskSpec objects programmatically.

Each difficulty level expands on the previous by adding compositional elements.
The factory produces tasks that are:
  - Hard through analytical composition, NOT confusing wording
  - Testable by the reference compiler
  - Diverse across scenarios and operator families

Operator coverage targets (paper Section 33):
  Level 1 (10%):  scalar_agg, filtered_agg
  Level 2 (20%):  grouped_rank, date_filtered_grouped_rank
  Level 3 (30%):  grouped_ratio_rank, filtered_grouped_ratio_rank
  Level 4 (25%):  period_comparison_rank (pct_change)
  Level 5 (15%):  multi-filter + ratio + threshold + tie-break + join
"""

from __future__ import annotations

import itertools
from typing import Any

from morphda.tasks.schema import (
    AggregationSpec,
    ComparisonSpec,
    DateScope,
    FilterSpec,
    JoinSpec,
    MetricSpec,
    PostFilterSpec,
    RankingSpec,
    TaskSpec,
)

# ─── Scenario vocabulary ──────────────────────────────────────────────────────

_RETAIL_FILTERS = [
    FilterSpec(column="order_status", operator="not_equal", value="cancelled"),
    FilterSpec(column="customer_type", operator="equal", value="new"),
    FilterSpec(column="region", operator="equal", value="US"),
    FilterSpec(column="order_status", operator="not_in", values=["cancelled", "refunded"]),
]

_RETAIL_DATES_2025 = DateScope(
    column="order_date",
    current_start="2025-01-01", current_end="2025-12-31",
    inclusive_bounds=True,
)
_RETAIL_DATES_Q2 = DateScope(
    column="order_date",
    current_start="2025-04-01", current_end="2025-06-30",
    previous_start="2024-04-01", previous_end="2024-06-30",
    inclusive_bounds=True,
)
_RETAIL_DATES_Q3 = DateScope(
    column="order_date",
    current_start="2025-07-01", current_end="2025-09-30",
    previous_start="2024-07-01", previous_end="2024-09-30",
    inclusive_bounds=True,
)

_WEB_DATES_2025 = DateScope(
    column="event_date",
    current_start="2025-01-01", current_end="2025-12-31",
    inclusive_bounds=True,
)
_WEB_DATES_Q2 = DateScope(
    column="event_date",
    current_start="2025-04-01", current_end="2025-06-30",
    previous_start="2024-04-01", previous_end="2024-06-30",
    inclusive_bounds=True,
)
_WEB_DATES_Q3 = DateScope(
    column="event_date",
    current_start="2025-07-01", current_end="2025-09-30",
    previous_start="2024-07-01", previous_end="2024-09-30",
    inclusive_bounds=True,
)

_WEB_FILTERS = [
    FilterSpec(column="customer_type", operator="equal", value="new"),
    FilterSpec(column="traffic_source", operator="not_equal", value="test"),
    FilterSpec(column="device", operator="equal", value="mobile"),
]

_MARKET_FILTERS = [
    FilterSpec(column="order_status", operator="not_equal", value="cancelled"),
    FilterSpec(column="fulfillment", operator="equal", value="marketplace"),
]


# ─── Level 1: Scalar aggregations ────────────────────────────────────────────

def _level1_tasks(scenario_id: str, table: str, start_idx: int) -> list[TaskSpec]:
    tasks = []
    date_col = "order_date" if "retail" in scenario_id or "market" in scenario_id else "event_date"
    rev_col  = "revenue" if "retail" in scenario_id else ("gmv" if "market" in scenario_id else "page_views")

    # Entity/ID column for count_distinct: market uses seller_id, others use customer_id
    entity_col = "seller_id" if "market" in scenario_id else "customer_id"
    for i, (op, col) in enumerate([
        ("sum",            rev_col),
        ("mean",           rev_col),
        ("count_distinct", entity_col),
    ]):
        task_id = f"{scenario_id}_l1_{start_idx + i:03d}"
        tasks.append(TaskSpec(
            task_id=task_id,
            scenario_id=scenario_id,
            question_family=f"scalar_{op}",
            difficulty_level=1,
            inputs=[table],
            metric=MetricSpec(name=f"{op}_{col}", operation=op, column=col),
            output_type="scalar",
            canonical_question=f"What is the {op.replace('_', ' ')} of {col}?",
        ))
    return tasks


# ─── Level 2: Grouped ranking ─────────────────────────────────────────────────

def _level2_tasks(scenario_id: str, table: str, start_idx: int) -> list[TaskSpec]:
    tasks = []
    # Correct measure column per scenario
    if "web" in scenario_id:
        rev_col = "page_views"
    elif "market" in scenario_id:
        rev_col = "gmv"
    else:
        rev_col = "revenue"

    group_col = "category"
    filter_list = _WEB_FILTERS[:1] if "web" in scenario_id else _RETAIL_FILTERS[:1]
    date_scope  = _WEB_DATES_2025 if "web" in scenario_id else _RETAIL_DATES_2025

    variants = [
        # (filters, date, direction, k, question)
        ([], None, "descending", 1,
         f"Which {group_col} had the highest total {rev_col}?"),
        (filter_list, None, "descending", 1,
         f"Among eligible records, which {group_col} had the highest total {rev_col}?"),
        ([], date_scope, "descending", 1,
         f"Which {group_col} had the highest total {rev_col} in the current period?"),
        (filter_list, date_scope, "descending", 1,
         f"Which {group_col} had the highest total {rev_col} among eligible records in the current period?"),
        (filter_list, date_scope, "ascending", 1,
         f"Which {group_col} had the lowest total {rev_col} among eligible records?"),
        (filter_list, date_scope, "descending", 3,
         f"What are the top 3 {group_col}s by total {rev_col} among eligible records?"),
    ]

    for i, (filters, date, direction, k, question) in enumerate(variants):
        output_type = "ranked_list" if k > 1 else "label"
        tasks.append(TaskSpec(
            task_id=f"{scenario_id}_l2_{start_idx + i:03d}",
            scenario_id=scenario_id,
            question_family="grouped_rank",
            difficulty_level=2,
            inputs=[table],
            filters=filters,
            date=date,
            metric=MetricSpec(name=f"total_{rev_col}", operation="sum", column=rev_col),
            group_by=[group_col],
            ranking=RankingSpec(direction=direction, k=k),
            output_type=output_type,
            canonical_question=question,
        ))
    return tasks


# ─── Level 3: Grouped ratio (conversion rate) ─────────────────────────────────

def _level3_tasks(scenario_id: str, table: str, start_idx: int) -> list[TaskSpec]:
    tasks = []

    if "web" in scenario_id:
        # Session-based conversion rate
        date_scope = _WEB_DATES_2025
        variants = [
            (_WEB_FILTERS[:1], None, 50,
             "Which category had the highest conversion rate among new users?"),
            (_WEB_FILTERS[:1], date_scope, 100,
             "Which category had the highest conversion rate among new users in 2025?"),
            (_WEB_FILTERS[:2], date_scope, 100,
             "Which category had the highest conversion rate excluding test traffic in 2025?"),
            (_WEB_FILTERS[2:3] + _WEB_FILTERS[:1], date_scope, 50,
             "Which category had the highest mobile conversion rate among new users in 2025?"),
        ]
        for i, (filters, date, thresh, question) in enumerate(variants):
            tasks.append(TaskSpec(
                task_id=f"{scenario_id}_l3_{start_idx + i:03d}",
                scenario_id=scenario_id,
                question_family="grouped_ratio_rank",
                difficulty_level=3,
                inputs=[table],
                filters=filters,
                date=date,
                metric=MetricSpec(
                    name="conversion_rate",
                    operation="ratio",
                    numerator=AggregationSpec(operation="count_distinct", column="customer_id"),
                    denominator=AggregationSpec(operation="count_distinct", column="session_id"),
                ),
                group_by=["category"],
                post_filter=PostFilterSpec(minimum_denominator=thresh),
                ranking=RankingSpec(direction="descending", k=1),
                output_type="label",
                canonical_question=question,
            ))
    else:
        # Revenue-per-order ratio
        rev_col = "gmv" if "market" in scenario_id else "revenue"
        filters = _MARKET_FILTERS if "market" in scenario_id else _RETAIL_FILTERS[:1]
        date_scope = _RETAIL_DATES_2025

        tasks.append(TaskSpec(
            task_id=f"{scenario_id}_l3_{start_idx:03d}",
            scenario_id=scenario_id,
            question_family="grouped_ratio_rank",
            difficulty_level=3,
            inputs=[table],
            filters=filters,
            date=date_scope,
            metric=MetricSpec(
                name="avg_order_value",
                operation="mean",
                column=rev_col,
            ),
            group_by=["category"],
            post_filter=PostFilterSpec(minimum_denominator=30),
            ranking=RankingSpec(direction="descending", k=1),
            output_type="label",
            canonical_question=(
                f"Which category had the highest average {rev_col} per eligible order in 2025?"
            ),
        ))

    return tasks


# ─── Level 4: Period comparison ───────────────────────────────────────────────

def _level4_tasks(scenario_id: str, table: str, start_idx: int) -> list[TaskSpec]:
    tasks = []

    date_variants = [
        ("Q2", _WEB_DATES_Q2 if "web" in scenario_id else _RETAIL_DATES_Q2),
        ("Q3", _WEB_DATES_Q3 if "web" in scenario_id else _RETAIL_DATES_Q3),
    ]
    filters = _WEB_FILTERS[:1] if "web" in scenario_id else _RETAIL_FILTERS[:1]
    if "web" in scenario_id:
        rev_col = "page_views"
    elif "market" in scenario_id:
        rev_col = "gmv"
    else:
        rev_col = "revenue"

    for i, (quarter, date_scope) in enumerate(date_variants):
        tasks.append(TaskSpec(
            task_id=f"{scenario_id}_l4_{start_idx + i:03d}",
            scenario_id=scenario_id,
            question_family="period_comparison_rank",
            difficulty_level=4,
            inputs=[table],
            filters=filters,
            date=date_scope,
            metric=MetricSpec(name=f"total_{rev_col}", operation="sum", column=rev_col),
            comparison=ComparisonSpec(operation="percentage_change"),
            group_by=["category"],
            ranking=RankingSpec(direction="descending", k=1),
            output_type="label",
            canonical_question=(
                f"Which category had the largest year-over-year percentage increase "
                f"in total {rev_col} in {quarter} 2025?"
            ),
        ))

    return tasks


# ─── Level 5: Multi-stage (ratio + comparison + threshold + tie-break) ────────

def _level5_tasks(scenario_id: str, table: str, start_idx: int) -> list[TaskSpec]:
    tasks = []
    if "web" not in scenario_id:
        return tasks  # Level 5 web-scenario only for now

    # YoY conversion-rate pct change, with session threshold and tie-break
    for i, (quarter, date_scope) in enumerate([
        ("Q2", _WEB_DATES_Q2),
        ("Q3", _WEB_DATES_Q3),
    ]):
        tasks.append(TaskSpec(
            task_id=f"{scenario_id}_l5_{start_idx + i:03d}",
            scenario_id=scenario_id,
            question_family="cohort_ratio_rank",
            difficulty_level=5,
            inputs=[table],
            filters=_WEB_FILTERS[:2],  # new customers, no test traffic
            date=date_scope,
            metric=MetricSpec(
                name="conversion_rate",
                operation="ratio",
                numerator=AggregationSpec(operation="count_distinct", column="customer_id"),
                denominator=AggregationSpec(operation="count_distinct", column="session_id"),
            ),
            comparison=ComparisonSpec(operation="percentage_change"),
            group_by=["category"],
            post_filter=PostFilterSpec(minimum_denominator=30),
            ranking=RankingSpec(direction="descending", k=1),
            output_type="label",
            canonical_question=(
                f"Among new customers (excluding test traffic) with at least 30 eligible sessions, "
                f"which category had the largest year-over-year percentage change in "
                f"conversion rate in {quarter} 2025?"
            ),
        ))
    return tasks


# ─── Public factory ───────────────────────────────────────────────────────────

# ─── Scenario vocabulary for new scenarios ───────────────────────────────────

def _scenario_vocab(scenario_id: str) -> dict:
    """Return the key column names for a given scenario."""
    return {
        "retail01":   {"table": "orders",        "measure": "revenue",      "entity": "customer_id", "date": "order_date",   "group": "category", "status_col": "order_status", "excl_val": "cancelled"},
        "web01":      {"table": "sessions",       "measure": "page_views",   "entity": "customer_id", "date": "event_date",   "group": "category", "status_col": "traffic_source","excl_val": "test"},
        "market01":   {"table": "seller_orders",  "measure": "gmv",          "entity": "seller_id",   "date": "order_date",   "group": "category", "status_col": "order_status", "excl_val": "cancelled"},
        "saas01":     {"table": "subscriptions",  "measure": "mrr",          "entity": "customer_id", "date": "start_date",   "group": "plan",     "status_col": "status",       "excl_val": "churned"},
        "mktg01":     {"table": "campaigns",      "measure": "spend",        "entity": "campaign_id", "date": "event_date",   "group": "channel",  "status_col": "status",       "excl_val": "test"},
        "payments01": {"table": "transactions",   "measure": "amount",       "entity": "customer_id", "date": "txn_date",     "group": "payment_method", "status_col": "status", "excl_val": "failed"},
        "ops01":      {"table": "shipments",      "measure": "cost",         "entity": "order_id",    "date": "ship_date",    "group": "category", "status_col": "status",       "excl_val": "lost"},
        "support01":  {"table": "tickets",        "measure": "resolution_hours", "entity": "customer_id", "date": "created_date", "group": "category", "status_col": "status", "excl_val": "open"},
    }.get(scenario_id, {})


def _generic_date_scope(vocab: dict, quarter: str) -> DateScope:
    date_col = vocab.get("date", "event_date")
    if quarter == "2025":
        return DateScope(column=date_col, current_start="2025-01-01", current_end="2025-12-31", inclusive_bounds=True)
    if quarter == "Q2":
        return DateScope(column=date_col, current_start="2025-04-01", current_end="2025-06-30",
                         previous_start="2024-04-01", previous_end="2024-06-30", inclusive_bounds=True)
    if quarter == "Q3":
        return DateScope(column=date_col, current_start="2025-07-01", current_end="2025-09-30",
                         previous_start="2024-07-01", previous_end="2024-09-30", inclusive_bounds=True)
    return DateScope(column=date_col, current_start="2025-01-01", current_end="2025-12-31", inclusive_bounds=True)


def _generic_tasks(scenario_id: str, start_idx: int) -> list[TaskSpec]:
    """Generate L1-L4 tasks for any scenario using its vocabulary."""
    vocab = _scenario_vocab(scenario_id)
    if not vocab:
        return []
    table     = vocab["table"]
    measure   = vocab["measure"]
    entity    = vocab["entity"]
    group     = vocab["group"]
    status_col = vocab["status_col"]
    excl_val  = vocab["excl_val"]
    date_2025 = _generic_date_scope(vocab, "2025")
    date_q2   = _generic_date_scope(vocab, "Q2")
    date_q3   = _generic_date_scope(vocab, "Q3")
    status_flt = FilterSpec(column=status_col, operator="not_equal", value=excl_val)

    tasks = []
    idx = start_idx

    # L1: scalar aggregations (3 tasks)
    for op, col in [("sum", measure), ("mean", measure), ("count_distinct", entity)]:
        tasks.append(TaskSpec(
            task_id=f"{scenario_id}_l1_{idx:03d}", scenario_id=scenario_id,
            question_family=f"scalar_{op}", difficulty_level=1,
            inputs=[table],
            metric=MetricSpec(name=f"{op}_{col}", operation=op, column=col),
            output_type="scalar",
            canonical_question=f"What is the {op.replace('_',' ')} of {col}?",
        ))
        idx += 1

    # L2: grouped ranking (6 tasks)
    for filters, date, direction, k, q in [
        ([],            None,     "descending", 1, f"Which {group} had the highest total {measure}?"),
        ([status_flt],  None,     "descending", 1, f"Among eligible records, which {group} had the highest total {measure}?"),
        ([],            date_2025,"descending", 1, f"Which {group} had the highest total {measure} in 2025?"),
        ([status_flt],  date_2025,"descending", 1, f"Which {group} had the highest total {measure} among eligible records in 2025?"),
        ([status_flt],  date_2025,"ascending",  1, f"Which {group} had the lowest total {measure} among eligible records in 2025?"),
        ([status_flt],  date_2025,"descending", 3, f"What are the top 3 {group}s by total {measure} among eligible records in 2025?"),
    ]:
        tasks.append(TaskSpec(
            task_id=f"{scenario_id}_l2_{idx:03d}", scenario_id=scenario_id,
            question_family="grouped_rank", difficulty_level=2,
            inputs=[table], filters=filters, date=date,
            metric=MetricSpec(name=f"total_{measure}", operation="sum", column=measure),
            group_by=[group],
            ranking=RankingSpec(direction=direction, k=k),
            output_type="ranked_list" if k > 1 else "label",
            canonical_question=q,
        ))
        idx += 1

    # L3: mean per group with threshold (1 task)
    tasks.append(TaskSpec(
        task_id=f"{scenario_id}_l3_{idx:03d}", scenario_id=scenario_id,
        question_family="grouped_mean_rank", difficulty_level=3,
        inputs=[table], filters=[status_flt], date=date_2025,
        metric=MetricSpec(name=f"avg_{measure}", operation="mean", column=measure),
        group_by=[group],
        post_filter=PostFilterSpec(minimum_denominator=30),
        ranking=RankingSpec(direction="descending", k=1),
        output_type="label",
        canonical_question=(
            f"Among eligible records in 2025, which {group} had the highest "
            f"average {measure} (at least 30 records)?"
        ),
    ))
    idx += 1

    # L4: period comparison (2 tasks)
    for quarter, date_scope in [("Q2", date_q2), ("Q3", date_q3)]:
        tasks.append(TaskSpec(
            task_id=f"{scenario_id}_l4_{idx:03d}", scenario_id=scenario_id,
            question_family="period_comparison_rank", difficulty_level=4,
            inputs=[table], filters=[status_flt], date=date_scope,
            metric=MetricSpec(name=f"total_{measure}", operation="sum", column=measure),
            comparison=ComparisonSpec(operation="percentage_change"),
            group_by=[group],
            ranking=RankingSpec(direction="descending", k=1),
            output_type="label",
            canonical_question=(
                f"Which {group} had the largest year-over-year percentage increase "
                f"in total {measure} in {quarter} 2025?"
            ),
        ))
        idx += 1

    return tasks


SCENARIO_TABLE_MAP: dict[str, str] = {
    "retail01":   "orders",
    "web01":      "sessions",
    "market01":   "seller_orders",
    "saas01":     "subscriptions",
    "mktg01":     "campaigns",
    "payments01": "transactions",
    "ops01":      "shipments",
    "support01":  "tickets",
}

# Difficulty level → count per scenario (approximately)
DIFFICULTY_COUNTS: dict[int, int] = {1: 3, 2: 6, 3: 4, 4: 2, 5: 2}


def generate_task_set(
    scenarios: list[str] | None = None,
) -> list[TaskSpec]:
    """
    Generate the full structured task set for the benchmark.

    Args:
        scenarios: List of scenario IDs. Defaults to all built-in scenarios.

    Returns:
        List of TaskSpec objects, ordered by difficulty level.
    """
    if scenarios is None:
        scenarios = list(SCENARIO_TABLE_MAP.keys())

    all_tasks: list[TaskSpec] = []
    idx = 0

    # Original 3 scenarios: use the specialized generators
    original_scenarios = {"retail01", "web01", "market01"}

    for scenario_id in scenarios:
        table = SCENARIO_TABLE_MAP.get(scenario_id, scenario_id)

        if scenario_id in original_scenarios:
            l1 = _level1_tasks(scenario_id, table, idx)
            idx += len(l1)
            l2 = _level2_tasks(scenario_id, table, idx)
            idx += len(l2)
            l3 = _level3_tasks(scenario_id, table, idx)
            idx += len(l3)
            l4 = _level4_tasks(scenario_id, table, idx)
            idx += len(l4)
            l5 = _level5_tasks(scenario_id, table, idx)
            idx += len(l5)
            all_tasks.extend(l1 + l2 + l3 + l4 + l5)
        else:
            # New scenarios: use the generic generator
            generic = _generic_tasks(scenario_id, idx)
            idx += len(generic)
            all_tasks.extend(generic)

    return all_tasks


def task_set_summary(tasks: list[TaskSpec]) -> str:
    from collections import Counter
    by_level = Counter(t.difficulty_level for t in tasks)
    by_scenario = Counter(t.scenario_id for t in tasks)
    by_family = Counter(t.question_family for t in tasks)
    lines = [
        f"Total tasks: {len(tasks)}",
        f"By difficulty: " + ", ".join(f"L{k}={v}" for k, v in sorted(by_level.items())),
        f"By scenario: " + ", ".join(f"{k}={v}" for k, v in sorted(by_scenario.items())),
        f"By family: " + ", ".join(f"{k}={v}" for k, v in sorted(by_family.items())),
    ]
    return "\n".join(lines)
