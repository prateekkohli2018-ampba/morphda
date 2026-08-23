"""
Seeded dataset generators for MORPH-DA benchmark tasks.

Each scenario produces one or more related DataFrames with realistic
but non-proprietary data. Generators support:
  - skewed numeric distributions, heavy tails, outliers
  - rare groups, near ties
  - duplicated IDs, null values
  - one-to-many join risks (dimension duplicates)
  - inclusive/exclusive date boundaries
  - out-of-scope extreme rows
  - subgroup-size thresholds
  - confounding columns with similar names
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ScenarioConfig:
    scenario_id: str
    tables: list["TableConfig"]
    date_range: tuple[str, str] = ("2025-01-01", "2025-12-31")
    prior_date_range: tuple[str, str] = ("2024-01-01", "2024-12-31")


@dataclass
class TableConfig:
    name: str
    n_rows: int
    columns: list["ColumnConfig"]
    pk_column: str | None = None          # primary key column name
    fk_to: str | None = None              # foreign key target table name
    fk_column: str | None = None          # which column is the FK


@dataclass
class ColumnConfig:
    name: str
    dtype: str                             # "int", "float", "category", "date", "string"
    values: list[Any] | None = None        # for category columns
    min_val: float | None = None
    max_val: float | None = None
    null_fraction: float = 0.0
    skew: str = "none"                     # "none", "left", "right", "heavy_tail"
    include_duplicates: bool = False       # inject duplicate IDs
    rare_group_fraction: float = 0.0       # fraction of rows assigned to rare categories


def generate_tables(
    config: ScenarioConfig,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate all tables for a scenario with the given random seed."""
    rng = np.random.default_rng(seed)
    tables: dict[str, pd.DataFrame] = {}

    for table_cfg in config.tables:
        df = _generate_table(table_cfg, config, rng)
        tables[table_cfg.name] = df

    return tables


def _generate_table(
    cfg: TableConfig,
    scenario: ScenarioConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n = cfg.n_rows
    data: dict[str, Any] = {}

    for col in cfg.columns:
        data[col.name] = _generate_column(col, n, scenario, rng)

    df = pd.DataFrame(data)

    # Inject duplicate PK values when configured
    if cfg.pk_column and any(c.include_duplicates for c in cfg.columns if c.name == cfg.pk_column):
        n_dups = max(1, n // 20)
        dup_ids = rng.choice(df[cfg.pk_column].values, size=n_dups, replace=False)
        dup_rows = df[df[cfg.pk_column].isin(dup_ids)].sample(
            n=n_dups, replace=True, random_state=int(rng.integers(0, 2**31))
        )
        df = pd.concat([df, dup_rows], ignore_index=True)

    return df.reset_index(drop=True)


def _generate_column(
    col: ColumnConfig,
    n: int,
    scenario: ScenarioConfig,
    rng: np.random.Generator,
) -> np.ndarray | list:
    if col.dtype == "int":
        lo = int(col.min_val or 1)
        hi = int(col.max_val or 10_000)
        values = rng.integers(lo, hi + 1, size=n).tolist()

    elif col.dtype == "float":
        lo = float(col.min_val or 0.0)
        hi = float(col.max_val or 1000.0)
        base = rng.uniform(lo, hi, size=n)
        if col.skew == "heavy_tail":
            # Pareto-distributed: most values small, occasional huge outlier
            pareto = rng.pareto(1.5, size=n)
            base = lo + (hi - lo) * pareto / (pareto.max() + 1)
        elif col.skew == "right":
            base = np.exp(rng.normal(0, 1, size=n)) * (hi - lo) / 20 + lo
        values = base.tolist()

    elif col.dtype == "category":
        cats = col.values or [f"Cat_{c}" for c in string.ascii_uppercase[:6]]
        if col.rare_group_fraction > 0:
            n_rare = max(1, int(n * col.rare_group_fraction))
            rare_cat = f"Rare_{cats[0]}"
            main_cats = cats
            chosen = rng.choice(main_cats, size=n - n_rare).tolist()
            chosen += [rare_cat] * n_rare
            rng.shuffle(chosen)
            values = chosen
        else:
            values = rng.choice(cats, size=n).tolist()

    elif col.dtype == "date":
        # Generate dates spanning BOTH current and prior date ranges so that
        # period-comparison tasks have data in both windows.
        curr_start, curr_end = scenario.date_range
        prior_start, prior_end = scenario.prior_date_range
        all_dates = []
        for s_str, e_str in [(prior_start, prior_end), (curr_start, curr_end)]:
            s = pd.Timestamp(s_str)
            e = pd.Timestamp(e_str)
            n_days = (e - s).days
            half_n = n // 2
            offsets = rng.integers(0, n_days + 1, size=half_n)
            all_dates.extend([(s + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d") for d in offsets])
        # If n is odd, add one more from current period
        if len(all_dates) < n:
            s = pd.Timestamp(curr_start)
            e = pd.Timestamp(curr_end)
            n_days = (e - s).days
            d = int(rng.integers(0, n_days + 1))
            all_dates.append((s + pd.Timedelta(days=d)).strftime("%Y-%m-%d"))
        rng.shuffle(all_dates)
        values = all_dates[:n]

    elif col.dtype == "string":
        length = 8
        values = [
            "".join(rng.choice(list(string.ascii_lowercase), size=length).tolist())
            for _ in range(n)
        ]

    else:
        values = [None] * n

    # Apply null fraction
    if col.null_fraction > 0:
        null_mask = rng.uniform(0, 1, size=n) < col.null_fraction
        values = [None if m else v for m, v in zip(null_mask, values)]

    return values


# ─────────────────────────────────────────
# Built-in scenarios
# ─────────────────────────────────────────

def retail_orders_scenario() -> ScenarioConfig:
    """Scenario 1: retail orders and products (single-table + product lookup)."""
    return ScenarioConfig(
        scenario_id="retail01",
        tables=[
            TableConfig(
                name="orders",
                n_rows=2000,
                pk_column="order_id",
                columns=[
                    ColumnConfig("order_id",     "int",      min_val=1, max_val=99999),
                    ColumnConfig("customer_id",  "int",      min_val=1, max_val=500),
                    ColumnConfig("category",     "category", values=["Electronics", "Clothing", "Home", "Sports", "Books", "Toys"]),
                    ColumnConfig("order_status", "category", values=["completed", "cancelled", "pending", "refunded"]),
                    ColumnConfig("customer_type","category", values=["new", "returning", "vip"]),
                    ColumnConfig("revenue",      "float",    min_val=5.0,  max_val=2000.0, skew="heavy_tail"),
                    ColumnConfig("quantity",     "int",      min_val=1,    max_val=20),
                    ColumnConfig("order_date",   "date"),
                    ColumnConfig("ship_date",    "date"),
                    ColumnConfig("region",       "category", values=["US", "EU", "APAC", "LATAM"]),
                ],
            ),
        ],
    )


def sessions_conversion_scenario() -> ScenarioConfig:
    """Scenario 2: website sessions and conversions (two tables)."""
    return ScenarioConfig(
        scenario_id="web01",
        tables=[
            TableConfig(
                name="sessions",
                n_rows=5000,
                pk_column="session_id",
                columns=[
                    ColumnConfig("session_id",      "int",      min_val=1, max_val=999999),
                    ColumnConfig("customer_id",     "int",      min_val=1, max_val=1000),
                    ColumnConfig("category",        "category", values=["Electronics", "Clothing", "Home", "Sports", "Books"]),
                    ColumnConfig("customer_type",   "category", values=["new", "returning"]),
                    ColumnConfig("device",          "category", values=["mobile", "desktop", "tablet"]),
                    ColumnConfig("traffic_source",  "category", values=["organic", "paid", "email", "direct", "test"]),
                    ColumnConfig("event_date",      "date"),
                    ColumnConfig("page_views",      "int",      min_val=1, max_val=50),
                    ColumnConfig("region",          "category", values=["US", "EU", "APAC"]),
                ],
            ),
            TableConfig(
                name="conversions",
                n_rows=800,
                pk_column="conversion_id",
                fk_to="sessions",
                fk_column="session_id",
                columns=[
                    ColumnConfig("conversion_id", "int",   min_val=1, max_val=99999),
                    ColumnConfig("session_id",    "int",   min_val=1, max_val=999999),
                    ColumnConfig("revenue",       "float", min_val=10.0, max_val=1500.0, skew="right"),
                    ColumnConfig("event_date",    "date"),
                ],
            ),
        ],
    )


def seller_marketplace_scenario() -> ScenarioConfig:
    """Scenario 3: seller marketplace performance."""
    return ScenarioConfig(
        scenario_id="market01",
        tables=[
            TableConfig(
                name="seller_orders",
                n_rows=3000,
                pk_column="order_id",
                columns=[
                    ColumnConfig("order_id",       "int",      min_val=1,   max_val=999999),
                    ColumnConfig("seller_id",      "int",      min_val=1,   max_val=200,   null_fraction=0.02),
                    ColumnConfig("category",       "category", values=["Fashion", "Electronics", "Home & Garden", "Sports", "Beauty"]),
                    ColumnConfig("order_status",   "category", values=["shipped", "cancelled", "returned", "processing"]),
                    ColumnConfig("gmv",            "float",    min_val=5.0, max_val=5000.0, skew="heavy_tail"),
                    ColumnConfig("units",          "int",      min_val=1,   max_val=50),
                    ColumnConfig("fulfillment",    "category", values=["seller", "marketplace"]),
                    ColumnConfig("order_date",     "date"),
                    ColumnConfig("ship_date",      "date",     null_fraction=0.05),
                    ColumnConfig("delivery_days",  "int",      min_val=1,   max_val=30,    null_fraction=0.08),
                ],
            ),
        ],
    )


def subscription_usage_scenario() -> ScenarioConfig:
    """Scenario 4: subscription SaaS usage and churn."""
    return ScenarioConfig(
        scenario_id="saas01",
        tables=[
            TableConfig(
                name="subscriptions",
                n_rows=3000,
                pk_column="sub_id",
                columns=[
                    ColumnConfig("sub_id",        "int",      min_val=1,    max_val=99999),
                    ColumnConfig("customer_id",   "int",      min_val=1,    max_val=800),
                    ColumnConfig("plan",          "category", values=["free", "starter", "pro", "enterprise"]),
                    ColumnConfig("status",        "category", values=["active", "churned", "paused", "trial"]),
                    ColumnConfig("region",        "category", values=["NA", "EMEA", "APAC", "LATAM"]),
                    ColumnConfig("mrr",           "float",    min_val=0.0,   max_val=5000.0, skew="heavy_tail"),
                    ColumnConfig("seats",         "int",      min_val=1,    max_val=500),
                    ColumnConfig("logins_30d",    "int",      min_val=0,    max_val=200,   null_fraction=0.05),
                    ColumnConfig("start_date",    "date"),
                    ColumnConfig("churn_date",    "date",     null_fraction=0.70),
                ],
            ),
        ],
    )


def marketing_campaigns_scenario() -> ScenarioConfig:
    """Scenario 5: marketing campaigns and attribution."""
    return ScenarioConfig(
        scenario_id="mktg01",
        tables=[
            TableConfig(
                name="campaigns",
                n_rows=2500,
                pk_column="event_id",
                columns=[
                    ColumnConfig("event_id",      "int",      min_val=1,    max_val=999999),
                    ColumnConfig("campaign_id",   "int",      min_val=1,    max_val=50),
                    ColumnConfig("channel",       "category", values=["email", "social", "search", "display", "affiliate"]),
                    ColumnConfig("campaign_type", "category", values=["acquisition", "retention", "upsell", "brand"]),
                    ColumnConfig("status",        "category", values=["active", "paused", "completed", "test"]),
                    ColumnConfig("impressions",   "int",      min_val=100,  max_val=1000000, skew="heavy_tail"),
                    ColumnConfig("clicks",        "int",      min_val=0,    max_val=50000),
                    ColumnConfig("spend",         "float",    min_val=0.0,  max_val=50000.0, skew="right"),
                    ColumnConfig("conversions",   "int",      min_val=0,    max_val=5000),
                    ColumnConfig("revenue",       "float",    min_val=0.0,  max_val=200000.0, skew="heavy_tail"),
                    ColumnConfig("event_date",    "date"),
                    ColumnConfig("region",        "category", values=["US", "EU", "APAC"]),
                ],
            ),
        ],
    )


def payments_refunds_scenario() -> ScenarioConfig:
    """Scenario 6: payments and refunds."""
    return ScenarioConfig(
        scenario_id="payments01",
        tables=[
            TableConfig(
                name="transactions",
                n_rows=4000,
                pk_column="txn_id",
                columns=[
                    ColumnConfig("txn_id",        "int",      min_val=1,    max_val=999999),
                    ColumnConfig("customer_id",   "int",      min_val=1,    max_val=1000),
                    ColumnConfig("order_id",      "int",      min_val=1,    max_val=200000),
                    ColumnConfig("txn_type",      "category", values=["payment", "refund", "chargeback", "adjustment"]),
                    ColumnConfig("payment_method","category", values=["card", "paypal", "bank", "crypto", "gift_card"]),
                    ColumnConfig("status",        "category", values=["completed", "failed", "pending", "disputed"]),
                    ColumnConfig("amount",        "float",    min_val=0.01, max_val=10000.0, skew="right"),
                    ColumnConfig("fee",           "float",    min_val=0.0,  max_val=500.0,  null_fraction=0.05),
                    ColumnConfig("txn_date",      "date"),
                    ColumnConfig("region",        "category", values=["US", "EU", "APAC", "LATAM"]),
                ],
            ),
        ],
    )


def inventory_fulfillment_scenario() -> ScenarioConfig:
    """Scenario 7: inventory and fulfillment operations."""
    return ScenarioConfig(
        scenario_id="ops01",
        tables=[
            TableConfig(
                name="shipments",
                n_rows=3500,
                pk_column="shipment_id",
                columns=[
                    ColumnConfig("shipment_id",   "int",      min_val=1,    max_val=999999),
                    ColumnConfig("order_id",      "int",      min_val=1,    max_val=200000),
                    ColumnConfig("warehouse",     "category", values=["WH_US_EAST", "WH_US_WEST", "WH_EU", "WH_APAC"]),
                    ColumnConfig("category",      "category", values=["Electronics", "Clothing", "Home", "Sports", "Perishable"]),
                    ColumnConfig("carrier",       "category", values=["FedEx", "UPS", "USPS", "DHL", "internal"]),
                    ColumnConfig("status",        "category", values=["delivered", "in_transit", "returned", "lost", "cancelled"]),
                    ColumnConfig("weight_kg",     "float",    min_val=0.1,  max_val=50.0),
                    ColumnConfig("cost",          "float",    min_val=1.0,  max_val=500.0, skew="right"),
                    ColumnConfig("promised_days", "int",      min_val=1,    max_val=14),
                    ColumnConfig("actual_days",   "int",      min_val=0,    max_val=30,   null_fraction=0.08),
                    ColumnConfig("ship_date",     "date"),
                    ColumnConfig("region",        "category", values=["US", "EU", "APAC"]),
                ],
            ),
        ],
    )


def support_tickets_scenario() -> ScenarioConfig:
    """Scenario 8: customer support tickets."""
    return ScenarioConfig(
        scenario_id="support01",
        tables=[
            TableConfig(
                name="tickets",
                n_rows=2000,
                pk_column="ticket_id",
                columns=[
                    ColumnConfig("ticket_id",     "int",      min_val=1,    max_val=99999),
                    ColumnConfig("customer_id",   "int",      min_val=1,    max_val=500),
                    ColumnConfig("category",      "category", values=["billing", "technical", "shipping", "returns", "account"]),
                    ColumnConfig("priority",      "category", values=["low", "medium", "high", "critical"]),
                    ColumnConfig("channel",       "category", values=["email", "chat", "phone", "self_service"]),
                    ColumnConfig("status",        "category", values=["resolved", "open", "escalated", "pending_customer"]),
                    ColumnConfig("resolution_hours", "float", min_val=0.1, max_val=336.0, skew="right", null_fraction=0.15),
                    ColumnConfig("csat_score",    "float",    min_val=1.0,  max_val=5.0,  null_fraction=0.30),
                    ColumnConfig("created_date",  "date"),
                    ColumnConfig("region",        "category", values=["US", "EU", "APAC"]),
                ],
            ),
        ],
    )


BUILTIN_SCENARIOS: dict[str, ScenarioConfig] = {
    "retail01":   retail_orders_scenario(),
    "web01":      sessions_conversion_scenario(),
    "market01":   seller_marketplace_scenario(),
    "saas01":     subscription_usage_scenario(),
    "mktg01":     marketing_campaigns_scenario(),
    "payments01": payments_refunds_scenario(),
    "ops01":      inventory_fulfillment_scenario(),
    "support01":  support_tickets_scenario(),
}


def generate_scenario(scenario_id: str, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate tables for a named built-in scenario."""
    if scenario_id not in BUILTIN_SCENARIOS:
        raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {list(BUILTIN_SCENARIOS)}")
    return generate_tables(BUILTIN_SCENARIOS[scenario_id], seed=seed)
