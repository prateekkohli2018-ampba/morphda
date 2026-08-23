"""Tests for the seeded data generator."""

import pandas as pd
import pytest

from morphda.data.generators import generate_scenario, BUILTIN_SCENARIOS


def test_retail_scenario_tables():
    tables = generate_scenario("retail01", seed=42)
    assert "orders" in tables
    df = tables["orders"]
    assert len(df) > 0
    assert "category" in df.columns
    assert "revenue" in df.columns
    assert "order_date" in df.columns


def test_web_scenario_tables():
    tables = generate_scenario("web01", seed=42)
    assert "sessions" in tables
    assert "conversions" in tables
    sessions = tables["sessions"]
    assert "customer_type" in sessions.columns
    assert "event_date" in sessions.columns


def test_different_seeds_produce_different_data():
    t1 = generate_scenario("retail01", seed=1)
    t2 = generate_scenario("retail01", seed=2)
    # revenue sums should differ across seeds
    assert t1["orders"]["revenue"].sum() != t2["orders"]["revenue"].sum()


def test_same_seed_reproducible():
    t1 = generate_scenario("retail01", seed=99)
    t2 = generate_scenario("retail01", seed=99)
    pd.testing.assert_frame_equal(t1["orders"], t2["orders"])


def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        generate_scenario("nonexistent_scenario")


def test_all_builtin_scenarios_generate():
    for sid in BUILTIN_SCENARIOS:
        tables = generate_scenario(sid, seed=0)
        assert len(tables) > 0
        for name, df in tables.items():
            assert len(df) > 0, f"Table '{name}' in scenario '{sid}' is empty"
