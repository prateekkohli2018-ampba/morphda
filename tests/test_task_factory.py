"""Tests for the task factory and validator."""

import pytest
from morphda.tasks.factory import generate_task_set, task_set_summary, SCENARIO_TABLE_MAP
from morphda.tasks.validators import validate_task
from morphda.tasks.schema import TaskSpec


class TestTaskFactory:
    def test_generates_tasks(self):
        tasks = generate_task_set()
        assert len(tasks) > 0

    def test_all_difficulty_levels_present(self):
        tasks = generate_task_set()
        levels = {t.difficulty_level for t in tasks}
        assert levels == {1, 2, 3, 4, 5}

    def test_all_scenarios_present(self):
        tasks = generate_task_set()
        scenario_ids = {t.scenario_id for t in tasks}
        assert scenario_ids == set(SCENARIO_TABLE_MAP.keys())

    def test_task_ids_unique(self):
        tasks = generate_task_set()
        ids = [t.task_id for t in tasks]
        assert len(ids) == len(set(ids)), "Duplicate task IDs detected"

    def test_all_tasks_have_canonical_questions(self):
        tasks = generate_task_set()
        for t in tasks:
            assert t.canonical_question, f"Task {t.task_id} has no canonical question"

    def test_level_distribution_roughly_correct(self):
        tasks = generate_task_set()
        from collections import Counter
        by_level = Counter(t.difficulty_level for t in tasks)
        # L2 is most common (6 variants × 3 scenarios = 18)
        assert by_level[2] > by_level[1], "L2 should exceed L1"
        # L5 should be fewest
        assert by_level[5] <= by_level[4], "L5 should not exceed L4"
        # All levels present
        assert all(by_level[lvl] > 0 for lvl in range(1, 6))

    def test_summary_runs_without_error(self):
        tasks = generate_task_set()
        summary = task_set_summary(tasks)
        assert "Total tasks:" in summary


class TestTaskValidator:
    """Integration tests: validate that reference programs pass MR checks."""

    def _simple_task(self) -> TaskSpec:
        """A simple L1 task that the compiler definitely handles correctly."""
        from morphda.tasks.schema import MetricSpec
        return TaskSpec(
            task_id="val_test_001",
            scenario_id="retail01",
            question_family="scalar_sum",
            difficulty_level=1,
            inputs=["orders"],
            metric=MetricSpec(name="total", operation="sum", column="revenue"),
            output_type="scalar",
        )

    def test_valid_reference_passes_validation(self):
        task = self._simple_task()
        result = validate_task(task, seeds=[42, 7])
        assert result.passed, f"Valid task failed: {result.failures}"

    def test_validation_produces_gold_answers(self):
        task = self._simple_task()
        result = validate_task(task, seeds=[42, 7])
        assert 42 in result.gold_answers
        assert 7 in result.gold_answers
        # Different seeds → different sums
        assert result.gold_answers[42] != result.gold_answers[7]

    def test_unknown_scenario_fails_gracefully(self):
        from morphda.tasks.schema import MetricSpec
        task = TaskSpec(
            task_id="val_bad_scenario",
            scenario_id="nonexistent_xyz",
            question_family="scalar_sum",
            difficulty_level=1,
            inputs=["t"],
            metric=MetricSpec(name="total", operation="sum", column="revenue"),
            output_type="scalar",
        )
        result = validate_task(task, seeds=[42])
        assert not result.passed
        assert any("not found" in f or "scenario" in f.lower() for f in result.failures)

    def test_l2_grouped_rank_validates(self):
        from morphda.tasks.schema import MetricSpec, RankingSpec, FilterSpec
        task = TaskSpec(
            task_id="val_l2_001",
            scenario_id="retail01",
            question_family="grouped_rank",
            difficulty_level=2,
            inputs=["orders"],
            filters=[FilterSpec(column="order_status", operator="not_equal", value="cancelled")],
            metric=MetricSpec(name="total", operation="sum", column="revenue"),
            group_by=["category"],
            ranking=RankingSpec(direction="descending", k=1),
            output_type="label",
        )
        result = validate_task(task, seeds=[42])
        assert result.passed, f"L2 task failed: {result.failures}"
        # Winner must be a real category
        gold = result.gold_answers[42]
        assert isinstance(gold, str), f"Expected string label, got {type(gold)}"
