"""Natural-agent evaluation: labels, conditions, and an end-to-end program check."""

from __future__ import annotations

import json

import pytest

from morphda.evaluation.natural import (
    evaluate_program,
    is_accidental_correct,
    load_source_programs,
    scope_is_discriminating,
    summarize,
)
from morphda.reference.compiler import compile_task
from morphda.tasks.factory import generate_task_set

TASKS = {t.task_id: t for t in generate_task_set()}


def _row(correct_source, held_out, flags, executed=True):
    return {
        "task_id": "t", "executed": executed, "correct_source": correct_source,
        "correct_held_out": {"7": held_out[0], "123": held_out[1]} if executed else {},
        "flags": {"42": flags[0], "7": flags[1]} if executed else {},
    }


def test_accidental_correct_needs_source_correct_and_some_held_out_failure():
    assert is_accidental_correct(_row(True, (True, False), (False, False)))
    assert not is_accidental_correct(_row(True, (True, True), (False, False)))   # truly correct
    assert not is_accidental_correct(_row(False, (False, False), (True, True)))  # wrong everywhere
    assert not is_accidental_correct(_row(False, (True, True), (False, False), executed=False))


def test_conditions_relabel_accidental_corrects_and_exclude_execution_failures():
    rows = [
        _row(True, (True, True), (False, False)),    # truly correct, not flagged        -> TN
        _row(True, (True, False), (True, False)),    # accidental correct, flagged       -> A: FP, B: TP
        _row(True, (True, False), (False, True)),    # accidental, only 2nd seed flags   -> B: FN, C: TP
        _row(False, (False, False), (True, True)),   # wrong, flagged                    -> TP
        _row(False, (False, False), (False, False)), # wrong, missed                     -> FN
        _row(False, None, None, executed=False),     # execution failure: excluded
    ]
    s = summarize(rows, n_boot=50)

    assert s["n_exec_failures"] == 1
    assert (s["n_correct_source_seed"], s["n_accidental_correct"], s["n_cross_seed_correct"]) == (3, 2, 1)
    assert s["accidental_correct_flagged"] == 1
    c = s["conditions"]
    assert (c["A_naive"]["tp"], c["A_naive"]["fp"], c["A_naive"]["tn"], c["A_naive"]["fn"]) == (1, 1, 2, 1)
    assert (c["B_cross_seed"]["tp"], c["B_cross_seed"]["fp"], c["B_cross_seed"]["tn"], c["B_cross_seed"]["fn"]) == (2, 0, 1, 2)
    assert (c["C_two_seeds"]["tp"], c["C_two_seeds"]["fn"]) == (3, 1)
    assert c["B_cross_seed"]["precision"] == 1.0 and c["B_cross_seed"]["recall"] == 0.5


def test_load_source_programs_keeps_last_record_and_only_source_seed(tmp_path):
    path = tmp_path / "programs.jsonl"
    records = [
        {"task_id": "a", "data_seed": 42, "generated_program": "old"},
        {"task_id": "a", "data_seed": 42, "generated_program": "new"},
        {"task_id": "a", "data_seed": 7, "generated_program": "other-seed"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\nnot json\n")
    assert load_source_programs(path)["a"]["generated_program"] == "new"


def test_scope_that_never_changes_the_answer_is_not_discriminating():
    assert not scope_is_discriminating(TASKS["market01_l2_033"])
    assert scope_is_discriminating(TASKS["retail01_l2_004"])


def test_reference_program_is_correct_on_every_seed_and_never_flagged():
    task = TASKS["retail01_l2_004"]
    row = evaluate_program((task, compile_task(task), (7, 123), (42,), 20))
    assert row["executed"] and row["correct_source"]
    assert row["correct_held_out"] == {"7": True, "123": True}
    assert row["flags"] == {"42": False}


def test_program_that_ignores_the_status_filter_is_flagged():
    task = TASKS["retail01_l2_004"]  # highest revenue category among non-cancelled orders
    program = (
        "def analyze(tables):\n"
        "    return tables['orders'].groupby('category')['revenue'].sum().idxmax()\n"
    )
    row = evaluate_program((task, program, (7, 123), (42,), 20))
    assert row["executed"]
    assert row["flags"]["42"] is True
