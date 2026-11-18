"""Tests for evaluation metrics."""

from morphda.evaluation.metrics import (
    compute_mutation_score,
    compute_repair_metrics,
    compute_verification_metrics,
)


def test_perfect_verifier():
    labels = [True, True, False, False]
    preds  = [True, True, False, False]
    m = compute_verification_metrics(labels, preds)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.false_positive_rate == 0.0
    assert m.accepted_answer_risk == 0.0


def test_all_flagged_verifier():
    labels = [True, False, False]
    preds  = [True, True,  True]
    m = compute_verification_metrics(labels, preds)
    assert m.recall == 1.0
    assert m.precision == pytest.approx(1 / 3)
    assert m.false_positive_rate == 1.0


def test_no_positives_does_not_crash():
    labels = [False, False, False]
    preds  = [False, False, True]
    m = compute_verification_metrics(labels, preds)
    assert m.recall == 0.0  # no true positives


def test_mutation_score_micro_and_macro():
    killed  = [True, True, False, True, False]
    families = ["filter", "filter", "filter", "agg", "agg"]
    m = compute_mutation_score(killed, families)
    assert m.micro_mutation_score == pytest.approx(3 / 5)
    # macro: filter=2/3, agg=1/2 → mean=7/12
    assert m.macro_mutation_score == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_mutation_score_empty():
    m = compute_mutation_score([], [])
    assert m.micro_mutation_score == 0.0


def test_repair_metrics():
    initially_wrong   = [True,  True,  False, False]
    repaired_correct  = [True,  False, False, False]
    initially_correct = [False, False, True,  True]
    regressed_wrong   = [False, False, True,  False]
    m = compute_repair_metrics(
        initially_wrong, repaired_correct,
        initially_correct, regressed_wrong,
    )
    assert m.repair_rate == pytest.approx(0.5)
    assert m.regression_rate == pytest.approx(0.5)
    assert m.net_correction_gain == pytest.approx(0.0)


import pytest
