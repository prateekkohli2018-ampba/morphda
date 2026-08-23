"""
MORPH-DA evaluation metrics.

All primary metrics as defined in paper Section 17.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class VerificationMetrics:
    """Section 17.2 — Verification metrics."""
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    accepted_answer_risk: float
    n_flagged: int
    n_correct_flagged: int
    n_incorrect_unflagged: int
    n_total_correct: int
    n_total_incorrect: int


@dataclass
class MutationMetrics:
    """Section 17.3 — Mutation testing metrics."""
    micro_mutation_score: float
    macro_mutation_score: float
    per_family_kill_rate: dict[str, float]
    equivalent_mutant_rate: float
    invalid_mutant_rate: float
    n_valid_mutants: int
    n_killed: int


@dataclass
class RepairMetrics:
    """Section 17.5 — Repair metrics."""
    repair_rate: float          # P(correct | initially wrong, retried)
    regression_rate: float      # P(wrong | initially correct, retried)
    net_correction_gain: float  # repair_rate - regression_rate
    n_repaired: int
    n_regressed: int
    n_wrong_retried: int
    n_correct_retried: int


def compute_verification_metrics(
    labels: Sequence[bool],      # True = incorrect program
    predictions: Sequence[bool], # True = flagged by verifier
) -> VerificationMetrics:
    """
    Compute precision, recall, F1, FPR, accepted-answer risk.

    Args:
        labels: Ground-truth correctness labels (True = incorrect).
        predictions: Verifier flags (True = flagged as suspicious).
    """
    assert len(labels) == len(predictions), "labels and predictions must have equal length"

    tp = sum(l and p for l, p in zip(labels, predictions))
    fp = sum(not l and p for l, p in zip(labels, predictions))
    fn = sum(l and not p for l, p in zip(labels, predictions))
    tn = sum(not l and not p for l, p in zip(labels, predictions))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # Among accepted (not flagged) programs, how many are wrong?
    accepted_risk = fn / (fn + tn) if (fn + tn) > 0 else 0.0

    return VerificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        accepted_answer_risk=accepted_risk,
        n_flagged=tp + fp,
        n_correct_flagged=fp,
        n_incorrect_unflagged=fn,
        n_total_correct=fp + tn,
        n_total_incorrect=tp + fn,
    )


def compute_mutation_score(
    killed: Sequence[bool],
    family_labels: Sequence[str],
) -> MutationMetrics:
    """
    Compute micro and macro mutation scores.

    Args:
        killed: True when the mutant was detected by at least one relation.
        family_labels: Fault family name for each mutant.
    """
    n = len(killed)
    if n == 0:
        return MutationMetrics(
            micro_mutation_score=0.0,
            macro_mutation_score=0.0,
            per_family_kill_rate={},
            equivalent_mutant_rate=0.0,
            invalid_mutant_rate=0.0,
            n_valid_mutants=0,
            n_killed=0,
        )

    n_killed = sum(killed)
    micro_ms = n_killed / n

    families: dict[str, list[bool]] = {}
    for k, f in zip(killed, family_labels):
        families.setdefault(f, []).append(k)

    per_family = {f: sum(ks) / len(ks) for f, ks in families.items()}
    macro_ms = sum(per_family.values()) / len(per_family) if per_family else 0.0

    return MutationMetrics(
        micro_mutation_score=micro_ms,
        macro_mutation_score=macro_ms,
        per_family_kill_rate=per_family,
        equivalent_mutant_rate=0.0,  # populated by pipeline
        invalid_mutant_rate=0.0,
        n_valid_mutants=n,
        n_killed=n_killed,
    )


def compute_repair_metrics(
    initially_wrong: Sequence[bool],
    repaired_correct: Sequence[bool],
    initially_correct: Sequence[bool],
    regressed_wrong: Sequence[bool],
) -> RepairMetrics:
    n_wrong = sum(initially_wrong)
    n_repaired = sum(r and w for r, w in zip(repaired_correct, initially_wrong))
    n_correct = sum(initially_correct)
    n_regressed = sum(r and c for r, c in zip(regressed_wrong, initially_correct))

    repair_rate = n_repaired / n_wrong if n_wrong > 0 else 0.0
    regression_rate = n_regressed / n_correct if n_correct > 0 else 0.0

    return RepairMetrics(
        repair_rate=repair_rate,
        regression_rate=regression_rate,
        net_correction_gain=repair_rate - regression_rate,
        n_repaired=n_repaired,
        n_regressed=n_regressed,
        n_wrong_retried=n_wrong,
        n_correct_retried=n_correct,
    )
