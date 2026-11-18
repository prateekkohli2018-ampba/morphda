"""
Continuous MORPH-DA verification score for AUROC/AUPRC curves.

Instead of binary pass/fail, produce a score in [0, 1]:
  0 = strong PASS (no violations)
  1 = strong FAIL (many high-severity violations)
"""

from __future__ import annotations

from morphda.verification.engine import VerificationReport


def morph_score(report: VerificationReport) -> float:
    """
    Continuous verification score for threshold-sweep metrics.

    Returns a float in [0, 1] where higher = more suspicious.
    """
    if not report.source_execution.success:
        return 0.5  # execution failure is informative but not conclusive

    if report.applicable_relations == 0:
        return 0.0

    # Fraction of applicable relations violated
    violation_fraction = report.violated_relations / report.applicable_relations

    # Weight by number of witnesses (more witnesses = more confidence)
    n_witnesses = len(report.witnesses)
    witness_boost = min(n_witnesses / 3.0, 1.0) * 0.1

    return min(1.0, violation_fraction + witness_boost)
