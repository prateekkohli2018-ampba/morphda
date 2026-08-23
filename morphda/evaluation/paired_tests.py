"""
Paired statistical tests for verification method comparisons.

Primary test: McNemar's test (paired binary outcomes on same programs).
Secondary: Task-clustered bootstrap (see bootstrap.py).

Paper Section 18.4:
  Primary paired comparisons (Holm-corrected):
  1. Full MORPH-DA vs execution-only
  2. Full MORPH-DA vs LLM judge
  3. Full MORPH-DA vs universal MRs
  4. Witness-guided repair vs random matched retry
  5. Witness-guided repair vs MORPH-gated generic retry
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class McnemarResult:
    """Result of McNemar's test for paired binary outcomes."""
    n_both_correct:   int   # A correct, B correct
    n_a_only:         int   # A correct, B wrong
    n_b_only:         int   # A wrong,   B correct
    n_both_wrong:     int   # A wrong,   B wrong
    statistic:        float # chi-squared statistic (with continuity correction)
    p_value:          float
    method_a_better:  bool  # True if A has significantly higher accuracy
    effect_size:      float  # |n_a_only - n_b_only| / n_discordant


def mcnemar_test(
    labels_a: Sequence[bool],   # True = method A correct
    labels_b: Sequence[bool],   # True = method B correct
    continuity_correction: bool = True,
) -> McnemarResult:
    """
    McNemar's test for paired binary outcomes.

    Args:
        labels_a: Correctness for method A on each program.
        labels_b: Correctness for method B on each program.
        continuity_correction: Apply Yates' continuity correction (recommended for n < 25).

    Returns:
        McnemarResult with chi-squared statistic, p-value, and contingency table.
    """
    assert len(labels_a) == len(labels_b), "lengths must match"

    n_both_correct = sum(a and b for a, b in zip(labels_a, labels_b))
    n_a_only       = sum(a and not b for a, b in zip(labels_a, labels_b))
    n_b_only       = sum(not a and b for a, b in zip(labels_a, labels_b))
    n_both_wrong   = sum(not a and not b for a, b in zip(labels_a, labels_b))

    # McNemar statistic on discordant pairs
    n_disc = n_a_only + n_b_only
    if n_disc == 0:
        return McnemarResult(
            n_both_correct=n_both_correct, n_a_only=0, n_b_only=0,
            n_both_wrong=n_both_wrong, statistic=0.0, p_value=1.0,
            method_a_better=False, effect_size=0.0,
        )

    if continuity_correction:
        chi2 = (abs(n_a_only - n_b_only) - 1.0) ** 2 / n_disc
    else:
        chi2 = (n_a_only - n_b_only) ** 2 / n_disc

    chi2 = max(0.0, chi2)
    p_value = _chi2_sf(chi2, df=1)

    effect = abs(n_a_only - n_b_only) / n_disc if n_disc > 0 else 0.0

    return McnemarResult(
        n_both_correct=n_both_correct,
        n_a_only=n_a_only,
        n_b_only=n_b_only,
        n_both_wrong=n_both_wrong,
        statistic=chi2,
        p_value=p_value,
        method_a_better=n_a_only > n_b_only,
        effect_size=effect,
    )


def holm_correct(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni correction for multiple comparisons.
    Returns corrected p-values (same order as input).
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected[orig_idx] = min(1.0, p * (n - rank))
    # Enforce monotonicity
    max_so_far = 0.0
    for _, (orig_idx, _) in enumerate(indexed):
        corrected[orig_idx] = max(corrected[orig_idx], max_so_far)
        max_so_far = corrected[orig_idx]
    return corrected


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function of chi-squared distribution (1 - CDF)."""
    # For df=1: chi2_sf(x) = erfc(sqrt(x/2)) ≈ 2 * Phi(-sqrt(x))
    if x <= 0:
        return 1.0
    z = math.sqrt(x / 2.0)
    return _erfc(z)


def _erfc(x: float) -> float:
    """Complementary error function (approximation)."""
    # Abramowitz & Stegun formula 7.1.26 (error < 1.5e-7)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return poly * math.exp(-x * x)


def detection_comparison_table(
    wrong_programs: list[dict],
    ver_by_id: dict,
    method_keys: list[str],
    method_labels: list[str],
) -> str:
    """
    Generate a comparison table with McNemar p-values for verification methods.

    wrong_programs: list of wrong-but-executable program dicts
    ver_by_id: dict mapping program_id → verification result
    method_keys: list of keys in ver_by_id['killed_by'] or similar
    """
    n = len(wrong_programs)
    lines = [
        f"\nVerification method comparison (n={n} wrong programs)",
        "=" * 70,
    ]

    # Build detection vectors for each method
    detected = {}
    for key in method_keys:
        detected[key] = []
        for p in wrong_programs:
            pid = p['program_id']
            v = ver_by_id.get(pid, {})
            # Support both the natural agent verification format and the RuleMut format
            if 'killed_by' in v:
                detected[key].append(v['killed_by'].get(key, False))
            else:
                # Natural agent format: decision field
                detected[key].append(v.get('decision') == 'fail')
        detected[key] = detected[key]

    # Pairwise McNemar vs full MORPH-DA (last method)
    full_key = method_keys[-1]
    p_values = []
    comparisons = []
    for key in method_keys[:-1]:
        result = mcnemar_test(detected[full_key], detected[key])
        p_values.append(result.p_value)
        comparisons.append((key, result))

    corrected = holm_correct(p_values)

    lines.append(f"{'Method':<30} {'Recall':>8} {'vs Full MORPH p':>16} {'p_corrected':>12}")
    lines.append("-" * 70)

    for i, (key, label) in enumerate(zip(method_keys, method_labels)):
        recall = sum(detected[key]) / n if n else 0.0
        if key == full_key:
            lines.append(f"  {label:<28} {recall:>8.1%}   (reference method)")
        else:
            result = comparisons[i]
            p_raw = result.p_value
            p_cor = corrected[i]
            sig = "***" if p_cor < 0.001 else ("**" if p_cor < 0.01 else ("*" if p_cor < 0.05 else "ns"))
            lines.append(f"  {label:<28} {recall:>8.1%}  {p_raw:>15.4f}  {p_cor:>10.4f} {sig}")

    lines.append("=" * 70)
    lines.append("* p<0.05, ** p<0.01, *** p<0.001, ns = not significant (Holm-corrected)")
    return "\n".join(lines)
