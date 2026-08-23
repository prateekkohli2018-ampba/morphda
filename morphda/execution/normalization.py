"""
Output normalization for candidate program results.

Outputs must be comparable across:
  - different formatting styles (0.21 vs 21.0 as percentage)
  - dtype differences (int vs float)
  - ordering of unordered collections
  - label casing
"""

from __future__ import annotations

import math
from typing import Any


def normalize_output(raw: Any, output_type: str) -> Any:
    """
    Normalize a program output for comparison.

    Args:
        raw: The raw output from analyze(tables).
        output_type: One of 'scalar', 'label', 'label_value_pairs', 'ranked_list'.

    Returns:
        Normalized output in a canonical form.

    Raises:
        ValueError: If output does not match the declared contract.
    """
    if raw is None:
        return None

    if output_type == "scalar":
        return _normalize_scalar(raw)
    elif output_type == "label":
        return _normalize_label(raw)
    elif output_type == "label_value_pairs":
        return _normalize_label_value_pairs(raw)
    elif output_type == "ranked_list":
        return _normalize_ranked_list(raw)
    else:
        raise ValueError(f"Unknown output_type: {output_type!r}")


def _normalize_scalar(raw: Any) -> float | int:
    # Accept Python int/float AND numpy scalar types (np.int64, np.float64, etc.)
    # hasattr("item") is the canonical numpy scalar check
    if hasattr(raw, "item"):
        raw = raw.item()  # convert numpy scalar → Python builtin
    if isinstance(raw, (int, float)) and not math.isnan(float(raw)):
        return float(raw)
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.replace(",", "").replace("%", "").strip())
        except ValueError:
            pass
    raise ValueError(f"Cannot normalize scalar from {type(raw).__name__}: {raw!r}")


def _normalize_label(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, (list, tuple)) and len(raw) == 1:
        return _normalize_label(raw[0])
    if isinstance(raw, dict):
        # {"label": "Electronics"} or {"category": "Electronics"}
        values = list(raw.values())
        if len(values) == 1 and isinstance(values[0], str):
            return values[0].strip()
    raise ValueError(f"Cannot normalize label from {type(raw).__name__}: {raw!r}")


def _normalize_label_value_pairs(raw: Any) -> list[tuple[str, float]]:
    if isinstance(raw, dict):
        return sorted(
            [(_normalize_label(k), _normalize_scalar(v)) for k, v in raw.items()]
        )
    if isinstance(raw, (list, tuple)):
        pairs = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((_normalize_label(item[0]), _normalize_scalar(item[1])))
            elif isinstance(item, dict):
                keys = list(item.keys())
                if len(keys) == 2:
                    pairs.append((
                        _normalize_label(item[keys[0]]),
                        _normalize_scalar(item[keys[1]]),
                    ))
        return sorted(pairs)
    raise ValueError(f"Cannot normalize label_value_pairs from {raw!r}")


def _normalize_ranked_list(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [_normalize_label(item) for item in raw]
    if isinstance(raw, str):
        return [raw.strip()]
    raise ValueError(f"Cannot normalize ranked_list from {raw!r}")


def outputs_equal(
    a: Any,
    b: Any,
    output_type: str,
    tolerance: float = 1e-9,
) -> bool:
    """
    Return True if two normalized outputs are semantically equal.

    Returns False (not equal) when normalization fails for either output —
    a type or structure mismatch is itself evidence of a semantic violation.
    """
    try:
        na = normalize_output(a, output_type)
        nb = normalize_output(b, output_type)
    except (ValueError, TypeError):
        return False

    if na is None and nb is None:
        return True
    if na is None or nb is None:
        return False

    if output_type == "scalar":
        na_f, nb_f = float(na), float(nb)
        # Use relative tolerance for large values to handle floating-point addition order
        scale = max(abs(na_f), abs(nb_f), 1.0)
        return abs(na_f - nb_f) <= max(tolerance, 1e-7 * scale)
    elif output_type == "label":
        return str(na).lower() == str(nb).lower()
    elif output_type in ("label_value_pairs", "ranked_list"):
        if len(na) != len(nb):  # type: ignore[arg-type]
            return False
        if output_type == "ranked_list":
            return all(
                str(a_).lower() == str(b_).lower()
                for a_, b_ in zip(na, nb)  # type: ignore[arg-type]
            )
        # label_value_pairs: sorted, compare element-wise
        for (la, va), (lb, vb) in zip(na, nb):  # type: ignore[misc]
            if str(la).lower() != str(lb).lower():
                return False
            if abs(float(va) - float(vb)) > tolerance:
                return False
        return True
    return na == nb
