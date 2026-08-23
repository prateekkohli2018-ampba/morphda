"""
B2: Static AST/question heuristics baseline.

Cheap, zero-LLM-call semantic checks based on the question text
and the generated program's AST.

Examples:
  - question says "median" but code calls .mean()
  - question says "distinct" but code uses .count()
  - question says "top 3" but no sort/head found
  - question mentions year 2025 but literal 2025 absent from code
  - question says "cancelled" in exclusion but code has no != check
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass
class HeuristicFlag:
    rule_id: str
    description: str
    confidence: float  # 0-1; used for AUROC curves


@dataclass
class HeuristicResult:
    flagged: bool
    flags: list[HeuristicFlag] = field(default_factory=list)
    checks_run: int = 0


def _ast_has_attr(source: str, *attrs: str) -> dict[str, bool]:
    """Check which method names appear in the program's AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {a: False for a in attrs}

    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in attrs:
            found.add(node.attr)
    return {a: (a in found) for a in attrs}


def _ast_literals(source: str) -> set:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
    }


def _question_lower(question: str) -> str:
    return question.lower()


def run_heuristics(
    question: str,
    program_source: str,
) -> HeuristicResult:
    """
    Run all static heuristics against the question + generated program.

    Returns HeuristicResult with any detected flags.
    """
    flags: list[HeuristicFlag] = []
    q = _question_lower(question)
    attrs = _ast_has_attr(
        program_source,
        "mean", "median", "sum", "count", "nunique",
        "sort_values", "nlargest", "nsmallest", "head", "tail",
        "groupby", "merge",
    )
    literals = _ast_literals(program_source)
    checks = 0

    # median in question but mean() in code
    checks += 1
    if re.search(r"\bmedian\b", q) and attrs.get("mean") and not attrs.get("median"):
        flags.append(HeuristicFlag(
            rule_id="H-MEAN-NOT-MEDIAN",
            description="Question mentions 'median' but code uses .mean() without .median()",
            confidence=0.80,
        ))

    # mean/average in question but median in code
    checks += 1
    if re.search(r"\b(average|mean)\b", q) and attrs.get("median") and not attrs.get("mean"):
        flags.append(HeuristicFlag(
            rule_id="H-MEDIAN-NOT-MEAN",
            description="Question mentions 'average/mean' but code uses .median() without .mean()",
            confidence=0.75,
        ))

    # distinct/unique in question but .count() (not .nunique()) in code
    checks += 1
    if re.search(r"\b(distinct|unique)\b", q) and attrs.get("count") and not attrs.get("nunique"):
        flags.append(HeuristicFlag(
            rule_id="H-COUNT-NOT-NUNIQUE",
            description="Question mentions 'distinct/unique' but code uses .count() without .nunique()",
            confidence=0.75,
        ))

    # "top N" in question but no sort
    checks += 1
    top_match = re.search(r"\btop[\s-]?(\d+)\b", q)
    if top_match:
        k = int(top_match.group(1))
        has_sort = attrs.get("sort_values") or attrs.get("nlargest") or attrs.get("nsmallest")
        if not has_sort:
            flags.append(HeuristicFlag(
                rule_id="H-TOP-K-NO-SORT",
                description=f"Question asks for top-{k} but code has no sort/nlargest operation",
                confidence=0.70,
            ))

    # Year literal in question but missing from code
    checks += 1
    year_matches = re.findall(r"\b(20\d{2})\b", question)
    for yr in year_matches:
        yr_int = int(yr)
        if yr_int not in literals and yr not in str(literals):
            flags.append(HeuristicFlag(
                rule_id="H-MISSING-YEAR-LITERAL",
                description=f"Question references year {yr} but literal not found in program",
                confidence=0.60,
            ))

    # "cancelled" / exclusion in question but no != in code
    checks += 1
    if re.search(r"\b(exclud|cancel|not include|without)\b", q):
        try:
            tree = ast.parse(program_source)
            has_not_eq = any(
                isinstance(node, ast.Compare) and any(isinstance(op, ast.NotEq) for op in node.ops)
                for node in ast.walk(tree)
            )
            if not has_not_eq:
                flags.append(HeuristicFlag(
                    rule_id="H-MISSING-EXCLUSION",
                    description="Question implies an exclusion filter but code has no != comparison",
                    confidence=0.55,
                ))
        except SyntaxError:
            pass

    # join in question but no .merge() in code
    checks += 1
    if re.search(r"\b(join|joining|across|between)\b", q) and not attrs.get("merge"):
        flags.append(HeuristicFlag(
            rule_id="H-MISSING-JOIN",
            description="Question implies a join but code has no .merge() call",
            confidence=0.55,
        ))

    return HeuristicResult(
        flagged=len(flags) > 0,
        flags=flags,
        checks_run=checks,
    )
