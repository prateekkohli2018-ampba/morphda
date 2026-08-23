# MORPH-DA

**Mutation-Grounded Benchmark for Metamorphic Verification of Data Analysis Agents**

---

## The Problem

AI agents that answer business questions work by writing and running Python code. A program can **execute successfully and return a confident answer that is completely wrong** — because it silently:

- Forgot to exclude cancelled orders
- Used `mean` instead of `sum`
- Applied the date filter to the wrong column
- Counted total sessions instead of distinct customers
- Used absolute change when the question asked for year-over-year percentage

There is no error. No crash. The answer just looks correct. We call these **wrong-but-executable programs**.

There are two ways they slip through:

**1. Execution-only testing catches 0%** — if the program runs without crashing, it passes.

**2. Single-seed gold-answer comparison is unreliable** — a program can accidentally return the correct answer on one dataset but fail on others. We call these **accidental corrects**: their filter or aggregation logic is wrong, but it doesn't happen to matter on that particular data distribution. In our experiments, **38–45 programs per model** passed single-seed evaluation this way.

---

## What MORPH-DA Does

MORPH-DA verifies data-analysis programs by running them on **controlled data transformations** and checking whether outputs satisfy mathematical invariants — without ever seeing the gold answer.

**Example 1 — Filter bug detection (MR-F1):**
Question asks for top revenue category among non-cancelled orders in 2025. MORPH-DA adds rows with `order_status='cancelled'` and extreme revenue values. A correct program ignores them. If the answer changes, the status filter is missing.

**Example 2 — Aggregation bug detection (MR-A1):**
MORPH-DA doubles all rows. The `mean` must be unchanged; the `sum` must double. A program using `sum` instead of `mean` is exposed.

**Example 3 — Period swap detection (MR-T4):**
MORPH-DA inserts a synthetic group with a huge year-over-year increase. A correct program reports it as the winner. A program with swapped current/prior periods does not.

This is **metamorphic testing**: instead of checking the final answer, check that the program *behaves correctly* under structured input variations.

---

## Key Results

### Controlled Mutation Detection (563 validated mutants)

| Method | Bugs Caught | 95% CI | McNemar p |
|---|---|---|---|
| Execution-only | 0% | — | — |
| Universal relations only | 1.6% | — | baseline |
| **Full MORPH-DA (single rng-seed)** | **64.7%** | [58.5%, 70.5%] | **p < 0.000001** |

MORPH-DA is **40× more effective** than the universal-only baseline.

**Per-fault-family detection rate:**

| Fault Family | Example bug | Kill rate | 95% CI |
|---|---|---|---|
| Hardcoding | Return `'Electronics'` instead of computing | **85.3%** | [77.6%, 92.0%] |
| Grouping | Wrong GROUP BY column | **81.0%** | [54.5%, 100%] |
| Ranking | Ascending instead of descending sort | **76.1%** | [65.8%, 85.6%] |
| Filter/Scope | Missing `status != 'cancelled'` filter | **67.6%** | [57.1%, 77.5%] |
| Aggregation | `.sum()` instead of `.mean()` | **28.2%** | [17.2%, 39.0%] |

*Note: Aggregation family is hardest — for label (ranking winner) output, the wrong aggregation may still produce the correct winner. MORPH-DA catches these when output type is scalar.*

### Natural Agent Evaluation (3 models × 3 seeds × 101 tasks)

Results are reported under three conditions with increasing rigor:

| Condition | What it measures |
|---|---|
| **A) Naive** | Single-seed evaluation, no correction. Represents production setting. |
| **B) Cross-seed corrected** | Programs correct on seed=42 but wrong on seeds 7 or 123 are reclassified as wrong (accidental corrects removed). 98-task set (3 filter-non-discriminating tasks excluded). |
| **C) Multi-seed MORPH-DA** | Condition B ground truth + MORPH-DA run with 2 transformation seeds (rng=42 and rng=7). Flagged if either fires. |

| Model | Condition | Precision | Recall | FPR | F1 |
|---|---|---|---|---|---|
| claude-haiku-4-5 | A) Naive | 62.0% | 70.2% | 26.8% | 65.8% |
| | B) Cross-seed corrected | **87.6%** | 67.9% | **11.4%** | 76.5% |
| | C) Multi-seed MORPH-DA | 86.5% | **78.2%** | 14.4% | **82.2%** |
| claude-sonnet-4-6 | A) Naive | 65.1% | 63.9% | 19.0% | 64.5% |
| | B) Cross-seed corrected | **84.9%** | 56.0% | **10.4%** | 67.5% |
| | C) Multi-seed MORPH-DA | 81.4% | **61.3%** | 14.6% | **70.0%** |
| claude-opus-4-5 | A) Naive | 61.4% | 63.1% | 23.8% | 62.2% |
| | B) Cross-seed corrected | **81.1%** | 60.1% | **13.9%** | 69.1% |
| | C) Multi-seed MORPH-DA | 80.0% | **72.7%** | 18.1% | **76.2%** |

McNemar χ² vs universal baseline: all p < 0.0001 (Holm-Bonferroni corrected).

**MORPH-DA catches 36–64% of accidental corrects** using single-seed transformations alone, without cross-seed access.

### Repair Experiment (n=91 wrong programs)

| Strategy | Description | Fix rate |
|---|---|---|
| R0 — No retry | Baseline | 0.0% |
| R2 — Generic feedback | "Your program has a bug" | 5.5% |
| R6 — Relation name | "You violated MR-F1 (filter/scope)" | **12.1%** |
| R7 — Witness-guided | Full counterexample + diagnosis | **12.1%** |

Naming the violated relation doubles the one-shot repair rate. For multi-round repair, use R7 witnesses iteratively.

---

## Repository Structure

```
morph-da/
├── morphda/                    # Core Python package
│   ├── tasks/                  # Task specification DSL (Pydantic) + 101-task factory
│   ├── data/                   # Seeded data generators (8 business scenarios)
│   ├── reference/              # Reference program compiler (L1–L5)
│   ├── relations/              # 20+ metamorphic relations across 8 families
│   ├── mutations/              # AST-level mutation operators (5 families)
│   ├── verification/           # Engine: relations → decision + witnesses
│   ├── execution/              # Sandboxed Python runner + output normalization
│   ├── evaluation/             # Metrics, bootstrap CI, McNemar tests
│   ├── baselines/              # Comparison baselines (B0–B5)
│   ├── agents/                 # LLM agent harness (Anthropic-compatible)
│   ├── repair/                 # Repair prompt strategies (R0–R7)
│   └── logging/                # Structured JSONL experiment logging
├── benchmark/
│   ├── task_specs/             # YAML task specifications
│   └── frozen_mutants/         # Mutant corpus statistics
├── configs/                    # Benchmark, model, relation configs
├── scripts/
│   ├── validate_references.py  # Phase 1: verify 0 false positives on references
│   ├── generate_rule_mutants.py
│   ├── run_verification.py
│   ├── run_natural_agents.py
│   ├── run_natural_agents_parallel.py
│   ├── run_repair.py
│   └── make_paper_figures.py
├── tests/                      # pytest test suite
├── runs/
│   ├── paper_results.json      # All computed statistics
│   ├── complete_honest_results.json  # Three-condition metrics
│   ├── filter_discriminability.json  # Filter non-discriminating task analysis
│   └── zero_cost_analyses.json       # Ablation, latency, PR tradeoff
├── CODEBASE_REVIEW.md          # Deep-dive: every module explained
└── CODEBASE_REVIEW.html        # Standalone interactive HTML (open directly)
```

---

## Metamorphic Relation Families

| Family | Relations | What it detects |
|---|---|---|
| **Universal** | MR-U1 to MR-U4 | Row order dependence, index use, column position access |
| **Filter/Scope** | MR-F1 to MR-F4 | Missing filters, wrong date bounds, AND→OR conversion |
| **Aggregation** | MR-A1 to MR-A9 | sum↔mean, count↔distinct, wrong denominator |
| **Grouping** | MR-G1 to MR-G4 | Wrong GROUP BY, hardcoded labels |
| **Time** | MR-T1 to MR-T5 | Period swap, absolute vs relative change, wrong date window |
| **Statistics** | MR-S* | Translation invariance, variance computation |
| **Joins** | MR-J* | Cartesian products, wrong join key |
| **Hardcoding** | MR-H* | Answers that ignore input data entirely |

---

## Quick Start

### Install

```bash
pip install -e ".[dev]"
```

### Validate the benchmark

```bash
python scripts/validate_references.py
# Expected: 101/101 PASS, 0 MR violations on reference programs
```

### Run agents

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Sequential (pilot)
python scripts/run_natural_agents.py --model claude-haiku-4-5 --seeds 42 7 123

# Parallel (full run)
python scripts/run_natural_agents_parallel.py \
  --model claude-haiku-4-5 --seeds 42 7 123 --workers 8
```

### Custom / compatible LLM endpoint

```bash
# Any Anthropic-compatible endpoint
export MORPH_DA_API_URL=https://your-proxy.example.com/v1/messages
export MORPH_DA_API_KEY=your-key

python scripts/run_natural_agents.py --model claude-haiku-4-5
```

### Verify a program directly

```python
from morphda.relations import ALL_RELATIONS
from morphda.verification.engine import VerificationEngine
from morphda.tasks.factory import generate_task_set
from morphda.data.generators import generate_scenario

task   = generate_task_set()[5]          # pick any task
tables = generate_scenario(task.scenario_id, seed=42)
engine = VerificationEngine(ALL_RELATIONS)

report = engine.verify(
    program_source=your_program_string,
    tables=tables,
    task_spec=task,
)
print(report.decision)      # "pass" | "fail" | "error"
print(report.witnesses)     # counterexamples if fail
```

---

## Benchmark Design Principles

- **101 tasks**, 8 business scenarios, 5 difficulty levels (L1 scalar → L5 cohort ratio + YoY)
- **Deterministic data**: every seed always produces the same tables — correctness is exact, not probabilistic
- **Filter discriminability**: data generators ensure required filters always change the answer
- **No gold answer in verifier**: MORPH-DA relations are mathematically derived — the verifier never sees the reference output
- **Mutation operators blind to relations**: mutation operators don't import relation code, ensuring mutation score is a genuine test

---

## Documentation

| File | Contents |
|---|---|
| `CODEBASE_REVIEW.md` | Every module explained: what it does, why, how it connects, known edge cases |
| `CODEBASE_REVIEW.html` | Same content as interactive webpage — open directly in any browser, no server needed |
| `runs/paper_results.json` | All computed statistics (precision, recall, CIs, McNemar tests) |
| `runs/complete_honest_results.json` | Three-condition metrics side by side |
| `paper/morph_da_paper.md` | Full paper draft |

---

## Citation

```bibtex
@misc{kohli2026morphda,
  title  = {{MORPH-DA}: Mutation-Grounded Benchmark for Metamorphic
             Verification of Data Analysis Agents},
  author = {Kohli, Prateek},
  year   = {2026},
  url    = {https://github.com/prateekkohli2018-ampba/morphda}
}
```

---

## License

MIT
