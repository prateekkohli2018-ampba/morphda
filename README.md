# MORPH-DA

**Mutation-Grounded Benchmark for Metamorphic Verification of Data Analysis Agents**

---

## The Problem

AI agents that answer business questions work by writing and running Python code. A program can **execute successfully and return a confident answer that is completely wrong** — because it silently:

- Forgot to exclude cancelled orders
- Used `mean` instead of `median`
- Applied the date filter to the wrong column
- Counted total sessions instead of distinct customers
- Used `sum` when the question asked for a year-over-year change

There is no error. No crash. The answer just looks correct.

We call these **wrong-but-executable programs**. The current industry-standard check — "did it run?" — catches **0% of them**.

---

## What MORPH-DA Does

MORPH-DA verifies data-analysis programs by running them on **controlled data transformations** and checking whether the outputs satisfy algebraic invariants — without ever seeing the gold answer.

**Example:** A question asks for the top revenue category in 2025. MORPH-DA secretly adds fake rows with dates in 2024 and extreme revenue values. A correct program ignores them (they're outside the requested year). If the answer changes, the program almost certainly has a date-filter bug.

This is **metamorphic testing** applied to analytical programs: instead of checking the final answer, we check that the program *behaves* correctly under structured data variations.

---

## Key Results

### Controlled Mutation Detection (563 known bugs)

| Method | Bugs Caught | Statistical Significance |
|---|---|---|
| Execution-only ("did it run?") | **0%** | — |
| Universal robustness checks | 1.6% | p=0.008 |
| MORPH-DA Filter+Aggregation | 61.5% | p<0.001 ★★★ |
| **Full MORPH-DA** | **64.7%** [60–69%] | **p<0.001 ★★★** |

McNemar's test (Holm-corrected). Task-clustered 95% CI.

**Per-fault-family detection rate:**

| Fault Type | Example | Caught |
|---|---|---|
| Hardcoding | Return `'Electronics'` regardless of data | **85%** |
| Wrong grouping | Group by `department` instead of `category` | **81%** |
| Wrong sort direction | Ascending instead of descending | **76%** |
| Missing filter | Forget `status != cancelled` | **68%** |
| Wrong aggregation | `sum` instead of `mean` | **28%** |

### Natural Agent Evaluation (909 programs, 3 models × 3 seeds)

We ran three Claude models on 101 business analytics tasks. A critical finding: **single-seed gold-answer comparison is too lenient** — 25–33% of programs labeled "correct" on one dataset fail on held-out datasets. After cross-seed correction:

| Model | Programs | True WER† | MORPH Recall | Precision | FPR |
|---|---|---|---|---|---|
| claude-haiku-4-5 | 300 | **53.7%** | 70% | **87%** | 12% |
| claude-sonnet-4-6 | 270 | **44.4%** | 61% | **79%** | 13% |
| claude-opus-4-5 | 300 | **48.3%** | 63% | **77%** | 17% |
| **Average** | **870** | **~49%** | **65%** | **81%** | **14%** |

†WER = wrong-but-executable rate; programs that run successfully but return wrong answers.

---

## Repository Structure

```
morph-da/
├── morphda/                    # Core Python package
│   ├── tasks/                  # Task specification DSL + factory + validator
│   ├── data/                   # Seeded data generators (8 business scenarios)
│   ├── reference/              # Reference program compiler (L1–L5 tasks)
│   ├── relations/              # 30 metamorphic relations across 8 families
│   ├── mutations/              # 34 AST-level mutation operators
│   ├── verification/           # Verification engine + scoring + witnesses
│   ├── execution/              # Safe sandbox + output normalization
│   ├── evaluation/             # Metrics, bootstrap CI, McNemar tests
│   ├── baselines/              # B0–B6 comparison baselines
│   ├── agents/                 # LLM agent + Code-Puppy gateway
│   ├── repair/                 # R0–R7 repair prompt strategies
│   └── logging/                # Structured JSONL experiment logging
├── benchmark/
│   ├── task_specs/             # YAML task specification examples
│   └── frozen_mutants/         # Mutant corpus statistics
├── configs/                    # Benchmark, model, relation configs
├── scripts/                    # Experiment runners
│   ├── validate_references.py  # Phase 1: verify reference programs (0 FP)
│   ├── generate_rule_mutants.py # Generate deterministic mutant corpus
│   ├── run_verification.py     # Run MORPH-DA on mutant corpus
│   ├── run_natural_agents.py   # Run LLM agents on benchmark tasks
│   ├── run_repair.py           # Repair experiment (R0–R7)
│   ├── make_paper_figures.py   # Figures 4 & 5
│   └── monitor_quota.py        # Gateway quota monitor
├── tests/                      # 103 pytest tests
├── paper/
│   ├── morph_da_paper.md       # Full paper draft
│   ├── executive_summary.md    # Non-technical summary
│   └── figures/                # Kill matrix, detection curve CSVs
└── runs/
    └── results_summary.json    # Machine-readable experiment results
```

---

## Metamorphic Relation Families

| Family | Example Relation | Detects |
|---|---|---|
| **Universal** | Row shuffle must not change output | Positional dependence, index use |
| **Filter/Scope** | Out-of-scope rows must be ignored | Missing filters, wrong date bounds |
| **Aggregation** | Row duplication: mean unchanged, sum doubles | Sum vs mean, count vs distinct |
| **Grouping** | Insert dominant group → winner must switch | Hardcoded labels, wrong groupby |
| **Time** | Prior-period boost → YoY metric changes directionally | Period swap, absolute vs. relative |
| **Statistics** | Translation invariance of variance | Wrong aggregation operator |
| **Join** | Ghost dimension rows must be ignored | Cartesian products, wrong join key |
| **Hardcoding** | 1000× scaling → output must change | Hardcoded answers, ignored data |

---

## Quick Start

### Install

```bash
cd morph-da
pip install -e ".[dev]"
```

### Validate the benchmark (Phase 1 exit condition)

```bash
python scripts/validate_references.py
# Expected: 101/101 PASS, 0 MR violations
```

### Generate and verify mutants

```bash
python scripts/generate_rule_mutants.py
python scripts/run_verification.py --workers 6
```

### Run agents (requires Anthropic API key or compatible gateway)

```bash
# With Anthropic API key
ANTHROPIC_API_KEY=... python scripts/run_natural_agents.py --model claude-haiku-4-5 --seeds 42 7 123

# With any OpenAI-compatible gateway
python scripts/run_natural_agents.py --model claude-opus-4-5
```

### Check quota and experiment status

```bash
python scripts/monitor_quota.py --interval 5
```

### Verify a single program (CLI)

```bash
python -m morphda info         # benchmark summary
python -m morphda validate     # Phase 1 check
```

---

## Benchmark Design

**101 tasks** across 8 realistic business scenarios:

| Scenario | Domain | Tables |
|---|---|---|
| `retail01` | Orders and products | orders |
| `web01` | Sessions and conversions | sessions, conversions |
| `market01` | Seller marketplace | seller_orders |
| `saas01` | SaaS subscriptions | subscriptions |
| `mktg01` | Marketing campaigns | campaigns |
| `payments01` | Payments and refunds | transactions |
| `ops01` | Fulfillment operations | shipments |
| `support01` | Customer tickets | tickets |

**5 difficulty levels:**

| Level | Operators | Example |
|---|---|---|
| L1 | Scalar aggregation | "What is the total revenue?" |
| L2 | Grouped ranking + optional date/filter | "Which category had the highest revenue in 2025?" |
| L3 | Ratio with minimum support | "Which category had the highest conversion rate (≥100 sessions)?" |
| L4 | Year-over-year percentage change | "Which category grew the most vs. last year?" |
| L5 | Multi-filter + ratio + YoY + threshold | "Among new customers, which category improved conversion rate the most, for categories with ≥30 sessions?" |

---

## The Lucky-Correct Problem

A key finding from our evaluation: **single-seed gold-answer comparison is insufficient** for evaluating agent correctness. Programs that happen to return the right answer on one dataset may fail on others — because their filter logic is wrong but the bug doesn't matter for that specific data distribution.

We call these **lucky-correct programs**. In our experiments, **25–33% of programs labeled "correct" by single-seed evaluation flip wrong on held-out seeds**. After correcting for this:

- Naive single-seed WER: **25–37%**
- Cross-seed corrected true WER: **44–54%**
- MORPH-DA precision rises from ~51% → **77–87%** (flagged programs are more reliably wrong)

**Recommendation:** Always evaluate agent correctness on ≥3 independent data seeds before drawing conclusions about accuracy.

---

## Citation

```bibtex
@misc{kohli2026morph,
  title  = {{MORPH-DA}: A Mutation-Grounded Benchmark for Metamorphic
             Verification of Data Analysis Agents},
  author = {Kohli, Prateek},
  year   = {2026},
  url    = {https://github.com/prateekkohli2018-ampba/morphda}
}
```

---

## License

MIT

---

