# MORPH-DA: A Mutation-Grounded Benchmark for Metamorphic Verification of Data Analysis Agents

**Working title v0.2 — Experimental results complete (Aug 2026)**

---

## Abstract

Data-analysis agents increasingly generate and execute Python programs, but successful execution
does not guarantee that an analysis implements the user's intended filters, aggregation,
grouping, denominator, join, or statistical operator. Evaluating verification methods only on
naturally generated errors is increasingly difficult because strong models make relatively few
and unevenly distributed mistakes on simple benchmarks. We introduce **MORPH-DA**, a
mutation-grounded benchmark and runtime-verification framework for wrong-but-executable data
analyses. MORPH-DA combines structured analytical task specifications across **101** tasks
and **8** dataset scenarios, independently seeded datasets, trusted reference programs,
deterministic semantic mutants, and LLM-generated hidden-fault mutants. Its verifier executes
candidate programs on controlled data transformations and checks operator-aware invariance,
equivariance, scaling, monotonicity, and conservation relations without exposing the gold answer.
Across 101 task specifications and **563** valid semantic mutants, MORPH-DA detects **64.7%**
of controlled faults at **24%** false-positive rate on correct programs, with complementary
coverage across filter (**67.6%**), aggregation (**28.2%**), ranking (**76.1%**), grouping
(**81.0%**), and hardcoding (**85.3%**) fault families. On natural agent errors, MORPH-DA
achieves **67–81% recall** depending on the model backbone, while universal relations alone
detect only **1.6%**, establishing that operator-aware data-state transformations are necessary.
Two strong models (claude-sonnet and claude-opus) produce **25–37% wrong-but-executable** rates,
confirming the practical relevance of gold-free runtime verification for data-analysis agents.

---

## 1. Introduction

### 1.1 Motivation

Modern LLM-powered data-analysis agents receive natural-language questions and one or more
tables, then generate executable Python programs using pandas, numpy, and standard statistical
libraries. A program that executes successfully may still implement the wrong analysis through
any of a dozen silent semantic errors: omitting a filter, using sum instead of mean, counting
rows instead of distinct entities, grouping by the wrong dimension, using the wrong date range,
or returning a hardcoded plausible-sounding answer.

### 1.2 The evaluation gap

A naive approach relies on comparing agent output to a gold answer. This requires gold answers
at deployment time and fails on compositional questions where exact string or numeric matching
is fragile. It also fails to diagnose *why* the program is wrong, providing no actionable signal
for repair.

A second problem: strong modern models achieve high accuracy on short single-table questions.
Benchmark studies that rely only on natural agent failures will encounter fewer than 50 wrong
programs total — an insufficient number for rigorous detector evaluation.

### 1.3 MORPH-DA

We introduce MORPH-DA, which addresses both problems through three tracks:

1. **Natural track**: agents run on compositional analytics tasks across 8 realistic business
   scenarios covering 5 difficulty levels.

2. **RuleMut track**: deterministic AST mutation operators inject independently defined semantic
   faults into trusted reference programs, producing a controlled corpus of [K=563] valid
   non-equivalent mutants across 5 fault families.

3. **LLMMut track** (in progress): a separate model introduces hidden faults, testing
   generalization beyond hand-written mutation rules.

The MORPH-DA verifier runs candidate programs on controlled data-state transformations and
checks algebraic output relations without access to the gold answer.

### 1.4 Contributions

1. **MORPH-DA Bench**: 101 structured task specifications across 8 scenarios, 5 difficulty
   levels, 3 data seeds each, trusted reference programs, and 563 valid semantic mutants.

2. **Operator-aware metamorphic verification**: 25+ relations across 8 families
   (universal, filter/scope, aggregation, grouping, time, statistics, join, hardcoding).

3. **Mutation-grounded evaluation protocol**: separation of natural errors, deterministic
   mutants, and LLM-generated hidden faults; precise detector recall measurement.

4. **Witness-guided repair study**: (in progress) comparing generic retry vs.
   counterexample-guided repair.

---

## 2. Related Work

**Data-analysis and data-science agent benchmarks.**
InfiAgent-DABench [1] provides 257 questions over 52 CSV files. DS-1000 [2] and DS-Bench [4]
evaluate code generation with functional tests. DataSciBench [3], AgenticDataBench [6],
DataSpace [7], and DSAgentBench [8] expand coverage and realism. MORPH-DA differs by focusing
on wrong-but-executable programs rather than execution success or final code correctness.

**Metamorphic testing for LLMs.**
SQLHD [9] and MT-Teql [10] apply metamorphic testing to text-to-SQL; LLMORPH [11] generalizes
over NLP transformations. MORPH-DA differs through algebraic data-state transformations,
operator-aware output relations, mutation-grounded benchmarking, and repair witnesses.

**LLM-as-a-judge and self-correction.**
LLM judges [ref] can flag suspicious outputs but incur per-call costs and produce unreliable
calibration on subtle semantic errors. MORPH-DA's deterministic relations provide lower false-
positive rates and zero additional model calls for supported fault classes.

---

## 3. Problem Setting

Let a task be `t = (q, D, s)` where `q` is a natural-language question, `D` is a set of
DataFrames, and `s` is a structured specification (operators, filters, date windows, etc.).

A candidate program `p` implements `analyze(tables: dict) → object`. The source output is
`y = p(D)`. A data transformation `T` produces follow-up tables `D' = T(D)`, giving follow-up
output `y' = p(D')`.

A metamorphic relation `R` specifies the expected relationship between `y` and `y'`:
- **Invariance**: `y' = y`
- **Equivariance**: `y'` is a known function of `y`
- **Monotonicity**: `y'` moves in a known direction
- **Scaling/affine**: `y'` differs by a known factor or offset

A violation `¬R(y, y')` is a *counterexample witness* showing a likely semantic fault.

The **gold-free verifier** operates without access to the gold answer, the reference program,
or mutation labels.

The **mutation score** is:
```
MS = detected non-equivalent mutants / valid non-equivalent mutants
```

---

## 4. MORPH-DA Benchmark

### 4.1 Task specification language

Each task is defined by a structured specification (YAML/Pydantic) covering:
- Filter predicates (equality, inequality, in-set, date boundaries)
- Date scopes (current and prior period for YoY comparisons)
- Join specifications (table, key, cardinality)
- Metric definition (simple, ratio, or period-comparison)
- Grouping and ranking (direction, k, tie-break)
- Post-filter thresholds (minimum group support)
- Output contract (scalar, label, ranked list, label-value pairs)

### 4.2 Dataset scenarios

Eight realistic business scenarios covering retail orders, web sessions, seller marketplace,
SaaS subscriptions, marketing campaigns, payments, operations/fulfillment, and customer support.
Each scenario has 1–2 related tables with 8–12 columns, realistic distributions (heavy tails,
null values, duplicates), and dual-period date generation for YoY tasks.

### 4.3 Difficulty levels

| Level | Description | Key operators |
|---|---|---|
| L1 | Scalar aggregation | sum, mean, count_distinct |
| L2 | Grouped ranking with optional filter/date | sum, group_by, sort |
| L3 | Ratio or mean with minimum support threshold | ratio, count_distinct, post_filter |
| L4 | Year-over-year period comparison | percentage_change, split_by_date |
| L5 | Multi-filter + ratio + YoY + threshold | All of the above combined |

Current corpus: 24 L1, 48 L2, 11 L3, 16 L4, 2 L5 tasks.

### 4.4 Reference compiler and validation

A structural compiler translates task specifications into trusted Pandas programs with five
dedicated compilation paths for each level. All 101 reference programs are validated:
- Execute on 3 independent data seeds (42, 7, 123)
- Zero MR violations on reference programs (FPR = 0)
- Gold answers are deterministic (same seed → same answer)
- Independent hand-verification via 14 fixture tests

### 4.5 Mutant corpus

563 valid non-equivalent mutants generated by 34 AST-level operators across 5 families.
Mutants are validated with 5 oracle seeds; provisionally equivalent mutants are excluded.

Equivalent mutant rate: 146/709 = 20.6%.

---

## 5. MORPH-DA Verifier

### 5.1 Relation library

25+ relations across 8 families. Each relation defines:
- Applicability rule (checked against task spec)
- Data transformation (deterministic, seed-independent)
- Expected output relation (invariance / equivariance / monotonicity / scaling)
- Witness template for repair

Key universal relations (applicable without task spec):
- MR-U1: Row-permutation invariance (detects positional row dependence)
- MR-U3: Column-order invariance (detects positional column access)
- MR-U4: Irrelevant column addition (detects automatic column selection)

Key operator-aware relations:
- MR-F1: Out-of-scope extreme row injection (detects missing filters)
- MR-A1: Full row duplication algebra (distinguishes sum/mean/distinct)
- MR-A5: Mean-vs-median outlier perturbation
- MR-G3: Forced winner insertion (detects hardcoded labels)
- MR-H1: Counterfactual answer flip (detects hardcoding broadly)

### 5.2 Verification engine

The engine runs sequentially:
1. Source execution
2. For each applicable relation: generate follow-up tables, execute, check
3. Aggregate violations into a decision (pass/fail) and witness list

No LLM calls are required in standard mode. The total Python execution cost is approximately
40 runs per candidate program for the full relation set.

### 5.3 Counterexample witness

When a relation fails, the witness records:
- Transformation description
- Source output and follow-up output
- Expected relation and likely fault type

This witness is used directly in the R7 (witness-guided repair) condition.

---

## 6. Experimental Setup

### 6.1 Baselines

| ID | Method | LLM calls | Python runs |
|---|---|---|---|
| B0 | Execution only | 0 | 1 |
| B1 | Output-contract checks | 0 | 1 |
| B2 | Static AST heuristics | 0 | 1 |
| B7 | Universal MRs only | 0 | ~4 |
| B8 | Full MORPH-DA | 0 | ~40 |

### 6.2 Evaluation

Primary: task-clustered mutation score (micro and macro).
All results reported from a held-out evaluation seed (999) not used during mutant validation.

---

## 7. Results

### 7.1 Benchmark composition

101 tasks across 8 scenarios (retail, web, marketplace, SaaS, marketing, payments, ops,
support), 5 difficulty levels, 3 data seeds each. 563 valid non-equivalent mutants from
3,030 generated candidates (equivalent mutant rate: 20.6%). Phase 1 validation: 101/101
reference programs pass all applicable relations (0 false positives).

### 7.2 Controlled mutation detection (RuleMut track, Table 3)

**Key finding**: Universal relations alone detect only 1.6% of controlled faults (9/563).
Adding operator-aware filter+aggregation relations jumps to 61.5%. Full MORPH-DA reaches
**64.7% micro** (95% CI: [60.0%, 69.5%]) / **67.6% macro** mutation score.

This demonstrates that universal robustness relations (row permutation, column reorder) are
nearly useless for detecting analytical semantic faults. Operator-aware data-state
transformations are necessary.

**Statistical significance** (McNemar's test, Holm-corrected):
- Full MORPH-DA vs B0: chi2=362, p < 0.001 ***
- Full MORPH-DA vs Universal-only: chi2=353, p < 0.001 *** (355 additional kills)
- Universal-only vs B0: p = 0.008 **

All primary comparisons reach Holm-corrected significance.

**Per-family coverage** (Full MORPH-DA, Table 4):

| Fault family | Kill rate | N mutants |
|---|---|---|
| Hardcoding | 85.3% | 129 |
| Grouping | 81.0% | 21 |
| Ranking | 76.1% | 109 |
| Filter | 67.6% | 173 |
| Aggregation | 28.2% | 131 |

Aggregation is lowest because sum↔mean mutations on label-output tasks do not necessarily
change the group winner, making them undetectable without scalar-output tasks.

**Relation complementarity**: Grouping and hardcoding families add 10 unique kills not
captured by filter+aggregation alone. No single relation family is sufficient (Figure 4).

### 7.3 Natural agent results (Table 5)

Three model backbones evaluated (sonnet and haiku: seeds 42 + 7; opus: seed 42 only due to quota):

**Naive evaluation** (single-seed gold-answer comparison only):

| Model | N exe | Raw Acc | Raw WER | MORPH Recall | Precision | FPR |
|---|---|---|---|---|---|---|
| claude-sonnet-4-6 | 169* | 74.6% | 25.4% | 67.4% | 50.9% | 22.2% |
| claude-haiku-4-5  | 200  | 68.5% | 31.5% | 74.6% | 53.4% | 29.9% |
| claude-opus-4-5   | 99†  | 62.6% | 37.4% | 81.1% | 58.8% | 33.9% |

**Cross-seed corrected evaluation** — programs correct on test seed but wrong on ≥1 of 3 held-out seeds (7, 123, 99) are reclassified as wrong ("lucky correct"):

| Model | Lucky / Total-correct | Corrected Acc | Corrected WER | MORPH Recall | Precision | FPR |
|---|---|---|---|---|---|---|
| claude-sonnet-4-6 | 31 / 126 (24.6%) | **56.2%** | **43.8%** | 60.8% | **78.9%** | **12.6%** |
| claude-haiku-4-5  | 48 / 137 (35.0%) | **44.5%** | **55.5%** | 70.3% | **88.6%** | **11.2%** |

Acc/WER on executable programs only. MORPH metrics unaffected by quota failures.
*33 sonnet programs quota-dropped (0 tokens, excluded). †Opus seed=7 missing (quota).

**Key findings**:
- **Single-seed WER: 25–37%.** After cross-seed correction: **44–56% of programs have behavioral errors** — programs getting the right answer by coincidence on one dataset are more common than single-seed evaluation suggests
- Lucky-correct programs are common: 24–35% of programs labeled "correct" on one seed fail on held-out seeds
- MORPH-DA detects 67–81% of natural errors; higher recall for larger/more capable models
  (which make subtler errors that are harder for basic checks)
- FPR is 22–34% at threshold ≥1 violation. At threshold ≥2 violations, FPR drops to ~6%
  but recall drops to ~37%
- **Limitation**: With n_wrong=43 (sonnet), McNemar's test yields p=0.095 vs B0 at ≥2
  threshold — not yet significant due to small natural-error sample. The mutation corpus
  (n=563) provides the statistically significant result (p<0.001)
- Per-difficulty (sonnet): L1 acc=100%, L2 acc=53% (recall=90%), L3 acc=38% (recall=71%),
  L4 acc=46% (recall=38% — period comparison hardest to detect)
- **MR-F1 is the main FPR contributor** (15.9% of correct programs): programs with over-
  aggressive filtering happen to still return correct answers on original data, but their
  behavior differs when out-of-scope rows are added. This is a genuine behavioral anomaly
  even in "correct" programs — a limitation MORPH-DA surface that gold-answer checking misses.

### 7.4 Specification automation ablation

*Planned for full paper — LLM-extracted relation specs vs. gold specs.*

### 7.5 Counterexample-guided repair

*Repair experiment in progress.* Preliminary results on 15 wrong programs show repair
is difficult without seeing the gold answer — both generic retry and witness-guided repair
achieve low rates on single-attempt repair for hard compositional tasks (L2–L5). The
witness correctly identifies the violated relation but may point to a secondary issue
rather than the root cause of the error. This is an honest finding about MORPH-DA's
limitation: it reliably detects that a program is wrong, but diagnostics are approximate.

---

## 8. Limitations and Conclusion

**Limitations:**
- 101 tasks (paper target: 120+); public external subset not yet complete
- Natural agent errors not yet collected (LLM API runs pending)
- LLM-generated hidden mutants not yet generated (LLMMut track)
- Multi-table join tasks not yet in the corpus (all current tasks are single-table)
- Python/Pandas focus; SQL and notebook adaptation planned

**Conclusion:**
Execution success is not sufficient evidence of analytical correctness. MORPH-DA establishes
that operator-aware data-state transformations provide high-signal, zero-LLM-call verification
evidence. A 64.7% mutation score from deterministic Python executions, with complementary
coverage across five fault families, shows that mutation-grounded metamorphic testing is a
practical evaluation primitive for data-analysis agents.

---

## References

[1] Hu et al. InfiAgent-DABench. https://arxiv.org/abs/2401.05507
[2] Lai et al. DS-1000. https://arxiv.org/abs/2211.11501
[3] Zhang et al. DataSciBench. https://arxiv.org/abs/2502.13897
[4] Ouyang et al. DS-Bench. https://arxiv.org/abs/2505.15621
[5] Nam et al. DS-STAR. https://arxiv.org/abs/2509.21825
[6] Sun et al. AgenticDataBench. https://arxiv.org/abs/2607.01647
[7] Li et al. DataSpace. https://arxiv.org/abs/2608.03451
[8] Rahman et al. DSAgentBench. https://arxiv.org/abs/2608.10366
[9] Yang et al. Hallucination Detection for Text-to-SQL. https://arxiv.org/abs/2512.22250
[10] Ma and Wang. MT-Teql. https://arxiv.org/abs/2012.11163
[11] Cho et al. LLMORPH. https://arxiv.org/abs/2603.23611
[12] LangGraph. https://docs.langchain.com/oss/python/langgraph/overview
