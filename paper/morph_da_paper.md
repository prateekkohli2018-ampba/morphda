# MORPH-DA: A Mutation-Grounded Benchmark for Metamorphic Verification of Data Analysis Agents

**Working paper v0.4 — All experiments complete (Aug 2026)**

---

## Abstract

Data-analysis agents increasingly generate and execute Python programs to answer business questions, but successful execution does not guarantee that an analysis implements the correct filters, aggregation operators, grouping dimensions, denominators, date windows, or statistical methods. We introduce **MORPH-DA**, a mutation-grounded benchmark and runtime-verification framework for detecting wrong-but-executable data-analysis programs. MORPH-DA combines 101 structured analytical tasks across 8 business scenarios and 5 difficulty levels with 563 validated semantic mutants, operator-aware metamorphic relations, and a counterfactual witness generator — all without exposing the gold answer. Across 563 mutants, MORPH-DA detects **64.7%** [58.5%, 70.5%] of controlled faults — **40× more effective than universal robustness checks alone** (1.6%) — with McNemar χ²=353, p<10⁻⁶. On natural agent programs from three Claude models, a critical finding emerges: single-seed gold-answer evaluation is unreliable because **38–45 programs per model** (37–44%) pass single-seed evaluation by coincidence but fail on held-out data seeds. After correcting for these **accidental corrects** via cross-seed validation, MORPH-DA achieves **81–88% precision** and **56–68% recall** at 10–14% false-positive rate (Condition B). Adding a second metamorphic transformation seed (Condition C) improves recall to **61–78%** at comparable precision with F1 gains of 6–7 percentage points. For repair, witness-guided feedback (R6/R7) achieves **12.1%** one-shot fix rate versus **5.5%** for generic feedback on 91 wrong programs, with relation-name identification being the key actionable signal.

---

## 1. Introduction

### 1.1 Motivation

Modern LLM-powered data-analysis agents receive natural-language questions and tabular data, then generate executable Python programs using pandas and numpy. A program that executes successfully may still implement the wrong analysis through any of a dozen silent semantic errors: omitting a filter, using sum instead of mean, counting rows instead of distinct entities, grouping by the wrong dimension, using the wrong date range, swapping current and prior periods in a year-over-year comparison, or returning a hardcoded answer.

We call these **wrong-but-executable programs (WEPs)**. The naive check — "did it run without error?" — catches 0% of them. Gold-answer comparison catches them, but requires knowing the correct answer at inference time and provides no diagnostic signal for repair.

### 1.2 The Accidental Correct Problem

A subtler evaluation failure mode: a program can accidentally return the correct answer on one data distribution but fail on others. For example, a program that omits the `order_status != 'cancelled'` filter may still return the correct top category on a particular dataset because the correct category dominates even with cancelled orders included. On a different data seed, the filter changes the winner. We term these **accidental corrects**: structurally wrong programs that pass single-seed gold-answer evaluation by coincidence.

In our experiments with three Claude models on 101 tasks across 3 seeds, **38–45 programs per model** (37–44% of single-seed "correct" programs) are accidental corrects. Single-seed evaluation overstates true accuracy by 15–19 percentage points. Cross-seed correction — running each model on seeds 7, 42, and 123 and requiring correctness on all three — produces a more reliable ground truth.

### 1.3 MORPH-DA

MORPH-DA addresses the WEP detection problem through three components:

1. **Natural track**: LLM agents run on compositional analytics tasks across 8 realistic business scenarios covering 5 difficulty levels (L1 scalar → L5 cohort ratio + YoY comparison).

2. **RuleMut track**: 563 validated deterministic semantic mutants (5 fault families) enable rigorous measurement of verifier detection coverage without relying on natural agent failure rates.

3. **Repair track**: Four feedback strategies (R0–R7) compare generic vs. witness-guided repair at scale.

The MORPH-DA verifier executes candidate programs on controlled data transformations and checks operator-aware invariance, equivariance, scaling, monotonicity, and conservation relations — entirely without the gold answer.

### 1.4 Contributions

1. **MORPH-DA Bench**: 101 tasks, 8 business scenarios, 5 difficulty levels, 3 seeds each, 563 valid semantic mutants, reference compiler, and full experimental results.

2. **Accidental correct analysis**: Cross-seed methodology revealing that 37–44% of single-seed "correct" programs are structurally wrong, with MORPH-DA catching 36–64% of them using only metamorphic testing.

3. **Three-condition evaluation framework**: Naive (production-realistic), cross-seed corrected (benchmark standard), and multi-seed MORPH-DA (recommended deployment), reported side-by-side.

4. **Operator-aware metamorphic verification**: 20+ relations across 8 families, with concrete detection mechanisms and failure mode analysis per family.

5. **Repair study at scale**: n=91 wrong programs, showing relation-name identification doubles one-shot repair rate vs generic feedback.

---

## 2. Related Work

**Data-analysis and data-science agent benchmarks.**
InfiAgent-DABench [1] provides 257 questions over 52 CSV files. DS-1000 [2] and DS-Bench [4] evaluate code generation with functional tests. DataSciBench [3], AgenticDataBench [6], DataSpace [7], and DSAgentBench [8] expand coverage and realism. MORPH-DA differs by focusing on wrong-but-executable programs, by measuring detector coverage via a controlled mutation corpus, and by identifying the accidental correct problem in single-seed evaluation.

**Metamorphic testing for LLMs.**
SQLHD [9] and MT-Teql [10] apply metamorphic testing to text-to-SQL; LLMORPH [11] generalizes over NLP transformations. MORPH-DA differs through algebraic data-state transformations, operator-aware output relations, mutation-grounded benchmarking, and repair witnesses.

**LLM-as-a-judge and self-correction.**
LLM judges can flag suspicious outputs but incur per-call costs and produce unreliable calibration on subtle semantic errors. MORPH-DA's deterministic relations provide lower false-positive rates and zero additional model calls for supported fault classes.

---

## 3. Problem Setting

Let a task be `t = (q, D, s)` where `q` is a natural-language question, `D` is a set of DataFrames, and `s` is a structured specification (operators, filters, date windows, etc.).

A candidate program `p` implements `analyze(tables: dict) → object`. The source output is `y = p(D)`. A data transformation `T` produces follow-up tables `D' = T(D)`, giving follow-up output `y' = p(D')`.

A metamorphic relation `R` specifies the expected relationship between `y` and `y'`:
- **Invariance**: `y' = y` (e.g., adding out-of-scope rows must not change the output)
- **Equivariance**: `y'` is a known function of `y` (e.g., doubling rows doubles the sum)
- **Monotonicity**: `y'` moves in a known direction (e.g., boosting current period increases YoY change)
- **Forced winner**: dominant sentinel group must become the output label

A violation `¬R(y, y')` is a **counterexample witness** showing a likely semantic fault. The **gold-free verifier** operates without access to the gold answer, the reference program, or mutation labels. The **mutation score** is: MS = detected non-equivalent mutants / valid non-equivalent mutants.

---

## 4. MORPH-DA Benchmark

### 4.1 Task specification language

Each task is defined by a structured Pydantic specification covering: filter predicates (equality, inequality, in-set, date boundaries), date scopes (current and prior period for YoY comparisons), join specifications, metric definition (simple aggregation, ratio, or period-comparison), grouping and ranking (direction, k, tie-break), post-filter thresholds (minimum group support), and output contract (scalar, label, ranked list).

### 4.2 Dataset scenarios and filter discriminability

Eight realistic business scenarios: retail orders, web sessions, seller marketplace, SaaS subscriptions, marketing campaigns, payments, operations/fulfillment, and customer support. Each scenario has 1–2 related tables with 8–12 columns, realistic distributions (heavy tails, null values, duplicates), and dual-period date generation for YoY tasks.

**Filter discriminability guarantee**: For each task with a required filter, we verify that applying the filter changes the gold answer on all three evaluation seeds. Three tasks failed this check and were excluded from the 98-task evaluation set used in Conditions B and C (Section 6). In cases where a filter is non-discriminating, programs that omit the filter would pass evaluation even when incorrect; excluding such tasks ensures the evaluation ground truth is clean.

### 4.3 Difficulty levels

| Level | Description | Key operators |
|---|---|---|
| L1 | Scalar aggregation | sum, mean, count_distinct |
| L2 | Grouped ranking with optional filter/date | sum, group_by, sort |
| L3 | Ratio or mean with minimum support threshold | ratio, count_distinct, post_filter |
| L4 | Year-over-year period comparison | percentage_change, split_by_date |
| L5 | Multi-filter + ratio + YoY + threshold | All combined |

Corpus: 24 L1, 48 L2, 11 L3, 16 L4, 2 L5 tasks.

### 4.4 Reference compiler and validation

A structural compiler translates task specifications into trusted Pandas programs via five compilation paths. All 101 reference programs are validated: execute on 3 independent seeds (7, 42, 123), zero MR violations (FPR = 0 on reference programs), deterministic gold answers, and independent hand-verification via fixture tests.

### 4.5 Mutant corpus

563 valid non-equivalent mutants from AST-level operators across 5 families. Mutants are validated with 5 oracle seeds; provisionally equivalent mutants are excluded. Equivalent mutant rate: 20.6% (146/709 candidates).

---

## 5. MORPH-DA Verifier

### 5.1 Relation library

20+ relations across 8 families. Each relation defines: an applicability rule (checked against the task spec), a deterministic data transformation, an expected output relation, and a witness template for repair.

Key universal relations (applicable without task spec):
- **MR-U1**: Row-permutation invariance
- **MR-U3**: Column-order invariance
- **MR-U4**: Irrelevant column addition

Key operator-aware relations:
- **MR-F1**: Out-of-scope extreme row injection (detects missing filters) — injects filter-violating rows with extreme metric values; correct programs ignore them
- **MR-F2**: In-scope sentinel sensitivity (detects hardcoded labels or over-filtering)
- **MR-A1**: Full row duplication algebra (distinguishes sum/mean/count/distinct)
- **MR-T4**: Forced YoY winner insertion (detects period swap, absolute vs. relative change)
- **MR-H1**: Counterfactual answer flip (detects hardcoding broadly)

### 5.2 Verification engine

The engine runs sequentially: (1) execute on source data, (2) for each applicable relation: generate follow-up tables, execute, check expected relation, (3) aggregate violations into pass/fail decision and witness list.

**Multi-seed MORPH-DA**: The engine can be run with multiple transformation random seeds (rng_seed). Running with rng_seeds {42, 7} generates different metamorphic test cases per seed; flagging if either fires increases recall at a small FPR cost. This is our recommended deployment configuration (Condition C, Section 6).

No LLM calls are required. Latency: 500ms median, 1.1s 95th percentile per task (20-task benchmark).

### 5.3 Counterexample witness

When a relation fails, the witness records the transformation description, source and follow-up outputs, expected relation, and likely fault type. This is used directly in the R6/R7 repair conditions.

---

## 6. Experimental Setup

### 6.1 Evaluation conditions

We report three conditions to give an honest picture across deployment scenarios:

| Condition | Ground truth | MORPH-DA config | Use case |
|---|---|---|---|
| **A — Naive** | Single-seed (seed=42) gold-answer comparison, 101 tasks | Single rng-seed (42) | Production: no cross-seed available |
| **B — Cross-seed corrected** | Programs correct on seed=42 but wrong on seeds 7 or 123 reclassified as wrong. 98-task set (3 non-discriminating tasks excluded) | Single rng-seed (42) | Benchmark standard |
| **C — Multi-seed MORPH-DA** | Same as B | Two rng-seeds (42, 7); flag if either fires | Recommended deployment |

**Accidental corrects** (Condition A→B correction): Data is generated deterministically, so a program failing on a different seed fails due to a structural bug, not randomness. Reclassifying these programs correctly reveals MORPH-DA's true detection performance.

### 6.2 Models and seeds

Three Claude model backbones evaluated: claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-5. Each run on seeds 7, 42, 123 (303 programs per model = 909 total).

### 6.3 Statistical tests

McNemar's test (continuity-corrected, χ² distribution, 1 df) for paired binary comparisons. Holm-Bonferroni correction for 3 simultaneous model comparisons. Task-clustered bootstrap (3,000 iterations) for confidence intervals — resamples tasks, not individual programs, to avoid pseudo-replication from multiple seeds per task.

### 6.4 Baselines

| ID | Method | LLM calls |
|---|---|---|
| B0 | Universal MRs only (MR-U1/U2/U3/U4) | 0 |
| B8 | Full MORPH-DA | 0 |

---

## 7. Results

### 7.1 Benchmark composition

101 tasks, 8 scenarios, 5 difficulty levels, 3 seeds. 563 valid semantic mutants (equivalent mutant rate: 20.6%). Phase 1 validation: 101/101 reference programs pass all applicable relations (FPR = 0).

### 7.2 Controlled mutation detection (Table 1)

**Table 1 — Mutation Score by Fault Family (563 mutants)**

| Fault family | Example bug | Kill rate | 95% CI | Universal rate |
|---|---|---|---|---|
| Hardcoding | Return constant instead of computing | **85.3%** | [77.6%, 92.0%] | 0.0% |
| Grouping | Wrong GROUP BY column | **81.0%** | [54.5%, 100%] | 0.0% |
| Ranking | Ascending instead of descending | **76.1%** | [65.8%, 85.6%] | 0.0% |
| Filter/Scope | Missing `status != 'cancelled'` filter | **67.6%** | [57.1%, 77.5%] | 0.6% |
| Aggregation | `.sum()` instead of `.mean()` | **28.2%** | [17.2%, 39.0%] | 6.1% |
| **Overall** | | **64.7%** | **[58.5%, 70.5%]** | **1.6%** |

McNemar vs Universal: χ²=353, p<10⁻⁶ (n₀₁=355, n₁₀=0). MORPH-DA is a strict detection superset of Universal: every bug Universal catches, MORPH-DA also catches, plus 355 additional detections.

**Why aggregation detection is lowest (28.2%)**: For label-output (ranking winner) tasks, `sum→mean` mutations may not change the winning category if the same group has both the highest sum and the highest mean. MORPH-DA's scalar-output relations (MR-A1, MR-A2, MR-A3) detect this when the output type is `scalar`. This is a fundamental limitation of oracle-free testing on ranking tasks: winner identity does not uniquely determine the underlying computation.

### 7.3 Accidental corrects and ground truth correction (Table 2)

**Table 2 — Accidental Corrects Found via Cross-Seed Testing**

| Model | Seed=42 correct | Truly correct (all 3 seeds) | Accidental corrects | MORPH-DA catches |
|---|---|---|---|---|
| claude-haiku-4-5 | 61/101 (60.4%) | 46/101 (45.5%) | **45** | 29 (64%) |
| claude-sonnet-4-6 | 65/101 (64.4%) | 51/101 (50.5%) | **42** | 15 (36%) |
| claude-opus-4-5 | 62/101 (61.4%) | 49/101 (48.5%) | **38** | 21 (55%) |

MORPH-DA catches 36–64% of accidental corrects using single-seed metamorphic testing alone — without access to held-out seeds. The remaining accidental corrects it misses represent structural bugs that no current metamorphic relation detects (primarily aggregation bugs in label-output tasks, Section 7.2).

### 7.4 Natural agent verification results (Table 3)

**Table 3 — MORPH-DA Verification Metrics Under Three Conditions**

| Model | Condition | Precision | Recall | F1 | FPR | AAR† |
|---|---|---|---|---|---|---|
| claude-haiku-4-5 | A) Naive (101 tasks) | 62.0% | 70.2% | 65.8% | 26.8% | 20.2% |
| | B) Cross-seed corrected (98 tasks) | **87.6%** | 67.9% | 76.5% | **11.4%** | 29.9% |
| | C) Multi-seed MORPH-DA | 86.5% | **78.2%** | **82.2%** | 14.4% | **23.1%** |
| claude-sonnet-4-6 | A) Naive | 65.1% | 63.9% | 64.5% | 19.0% | 19.8% |
| | B) Cross-seed corrected | **84.9%** | 56.0% | 67.5% | **10.4%** | 33.9% |
| | C) Multi-seed MORPH-DA | 81.4% | **61.3%** | **70.0%** | 14.6% | **32.0%** |
| claude-opus-4-5 | A) Naive | 61.4% | 63.1% | 62.2% | 23.8% | 22.5% |
| | B) Cross-seed corrected | **81.1%** | 60.1% | 69.1% | **13.9%** | 31.5% |
| | C) Multi-seed MORPH-DA | 80.0% | **72.7%** | **76.2%** | 18.1% | **24.8%** |

†AAR = Accepted-Answer Risk = FN/(FN+TN). Fraction of programs MORPH-DA passes that are actually wrong.

McNemar significance (MORPH-DA full vs Universal-only):

| Model | χ² | p-value (Holm) | n₀₁ | n₁₀ |
|---|---|---|---|---|
| claude-haiku-4-5 | 127.0 | **p < 0.0001** | 129 | 0 |
| claude-sonnet-4-6 | 104.0 | **p < 0.0001** | 106 | 0 |
| claude-opus-4-5 | 116.0 | **p < 0.0001** | 118 | 0 |

All n₁₀ = 0: MORPH-DA strictly dominates universal-only detection.

**Reading the conditions:**

- **Why Condition A has low precision (61–65%)**: Accidental corrects are labeled as "correct" in the ground truth. When MORPH-DA correctly flags their structural bugs, these are counted as false positives under the naive ground truth — a labeling artifact, not a MORPH-DA failure.

- **Why Condition B improves precision to 81–88%**: Cross-seed correction relabels accidental corrects as wrong. MORPH-DA flags that were classified as FPs under Condition A become TPs, revealing MORPH-DA's true detection performance.

- **Why Condition C improves recall (+8–13pp) at small FPR cost (+3–4pp)**: A second transformation seed generates different test cases, exposing bugs that one seed misses. F1 improves by 6–7pp across all models.

- **Accepted-answer risk (AAR)**: Under Condition B, 30–34% of programs MORPH-DA passes as correct are actually wrong — the primary limitation motivating multi-seed deployment (Condition C reduces AAR to 23–32%).

**False positive breakdown by relation family:**

| Relation | Haiku FPs | Sonnet FPs | Opus FPs | Root cause |
|---|---|---|---|---|
| MR-F1 | 45 | 25 | 41 | Extreme out-of-scope rows affect programs with imprecise filter logic |
| MR-G3 | 12 | 21 | 7 | Tie-break sensitivity on groups with equal metric values |
| MR-H1 | 12 | 20 | 7 | Hardcoding detector fires on legal computed constants |
| MR-F2 | 13 | 19 | 6 | Sentinel group insufficient margin on some tasks |
| MR-T1 | 8 | 10 | 10 | Outside-window rows affect tasks without date-gated metric |

### 7.5 Repair experiment (Table 4)

**Table 4 — One-Shot Repair Results (n=91 wrong-but-executable programs)**

| Strategy | Feedback provided | Fixed | Rate |
|---|---|---|---|
| R0 — No retry | None (baseline) | 0/91 | 0.0% |
| R2 — Generic feedback | "Your program has a bug, please fix it" | 5/91 | **5.5%** |
| R6 — Relation name | "You may have violated MR-F1 (filter/scope)" | 11/91 | **12.1%** |
| R7 — Witness-guided | Source output, follow-up output, transformation description, likely issue | 11/91 | **12.1%** |

Naming the violated relation (R6) doubles the one-shot repair rate vs generic feedback (12.1% vs 5.5%). Providing the full counterexample witness (R7) achieves the same rate as relation-name identification in a single attempt. The bottleneck is not information richness but number of repair rounds. For R6 vs R7: at n=91 with observed discordant pairs of 0 (both strategies fix the same programs), McNemar's test is uninformative — a larger study with multi-round repair loops is needed to distinguish R6 from R7.

**Recommended repair architecture**: Apply R7 witnesses in a multi-round loop (verify → repair → verify, up to 3–5 rounds). The structured counterexample supports iterative reasoning even when a single attempt fails.

---

## 8. Limitations

1. **Accidental corrects in Condition A**: Without cross-seed testing, programs that pass evaluation by coincidence inflate the FP count. Condition A numbers represent the realistic production scenario where cross-seed testing is unavailable. Condition B numbers are more appropriate for benchmarking but require 3× the LLM evaluation cost.

2. **Filter non-discriminating data**: 3/101 tasks had filters that were empirically non-discriminating across all evaluation seeds (answer unchanged with or without filter). These were excluded from Conditions B and C. Future benchmark releases should verify filter discriminability by construction during data generation.

3. **Aggregation family detection gap (28.2%)**: For label-output ranking tasks, wrong aggregation operators may produce the same winner. This is a fundamental limitation of oracle-free testing on ranking: winner identity does not uniquely determine the computation.

4. **Single-table focus**: All 101 current tasks use single-table scenarios. Multi-table join tasks are planned for future releases.

5. **Repair study underpowered for R6 vs R7**: With n=91 and 0 discordant pairs between R6 and R7, the repair study cannot distinguish the two strategies. Multi-round repair experiments are needed.

6. **Natural errors only from Anthropic models**: Generalizability to other model families (GPT-4o, Gemini) is not established.

---

## 9. Conclusion

Execution success is not evidence of analytical correctness. Single-seed gold-answer evaluation further understates the problem: 37–44% of programs that pass single-seed evaluation are structurally wrong accidental corrects. MORPH-DA addresses both problems through operator-aware metamorphic testing.

Across 563 controlled semantic mutants, MORPH-DA detects 64.7% of faults at 40× the detection rate of universal robustness checks (p<10⁻⁶). On natural agent programs with cross-seed corrected ground truth, MORPH-DA achieves 81–88% precision and 56–68% recall at 10–14% FPR. Multi-seed MORPH-DA improves recall by 8–13pp with F1 gains of 6–7pp. Relation-name-guided repair doubles the one-shot fix rate over generic feedback. These results establish operator-aware data-state metamorphic testing as a practical, zero-LLM-call verification primitive for data-analysis agents.

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
