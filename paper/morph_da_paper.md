# MORPH-DA: A Mutation-Grounded Benchmark for Metamorphic Verification of Data Analysis Agents

**Working paper v0.5 — All experiments complete, methodology corrected (Aug 2026)**

---

## Abstract

Data-analysis agents increasingly generate and execute Python programs to answer business questions, but successful execution does not guarantee that an analysis implements the correct filters, aggregation operators, grouping dimensions, denominators, or date windows. We introduce **MORPH-DA**, a mutation-grounded benchmark and runtime-verification framework for detecting wrong-but-executable data-analysis programs. MORPH-DA combines 101 structured analytical tasks across 8 business scenarios and 5 difficulty levels with 563 validated semantic mutants, operator-aware metamorphic relations, and a counterfactual witness generator — all without exposing the gold answer. The verifier is **gold-answer-free but specification-conditioned**: it receives a structured task specification (filters, metric operations, grouping, date windows) but not the correct output.

Across 563 mutants, MORPH-DA detects **64.7%** [58.5%, 70.5%] of controlled faults versus 1.6% for universal robustness checks — 364 vs 9 detections, McNemar χ²=353, p=9.4×10⁻⁷⁹. On natural agent programs from three Claude models, a key finding emerges: single-seed gold-answer evaluation is unreliable because **13–16 programs per model** (20–26% of programs that appear correct on seed=42) fail when the same program is re-executed unchanged on held-out seeds 7 and 123. These **accidental corrects** are structurally wrong programs that coincidentally pass one evaluation. After correcting for them, MORPH-DA achieves **79–89% precision** and **67–79% recall** at 8–21% false-positive rate. Adding a second metamorphic transformation seed improves recall by 2–8 points at comparable precision. For repair, naming the violated relation achieves **12.1%** one-shot fix rate versus 5.5% for generic feedback on 91 wrong programs.

---

## 1. Introduction

### 1.1 Motivation

Modern LLM-powered data-analysis agents receive natural-language questions and tabular data, then generate executable Python programs using pandas and numpy. A program that executes successfully may still implement the wrong analysis through silent semantic errors: omitting a filter, using sum instead of mean, counting rows instead of distinct entities, grouping by the wrong dimension, using the wrong date range, or swapping current and prior periods in a year-over-year comparison.

We call these **wrong-but-executable programs (WEPs)**. The naive check — "did it run without error?" — catches 0% of them. Gold-answer comparison catches them, but requires knowing the correct answer at inference time and provides no diagnostic signal for repair.

### 1.2 The Accidental Correct Problem

A subtler evaluation failure mode: a program can accidentally return the correct answer on one data distribution but fail on others. A program that omits the `order_status != 'cancelled'` filter may still return the correct top category on seed=42 because the correct category dominates even with cancelled orders included. On seed=7, the filter changes the winner.

We term these **accidental corrects**: structurally wrong programs that pass single-seed gold-answer evaluation by coincidence. In our experiments, **13–16 programs per model** (20–26% of programs that appear correct on seed=42) are accidental corrects, identified by re-executing the same seed=42 program unchanged on seeds 7 and 123. MORPH-DA detects 54–81% of them through metamorphic testing alone — without access to the held-out seeds.

### 1.3 MORPH-DA

MORPH-DA addresses WEP detection through three tracks:

1. **Natural track**: LLM agents run on compositional analytics tasks across 8 synthetic business scenarios covering 5 difficulty levels (L1 scalar → L5 cohort ratio + YoY comparison). All 101 tasks are currently single-table; join tasks are future work.

2. **RuleMut track**: 563 validated deterministic semantic mutants (5 fault families) enable rigorous measurement of verifier detection coverage without relying on natural agent failure rates.

3. **Repair track**: Four feedback strategies (R0, R2, R6, R7) compare generic vs. witness-guided one-shot repair on 91 wrong programs.

**What MORPH-DA receives**: The verifier receives a structured `TaskSpec` — filters, metric operation, grouping columns, date scope, ranking direction — and the candidate program source. It does **not** receive the gold answer or the reference program. The LLM agent receives only the natural-language question and a schema summary (column names and types), not the TaskSpec.

### 1.4 Contributions

1. **MORPH-DA Bench**: 101 single-table tasks, 8 synthetic business scenarios, 5 difficulty levels, 563 valid semantic mutants (5 families), and a deterministic reference compiler.

2. **Accidental correct analysis**: Procedure for identifying programs that pass single-seed evaluation by coincidence; MORPH-DA catches 54–81% without cross-seed access.

3. **Three-condition evaluation framework**: Naive (production-realistic), cross-seed corrected (benchmark standard), and multi-seed MORPH-DA (recommended deployment), with honest comparison of what each assumes.

4. **Operator-aware metamorphic verification**: 20+ relations across 8 families — universal, filter/scope, aggregation, grouping, time/period, statistics, join (DSL only), hardcoding — with failure-mode analysis per family.

5. **Repair study**: n=91 wrong programs across 4 strategies; relation-name feedback achieves 2.2× the one-shot fix rate of generic feedback.

---

## 2. Related Work

**Data-analysis and data-science agent benchmarks.**
InfiAgent-DABench [1] provides 257 questions over 52 CSV files. DS-1000 [2] evaluates code generation with functional tests. MORPH-DA differs by targeting wrong-but-executable programs rather than execution success, by providing a controlled mutation corpus for rigorous detector evaluation, and by identifying accidental correctness as a measurement problem in single-seed evaluation.

**Metamorphic testing for LLMs and databases.**
MT-Teql [10] applies metamorphic testing to text-to-SQL systems by generating semantically equivalent SQL variants. SQLHD [9] targets text-to-SQL hallucinations. LLMORPH [11] applies NLP-level transformations to language models. MORPH-DA differs through algebraic data-state transformations on tabular inputs, operator-aware output relations, mutation-grounded benchmarking, and counterexample witnesses for repair.

**Mutation testing.**
Classical mutation testing (Offutt, Jia & Harman) injects known faults to measure test-suite coverage. MORPH-DA applies this methodology to metamorphic verifiers rather than unit tests, and pairs each mutant family with the relation family designed to detect it, enabling per-family coverage measurement.

**LLM-as-a-judge.**
LLM judges can flag suspicious outputs but incur per-call costs and produce unreliable calibration on subtle semantic errors. MORPH-DA's deterministic relations require zero additional model calls and provide structured counterexamples rather than opaque verdicts.

---

## 3. Problem Setting

Let a task be `t = (q, D, s)` where `q` is a natural-language question, `D` is a set of DataFrames, and `s` is a structured TaskSpec (filters, metric operations, grouping, date windows, ranking).

A candidate program `p` implements `analyze(tables: dict) → object`. The source output is `y = p(D)`. A data transformation `T` produces follow-up tables `D' = T(D)`, giving follow-up output `y' = p(D')`.

A metamorphic relation `R` specifies the expected relationship between `y` and `y'`:
- **Invariance**: `y' = y` (e.g., adding out-of-scope rows must not change the output)
- **Equivariance**: `y'` is a known function of `y` (e.g., doubling rows doubles the sum)
- **Monotonicity**: `y'` moves in a known direction
- **Forced winner**: a dominant sentinel group must become the output label

A violation `¬R(y, y')` is a **counterexample witness**. The verifier is **gold-answer-free** (never sees the reference output) and **specification-conditioned** (uses TaskSpec to determine which relations apply and how to construct transformations).

The **mutation score** MS = killed non-equivalent mutants / valid non-equivalent mutants.

---

## 4. MORPH-DA Benchmark

### 4.1 Task specification language

Each task is defined by a structured Pydantic `TaskSpec` covering: filter predicates (equality, inequality, in-set, date boundaries), date scopes (current and prior period for YoY comparisons), join specifications (DSL only; not evaluated in current corpus), metric definition (simple aggregation, ratio, or period-comparison), grouping and ranking (direction, k), post-filter thresholds (minimum group support), and output contract (scalar, label, ranked list).

The natural-language question is derived from the TaskSpec via template. The LLM agent receives only the question and a schema summary; the TaskSpec is used exclusively by the verifier and reference compiler.

### 4.2 Dataset scenarios and filter discriminability

Eight synthetic business scenarios: retail orders, web sessions, seller marketplace, SaaS subscriptions, marketing campaigns, payments, operations/fulfillment, and customer support. Each scenario has 1–2 tables with 8–12 columns, realistic distributions (Pareto-distributed revenue, null values, duplicates), and dual-period date generation for YoY tasks. Data is deterministic: seed=42 always produces the same tables.

**Filter discriminability**: For each task with a required filter, we verify that applying the filter changes the gold answer on all three evaluation seeds (7, 42, 123). Three tasks failed this check — the filter was empirically non-discriminating — and are excluded from the 98-task evaluation set used in Conditions B and C. In such tasks, programs omitting the required filter would pass evaluation regardless; including them would inflate the denominator of correctly evaluated programs.

### 4.3 Difficulty levels

| Level | Description | Key operators |
|---|---|---|
| L1 | Scalar aggregation | sum, mean, count\_distinct |
| L2 | Grouped ranking with optional filter/date | sum, group\_by, sort |
| L3 | Ratio or mean with minimum support threshold | ratio, count\_distinct, post\_filter |
| L4 | Year-over-year period comparison | percentage\_change, split\_by\_date |
| L5 | Multi-filter + ratio + YoY + threshold | All combined |

Current corpus: 24 L1, 48 L2, 11 L3, 16 L4, 2 L5 tasks. All 101 tasks are single-table; join tasks are future work.

### 4.4 Reference compiler and benchmark consistency

A structural compiler (`morphda/reference/compiler.py`) translates TaskSpecs into trusted Pandas programs via five compilation paths — one per difficulty level. All 101 reference programs execute successfully on seeds 7, 42, and 123, and pass all applicable metamorphic relations.

**Important caveat**: Reference programs and relations were co-developed within the same benchmark using the same task specifications. The FPR=0 result on reference programs validates **benchmark internal consistency** — the compiler and relations implement the same TaskSpec — but is not an independent false-positive measurement on programs authored separately. A proper FPR estimate requires programs written independently of the task specs; we obtain this from the natural agent experiments (Section 6).

### 4.5 Mutant corpus

563 valid non-equivalent mutants generated by AST-level operators across 5 fault families. Mutants are validated with 5 oracle seeds; equivalent mutants are excluded. Equivalent mutant rate: 146/709 = 20.6%.

---

## 5. MORPH-DA Verifier

### 5.1 Relation library

20+ relations across 8 families. Each relation defines: applicability rule (checked against TaskSpec), deterministic data transformation, expected output relation, and witness template. Key relations:

**Universal** (all tasks):
- **MR-U1**: Row-permutation invariance — detects positional row dependence
- **MR-U3**: Column-order invariance — detects positional column access

**Filter/Scope**:
- **MR-F1**: Out-of-scope extreme row injection — injects filter-violating rows with extreme metric values; correct programs ignore them. The primary filter-bug detector.
- **MR-F2**: In-scope sentinel sensitivity — dominant group with sentinel label must become winner

**Aggregation**:
- **MR-A1**: Full row duplication algebra — doubling rows: mean/median/distinct unchanged, sum/count doubles. Distinguishes sum from mean.
- **MR-A5**: Non-median extreme perturbation — mean changes, median does not

**Time/Period**:
- **MR-T4**: Forced YoY winner insertion — dominant group with massive YoY growth must be reported as winner. Detects period swap, absolute vs. relative change.

**Hardcoding**:
- **MR-H1**: Counterfactual answer flip — multiplying group keys makes hardcoded labels impossible

Supported metric operations: sum, mean, median, count, count\_distinct, min, max, std, variance, quantile, ratio, percentage\_change, correlation. Note: `weighted_mean` is in the schema DSL but has no implemented relations or mutation operators in the current corpus.

### 5.2 Verification engine

The engine runs sequentially: (1) execute on source data, (2) for each applicable relation: generate follow-up tables, execute, check expected relation, (3) aggregate violations into pass/fail decision and witness list. No LLM calls are required.

**Multi-seed MORPH-DA**: The engine accepts a random seed (`rng_seed`) that controls which specific rows are selected for transformations (e.g., which rows become sentinel groups, which rows are injected). Running with `rng_seed={42,7}` generates two different sets of test cases on the same candidate program and the same task data. A program is flagged if either pass fires a violation. This is Condition C.

**Latency** (20-task pilot, reference programs, raw `time.perf_counter()` timestamps): mean=500ms, median=525ms, p95=1143ms. Python execution count per task: actual count from `VerificationReport.total_python_runs` — approximately 40 for the full relation set.

### 5.3 Counterexample witness

When a relation fails, the witness records: transformation description, source output, follow-up output, expected relation, and diagnosed likely fault type. This is used in the R6/R7 repair conditions.

---

## 6. Experimental Setup

### 6.1 Evaluation conditions

| Condition | Ground truth | MORPH-DA config | Use case |
|---|---|---|---|
| **A — Naive** | Seed=42 gold-answer comparison, 101 tasks | Single rng-seed (42) | Production: cross-seed unavailable |
| **B — Cross-seed corrected** | Same seed=42 program re-executed unchanged on seeds 7 and 123; fails = accidental correct → reclassify as wrong. 98-task set. | Single rng-seed (42) | Benchmark standard |
| **C — Multi-seed MORPH-DA** | Same cross-seed ground truth as B | Two rng-seeds {42, 7}; flag if either fires | Recommended deployment |

**Accidental correct procedure (Condition B)**: For each task, the seed=42 program source code is saved. That same program is executed unchanged on the seed=7 dataset and seed=123 dataset. If it produces the wrong answer on either, the seed=42 program is classified as an accidental correct. Note that independent programs are generated for seeds 7 and 123 as well — the accidental-correct label is determined by the seed=42 program alone, not by whether other independently-generated programs fail their respective seeds.

### 6.2 Models, seeds, and programs

Three model backbones: claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-5. Each generates one program per task per seed via a single LLM call; independent programs are generated for each (task, seed) pair. Evaluation uses the seed=42 programs (101 per model). Programs are generated using only the natural-language question and a schema summary — the TaskSpec is not provided to the agent.

### 6.3 Statistical tests

McNemar's test (continuity-corrected, χ² distribution, 1 df). Holm-Bonferroni correction for 3 simultaneous model comparisons. Task-clustered bootstrap (3,000 iterations) for confidence intervals — resamples tasks to avoid pseudo-replication from multiple verification seeds per program.

### 6.4 Intermediate ablation baseline

The intermediate baseline (Universal + Filter + Aggregation relations, 61.5%) was produced by a genuine verifier run in which each mutant's `killed_by` field records three flags: `universal_only`, `filter_agg`, and `full_morph_da`. These flags were set simultaneously during a single pipeline pass, not reconstructed afterward. The 18 additional detections (64.7% − 61.5% = 3.2pp) come from grouping, hardcoding, time, and statistics relation families.

---

## 7. Results

### 7.1 Benchmark composition

101 tasks, 8 scenarios, 5 difficulty levels, 3 seeds. 563 valid semantic mutants (equivalent mutant rate: 20.6%). 3 tasks excluded from natural-agent evaluation for filter non-discriminability (98-task evaluation set).

### 7.2 Controlled mutation detection (Table 1)

**Table 1 — Mutation Score by Method**

| Method | Killed | Total | Micro MS | 95% CI | McNemar p |
|---|---|---|---|---|---|
| Execution only | 0 | 563 | 0.0% | — | — |
| Universal-only | 9 | 563 | 1.6% | — | baseline |
| Universal + Filter + Agg | 346 | 563 | 61.5% | — | — |
| **Full MORPH-DA** | **364** | **563** | **64.7%** | **[58.5%, 70.5%]** | **p=9.4×10⁻⁷⁹** |

McNemar full vs universal: χ²=353, n₀₁=355, n₁₀=0. MORPH-DA is a strict superset of universal detection. The intermediate step shows filter and aggregation relations account for most detection (346/364 kills); remaining relation families add 18 detections. The 64.7/1.6 ≈ 40× ratio reflects the weakness of the universal-only baseline, which cannot detect operator-specific faults by design.

**Per-family detection** (Table 2 in result_tables.txt): hardcoding 85.3%, grouping 81.0%, ranking 76.1%, filter 67.6%, aggregation 28.2%. Aggregation family is lowest because sum↔mean mutations on label-output ranking tasks may not change the winner — the same category can have both the highest sum and the highest mean. This is a fundamental limitation of oracle-free testing on ranking tasks.

### 7.3 Accidental correct analysis (Table 3)

**Procedure**: Same seed=42 program re-executed unchanged on seeds 7 and 123.

| Model | Seed=42 correct | Truly correct | Accidental corrects | MORPH-DA catches |
|---|---|---|---|---|
| claude-haiku-4-5 | 61/101 (60.4%) | **45/101 (44.6%)** | **16 (26.2%)** | 13 (81%) |
| claude-sonnet-4-6 | 65/101 (64.4%) | **52/101 (51.5%)** | **13 (20.0%)** | 7 (54%) |
| claude-opus-4-5 | 62/101 (61.4%) | **49/101 (48.5%)** | **13 (21.0%)** | 10 (77%) |

MORPH-DA detects 54–81% of accidental corrects using only seed=42 metamorphic transformations — without access to held-out seeds. The remaining accidental corrects it misses represent programs whose structural bug does not manifest in any current metamorphic transformation (primarily aggregation bugs in label-output tasks).

### 7.4 Natural agent verification results (Table 4)

**Table 4 — Three-Condition Verification Metrics (98-task evaluation set)**

| Model | Condition | Precision | 95% CI | Recall | 95% CI | FPR | 95% CI | F1 | AAR |
|---|---|---|---|---|---|---|---|---|---|
| Haiku | A) Naive | 58.8% | — | 79.0% | — | 34.4% | — | 67.4% | 16.7% |
| | B) Cross-seed | **87.5%** | [77.1%, 95.8%] | **79.2%** | [67.9%, 90.6%] | **14.0%** | [4.7%, 25.6%] | 83.2% | 22.9% |
| | C) Multi-seed | 86.0% | — | 81.1% | — | 16.3% | — | 83.5% | 21.7% |
| Sonnet | A) Naive | 66.7% | — | 72.2% | — | 20.0% | — | 69.3% | 16.1% |
| | B) Cross-seed | **89.2%** | [78.4%, 97.3%] | **67.3%** | [53.1%, 79.6%] | **8.2%** | [2.0%, 18.4%] | 76.7% | 26.2% |
| | C) Multi-seed | 86.8% | — | 67.3% | — | 10.2% | — | 75.9% | 26.7% |
| Opus | A) Naive | 58.8% | — | 81.1% | — | 33.9% | — | 68.2% | 14.6% |
| | B) Cross-seed | **79.2%** | [68.8%, 89.6%] | **79.2%** | [66.7%, 89.6%] | **20.8%** | [10.4%, 33.3%] | 79.2% | 20.8% |
| | C) Multi-seed | 78.0% | — | 81.2% | — | 22.9% | — | 79.6% | 19.6% |

CIs: task-clustered bootstrap, 3,000 iterations. AAR = Accepted-Answer Risk = FN/(FN+TN).

**McNemar significance (MORPH-DA full vs Universal-only, Holm-corrected):**

| Model | χ² | p-value | n₀₁ | n₁₀ |
|---|---|---|---|---|
| claude-haiku-4-5 | 45.02 | p=1.95×10⁻¹¹ | 47 | 0 |
| claude-sonnet-4-6 | 35.03 | p=3.25×10⁻⁹ | 37 | 0 |
| claude-opus-4-5 | 46.02 | p=1.17×10⁻¹¹ | 48 | 0 |

**Reading the conditions**: Condition A precision (59–67%) is low because accidental corrects are labeled as "correct" in the naive ground truth — when MORPH-DA correctly flags their structural bugs, these count as false positives. Condition B relabels accidental corrects as wrong; MORPH-DA's flags become true positives, revealing true detection performance. Condition C adds a second transformation seed: recall improves 0–8pp with small FPR cost.

**Relation-level false positive breakdown** (correctly labeled programs incorrectly flagged, Condition B):

| Relation | Haiku FPs | Sonnet FPs | Opus FPs | Root cause |
|---|---|---|---|---|
| MR-F1 | 4 | 2 | 6 | Extreme out-of-scope rows affect programs with imprecise filter logic |
| MR-G3 | 1 | 1 | 2 | Tie-break sensitivity on groups with equal metric values |
| MR-H1 | 1 | 1 | 2 | Hardcoding detector fires on programs with legal computed constants |

### 7.5 Repair experiment (Table 5)

**Procedure**: 91 wrong-but-executable seed=42 programs (55 distinct tasks, raw logs in `runs/repair/repair_results.jsonl`), one repair attempt per program per strategy.

| Strategy | Feedback provided | Fixed | Rate |
|---|---|---|---|
| R0 — No retry | None (baseline) | 0/91 | 0.0% |
| R2 — Generic | "Your program has a bug, please fix it" | 5/91 | **5.5%** |
| R6 — Relation name | "You may have violated MR-F1 (filter/scope)" | 11/91 | **12.1%** |
| R7 — Witness | Source=X, follow-up=Y, transformation, likely issue | 11/91 | **12.1%** |

Naming the violated relation doubles the one-shot fix rate vs generic feedback. R6 and R7 achieve identical rates at n=91 (0 discordant pairs between R6 and R7; the study is underpowered to distinguish them). For programs where witnesses exist, multi-round R7 is recommended.

---

## 8. Limitations

1. **Specification conditioning**: The verifier uses a structured TaskSpec. In deployment, this must be provided by analysts, derived from existing business metric definitions, or extracted by a separate LLM — this is outside MORPH-DA's current scope.

2. **Reference program circularity**: Reference programs are generated by the task-spec compiler. FPR=0 on references validates benchmark consistency but is not an independent false-positive measurement. Independent FPR estimates come from the natural agent experiments (8–21% depending on condition and model).

3. **Single-table only**: All 101 evaluated tasks are single-table. JoinSpec exists in the DSL but no evaluated task uses it.

4. **Aggregation family gap (28.2%)**: Aggregation bugs on label-output ranking tasks are undetectable when the wrong aggregation produces the same winner. This is fundamental to oracle-free testing on ranking.

5. **Three Anthropic model variants**: Results may not generalize to other model families.

6. **Repair underpowered**: At n=91 with 0 discordant R6/R7 pairs, the repair study cannot distinguish R6 from R7. Multi-round experiments are needed.

---

## 9. Conclusion

Execution success is not evidence of analytical correctness. Single-seed gold-answer evaluation further understates the problem: 20–26% of programs that appear correct on one dataset are structurally wrong accidental corrects. MORPH-DA addresses both problems through specification-conditioned metamorphic testing — without requiring the gold answer.

Across 563 controlled mutants, MORPH-DA detects 64.7% of faults at 364 detections versus 9 for universal-only (χ²=353, p=9.4×10⁻⁷⁹). On natural agent programs with cross-seed corrected ground truth, MORPH-DA achieves 79–89% precision and 67–79% recall at 8–21% FPR. Naming the violated relation in feedback doubles one-shot repair success over generic feedback. These results establish operator-aware, specification-conditioned metamorphic testing as a practical, zero-LLM-call verification primitive for data-analysis agents.

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
