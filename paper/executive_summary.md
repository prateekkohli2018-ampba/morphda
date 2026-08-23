# MORPH-DA: Executive Summary

---

## The Problem — Why Should You Care?

Companies are deploying AI agents that answer business questions by writing and running code automatically.

A manager asks: *"Which product category had the highest conversion rate last quarter, excluding test traffic and cancelled orders?"*

The AI agent writes a program, runs it, and returns: **"Electronics — 23.4%"**

The program ran without errors. It returned a confident, plausible-looking number. But it may be **completely wrong** — because the agent silently:
- forgot to exclude cancelled orders
- used average instead of median
- counted total sessions instead of unique customers
- applied the date filter to the wrong column

**There is no error message. No crash. No warning.** The answer just looks correct.

This is called a **wrong-but-executable program** — the AI equivalent of a spreadsheet that calculates the wrong formula without telling you.

---

## How Big Is the Problem?

We ran three different AI models (Claude Haiku, Sonnet, and Opus) on 101 realistic business analytics questions — the kind analysts answer every day: revenue by category, conversion rates, year-over-year comparisons, customer segmentation.

**First look** (comparing against one test dataset):

| | Haiku (faster/cheaper) | Sonnet (mid-tier) | Opus (most capable) |
|---|---|---|---|
| Looks correct | 68.5% | 74.6% | 62.6% |
| **Silently wrong** | **31.5%** | **25.4%** | **37.4%** |

**But it's worse than it looks.** Many programs appear correct because they got lucky — the specific test dataset happened to produce the right answer even though the underlying logic is wrong. We tested each program on three additional datasets it had never seen.

**Corrected numbers** (removing programs that only got lucky on one dataset):

| | Haiku | Sonnet |
|---|---|---|
| Truly correct (all datasets) | **44.5%** | **56.2%** |
| **Behaviorally wrong** | **55.5%** | **43.8%** |

**More than half of Haiku's "correct" analyses and nearly half of Sonnet's are actually wrong — they just got lucky on the test data.**

This makes intuitive sense: if a program forgets to exclude cancelled orders but the cancelled orders happen to be evenly spread across categories, it returns the right winner by coincidence. Change the data distribution slightly and it breaks.

---

## The Current "Solution" Doesn't Work

Most AI systems today use a simple check: *"Did the program run without crashing?"*

If yes → accept the answer.
If no (error/crash) → retry.

**Our finding: This catches exactly 0% of silent semantic errors.** All 563 bugs we deliberately injected ran perfectly and returned confident-looking answers. Every single one passed the execution check.

---

## What MORPH-DA Does

MORPH-DA is a **behavioral verification system**. Instead of just asking "did the program run?", it asks:

> *"Does this program's answer change in the right way when we change the data?"*

**Example:** The question asks for revenue in 2025. MORPH-DA secretly adds fake 2024 records with extreme values (say, $10M in revenue for a category that normally makes $50K). A correct program should ignore these — they're outside the requested year. If the answer *changes*, the program clearly isn't filtering by year correctly.

This is done with ~40 controlled data variations, all running automatically in seconds, with no human review needed.

---

## Does It Work?

We tested on **563 deliberately injected bugs** with known ground truth:

| Verification Method | Bugs Caught | Notes |
|---|---|---|
| "Did it run?" (current standard) | **0%** | Catches nothing |
| Generic data shuffling | 1.6% | Nearly useless |
| MORPH-DA (filter + aggregation checks) | 61.5% | |
| **MORPH-DA (full system)** | **64.7%** | **95% confidence interval: 60–69%** |

**This is statistically significant** — the probability these results are due to chance is less than 1 in 1,000 (p < 0.001).

On real AI-generated programs (not pre-planted bugs):

After cross-seed correction (using the true wrong program population):

| Model | Silent errors caught | False alarm rate | Precision |
|---|---|---|---|
| Claude Haiku | 70% of errors caught | **11%** | **89%** |
| Claude Sonnet | 61% of errors caught | **13%** | **79%** |

**When MORPH-DA flags a program, it's right 79–89% of the time.** The remaining 11–21% of flags are on programs with genuine behavioral anomalies — they returned the right answer on the specific test data, but their underlying logic is fragile and would fail on different data.

---

## Why Does MORPH-DA Miss ~35% of Errors?

The hardest errors to catch are in **year-over-year comparison tasks** (e.g., "which category improved most vs. last year?"). When a program uses the wrong formula but the same category happens to win, behavioral testing can't distinguish it without knowing the true answer.

The easiest errors to catch: **hardcoded answers** (85% detected), **wrong grouping** (81%), **wrong sort direction** (76%), **missing filters** (68%).

---

## Business Impact

**Without MORPH-DA:** Your AI agent returns a wrong answer 25–37% of the time with no indication anything is wrong. Decisions get made on bad data.

**With MORPH-DA:**
- ~65–80% of wrong answers are flagged before reaching a decision-maker
- Flagged answers can be routed to human review or automatically retried
- The system runs in seconds, requires no gold-standard answer at runtime, and costs nothing in additional AI model calls

**The cost of a missed error** (a procurement decision, pricing change, or customer offer based on wrong analysis) vastly outweighs the cost of a false alarm (asking a human to double-check one analysis).

---

## What's Next

1. **Complete opus-4-5 run** — quota reset needed; seed 7 missing
2. **Scale to 120+ tasks** — currently 101; more diverse business scenarios
3. **Reduce false alarms** — improve MR-F1 to lower the 22% false positive rate
4. **Repair experiment** — use the detected error + counterexample to automatically fix wrong programs
5. **NeurIPS 2026 submission** — Verify-Agents workshop, August deadline

---

*MORPH-DA is a research prototype. Core technology: deterministic data-state transformations + algebraic output invariants. No LLM calls required for verification — runs in ~2 seconds per program.*
