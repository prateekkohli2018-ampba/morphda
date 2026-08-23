# MORPH-DA: Executive Summary

---

## The Problem — Why Should You Care?

Companies are deploying AI agents that answer business questions by writing and running code automatically.

A manager asks: *"Which product category had the highest conversion rate last quarter, excluding test traffic and cancelled orders?"*

The AI agent writes a program, runs it, and returns: **"Electronics — 23.4%"**

The program ran without errors. It returned a confident, plausible-looking number. But it may be **completely wrong** — because the agent silently:
- forgot to exclude cancelled orders
- used average instead of sum
- counted total sessions instead of unique customers
- applied the date filter to the wrong column
- hardcoded the answer from the sample data it was shown

**There is no error message. No crash. No warning.** The answer just looks correct.

This is called a **wrong-but-executable program** — the AI equivalent of a spreadsheet that calculates the wrong formula without telling you.

---

## How Big Is the Problem?

We ran three different AI models (Claude Haiku, Sonnet, and Opus) on 101 realistic business analytics questions — the kind analysts answer every day: revenue by category, conversion rates, year-over-year comparisons, customer segmentation.

**First look** (single dataset, seed=42):

| | Haiku | Sonnet | Opus |
|---|---|---|---|
| Appears correct | 60.4% | 64.4% | 61.4% |
| **Silently wrong** | **37.6%** | **35.6%** | **36.6%** |

**But it gets worse.** Many programs appear correct because they got lucky — the test dataset happened to produce the right answer even though the underlying logic is wrong. We tested each program on two additional datasets it had never seen (different random seeds).

**Corrected numbers** (removing "accidental corrects" — programs that pass one dataset but fail others):

| | Haiku | Sonnet | Opus |
|---|---|---|---|
| Truly correct (all 3 datasets) | **45.5%** | **50.5%** | **48.5%** |
| **Structurally wrong** | **54.5%** | **49.5%** | **51.5%** |
| Accidental corrects found | **45 programs** | **42 programs** | **38 programs** |

**Nearly half of all programs that look correct on one dataset are actually wrong.** They just got lucky. Change the data distribution slightly and they break.

Why does this happen? A program that forgets to exclude cancelled orders might still return the correct top category on one dataset, because the correct category dominates even with cancelled orders included. On a different dataset, the cancelled orders tip the balance — and the wrong answer becomes visible.

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

**Example 1 — Filter bug detection:** The question asks for revenue among non-cancelled orders. MORPH-DA secretly adds cancelled orders with enormous revenue values. A correct program ignores them. If the answer changes, the program forgot to filter.

**Example 2 — Aggregation bug detection:** MORPH-DA doubles all rows. A correct mean must stay the same; a correct sum must double. A program using sum instead of mean is exposed immediately.

**Example 3 — Period comparison bug:** MORPH-DA inserts a group that grew 1000% year-over-year. A correct program reports it as the winner. A program that swapped the current and prior year does not.

This runs ~40 controlled data variations automatically in under a second, with no human review needed and no need to know the correct answer.

---

## Does It Work?

**On 563 deliberately injected bugs with known ground truth:**

| Verification method | Bugs caught | Notes |
|---|---|---|
| "Did it run?" (current standard) | **0%** | Catches nothing |
| Generic data shuffling | 1.6% | Nearly useless |
| MORPH-DA filter + aggregation checks | 61.5% | |
| **MORPH-DA full system** | **64.7%** | **95% CI: [58.5%, 70.5%]** |

**This is 40× more effective than the best simple alternative** (p < 0.000001, statistically certain).

**On real AI-generated programs (not pre-planted bugs) — three evaluation conditions:**

| Condition | What it measures | Haiku Precision | Haiku Recall | Haiku FPR |
|---|---|---|---|---|
| Naive (production) | Single dataset, no correction | 62% | 70% | 27% |
| **Cross-seed corrected** | True structural bugs identified | **88%** | **68%** | **11%** |
| Multi-seed MORPH-DA | More verification passes | **87%** | **78%** | 14% |

**When MORPH-DA flags a program (cross-seed corrected ground truth), it is right 81–88% of the time.**

*Why three conditions?* The "naive" condition represents what you'd see in production — where you can't run every program on 3 datasets. The "cross-seed corrected" condition is the true scientific ground truth. The "multi-seed MORPH-DA" condition runs verification with two different random transformation sets — our recommended deployment setting, which catches 8–13% more bugs at comparable precision.

---

## Breakdown: What Types of Bugs Does MORPH-DA Catch?

| Bug type | Example | Detection rate | 95% confidence |
|---|---|---|---|
| **Hardcoded answers** | Returns `"Electronics"` always, ignores data | **85%** | [78%, 92%] |
| **Wrong grouping** | Groups by `date` instead of `category` | **81%** | [55%, 100%] |
| **Wrong sort direction** | Returns lowest instead of highest | **76%** | [66%, 86%] |
| **Missing filter** | Forgets to exclude cancelled orders | **68%** | [57%, 78%] |
| **Wrong aggregation** | Uses sum instead of mean | **28%** | [17%, 39%] |

**Why does wrong aggregation have only 28% detection?** If a program uses sum instead of mean to find the top category, but the same category has both the highest sum and the highest mean, the wrong answer happens to look right. This is a known limitation of verification without knowing the correct answer — the winner's identity doesn't uniquely tell you which formula was used.

---

## Can MORPH-DA Help Fix Wrong Programs?

Yes — MORPH-DA produces a **counterexample**: it shows the program what it returned on the original data vs. what it returned on the modified data, and what should have stayed the same.

We tested whether feeding this back to the AI model helps it fix its own mistake (91 wrong programs):

| Feedback type | Fix rate |
|---|---|
| No feedback (retry blindly) | 0% |
| Generic: "your program has a bug" | 5.5% |
| **Named violation: "you missed the filter on order_status"** | **12.1%** |
| Full counterexample with data | **12.1%** |

**Naming the specific problem doubles the fix rate.** Simply telling the model which filter or operator it violated is as effective as showing it the full data counterexample in a single attempt. For harder bugs, the counterexample helps across multiple repair rounds.

---

## Business Impact

**The two types of error — and their costs:**

| Error type | What it means | Business cost |
|---|---|---|
| **Missed bug** (MORPH-DA passes wrong program) | Wrong analysis reaches decision-maker | Revenue impact, bad decisions, lost trust in AI |
| **False alarm** (MORPH-DA flags correct program) | Correct analysis gets double-checked | Analyst time to verify (~1 minute) |

**Without MORPH-DA:** ~37–54% of programs reaching decision-makers are wrong. No warning.

**With MORPH-DA (cross-seed ground truth):**
- 68–78% of wrong programs are caught before reaching a decision-maker
- Precision 81–88%: fewer than 1 in 5 flags is a false alarm
- False alarm rate 11–14%: ~1 in 8 correct analyses gets flagged for review
- Runs in under 1 second, requires no gold-standard answer, costs zero additional AI calls

**The cost calculation is clear:** One major wrong business decision (pricing, procurement, customer offer) based on a wrong analysis vastly outweighs the cost of manually verifying a few flagged results per day.

---

## What's Complete

| Item | Status |
|---|---|
| Benchmark: 101 tasks, 8 scenarios | ✅ Complete |
| 563 validated semantic mutants | ✅ Complete |
| 3 models × 3 seeds × 101 tasks | ✅ Complete (909 programs) |
| Mutation score: 64.7% [58.5%, 70.5%] | ✅ Published |
| Natural agent metrics (3 conditions) | ✅ Published |
| Repair experiment n=91 | ✅ Complete |
| Multi-seed MORPH-DA analysis | ✅ Complete |
| Accidental corrects analysis | ✅ Complete |
| Statistical tests (McNemar, Holm) | ✅ Published |

---

*MORPH-DA is a research system. Core technology: deterministic data-state transformations + algebraic output invariants. No LLM calls required for verification. Verification latency: ~500ms median per program.*
