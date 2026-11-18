---
pretty_name: MORPH-DA
license: mit
language:
- en
size_categories:
- 10K<n<100K
task_categories:
- text-generation
- question-answering
- table-question-answering
tags:
- data-analysis-agents
- llm-agents
- agent-evaluation
- agent-verification
- metamorphic-testing
- mutation-testing
- benchmark
- llm-evaluation
- data-science-agents
- pandas
- code-generation
- text-to-sql
- synthetic
configs:
{{CONFIGS_YAML}}
---

# MORPH-DA: a mutation-grounded benchmark for metamorphic verification of data-analysis agents

MORPH-DA is a benchmark for **LLM data-analysis agents** and for the verifiers that check them.
Agents write Python/pandas programs that can run cleanly and still answer the wrong question:
a missing filter, `sum` instead of `mean`, swapped periods. MORPH-DA keeps the program fixed,
transforms its input tables with specification-conditioned **metamorphic relations**, and checks
necessary output properties — with **no numeric gold answer and no additional model call** at
verification time. A corpus of **seeded faults (mutation testing)** measures how good a verifier is.

- Paper: [`paper/morphda_paper.pdf`](paper/morphda_paper.pdf) · [OpenReview](https://openreview.net/forum?id=JsrKw3nUSV)
- Poster: [`paper/morphda_poster.pdf`](paper/morphda_poster.pdf)
- Code (verifier, relations, agent harness, evaluation): <https://github.com/prateekkohli2018-ampba/morphda>
- Authors: Prateek Kohli, Karun Thankachan — NeurIPS 2026 Workshop *Who Verifies the Agents?*

## What is in this dataset

| Config | Rows | What it contains |
|---|---|---|
| `tasks` | {{N_TASKS}} | Question, structured task specification, reference answers on three data seeds |
| *one per table* (see below) | 800 – 5,000 per seed | The synthetic input tables, one split per data seed (`seed42`, `seed7`, `seed123`) |
| `mutants` | {{N_MUTANT_ROWS}} | Programs with one injected fault, with fault family and mutation operator |
| `agent_programs` | {{N_PROGRAMS}} | Programs written by three hosted models, with per-seed correctness |

Input tables (one config each):

| Config | Scenario(s) |
|---|---|
{{TABLE_ROWS}}

## Load it

```python
from datasets import load_dataset

tasks    = load_dataset("{{REPO_ID}}", "tasks", split="train")
orders   = load_dataset("{{REPO_ID}}", "orders", split="seed42")      # retail01 table
mutants  = load_dataset("{{REPO_ID}}", "mutants", split="train")
programs = load_dataset("{{REPO_ID}}", "agent_programs", split="train")

print(tasks[3]["question"])           # "Which category had the highest total revenue?"
print(tasks[3]["gold_seed42"])        # reference answer, JSON-encoded: "\"Books\""
```

Every task is answered by a function `analyze(tables)` that receives a dict of pandas DataFrames
and returns a scalar, a label, or a table. Reference answers are JSON strings because the output
type differs between tasks (`output_type`).

## Fields

**`tasks`** — `task_id`, `scenario_id`, `tables`, `question`, `question_family`, `difficulty_level`
(1–5), `output_type`, `scope_discriminating` (whether the task's filter or date scope changes the
answer; used to define the scope-discriminating evaluation subset), `gold_seed42`, `gold_seed7`,
`gold_seed123`, and `task_spec` (the full machine-readable specification as a JSON string).

**Tables** — generated deterministically per `(scenario, seed)`; dates are ISO strings:

{{TABLE_SCHEMAS}}

**`mutants`** — `mutant_id`, `task_id`, `mutation_family` (aggregation, filter, grouping,
hardcoding, ranking), `mutation_operator`, `mutated_program`, `non_equivalent_seeds` (data seeds on
which the mutant's output differs from the reference).

**`agent_programs`** — `model`, `task_id`, `scenario_id`, `generated_program`,
`executed_on_seed42`, `correct_seed42`, and `correct_seed7` / `correct_seed123` (the same program
re-run unchanged on held-out data seeds; `null` when the program did not execute). A program that
is correct on seed 42 but wrong on a held-out seed is an *accidental correct*.

## How to use it

1. **Evaluate an agent:** give it `question` and the tables of a scenario (seed 42), run the
   program it writes, and compare with `gold_seed42`. Re-run the same program on seeds 7 and 123
   and compare with `gold_seed7` / `gold_seed123` to find accidental corrects.
2. **Evaluate a verifier or test suite:** run it on `mutants` and count the seeded faults it
   catches, per fault family. The reference verifier is in the GitHub repository.
3. **Study agent errors:** `agent_programs` has full program text for correct, wrong and
   accidentally-correct programs.

## Notes and limitations

- All data are **synthetic and single-table per task** (one scenario uses two tables).
  Results may not transfer to real warehouses or multi-table SQL workloads.
- The verifier relies on a **trusted task specification**; passing the checks is evidence, not a
  proof of correctness.
- `mutants` has {{N_MUTANT_ROWS}} rows. Some rows are exact duplicates of others (the same
  `mutant_id`); deduplicate on `mutant_id` if you need distinct programs.
- The tasks, relations and executor target Python/pandas programs. SQL is not executed, so this is
  not a text-to-SQL benchmark, although the metamorphic idea has also been applied to text-to-SQL
  systems in prior work.
- `agent_programs` contains model-written code. Run it in a container or VM, not in your main
  environment.

## Reproduce this dataset

```bash
git clone https://github.com/prateekkohli2018-ampba/morphda && cd morphda
bash scripts/setup.sh && source .venv/bin/activate
python scripts/build_hf_dataset.py --repo-id {{REPO_ID}} --out ../morphda-hf
```

## License

MIT (see `LICENSE`). The data are synthetic and contain no personal information.

## Citation

```bibtex
@inproceedings{kohli2026morphda,
  title     = {{MORPH}-{DA}: A Mutation-Grounded Benchmark for Metamorphic Verification of Data-Analysis Agents},
  author    = {Prateek Kohli and Karun Thankachan},
  booktitle = {NeurIPS 2026 Workshop Who Verifies the Agents? Toward Reliable Agent Development},
  year      = {2026},
  url       = {https://openreview.net/forum?id=JsrKw3nUSV}
}
```
