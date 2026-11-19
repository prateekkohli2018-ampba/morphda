"""The Hugging Face package is self-consistent: every config points at a file that exists."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("hf") / "ds"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_hf_dataset.py"),
         "--repo-id", "someone/morph-da", "--out", str(out)],
        check=True, capture_output=True,
    )
    return out


def _front_matter(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---")[1])


def test_every_config_file_exists_and_card_has_no_placeholders(dataset):
    card = (dataset / "README.md").read_text(encoding="utf-8")
    assert not re.findall(r"\{\{[A-Z_]+\}\}", card)
    configs = _front_matter(dataset / "README.md")["configs"]
    names = {c["config_name"] for c in configs}
    assert {"tasks", "mutants", "agent_programs", "orders", "sessions", "conversions"} <= names
    for c in configs:
        for f in c["data_files"]:
            assert (dataset / f["path"]).is_file(), f["path"]


def test_dataset_is_licensed_cc_by_4_0(dataset):
    assert _front_matter(dataset / "README.md")["license"] == "cc-by-4.0"
    assert (dataset / "LICENSE").read_text(encoding="utf-8").startswith("Attribution 4.0 International")


def test_tasks_carry_the_question_and_a_reference_answer_per_seed(dataset):
    tasks = [json.loads(l) for l in (dataset / "data" / "tasks.jsonl").read_text().splitlines()]
    assert len(tasks) == 101
    t = next(x for x in tasks if x["task_id"] == "retail01_l2_003")
    assert t["question"] == "Which category had the highest total revenue?"
    assert all(json.loads(t[f"gold_seed{s}"]) for s in (42, 7, 123))
    assert sum(not x["scope_discriminating"] for x in tasks) == 3


def test_agent_programs_distinguish_accidental_corrects(dataset):
    rows = [json.loads(l) for l in (dataset / "data" / "agent_programs.jsonl").read_text().splitlines()]
    assert len(rows) == 303
    assert any(r["correct_seed42"] and r["correct_seed7"] is False for r in rows)
    assert all(set(r) == {"model", "task_id", "scenario_id", "generated_program", "executed_on_seed42",
                          "correct_seed42", "correct_seed7", "correct_seed123"} for r in rows)


def test_card_lists_dois_only_when_given(dataset, tmp_path):
    assert "Persistent identifiers" not in (dataset / "README.md").read_text(encoding="utf-8")
    out = tmp_path / "with_dois"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_hf_dataset.py"), "--repo-id", "someone/morph-da",
         "--out", str(out), "--dataset-doi", "10.57967/hf/1", "--software-doi", "10.5281/zenodo.2"],
        check=True, capture_output=True,
    )
    card = (out / "README.md").read_text(encoding="utf-8")
    assert "https://doi.org/10.57967/hf/1" in card and "https://doi.org/10.5281/zenodo.2" in card
    assert _front_matter(out / "README.md")["license"] == "cc-by-4.0"
