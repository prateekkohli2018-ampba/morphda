"""set_dois fills the DOI slots, leaves them invisible when empty, and rejects bad input."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("set_dois", ROOT / "scripts" / "set_dois.py")
set_dois = importlib.util.module_from_spec(spec)
spec.loader.exec_module(set_dois)


@pytest.fixture()
def tree(tmp_path):
    (tmp_path / "scripts").mkdir()
    for rel in ("README.md", "CITATION.cff"):
        (tmp_path / rel).write_text((ROOT / rel).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def test_repository_keeps_its_doi_slots_and_has_no_placeholder_text():
    for rel, start in (("README.md", set_dois.MD_START), ("CITATION.cff", set_dois.CFF_START)):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert start in text
        assert "YOUR" + "_" not in text


def test_both_dois_land_in_readme_and_citation(tree):
    set_dois.apply(tree, "10.5281/zenodo.123", "10.57967/hf/456")
    readme = (tree / "README.md").read_text(encoding="utf-8")
    assert "zenodo.org/badge/DOI/10.5281/zenodo.123.svg" in readme
    assert "- **Dataset DOI:** [10.57967/hf/456](https://doi.org/10.57967/hf/456)" in readme
    cff = yaml.safe_load((tree / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["doi"] == "10.5281/zenodo.123"
    assert {i["value"] for i in cff["identifiers"]} == {"10.5281/zenodo.123", "10.57967/hf/456"}


def test_rerunning_with_a_new_doi_replaces_instead_of_duplicating(tree):
    set_dois.apply(tree, "10.5281/zenodo.1", None)
    set_dois.apply(tree, "10.5281/zenodo.2", None)
    readme = (tree / "README.md").read_text(encoding="utf-8")
    assert "zenodo.2" in readme and "zenodo.1" not in readme


def test_doi_urls_are_normalized_and_garbage_is_rejected():
    assert set_dois.normalize("https://doi.org/10.5281/zenodo.99") == "10.5281/zenodo.99"
    assert set_dois.normalize("doi:10.57967/hf/11699") == "10.57967/hf/11699"
    with pytest.raises(SystemExit):
        set_dois.normalize("not-a-doi")
