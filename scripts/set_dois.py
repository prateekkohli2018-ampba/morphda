#!/usr/bin/env python3
"""
Record the dataset DOI (Hugging Face) and software DOI (Zenodo) in the repository.

Fills the marked slots in README.md and CITATION.cff. (The dataset card gets its DOIs from
`build_hf_dataset.py --dataset-doi/--software-doi`.) Until a DOI is
given, the slots are invisible comments, so nothing placeholder-like is ever published.

Usage:
    python scripts/set_dois.py --software 10.5281/zenodo.1234567 --dataset 10.57967/hf/1234
    python scripts/set_dois.py --software https://doi.org/10.5281/zenodo.1234567   # either alone
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
MD_START, MD_END = "<!-- dois:start -->", "<!-- dois:end -->"
CFF_START, CFF_END = "# dois:start", "# dois:end"


def normalize(value: str) -> str:
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.IGNORECASE)
    if not DOI_RE.match(doi):
        raise SystemExit(f"Not a DOI: {value!r} (expected like 10.5281/zenodo.1234567)")
    return doi


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    if start not in text or end not in text:
        raise SystemExit(f"Marker {start!r}/{end!r} not found")
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    return pattern.sub(lambda _: f"{start}\n{body}{end}" if body else f"{start}\n{end}", text, count=1)


def _readme_badge(software: str | None, dataset: str | None) -> str:
    if not software:
        return ""
    return f"[![DOI](https://zenodo.org/badge/DOI/{software}.svg)](https://doi.org/{software})\n"


def _readme_section(software: str | None, dataset: str | None) -> str:
    lines = []
    if dataset:
        lines.append(f"- **Dataset DOI:** [{dataset}](https://doi.org/{dataset})\n")
    if software:
        lines.append(f"- **Software DOI:** [{software}](https://doi.org/{software})\n")
    return "\n## Persistent identifiers\n\n" + "".join(lines) if lines else ""


def _cff_body(software: str | None, dataset: str | None) -> str:
    out = []
    if software:
        out.append(f"doi: {software}\n")
    ids = []
    if software:
        ids.append(f'  - type: doi\n    value: {software}\n    description: "Software archive (Zenodo)"\n')
    if dataset:
        ids.append(f'  - type: doi\n    value: {dataset}\n    description: "Dataset (Hugging Face)"\n')
    if ids:
        out.append("identifiers:\n" + "".join(ids))
    return "".join(out)


def apply(root: Path, software: str | None, dataset: str | None) -> list[str]:
    targets = [
        (root / "README.md", MD_START, MD_END, _readme_badge),
        (root / "README.md", "<!-- dois-section:start -->", "<!-- dois-section:end -->", _readme_section),
        (root / "CITATION.cff", CFF_START, CFF_END, _cff_body),
    ]
    touched = []
    for path, start, end, body in targets:
        text = path.read_text(encoding="utf-8")
        new = _replace_block(text, start, end, body(software, dataset))
        if new != text:
            path.write_text(new, encoding="utf-8")
            name = str(path.relative_to(root))
            if name not in touched:
                touched.append(name)
    return touched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--software", help="Zenodo DOI of the code archive")
    ap.add_argument("--dataset", help="DOI of the Hugging Face dataset")
    args = ap.parse_args()
    if not (args.software or args.dataset):
        ap.error("give --software and/or --dataset")
    software = normalize(args.software) if args.software else None
    dataset = normalize(args.dataset) if args.dataset else None
    touched = apply(ROOT, software, dataset)
    print("Updated:", ", ".join(touched) if touched else "nothing changed")
    print("Dataset card: DATASET_DOI=... SOFTWARE_DOI=... bash upload_hf_dataset.sh <repo-id>")


if __name__ == "__main__":
    main()
