"""
Append-only JSONL log writer.

All results are written atomically — one JSON object per line.
Never mutate existing log files; always append or create.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from morphda.logging.schemas import record_to_dict


class LogWriter:
    """
    Thread-safe JSONL writer for experiment results.

    Usage:
        with LogWriter("runs/exp01/programs.jsonl") as w:
            w.write(program_record)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = None

    def __enter__(self) -> "LogWriter":
        self._fh = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *args: Any) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def write(self, record: Any) -> None:
        """Serialize and append one record."""
        d = record_to_dict(record) if not isinstance(record, dict) else record
        line = json.dumps(d, default=str, ensure_ascii=False)
        with self._lock:
            if self._fh:
                self._fh.write(line + "\n")
            else:
                # One-shot write without context manager
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def write_many(self, records: list[Any]) -> None:
        for r in records:
            self.write(r)


def load_jsonl(path: str | Path) -> list[dict]:
    """Load all records from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return []
    results = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results
