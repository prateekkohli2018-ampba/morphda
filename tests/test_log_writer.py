from morphda.logging.writer import LogWriter, load_jsonl


def test_rerun_with_overwrite_does_not_duplicate_records(tmp_path):
    path = tmp_path / "out.jsonl"
    for _ in range(2):  # the same script executed twice
        with LogWriter(path, overwrite=True) as w:
            w.write({"id": 1})
            w.write({"id": 2})
    assert [r["id"] for r in load_jsonl(path)] == [1, 2]


def test_default_mode_appends_for_resumable_logs(tmp_path):
    path = tmp_path / "out.jsonl"
    for i in (1, 2):
        with LogWriter(path) as w:
            w.write({"id": i})
    assert [r["id"] for r in load_jsonl(path)] == [1, 2]
