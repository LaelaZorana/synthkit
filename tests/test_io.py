"""Tests for dataset IO: jsonl/json/csv round-trips and error paths."""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthkit.io_utils import read_records, write_jsonl
from synthkit.models import SynthkitError


def _tmp(suffix: str, text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def test_jsonl_roundtrip():
    recs = [{"prompt": "a"}, {"prompt": "b", "domain": "x"}]
    path = _tmp(".jsonl", "")
    write_jsonl(path, recs)
    try:
        assert read_records(path) == recs
    finally:
        os.remove(path)


def test_json_list_and_dict_wrapper():
    p1 = _tmp(".json", json.dumps([{"x": 1}, {"x": 2}]))
    p2 = _tmp(".json", json.dumps({"data": [{"x": 1}]}))
    try:
        assert read_records(p1) == [{"x": 1}, {"x": 2}]
        assert read_records(p2) == [{"x": 1}]
    finally:
        os.remove(p1)
        os.remove(p2)


def test_csv():
    path = _tmp(".csv", "prompt,domain\nhello,x\nworld,y\n")
    try:
        rows = read_records(path)
        assert rows[0]["prompt"] == "hello" and rows[1]["domain"] == "y"
    finally:
        os.remove(path)


def test_unsupported_format_raises():
    path = _tmp(".parquet", "junk")
    try:
        read_records(path)
        raise AssertionError("should have raised")
    except SynthkitError:
        pass
    finally:
        os.remove(path)


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
