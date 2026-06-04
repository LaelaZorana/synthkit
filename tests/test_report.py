"""Tests for report rendering — and that untrusted record text is HTML-escaped."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthkit.grading import grade_dataset
from synthkit.report import to_html, to_json


def test_json_report_is_valid_and_complete():
    report = grade_dataset([{"prompt": "a sufficiently long standalone prompt for grading"}])
    d = json.loads(to_json(report, "ds"))
    assert d["grade"] == report.grade and d["records"] == 1
    assert {x["key"] for x in d["dimensions"]} >= {"validity", "uniqueness", "diversity"}


def test_html_escapes_untrusted_record_text():
    # A leaked record carrying an XSS payload must surface escaped in the report.
    payload = "<script>alert(1)</script> with enough trailing words to be a real record"
    html = to_html(grade_dataset([{"prompt": payload}],
                                 against=[{"prompt": payload}], ngram=3), "ds")
    assert "<script>alert(1)</script>" not in html   # raw payload must NOT appear
    assert "&lt;script&gt;" in html                  # escaped form must appear


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
