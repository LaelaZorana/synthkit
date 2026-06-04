"""End-to-end CLI smoke tests via subprocess."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cli(*args):
    return subprocess.run([sys.executable, "main.py", *args],
                          cwd=_ROOT, capture_output=True, text=True)


def test_version():
    r = _cli("--version")
    assert r.returncode == 0 and "synthkit" in r.stdout


def test_grade_and_min_grade_gate():
    # 6 exact duplicates ⇒ uniqueness tanks ⇒ the grade is well below A+.
    recs = [{"prompt": "the quick brown fox jumps over the lazy dog"}] * 6 + [
        {"prompt": "an entirely separate standalone sentence about astronomy and stars"},
        {"prompt": "yet another distinct sentence concerning marine biology and coral"},
    ]
    path = os.path.join(tempfile.mkdtemp(), "d.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    assert _cli("grade", path, "--no-color").returncode == 0
    # the CI gate must fail (non-zero) when the grade is below the threshold
    assert _cli("grade", path, "--no-color", "--min-grade", "A+").returncode == 1


def test_text_gen_demo_runs():
    r = _cli("text", "gen", "--demo", "--no-color")
    assert r.returncode == 0
    assert os.path.exists(os.path.join(_ROOT, "synthkit_demo.jsonl"))


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
