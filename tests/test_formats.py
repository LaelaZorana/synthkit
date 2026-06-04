"""Tests for fine-tuning output formats."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthkit.formats import to_format

_REC = [{"instruction": "Sort a list in Python", "input": "",
         "output": "Use sorted().", "system": "You are helpful."}]


def test_alpaca_shape():
    o = to_format(_REC, "alpaca")[0]
    assert set(o) == {"instruction", "input", "output"}


def test_sharegpt_roles_and_value():
    o = to_format(_REC, "sharegpt")[0]
    assert [m["from"] for m in o["conversations"]] == ["system", "human", "gpt"]
    assert o["conversations"][-1]["value"] == "Use sorted()."


def test_openai_roles():
    o = to_format(_REC, "openai")[0]
    assert [m["role"] for m in o["messages"]] == ["system", "user", "assistant"]


def test_eval_prompt_maps_to_user_turn():
    o = to_format([{"prompt": "What is 2+2?"}], "openai")[0]
    assert o["messages"][0] == {"role": "user", "content": "What is 2+2?"}


def test_raw_is_passthrough():
    assert to_format(_REC, "raw") is _REC


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
