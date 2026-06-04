"""Tests for text generation: counts, determinism, dedup, slot exhaustion."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthkit.models import SynthkitError
from synthkit.text.generate import generate, sample_prompts
from synthkit.text.seeds import DEMO_EVAL, DEMO_INSTRUCTION


def test_eval_generation_count_and_shape():
    data = generate(DEMO_EVAL, 50, seed=17)
    assert len(data) == 50
    assert all("prompt" in r for r in data)


def test_generation_is_deterministic():
    assert generate(DEMO_EVAL, 30, seed=123) == generate(DEMO_EVAL, 30, seed=123)


def test_dedup_skips_exact_repeats():
    prompts = [p for p, _ in sample_prompts(DEMO_EVAL, 200, seed=5, dedup=True)]
    assert len(prompts) == len(set(prompts))


def test_slot_exhaustion_caps_output():
    tiny = {"kind": "eval", "templates": ["{a} {b}"],
            "slots": {"a": ["x", "y"], "b": ["1", "2"]}, "response": {"mode": "none"}}
    assert len(generate(tiny, 100, seed=1)) == 4  # only 2×2 unique combos exist


def test_rule_mode_fills_output_offline():
    data = generate(DEMO_INSTRUCTION, 10, seed=2)  # response.mode == "rule"
    assert all("instruction" in r and r["output"] for r in data)


class _HashEmbedder:
    """Fixed-vocab bag-of-words vectors (stable across separate embed() calls)."""

    D = 128

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.D
            for w in t.lower().split():
                v[sum(ord(c) for c in w) % self.D] += 1.0
            out.append(v)
        return out


def test_semantic_dedup_during_generation():
    # 5 prompts differing only by one adjective ⇒ near-identical meaning vectors
    spec = {"kind": "eval", "templates": ["the {a} cat sat on the mat"],
            "slots": {"a": ["happy", "sad", "sleepy", "tiny", "huge"]},
            "response": {"mode": "none"}}
    stats = {}
    data = generate(spec, 5, seed=1, dedup_embedder=_HashEmbedder(),
                    dedup_threshold=0.85, stats=stats)
    assert len(data) == 1, len(data)              # only the first survives
    assert stats["rejected_semantic"] == 4, stats


def test_template_injection_is_inert():
    # A pasted template trying CWE-134 format-string tricks must stay LITERAL.
    spec = {"kind": "eval",
            "templates": ["leak {b.__class__.__mro__} pad {b:>999999}"],
            "slots": {"b": ["x"]}, "response": {"mode": "none"},
            "constraints": {"min_words": 0}}
    p = generate(spec, 1, seed=1)[0]["prompt"]
    assert "<class" not in p and len(p) < 200      # no attribute walk, no format-spec DoS


def test_missing_slot_raises_synthkit_error():
    try:
        generate({"kind": "eval", "templates": ["{nope}"],
                  "slots": {"b": ["x"]}, "response": {"mode": "none"}}, 1)
        raise AssertionError("should have raised")
    except SynthkitError:
        pass


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
