"""Smoke tests for the grading engine.

Run with pytest, or standalone with no dependencies:
    python3 tests/test_grading.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthkit.grading import grade_dataset


def test_exact_duplicates_drop_uniqueness():
    recs = [{"prompt": "the quick brown fox jumps over the lazy dog"}] * 4
    u = grade_dataset(recs).dim("uniqueness")
    assert u.stats["exact"] == 3, u.stats
    assert u.score < 60, u.score


def test_near_duplicate_detected():
    recs = [
        {"prompt": "please write a function that reverses a string in python carefully"},
        {"prompt": "please write a function that reverses a string in python carefully now"},
        {"prompt": "explain photosynthesis to a five year old child in simple terms"},
    ]
    assert grade_dataset(recs).dim("uniqueness").stats["near"] >= 1


def test_diversity_high_for_varied_low_for_repetitive():
    varied = [{"prompt": p} for p in [
        "how do volcanoes form deep beneath the ocean floor over time",
        "what causes the northern lights to shimmer green across the sky",
        "explain compound interest with a simple worked numerical example",
        "describe how vaccines train the human immune system to respond",
        "why does the sky look blue at noon and red at sunset",
    ]]
    repetitive = [
        {"prompt": "the quick brown fox jumps over the lazy sleeping dog by the river " + w}
        for w in "alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
    ]
    hi = grade_dataset(varied).dim("diversity").score
    lo = grade_dataset(repetitive).dim("diversity").score
    assert hi > lo, (hi, lo)


def test_contamination_flags_leaked_eval():
    eval_set = [{"prompt": "what is the capital city of australia"}]
    train = [
        {"prompt": "what is the capital city of australia"},        # leaked verbatim
        {"prompt": "name three primary colors used in oil painting"},  # clean
    ]
    c = grade_dataset(train, against=eval_set, ngram=5).dim("contamination")
    assert c.stats["flagged"] == 1, c.stats


def test_contamination_na_without_eval():
    r = grade_dataset([{"prompt": "hello world this is a test prompt for grading"}])
    assert r.dim("contamination").score is None


def test_validity_flags_empty_and_short():
    recs = [
        {"prompt": ""},
        {"prompt": "hi"},
        {"prompt": "a perfectly fine and sufficiently long prompt to keep here"},
    ]
    v = grade_dataset(recs, min_words=3).dim("validity")
    assert v.stats["empty"] == 1 and v.stats["too_short"] == 1, v.stats


class _FakeEmbedder:
    """Deterministic bag-of-words vectors — identical text ⇒ identical vector."""

    def embed(self, texts):
        vocab = sorted({w for t in texts for w in t.lower().split()})
        idx = {w: i for i, w in enumerate(vocab)}
        out = []
        for t in texts:
            v = [0.0] * len(vocab)
            for w in t.lower().split():
                v[idx[w]] += 1.0
            out.append(v)
        return out


def test_semantic_axis_flags_duplicate_meanings():
    recs = [
        {"prompt": "the cat sat on the warm mat"},
        {"prompt": "the cat sat on the warm mat"},          # identical ⇒ semantic dup
        {"prompt": "quantum entanglement links distant particles instantly"},
    ]
    s = grade_dataset(recs, embedder=_FakeEmbedder(), semantic_threshold=0.9).dim("semantic")
    assert s is not None and s.stats["semantic_dups"] == 1, s.stats


def test_semantic_axis_absent_without_embedder():
    assert grade_dataset([{"prompt": "a sufficiently long standalone prompt here"}]).dim("semantic") is None


def test_empty_dataset_guard():
    r = grade_dataset([])
    assert r.n_records == 0 and r.grade == "F" and r.dimensions == []


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
