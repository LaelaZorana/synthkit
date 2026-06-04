"""Tests for provider / embedder selection (no network calls)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthkit.models import SynthkitError
from synthkit.providers import OllamaEmbedder, OllamaProvider, get_embedder, get_provider


def test_none_returns_none():
    assert get_provider("none") is None
    assert get_provider(None) is None
    assert get_embedder("none") is None


def test_ollama_constructs_without_network():
    p = get_provider("ollama", "llama3.2")
    assert isinstance(p, OllamaProvider) and p.model == "llama3.2"
    assert isinstance(get_embedder("ollama"), OllamaEmbedder)


def test_unknown_name_raises():
    for fn in (get_provider, get_embedder):
        try:
            fn("bogus")
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
