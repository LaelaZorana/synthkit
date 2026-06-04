"""Small shared utilities."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def pmap(fn: Callable[[T], R], items: Iterable[T], concurrency: int = 1,
         progress: Optional[Callable[[int, int], None]] = None) -> List[R]:
    """Map fn over items, optionally across threads, preserving input order.

    Exceptions propagate (the first one raised wins). `progress(done, total)`
    is called after each item completes.
    """
    items = list(items)
    total = len(items)
    results: List[Optional[R]] = [None] * total
    if concurrency <= 1:
        for i, it in enumerate(items):
            results[i] = fn(it)
            if progress:
                progress(i + 1, total)
        return results  # type: ignore[return-value]

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done += 1
            if progress:
                progress(done, total)
    return results  # type: ignore[return-value]


def unit(vec: Sequence[float]) -> List[float]:
    """L2-normalize a vector (a zero vector maps to itself)."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def max_cosine(u: Sequence[float], units: Sequence[Sequence[float]]) -> float:
    """Max cosine similarity between unit vector u and a list of unit vectors."""
    best = 0.0
    for w in units:
        s = 0.0
        for a, b in zip(u, w):
            s += a * b
        if s > best:
            best = s
    return best
