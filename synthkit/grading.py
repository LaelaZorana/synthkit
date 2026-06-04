"""The grading engine — the part every synthkit product shares.

Given a list of records, score the dataset on four axes:

  validity       structurally sound records (required fields, non-empty, sane length)
  uniqueness     free of exact and near-duplicate records (MinHash + LSH)
  diversity      lexical variety across the set (distinct-n, self-similarity)
  contamination  overlap with a held-out eval/benchmark set (n-gram containment)

Everything here is standard-library only and deterministic for a fixed seed.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from synthkit.models import DimensionScore, GradeReport, to_grade

_MERSENNE = (1 << 61) - 1
_WORD = re.compile(r"[a-z0-9]+")
# Content fields we analyze by default (role/config keys like "system" excluded).
_KNOWN_FIELDS = ("instruction", "input", "prompt", "question",
                 "output", "response", "answer", "text")


# ---- text helpers ------------------------------------------------------------

def record_text(rec: Dict[str, Any], fields: Optional[Sequence[str]]) -> str:
    """Flatten the fields we analyze into a single string."""
    if fields:
        vals = [rec.get(f, "") for f in fields]
    else:
        keys = [k for k in _KNOWN_FIELDS if k in rec]
        if not keys:
            keys = [k for k, v in rec.items() if isinstance(v, str)]
        vals = [rec.get(k, "") for k in keys]
    return "  ".join(str(v) for v in vals if v is not None)


def tokens(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def shingles(toks: Sequence[str], k: int) -> Set[str]:
    if not toks:
        return set()
    if len(toks) < k:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def ngram_set(toks: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
    """Word n-grams as tuples; items shorter than n contribute one whole tuple."""
    if not toks:
        return set()
    if len(toks) <= n:
        return {tuple(toks)}
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


# ---- MinHash + LSH near-duplicate detection ----------------------------------

class _MinHasher:
    def __init__(self, num_perm: int, seed: int) -> None:
        rng = random.Random(seed)
        self.a = [rng.randrange(1, _MERSENNE) for _ in range(num_perm)]
        self.b = [rng.randrange(0, _MERSENNE) for _ in range(num_perm)]

    def sign(self, shs: Set[str]) -> Optional[Tuple[int, ...]]:
        if not shs:
            return None
        base = [_stable_hash(s) for s in shs]
        return tuple(min((a * h + b) % _MERSENNE for h in base)
                     for a, b in zip(self.a, self.b))


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # keep the smaller index as root (the "original" of the cluster)
        if rx < ry:
            self.parent[ry] = rx
        else:
            self.parent[rx] = ry


class _LSHIndex:
    """MinHash + LSH index over a list of shingle sets.

    Build once, then `candidates(shingles)` returns the (small) set of indices that
    share at least one band with the query — turning all-pairs similarity work into
    near-linear candidate lookups. Used for both near-dup detection and the
    contamination check so neither is quadratic in the dataset size.
    """

    def __init__(self, shingle_sets: List[Set[str]], *, num_perm: int = 64,
                 bands: int = 16, seed: int = 17) -> None:
        self._hasher = _MinHasher(num_perm, seed)
        self._bands = bands
        self._rows = num_perm // bands
        self._buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
        for idx, shingset in enumerate(shingle_sets):
            sig = self._hasher.sign(shingset)
            if sig is None:
                continue
            for band in range(bands):
                key = (band, sig[band * self._rows:(band + 1) * self._rows])
                self._buckets.setdefault(key, []).append(idx)

    def candidates(self, shingset: Set[str]) -> Set[int]:
        sig = self._hasher.sign(shingset)
        if sig is None:
            return set()
        out: Set[int] = set()
        for band in range(self._bands):
            key = (band, sig[band * self._rows:(band + 1) * self._rows])
            out.update(self._buckets.get(key, ()))
        return out


def _duplicate_map(shingle_sets: List[Set[str]], *, threshold: float = 0.8,
                   seed: int = 17) -> Dict[int, int]:
    """Return {pos: root_pos} for every entry that near-duplicates an earlier one."""
    index = _LSHIndex(shingle_sets, seed=seed)
    uf = _UnionFind(len(shingle_sets))
    for i, shingset in enumerate(shingle_sets):
        if not shingset:
            continue
        for j in index.candidates(shingset):
            if j >= i:
                continue
            other = shingle_sets[j]
            if other and len(shingset & other) / len(shingset | other) >= threshold:
                uf.union(i, j)

    dup_of: Dict[int, int] = {}
    for idx in range(len(shingle_sets)):
        root = uf.find(idx)
        if root != idx:
            dup_of[idx] = root
    return dup_of


# ---- the four dimensions -----------------------------------------------------

def _validity_dim(records, texts, *, min_words, max_words) -> DimensionScore:
    n = len(records)
    empty = short = long = 0
    examples: List[str] = []
    for text in texts:
        wc = len(tokens(text))
        if not text.strip():
            empty += 1
            if len(examples) < 3:
                examples.append("empty record")
        elif wc < min_words:
            short += 1
            if len(examples) < 3:
                examples.append(f"only {wc} words: {text[:60]!r}")
        elif wc > max_words:
            long += 1
            if len(examples) < 3:
                examples.append(f"{wc} words (over {max_words})")
    bad = empty + short + long
    score = 100.0 * (1 - bad / n) if n else 0.0
    findings: List[str] = []
    if empty:
        findings.append(f"{empty} empty record(s)")
    if short:
        findings.append(f"{short} below {min_words} words")
    if long:
        findings.append(f"{long} above {max_words} words")
    findings += examples
    return DimensionScore(
        "validity", "Validity", round(score, 1),
        f"{n - bad}/{n} records well-formed", findings,
        {"empty": empty, "too_short": short, "too_long": long, "n": n}, weight=0.25)


def _uniqueness_dim(records, texts, shingle_sets, *, seed) -> DimensionScore:
    n = len(records)
    seen: Dict[str, int] = {}
    exact_dup: Set[int] = set()
    rep_indices: List[int] = []
    for idx, text in enumerate(texts):
        norm = " ".join(tokens(text))
        if norm and norm in seen:
            exact_dup.add(idx)
        else:
            if norm:
                seen[norm] = idx
            rep_indices.append(idx)
    # near-dup search runs only over exact-unique representatives (keeps it cheap)
    rep_shingles = [shingle_sets[i] for i in rep_indices]
    local_dup = _duplicate_map(rep_shingles, seed=seed)
    near_dup: Set[int] = set()
    clusters: Set[int] = set()
    example = ""
    for local_idx, local_root in local_dup.items():
        gi, gr = rep_indices[local_idx], rep_indices[local_root]
        near_dup.add(gi)
        clusters.add(gr)
        if not example:
            example = f"e.g. #{gi} ≈ #{gr}: {texts[gi][:64]!r}"
    dup_total = len(exact_dup) + len(near_dup)
    score = 100.0 * (1 - dup_total / n) if n else 0.0
    findings: List[str] = []
    if exact_dup:
        findings.append(f"{len(exact_dup)} exact duplicate(s)")
    if near_dup:
        findings.append(f"{len(near_dup)} near-duplicate(s) in {len(clusters)} cluster(s)")
    if example:
        findings.append(example)
    return DimensionScore(
        "uniqueness", "Uniqueness", round(score, 1),
        f"{n - dup_total}/{n} unique  ({len(exact_dup)} exact, {len(near_dup)} near)",
        findings,
        {"exact": len(exact_dup), "near": len(near_dup),
         "clusters": len(clusters), "n": n}, weight=0.30)


def _diversity_dim(texts, shingle_sets, *, seed) -> DimensionScore:
    unigrams: List[str] = []
    bigrams: List[Tuple[str, str]] = []
    for text in texts:
        ts = tokens(text)
        unigrams.extend(ts)
        bigrams.extend(zip(ts, ts[1:]))
    d1 = len(set(unigrams)) / len(unigrams) if unigrams else 0.0
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    # self-similarity: mean Jaccard over a seeded sample of record pairs
    rng = random.Random(seed)
    idxs = [i for i, s in enumerate(shingle_sets) if s]
    sims: List[float] = []
    if len(idxs) >= 2:
        for _ in range(min(2000, len(idxs) * 4)):
            i, j = rng.sample(idxs, 2)
            a, b = shingle_sets[i], shingle_sets[j]
            sims.append(len(a & b) / len(a | b))
    self_sim = sum(sims) / len(sims) if sims else 0.0
    # Pairwise distinctness (1 - self-similarity) is the size-stable signal and
    # leads; distinct-2 is a secondary lexical-variety term with a lenient target
    # (corpus-level distinct-n shrinks as the set grows). Raw numbers are reported
    # in stats either way so the letter is never the whole story.
    score = 100.0 * (0.6 * (1 - self_sim) + 0.4 * min(1, d2 / 0.25))
    findings: List[str] = []
    if d2 < 0.4:
        findings.append("low bigram diversity — templates may be too repetitive")
    if self_sim > 0.3:
        findings.append(f"records are {self_sim * 100:.0f}% similar on average")
    return DimensionScore(
        "diversity", "Diversity", round(score, 1),
        f"distinct-2 {d2:.2f} · distinct-1 {d1:.2f} · self-sim {self_sim:.2f}",
        findings,
        {"distinct_1": round(d1, 4), "distinct_2": round(d2, 4),
         "self_similarity": round(self_sim, 4), "vocab": len(set(unigrams))},
        weight=0.25)


def _contamination_dim(texts, shingle_sets, against_texts, *, ngram) -> DimensionScore:
    if against_texts is None:
        return DimensionScore(
            "contamination", "Contamination", None,
            "no eval set provided (pass --against to check)", [], {}, weight=0.20)
    eval_ngrams: Set[Tuple[str, ...]] = set()
    eval_shingles: List[Set[str]] = []
    for t in against_texts:
        ts = tokens(t)
        eval_ngrams |= ngram_set(ts, ngram)
        eval_shingles.append(shingles(ts, 5))
    eval_index = _LSHIndex(eval_shingles, seed=17)     # avoid the O(records×eval) scan
    flagged: List[Tuple[int, str, str]] = []
    for idx, text in enumerate(texts):
        ts = tokens(text)
        sh = shingle_sets[idx]
        hit, reason = False, ""
        if sh:                                         # near-duplicate of an eval item
            for j in eval_index.candidates(sh):
                es = eval_shingles[j]
                if es and len(sh & es) / len(sh | es) >= 0.7:
                    hit, reason = True, "near-duplicate of an eval item"
                    break
        if not hit:
            # n-gram containment: what fraction of THIS record's n-grams are in the
            # eval set. Robust to shared template boilerplate (only a few n-grams),
            # which a raw "shares any n-gram" check would over-flag.
            grams = ngram_set(ts, ngram)
            if grams:
                contained = sum(1 for g in grams if g in eval_ngrams) / len(grams)
                if contained >= 0.8:
                    hit = True
                    reason = f"{contained * 100:.0f}% of its {ngram}-grams are in the eval set"
        if hit:
            flagged.append((idx, reason, text[:64]))
    n = len(texts)
    score = 100.0 * (1 - len(flagged) / n) if n else 100.0
    summary = (f"{len(flagged)}/{n} records overlap the eval set" if flagged
               else f"clean — 0/{n} overlap the eval set")
    findings = [f"#{i}: {why}: {snip!r}" for i, why, snip in flagged[:4]]
    return DimensionScore(
        "contamination", "Contamination", round(score, 1), summary, findings,
        {"flagged": len(flagged), "n": n, "ngram": ngram}, weight=0.20)


# ---- optional semantic axis (embedding-based) --------------------------------

def _unit(v: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cos_unit(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _semantic_dups(units: List[List[float]], threshold: float) -> Set[int]:
    n = len(units)
    dup: Set[int] = set()
    try:
        import numpy as np  # fast path if available
        sims = np.asarray(units) @ np.asarray(units).T
        for a in range(n):
            if a in dup:
                continue
            row = sims[a]
            for b in range(a + 1, n):
                if b not in dup and row[b] >= threshold:
                    dup.add(b)
    except ImportError:
        for a in range(n):
            if a in dup:
                continue
            ua = units[a]
            for b in range(a + 1, n):
                if b not in dup and _cos_unit(ua, units[b]) >= threshold:
                    dup.add(b)
    return dup


def _semantic_dim(texts, embedder, *, threshold, seed, max_n=400) -> DimensionScore:
    idx = [i for i, t in enumerate(texts) if t.strip()]
    note = ""
    if len(idx) > max_n:
        idx = sorted(random.Random(seed).sample(idx, max_n))
        note = f" (sampled {max_n})"
    units = [_unit(v) for v in embedder.embed([texts[i] for i in idx])]
    dup = _semantic_dups(units, threshold)
    rng = random.Random(seed)
    sims = []
    if len(units) >= 2:
        for _ in range(min(2000, len(units) * 4)):
            a, b = rng.sample(range(len(units)), 2)
            sims.append(_cos_unit(units[a], units[b]))
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    n = len(idx)
    score = 100.0 * (1 - len(dup) / n) if n else 100.0
    findings = []
    if dup:
        findings.append(f"{len(dup)} semantic near-duplicate(s) at cosine ≥ {threshold} "
                        "(paraphrases lexical dedup misses)")
    summary = f"{n - len(dup)}/{n} semantically distinct · mean cosine {mean_sim:.2f}{note}"
    return DimensionScore(
        "semantic", "Semantic dedup", round(score, 1), summary, findings,
        {"semantic_dups": len(dup), "mean_cosine": round(mean_sim, 4),
         "n": n, "threshold": threshold}, weight=0.20)


# ---- public API --------------------------------------------------------------

def grade_dataset(records: List[Dict[str, Any]], *,
                  fields: Optional[Sequence[str]] = None,
                  against: Optional[List[Dict[str, Any]]] = None,
                  against_fields: Optional[Sequence[str]] = None,
                  ngram: int = 8, min_words: int = 3, max_words: int = 512,
                  seed: int = 17, embedder=None,
                  semantic_threshold: float = 0.83) -> GradeReport:
    if not records:
        return GradeReport(grade="F", score=0.0, n_records=0, dimensions=[],
                           meta={"note": "empty dataset"})

    texts = [record_text(r, fields) for r in records]
    shingle_sets = [shingles(tokens(t), 5) for t in texts]
    against_texts = None
    if against is not None:
        against_texts = [record_text(r, against_fields or fields) for r in against]

    dims = [
        _validity_dim(records, texts, min_words=min_words, max_words=max_words),
        _uniqueness_dim(records, texts, shingle_sets, seed=seed),
        _diversity_dim(texts, shingle_sets, seed=seed),
        _contamination_dim(texts, shingle_sets, against_texts, ngram=ngram),
    ]
    if embedder is not None:
        dims.append(_semantic_dim(texts, embedder, threshold=semantic_threshold, seed=seed))

    applicable = [d for d in dims if d.applicable]
    wsum = sum(d.weight for d in applicable) or 1.0
    overall = sum(d.score * d.weight for d in applicable) / wsum
    return GradeReport(
        grade=to_grade(overall), score=round(overall, 1), n_records=len(records),
        dimensions=dims,
        meta={"fields": list(fields) if fields else "auto",
              "has_eval": against_texts is not None,
              "semantic": embedder is not None})
