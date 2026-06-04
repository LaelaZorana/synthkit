"""Generate text records (eval prompts or instruction→output pairs) from a seed spec.

Two phases:
  1. sample prompts — deterministic for a fixed seed; exact-duplicate prompts are
     skipped by default so naive slot collisions don't pad the dataset.
  2. fill responses — only for instruction data with response.mode == 'provider';
     runs concurrently with a progress callback.

Optional: pass a `dedup_embedder` to dedup *by meaning as you generate* — each
candidate is embedded and rejected if it's within `dedup_threshold` cosine of an
already-accepted record.

Templates are rendered with a safe `{slot}`-only substitution (NOT str.format):
attribute/index access and format specs are treated as literal text, so an
untrusted template can't reach object internals or trigger a format-spec blow-up.
"""
from __future__ import annotations

import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from synthkit.grading import record_text
from synthkit.models import SynthkitError
from synthkit.providers import Embedder, Provider
from synthkit.util import max_cosine, pmap, unit

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_MAX_RECORD_CHARS = 100_000  # guards against a pathological slot value


def render(template: str, fill: Dict[str, str]) -> str:
    """Substitute only bare ``{slot}`` placeholders; everything else stays literal."""
    def repl(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in fill:
            raise SynthkitError(f"template slot '{key}' is missing from 'slots'")
        return str(fill[key])

    out = _PLACEHOLDER.sub(repl, template)
    if len(out) > _MAX_RECORD_CHARS:
        raise SynthkitError(f"rendered record exceeds {_MAX_RECORD_CHARS} characters")
    return out


def _rule_response(resp_cfg: Dict[str, Any], fill: Dict[str, str]) -> str:
    return render(str(resp_cfg.get("template", "")), fill)


def _shape_record(kind: str, prompt: str, response: str,
                  system: str, domain: str) -> Dict[str, Any]:
    if kind == "instruction":
        rec: Dict[str, Any] = {"instruction": prompt, "input": "", "output": response}
        if system:
            rec["system"] = system
    else:
        rec = {"prompt": prompt}
    if domain:
        rec["domain"] = domain
    return rec


def _response_for(kind: str, mode: str, resp_cfg: Dict[str, Any], fill: Dict[str, str],
                  prompt: str, system: str, provider: Optional[Provider]) -> str:
    if kind != "instruction":
        return ""
    if mode == "none":
        return ""
    if mode == "rule":
        return _rule_response(resp_cfg, fill)
    if mode == "provider":
        return provider.generate(prompt, system)  # type: ignore[union-attr]
    raise SynthkitError(f"unknown response.mode {mode!r}")


def sample_prompts(spec: Dict[str, Any], n: int, *, seed: int = 17,
                   dedup: bool = True,
                   max_attempts: Optional[int] = None
                   ) -> List[Tuple[str, Dict[str, str]]]:
    """Phase 1 — return up to n (prompt, slot-fill) pairs."""
    templates = spec.get("templates") or []
    if not templates:
        raise SynthkitError("seed spec has no 'templates'")
    slots: Dict[str, List[str]] = spec.get("slots") or {}
    min_words = int((spec.get("constraints") or {}).get("min_words", 0))

    rng = random.Random(seed)
    out: List[Tuple[str, Dict[str, str]]] = []
    seen: set = set()
    attempts = 0
    cap = max_attempts if max_attempts is not None else max(n * 50, 200)
    while len(out) < n and attempts < cap:
        attempts += 1
        template = rng.choice(templates)
        fill = {k: rng.choice(v) for k, v in slots.items()}
        prompt = render(template, fill)
        if len(prompt.split()) < min_words:
            continue
        if dedup:
            key = " ".join(prompt.lower().split())
            if key in seen:
                continue
            seen.add(key)
        out.append((prompt, fill))
    return out


def _generate_dedup(spec, n, provider, seed, dedup, progress,
                    embedder: Embedder, threshold: float,
                    stats: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kind = spec.get("kind", "eval")
    system = spec.get("system", "")
    domain = spec.get("domain", "")
    resp_cfg = spec.get("response") or {"mode": "none"}
    mode = resp_cfg.get("mode", "none")

    pool = sample_prompts(spec, max(n * 5, n + 100), seed=seed, dedup=dedup)
    records: List[Dict[str, Any]] = []
    units: List[List[float]] = []
    rejected = attempts = 0
    for prompt, fill in pool:
        if len(records) >= n:
            break
        attempts += 1
        resp = _response_for(kind, mode, resp_cfg, fill, prompt, system, provider)
        rec = _shape_record(kind, prompt, resp, system, domain)
        u = unit(embedder.embed([record_text(rec, None)])[0])
        if units and max_cosine(u, units) >= threshold:
            rejected += 1
        else:
            records.append(rec)
            units.append(u)
        if progress:
            progress(len(records), n)
    if stats is not None:
        stats["rejected_semantic"] = rejected
        stats["attempts"] = attempts
        stats["pool"] = len(pool)
    return records


def generate(spec: Dict[str, Any], n: int, *, provider: Optional[Provider] = None,
             seed: int = 17, dedup: bool = True,
             max_attempts: Optional[int] = None, concurrency: int = 1,
             progress: Optional[Callable[[int, int], None]] = None,
             dedup_embedder: Optional[Embedder] = None,
             dedup_threshold: float = 0.9,
             stats: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    kind = spec.get("kind", "eval")
    system = spec.get("system", "")
    domain = spec.get("domain", "")
    resp_cfg = spec.get("response") or {"mode": "none"}
    mode = resp_cfg.get("mode", "none")

    if mode == "provider" and provider is None:
        raise SynthkitError("response.mode is 'provider' but no provider was selected "
                            "(pass --provider ollama|anthropic|openai)")

    if dedup_embedder is not None:
        return _generate_dedup(spec, n, provider, seed, dedup, progress,
                               dedup_embedder, dedup_threshold, stats)

    prompts = sample_prompts(spec, n, seed=seed, dedup=dedup, max_attempts=max_attempts)
    responses: List[str] = [""] * len(prompts)
    if kind == "instruction" and mode != "none":
        if mode == "rule":
            responses = [_rule_response(resp_cfg, fill) for _, fill in prompts]
        elif mode == "provider":
            responses = pmap(lambda pf: provider.generate(pf[0], system),
                             prompts, concurrency=concurrency, progress=progress)
        else:
            raise SynthkitError(f"unknown response.mode {mode!r}")

    return [_shape_record(kind, prompt, resp, system, domain)
            for (prompt, _fill), resp in zip(prompts, responses)]
