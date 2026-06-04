# synthkit

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-live%20demo-yellow)](https://huggingface.co/spaces/LaelaZ/synthkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

**Generate synthetic data — and grade it before you train on it.**

> **Live demo** — try the grader in your browser: **[huggingface.co/spaces/LaelaZ/synthkit](https://huggingface.co/spaces/LaelaZ/synthkit)**

Anyone can generate synthetic data. The hard part is knowing whether it's any
good: Is it full of near-duplicates? Is it diverse enough to teach anything? Is
it secretly leaking your eval set into your training set? `synthkit` generates
data *and* grades it on those axes, with a single A+→F headline and a per-axis
breakdown.

```
$ synthkit text gen --demo

  ┌─ synthkit · data quality ──────────────────────────
  │
  │  Quality grade   B    (84.3/100)
  │  Records         200
  │  Dataset         synthkit_demo.jsonl
  │
  └────────────────────────────────────────────────────

  DIMENSIONS

  ● Validity        100  █████████████████████  200/200 records well-formed
  ● Uniqueness       98  ████████████████████░  197/200 unique (0 exact, 3 near)
  ● Diversity        72  ███████████████░░░░░░  distinct-2 0.66 · self-sim 0.18
  ● Contamination    96  ████████████████████░  8/200 records overlap the eval set
```

Zero dependencies for the core — it runs on the Python standard library alone.

---

## Three products, one core

synthkit is built as a toolkit. The generator differs per data type; the
**grading engine, providers, and report writer are shared**.

| Module | Makes | Status |
|---|---|---|
| **`text`** (A) | LLM instruction & evaluation datasets | **live** |
| **`tabular`** (B) | schema-aware fixtures with referential integrity | roadmap |
| **`privacy`** (C) | privacy-safe synthetic twins of real datasets | roadmap |

---

## Quickstart

No install needed:

```bash
python3 main.py text gen --demo        # generate + grade a built-in coding eval set
python3 main.py grade my_data.jsonl    # grade any dataset you already have
python3 main.py list                   # products + built-in seeds
```

Or install the `synthkit` command:

```bash
pip install -e .
synthkit text gen --demo
```

## Generate from your own seed

A **seed spec** is a small JSON (or YAML) file: templates with `{slots}`, the
fillers for each slot, and how to produce the response.

```bash
# prompt-only eval set, fully offline
synthkit text gen --seed examples/eval.support.json -n 300 -o support_eval.jsonl

# instruction→output pairs, responses from a local (free) Ollama model,
# written as ready-to-train alpaca JSONL, 8 calls in flight at once
synthkit text gen --seed examples/instruction.datascience.json \
    --provider ollama --model qwen2.5-coder:14b -n 500 \
    --format alpaca --concurrency 8 -o ds_sft.jsonl
```

**Output formats** (`--format`): `raw` (default), `alpaca` (`instruction/input/output`),
`sharegpt` (`conversations`), `openai` (`messages`) — pick whatever your trainer expects.

**Clean by construction** (`--dedup-semantic`): embed each candidate as it's generated
and reject it if it's within `--dedup-threshold` cosine of a record you've already kept,
so the output has no meaning-duplicates. It also tells you the truth about your seed — one
that emits 36 lexically-unique prompts but only ~6 distinct *meanings* yields 6 and says so,
instead of padding the file with redundancy.

Response providers (only needed for instruction→output pairs):

| `--provider` | Cost | Needs |
|---|---|---|
| `none` (default) | free | nothing — prompt-only eval sets |
| `ollama` | free, local | a running Ollama daemon |
| `anthropic` | paid | `pip install anthropic`, `ANTHROPIC_API_KEY` |
| `openai` | paid | `pip install openai`, `OPENAI_API_KEY` |

## The grading axes

| Axis | What it measures | How |
|---|---|---|
| **Validity** | structurally sound records | required fields present, non-empty, sane length |
| **Uniqueness** | free of duplicates | exact match + near-dup via MinHash + LSH |
| **Diversity** | lexical variety | distinct-1/2, mean self-similarity |
| **Contamination** | train/eval leakage | n-gram containment vs. a held-out set (`--against`) |
| **Semantic** *(opt-in)* | meaning-level redundancy | embedding cosine near-dup (`--semantic`) |

`--semantic` is the upgrade that earns its keep: a dataset can be **100% lexically
unique yet full of meaning-duplicates** (e.g. an LLM answering five different prompts
the same way). It embeds each record — local `nomic-embed-text` via Ollama by default,
free — and flags cosine near-duplicates the MinHash pass can't see.

The headline grade is a weighted blend of whichever axes apply (contamination
only counts with `--against`; semantic only with `--semantic`). Raw numbers for
every axis are in the JSON report — the letter is a convenience, not a claim of
ground truth.

```bash
# catch eval-set leakage in training data
synthkit grade train.jsonl --against benchmark.jsonl --html report.html

# CI gate: fail the build if quality drops below B
synthkit grade train.jsonl --min-grade B
```

## Output

Every run prints the terminal report and can also write:

- `--report-json` / `--json` — machine-readable, every raw metric
- `--html` — a standalone, shareable report

## Roadmap

- **B · tabular** — point at a DB schema or sample rows, get realistic fixtures
  with foreign-key integrity; adds referential-integrity + PII-safety axes.
- **C · privacy** — fit a generator to a real dataset, emit a statistically
  similar twin; adds distribution-fidelity + membership-inference-distance axes.

Both reuse the Product A core unchanged.

## License

MIT.
