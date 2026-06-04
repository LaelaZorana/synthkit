"""Command-line interface for synthkit."""
from __future__ import annotations

import argparse
import sys
from typing import List

from synthkit import __version__
from synthkit.formats import FORMATS, to_format
from synthkit.grading import grade_dataset
from synthkit.io_utils import load_spec, read_records, write_jsonl, write_text
from synthkit.models import SynthkitError
from synthkit.providers import get_embedder, get_provider
from synthkit.report import print_report, to_html, to_json
from synthkit.text.generate import generate
from synthkit.text.seeds import BUILTIN_SEEDS, DEMO_EVAL

_GRADE_ORDER = ["F", "D", "C", "B", "A", "A+"]


def _progress(done: int, total: int) -> None:
    end = "\n" if done >= total else ""
    print(f"\r  generated {done}/{total}", end=end, file=sys.stderr, flush=True)


def _maybe_embedder(args):
    if not getattr(args, "semantic", False):
        return None
    return get_embedder(getattr(args, "embed_provider", "ollama"),
                        getattr(args, "embed_model", ""))


# ---- text gen ----------------------------------------------------------------

def _run_demo(args) -> int:
    print("synthkit demo: generating a coding eval set, then grading it…",
          file=sys.stderr)
    # Train and eval are drawn from DISJOINT tasks (a genuine held-out split), then a
    # known handful of eval records are deliberately leaked into train, so the
    # contamination axis reflects real leakage, not one generator overlapping itself.
    tasks = DEMO_EVAL["slots"]["task"]
    train_spec = {**DEMO_EVAL, "slots": {**DEMO_EVAL["slots"], "task": tasks[:7]}}
    eval_spec = {**DEMO_EVAL, "slots": {**DEMO_EVAL["slots"], "task": tasks[7:]}}
    data = generate(train_spec, 195, seed=17)
    bench = generate(eval_spec, 40, seed=99)
    leaks = [dict(r) for r in bench[:5]]              # 5 genuine, verbatim leaks
    data = data + leaks
    write_jsonl("synthkit_demo.jsonl", data)
    write_jsonl("synthkit_demo.benchmark.jsonl", bench)

    report = grade_dataset(data, against=bench, ngram=args.ngram)
    print_report(report, dataset="synthkit_demo.jsonl", use_color=not args.no_color)
    write_text("synthkit_demo.report.json", to_json(report, "synthkit_demo.jsonl"))
    write_text("synthkit_demo.report.html", to_html(report, "synthkit_demo.jsonl"))
    print(f"  wrote synthkit_demo.jsonl ({len(data)} records) + benchmark ({len(bench)}) "
          "+ .report.json + .report.html", file=sys.stderr)
    print("  note: eval uses HELD-OUT tasks; 5 records were deliberately leaked into "
          "train, so contamination flags exactly those real leaks.", file=sys.stderr)
    return 0


def cmd_text_gen(args) -> int:
    if args.demo:
        return _run_demo(args)
    if not args.seed:
        sys.exit("error: provide --seed FILE (a JSON/YAML seed spec) or use --demo")

    spec = load_spec(args.seed)
    if args.kind:
        spec["kind"] = args.kind
    provider = get_provider(args.provider, args.model)
    dedup_embedder = get_embedder(args.embed_provider, args.embed_model) if args.dedup_semantic else None
    busy = args.provider != "none" or args.dedup_semantic
    show = _progress if (busy and not args.no_progress) else None
    gstats: dict = {}

    print(f"synthkit: generating {args.num} records from {args.seed}…", file=sys.stderr)
    data = generate(spec, args.num, provider=provider, seed=args.seed_int,
                    dedup=not args.no_dedup, concurrency=args.concurrency, progress=show,
                    dedup_embedder=dedup_embedder, dedup_threshold=args.dedup_threshold,
                    stats=gstats)
    if args.dedup_semantic and gstats.get("rejected_semantic"):
        print(f"  semantic dedup: rejected {gstats['rejected_semantic']} of "
              f"{gstats['attempts']} candidates (cosine ≥ {args.dedup_threshold})",
              file=sys.stderr)
    if len(data) < args.num:
        reason = ("raise --dedup-threshold or add slot variety" if args.dedup_semantic
                  else "the seed's template×slot space is exhausted "
                       "(add slot variety or pass --no-dedup)")
        print(f"  note: produced {len(data)} of {args.num} requested, {reason}.",
              file=sys.stderr)

    out = args.out or "synth_text.jsonl"
    write_jsonl(out, to_format(data, args.format))
    print(f"  wrote {len(data)} records to {out}"
          + (f"  ({args.format} format)" if args.format != "raw" else ""))

    if not args.no_grade:
        against = read_records(args.against) if args.against else None
        report = grade_dataset(data, against=against, ngram=args.ngram,
                               embedder=_maybe_embedder(args))
        print_report(report, dataset=out, use_color=not args.no_color)
        if args.report_json:
            write_text(args.report_json, to_json(report, out))
        if args.html:
            write_text(args.html, to_html(report, out))
    return 0


# ---- grade -------------------------------------------------------------------

def cmd_grade(args) -> int:
    records = read_records(args.path)
    if not records:
        sys.exit(f"error: no records found in {args.path}")
    against = read_records(args.against) if args.against else None
    report = grade_dataset(records, fields=args.field or None,
                           against=against, ngram=args.ngram,
                           embedder=_maybe_embedder(args))
    print_report(report, dataset=args.path, use_color=not args.no_color)
    if args.json:
        write_text(args.json, to_json(report, args.path))
        print(f"  JSON written to {args.json}")
    if args.html:
        write_text(args.html, to_html(report, args.path))
        print(f"  HTML written to {args.html}")
    if args.min_grade:
        if _GRADE_ORDER.index(report.grade) < _GRADE_ORDER.index(args.min_grade):
            print(f"  grade {report.grade} is below --min-grade {args.min_grade}",
                  file=sys.stderr)
            return 1
    return 0


# ---- list / roadmap ----------------------------------------------------------

def cmd_list(args) -> int:
    print("\nsynthkit products\n")
    print("  text      (A) live     LLM instruction & eval datasets + quality grading")
    print("  tabular   (B) roadmap  schema-aware fixtures with referential integrity")
    print("  privacy   (C) roadmap  privacy-safe synthetic twins of real datasets")
    print("\n  built-in text seeds (use with: text gen --demo, or copy from examples/)\n")
    for name, spec in BUILTIN_SEEDS.items():
        t = len(spec["templates"])
        slots = " × ".join(f"{len(v)} {k}" for k, v in spec["slots"].items())
        print(f"    {name:<12} {spec['kind']:<12} {t} templates · {slots}")
    print()
    return 0


def cmd_coming_soon(args) -> int:
    print(f"\n  synthkit {args.product}: {args.blurb}")
    print("  On the roadmap. Product A (`synthkit text`) is live today and B/C share")
    print("  the same core: providers, the grading engine, and the report writer.\n")
    return 0


# ---- parser ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synthkit",
        description="Generate synthetic data and grade it for quality "
                    "(validity, uniqueness, diversity, contamination).")
    p.add_argument("--version", action="version", version=f"synthkit {__version__}")
    sub = p.add_subparsers(dest="cmd")

    # text (Product A) with its own subcommands
    text_p = sub.add_parser("text", help="Product A: LLM instruction & eval datasets")
    text_sub = text_p.add_subparsers(dest="text_cmd")
    gen = text_sub.add_parser("gen", help="generate a text dataset and grade it")
    gen.add_argument("--demo", action="store_true",
                     help="generate + grade a built-in coding eval set (no setup)")
    gen.add_argument("--seed", help="path to a JSON/YAML seed spec")
    gen.add_argument("-n", "--num", type=int, default=200, help="records to generate")
    gen.add_argument("--kind", choices=["eval", "instruction"],
                     help="override the spec's kind")
    gen.add_argument("--provider", default="none",
                     choices=["none", "ollama", "anthropic", "openai"],
                     help="response generator for instruction data (default: none)")
    gen.add_argument("--model", default="", help="model name for the provider")
    gen.add_argument("-o", "--out", help="output JSONL path (default: synth_text.jsonl)")
    gen.add_argument("--seed-int", type=int, default=17, help="RNG seed (default: 17)")
    gen.add_argument("--no-dedup", action="store_true",
                     help="keep exact-duplicate prompts instead of skipping them")
    gen.add_argument("--no-grade", action="store_true", help="skip grading the output")
    gen.add_argument("--against", help="held-out eval set to check contamination against")
    gen.add_argument("--ngram", type=int, default=8, help="contamination n-gram size")
    gen.add_argument("--format", default="raw", choices=list(FORMATS),
                     help="output schema: raw|alpaca|sharegpt|openai (default: raw)")
    gen.add_argument("--concurrency", type=int, default=4,
                     help="parallel provider calls when filling responses (default: 4)")
    gen.add_argument("--no-progress", action="store_true",
                     help="hide the response progress line")
    gen.add_argument("--semantic", action="store_true",
                     help="add an embedding-based semantic-dedup axis to grading")
    gen.add_argument("--embed-provider", default="ollama", choices=["ollama", "openai"],
                     help="embedder for --semantic (default: ollama)")
    gen.add_argument("--embed-model", default="", help="embedding model name")
    gen.add_argument("--dedup-semantic", action="store_true",
                     help="reject semantically-similar records during generation "
                          "(clean-by-construction; uses --embed-provider)")
    gen.add_argument("--dedup-threshold", type=float, default=0.9,
                     help="cosine ≥ this ⇒ reject as a semantic duplicate (default: 0.9)")
    gen.add_argument("--report-json", help="write the grade report as JSON")
    gen.add_argument("--html", help="write the grade report as HTML")
    gen.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    gen.set_defaults(func=cmd_text_gen)

    # grade (shared across all products)
    gr = sub.add_parser("grade", help="grade any dataset (jsonl/json/csv) for quality")
    gr.add_argument("path", help="dataset to grade")
    gr.add_argument("--against", help="held-out eval set for the contamination check")
    gr.add_argument("--field", action="append", default=[],
                    help="field(s) to analyze (repeatable; default: auto-detect)")
    gr.add_argument("--ngram", type=int, default=8, help="contamination n-gram size")
    gr.add_argument("--min-grade", choices=_GRADE_ORDER,
                    help="exit non-zero if the grade is below this (CI gate)")
    gr.add_argument("--semantic", action="store_true",
                    help="add an embedding-based semantic-dedup axis")
    gr.add_argument("--embed-provider", default="ollama", choices=["ollama", "openai"],
                    help="embedder for --semantic (default: ollama)")
    gr.add_argument("--embed-model", default="", help="embedding model name")
    gr.add_argument("--json", help="write the report as JSON")
    gr.add_argument("--html", help="write the report as HTML")
    gr.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    gr.set_defaults(func=cmd_grade)

    # list
    ls = sub.add_parser("list", help="list products and built-in seeds")
    ls.set_defaults(func=cmd_list)

    # roadmap stubs
    tb = sub.add_parser("tabular", help="Product B: schema-aware fixtures [roadmap]")
    tb.set_defaults(func=cmd_coming_soon, product="tabular",
                    blurb="schema-aware fixtures with referential integrity")
    pv = sub.add_parser("privacy", help="Product C: privacy-safe twins [roadmap]")
    pv.set_defaults(func=cmd_coming_soon, product="privacy",
                    blurb="privacy-safe synthetic twins of real datasets")

    return p


def main(argv: List[str] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        sys.exit(0)
    # bare `synthkit text` with no subcommand
    if args.cmd == "text" and not getattr(args, "func", None):
        parser.parse_args(["text", "--help"])
    try:
        sys.exit(args.func(args))
    except SynthkitError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
