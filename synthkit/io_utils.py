"""Read and write datasets as JSONL, JSON, or CSV — standard library only."""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List


def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def read_records(path: str) -> List[Dict[str, Any]]:
    """Load records from .jsonl / .json / .csv / .tsv."""
    ext = _ext(path)
    with open(path, "r", encoding="utf-8") as fh:
        if ext in (".jsonl", ".ndjson"):
            return [json.loads(line) for line in fh if line.strip()]
        if ext == ".json":
            data = json.load(fh)
            if isinstance(data, dict):
                # common wrappers: {"data": [...]} / {"records": [...]}
                for key in ("data", "records", "rows", "examples"):
                    if isinstance(data.get(key), list):
                        return data[key]
                return [data]
            return list(data)
        if ext in (".csv", ".tsv"):
            delim = "\t" if ext == ".tsv" else ","
            return list(csv.DictReader(fh, delimiter=delim))
    raise SystemExit(f"error: unsupported input format {ext or path!r}")


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_text(path: str, text: str) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def load_spec(path: str) -> Dict[str, Any]:
    """Load a seed spec from JSON (always) or YAML (if pyyaml is installed)."""
    ext = _ext(path)
    with open(path, "r", encoding="utf-8") as fh:
        if ext in (".yaml", ".yml"):
            try:
                import yaml  # optional dependency
            except ImportError as exc:  # pragma: no cover
                raise SystemExit(
                    "error: reading YAML seeds needs pyyaml — "
                    "`pip install pyyaml`, or use a .json seed."
                ) from exc
            return yaml.safe_load(fh)
        return json.load(fh)


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
