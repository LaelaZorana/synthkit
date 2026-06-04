"""synthkit — generate synthetic data and grade it for quality.

Three products on one core:
  • text     (A) — instruction / eval datasets for training & evaluating LLMs   [live]
  • tabular  (B) — schema-aware fixtures with referential integrity            [roadmap]
  • privacy  (C) — privacy-safe synthetic twins of real datasets               [roadmap]

The generator layer differs per product; the grading engine, providers, and
report writers are shared.
"""
from __future__ import annotations

__version__ = "0.4.0"
