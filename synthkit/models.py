"""Core data types shared across every synthkit product."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class SynthkitError(Exception):
    """User-facing error (bad seed spec, unreadable input, provider failure).

    Library code raises this instead of calling sys.exit/SystemExit, so that
    callers embedding the library (the Gradio app, the tests) can catch it.
    The CLI converts it into a clean non-zero exit.
    """


# Letter grades, best to worst — same scale as the rest of the portfolio.
GRADE_BANDS = [
    (97, "A+"), (93, "A"), (85, "B"), (75, "C"), (65, "D"), (0, "F"),
]


def to_grade(score: float) -> str:
    for cutoff, letter in GRADE_BANDS:
        if score >= cutoff:
            return letter
    return "F"


@dataclass
class DimensionScore:
    """One quality axis (validity, uniqueness, diversity, contamination)."""

    key: str
    title: str
    score: Optional[float]                                # 0–100, or None when N/A
    summary: str = ""
    findings: List[str] = field(default_factory=list)     # human-readable notes
    stats: Dict[str, Any] = field(default_factory=dict)   # raw numbers
    weight: float = 1.0

    @property
    def applicable(self) -> bool:
        return self.score is not None


@dataclass
class GradeReport:
    """The graded result for a dataset."""

    grade: str
    score: float                                          # 0–100 overall
    n_records: int
    dimensions: List[DimensionScore] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def dim(self, key: str) -> Optional[DimensionScore]:
        for d in self.dimensions:
            if d.key == key:
                return d
        return None
