"""Render a GradeReport: terminal, JSON, and a standalone HTML report."""
from __future__ import annotations

import html
import json
from typing import List

from synthkit import __version__
from synthkit.models import GradeReport, to_grade

# ---- ANSI helpers ------------------------------------------------------------
_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
}
GRADE_COLOR = {"A+": "green", "A": "green", "B": "cyan", "C": "yellow", "D": "yellow", "F": "red"}
GRADE_HEX = {"A+": "#16a34a", "A": "#16a34a", "B": "#0891b2", "C": "#ca8a04",
             "D": "#ea580c", "F": "#dc2626"}


def _c(text: str, color: str) -> str:
    return f"{_C.get(color, '')}{text}{_C['reset']}"


def _bar(score: float, width: int = 21) -> str:
    fill = int(round(score / 100 * width))
    return "█" * fill + "░" * (width - fill)


def _score_color(score: float) -> str:
    return GRADE_COLOR.get(to_grade(score), "yellow")


def _hex(score: float) -> str:
    return GRADE_HEX.get(to_grade(score), "#ca8a04")


def print_report(report: GradeReport, dataset: str = "", use_color: bool = True) -> None:
    def col(t, c):
        return _c(t, c) if use_color else t

    g = report.grade
    print()
    print(col("  ┌─ synthkit · data quality ──────────────────────────", "dim"))
    print("  │")
    print(f"  │  Quality grade   {col(g, GRADE_COLOR.get(g, 'yellow'))}    ({report.score}/100)")
    print(f"  │  Records         {report.n_records}")
    if dataset:
        print(f"  │  Dataset         {dataset}")
    print("  │")
    print(col("  └────────────────────────────────────────────────────", "dim"))

    print(f"\n  {col('DIMENSIONS', 'bold')}\n")
    for d in report.dimensions:
        if d.applicable:
            mark = col("●", _score_color(d.score))
            bar = col(_bar(d.score), _score_color(d.score))
            print(f"  {mark} {d.title:<14}{d.score:>5.0f}  {bar}  {col(d.summary, 'dim')}")
        else:
            print(f"  {col('○', 'dim')} {d.title:<14}{col('  n/a', 'dim')}  {col(d.summary, 'dim')}")

    notes = [(d.title, f) for d in report.dimensions for f in d.findings]
    if notes:
        print(f"\n  {col('NOTES', 'bold')}\n")
        for title, f in notes:
            print(f"  {col('▸ ' + title + ':', 'cyan')} {f}")

    print(f"\n  {col('Grade = weighted blend of the applicable axes · tune with --against / --field', 'dim')}")
    print()


def to_json(report: GradeReport, dataset: str = "") -> str:
    return json.dumps({
        "dataset": dataset,
        "grade": report.grade,
        "score": report.score,
        "records": report.n_records,
        "dimensions": [
            {"key": d.key, "title": d.title, "score": d.score,
             "summary": d.summary, "findings": d.findings, "stats": d.stats}
            for d in report.dimensions
        ],
        "meta": report.meta,
    }, indent=2)


def to_html(report: GradeReport, dataset: str = "") -> str:
    gcolor = GRADE_HEX.get(report.grade, "#ca8a04")

    rows: List[str] = []
    for d in report.dimensions:
        if d.applicable:
            bc = _hex(d.score)
            rows.append(f"""
        <div class="dim">
          <div class="dim-head">
            <span class="dt">{html.escape(d.title)}</span>
            <span class="ds" style="color:{bc}">{d.score:.0f}</span>
          </div>
          <div class="track"><div class="fill" style="width:{d.score:.0f}%;background:{bc}"></div></div>
          <div class="dsum">{html.escape(d.summary)}</div>
        </div>""")
        else:
            rows.append(f"""
        <div class="dim">
          <div class="dim-head">
            <span class="dt">{html.escape(d.title)}</span>
            <span class="ds na">n/a</span>
          </div>
          <div class="dsum">{html.escape(d.summary)}</div>
        </div>""")

    notes = [f'<li><b>{html.escape(d.title)}:</b> {html.escape(f)}</li>'
             for d in report.dimensions for f in d.findings]

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>synthkit report</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.55 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
          background: #0b1020; color: #e5e7eb; }}
  .wrap {{ max-width: 760px; margin: 0 auto; padding: 40px 24px 80px; }}
  h1 {{ font-size: 20px; letter-spacing: .3px; margin: 0 0 4px; }}
  .sub {{ color: #94a3b8; margin: 0 0 28px; font-size: 13px; }}
  .hero {{ display: flex; gap: 24px; align-items: center; background: #111934;
           border: 1px solid #1e293b; border-radius: 14px; padding: 24px; margin-bottom: 28px; }}
  .grade {{ font-size: 56px; font-weight: 800; line-height: 1; color: {gcolor}; }}
  .meta {{ flex: 1; }}
  .meta .big {{ font-size: 15px; margin-bottom: 6px; }}
  .meta .small {{ color: #94a3b8; font-size: 13px; }}
  h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8;
        margin: 32px 0 14px; }}
  .dim {{ background: #111934; border: 1px solid #1e293b; border-radius: 12px;
          padding: 14px 18px; margin-bottom: 10px; }}
  .dim-head {{ display: flex; justify-content: space-between; align-items: baseline; }}
  .dt {{ font-weight: 700; }}
  .ds {{ font-size: 22px; font-weight: 800; }}
  .ds.na {{ color: #64748b; font-size: 15px; font-weight: 600; }}
  .track {{ height: 8px; background: #0b1020; border-radius: 999px; margin: 10px 0 8px; overflow: hidden; }}
  .fill {{ height: 100%; border-radius: 999px; }}
  .dsum {{ color: #cbd5e1; font-size: 13px; }}
  ul.notes {{ list-style: none; padding: 0; margin: 0; }}
  ul.notes li {{ background: #0e1830; border: 1px solid #1e293b; border-left: 3px solid #38bdf8;
                 border-radius: 8px; padding: 9px 14px; margin-bottom: 8px; font-size: 13.5px; }}
  ul.notes b {{ color: #818cf8; }}
  .foot {{ margin-top: 36px; color: #64748b; font-size: 12px; }}
</style></head><body><div class="wrap">
  <h1>synthkit — synthetic data quality report</h1>
  <p class="sub">validity · uniqueness · diversity · contamination</p>
  <div class="hero">
    <div class="grade">{report.grade}</div>
    <div class="meta">
      <div class="big">Quality score <b>{report.score}/100</b></div>
      <div class="small">{report.n_records} records{(' · ' + html.escape(dataset)) if dataset else ''}</div>
    </div>
  </div>
  <h2>Dimensions</h2>
  {''.join(rows)}
  <h2>Notes</h2>
  {('<ul class="notes">' + ''.join(notes) + '</ul>') if notes else '<p style="color:#86efac">Nothing flagged.</p>'}
  <p class="foot">Generated by synthkit v{__version__} · grade is a weighted blend of the applicable axes.</p>
</div></body></html>"""
