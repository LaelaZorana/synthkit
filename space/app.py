"""synthkit — Hugging Face Space (Gradio).

A small live demo of the synthkit quality grader and offline generator. This
Space runs fully offline: template-based generation (no API key) and the lexical
grading axes. The CLI adds LLM-backed generation, fine-tuning output formats, and
an embedding-based semantic-dedup axis. Source: https://github.com/LaelaZorana/synthkit
"""
from __future__ import annotations

import html
import json

import gradio as gr

from synthkit import __version__
from synthkit.grading import grade_dataset
from synthkit.report import to_html, to_json
from synthkit.text.generate import generate
from synthkit.text.seeds import BUILTIN_SEEDS

GH = "https://github.com/LaelaZorana/synthkit"


_MAX_INPUT_BYTES = 4_000_000   # ~4 MB ceiling on pasted/uploaded data (public demo)
_MAX_RECORDS = 5000            # cap rows graded per request


def _records_from(text: str, file) -> list:
    raw = ""
    if file is not None:
        path = file if isinstance(file, str) else getattr(file, "name", None)
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read(_MAX_INPUT_BYTES + 1)
    elif text and text.strip():
        raw = text
    if len(raw) > _MAX_INPUT_BYTES:
        raise ValueError(f"input too large (limit {_MAX_INPUT_BYTES // 1_000_000} MB)")
    raw = raw.strip()
    if not raw:
        raise ValueError("Paste some JSONL/JSON or upload a file first.")
    if raw[0] == "[":
        recs = json.loads(raw)
    else:
        recs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(recs, list) or not recs:
        raise ValueError("No records found.")
    if len(recs) > _MAX_RECORDS:
        raise ValueError(f"too many records ({len(recs)}); this demo caps at {_MAX_RECORDS}")
    return recs


def _iframe(report_html: str) -> str:
    return (f'<iframe srcdoc="{html.escape(report_html, quote=True)}" '
            'style="width:100%;height:760px;border:0;border-radius:14px;background:#0b1020">'
            '</iframe>')


def _err(exc: Exception) -> str:
    return f'<p style="color:#dc2626;font:14px sans-serif">⚠ {html.escape(str(exc))}</p>'


def grade_ui(dataset_text, dataset_file, eval_text):
    try:
        records = _records_from(dataset_text, dataset_file)
        against = None
        if eval_text and eval_text.strip():
            against = [json.loads(ln) for ln in eval_text.splitlines() if ln.strip()][:_MAX_RECORDS]
        report = grade_dataset(records, against=against)
        return _iframe(to_html(report, "uploaded dataset")), to_json(report, "uploaded dataset")
    except Exception as exc:  # noqa: BLE001 - surface any parse/grade error to the UI
        return _err(exc), ""


def generate_ui(seed_choice, seed_text, n):
    try:
        spec = json.loads(seed_text) if seed_text and seed_text.strip() else BUILTIN_SEEDS[seed_choice]
        data = generate(spec, max(1, min(int(n), 300)), seed=17)   # hard-cap N on the public demo
        report = grade_dataset(data)
        jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in data)
        return _iframe(to_html(report, "generated dataset")), jsonl
    except Exception as exc:  # noqa: BLE001
        return _err(exc), ""


with gr.Blocks(title="synthkit", theme=gr.themes.Soft(), css="footer{visibility:hidden}") as demo:
    gr.Markdown(
        "# 🧪 synthkit\n"
        "**Generate synthetic data, and grade it before you train on it.** "
        "Scored on validity · uniqueness · diversity · contamination, with an A+→F headline.\n\n"
        f"This Space runs offline (template generation + lexical grading). The full "
        f"[CLI]({GH}) adds LLM-backed instruction data, fine-tuning formats "
        f"(alpaca/sharegpt/openai), and an embedding semantic-dedup axis. · v{__version__}")

    with gr.Tab("Grade a dataset"):
        gr.Markdown("Upload or paste a `.jsonl` (one JSON record per line). Optionally "
                    "paste a held-out eval set to check **contamination** (train/eval leakage).")
        with gr.Row():
            with gr.Column():
                g_file = gr.File(label="Dataset (.jsonl / .json)",
                                 file_types=[".jsonl", ".json", ".txt"])
                g_text = gr.Textbox(label="…or paste records", lines=8,
                                    placeholder='{"prompt": "Explain gradient descent"}\n'
                                                '{"prompt": "What is a hash map?"}')
                g_eval = gr.Textbox(label="Held-out eval set for contamination (optional)",
                                    lines=3, placeholder='{"prompt": "Explain gradient descent"}')
                g_btn = gr.Button("Grade", variant="primary")
            with gr.Column():
                g_html = gr.HTML()
        g_json = gr.Code(label="report.json", language="json")
        g_btn.click(grade_ui, [g_text, g_file, g_eval], [g_html, g_json])

    with gr.Tab("Generate (offline)"):
        gr.Markdown("Generate eval prompts from a built-in seed, or paste your own seed "
                    "spec (templates + slots). The result is graded automatically.")
        with gr.Row():
            with gr.Column():
                s_choice = gr.Dropdown(list(BUILTIN_SEEDS), value="eval", label="Built-in seed")
                s_n = gr.Slider(10, 300, value=120, step=10, label="How many records")
                s_text = gr.Textbox(
                    label="…or paste a seed spec (JSON)", lines=8,
                    placeholder='{"kind":"eval","templates":["{a} about {b}"],'
                                '"slots":{"a":["Explain","Summarize"],"b":["TLS","DNS","OAuth"]},'
                                '"response":{"mode":"none"}}')
                s_btn = gr.Button("Generate + grade", variant="primary")
            with gr.Column():
                s_html = gr.HTML()
        s_jsonl = gr.Code(label="dataset.jsonl")
        s_btn.click(generate_ui, [s_choice, s_text, s_n], [s_html, s_jsonl])

    gr.Markdown(f"[⭐ Source on GitHub]({GH}) · MIT-licensed · zero-dependency core")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2, max_size=24).launch()
