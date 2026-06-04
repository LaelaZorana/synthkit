"""Built-in seed specs used by `synthkit text gen --demo`.

A seed spec is plain data (works as JSON or YAML on disk too):

  kind        "eval" (prompt-only) or "instruction" (instruction→output)
  templates   sentence templates with {slot} placeholders
  slots       lists of fillers, one per placeholder name
  response    how to fill the output for instruction data:
                {"mode": "none"}                  leave blank (eval sets)
                {"mode": "rule", "template": ...} fill a string template (offline)
                {"mode": "provider"}              call an LLM provider
  constraints optional, e.g. {"min_words": 4}
"""
from __future__ import annotations

_LANGS = ["Python", "JavaScript", "Rust", "Go", "TypeScript", "Java", "C++", "Ruby"]
_TASKS = [
    "reverse a string",
    "check whether a number is prime",
    "merge two sorted lists",
    "find the longest common subsequence of two strings",
    "parse an ISO-8601 date",
    "debounce a function",
    "flatten a deeply nested list",
    "compute a moving average over a stream",
    "detect a cycle in a linked list",
    "implement binary search",
]

DEMO_EVAL = {
    "kind": "eval",
    "domain": "coding",
    "templates": [
        "Write a {language} function that {task}.",
        "How would you {task} in {language}? Walk through your reasoning.",
        "Review this {language} snippet that is meant to {task} and point out the bugs.",
        "Explain to a beginner how to {task} using {language}.",
        "What's the most efficient way to {task} in {language}, and why?",
        "Refactor a {language} program that {task} to be more readable.",
    ],
    "slots": {"language": _LANGS, "task": _TASKS},
    "constraints": {"min_words": 4},
    "response": {"mode": "none"},
}

DEMO_INSTRUCTION = {
    "kind": "instruction",
    "domain": "coding",
    "system": "You are a precise, helpful coding assistant.",
    "templates": [
        "Write a {language} function that {task}.",
        "Show me how to {task} in {language}.",
        "I need {language} code to {task}. Include a short explanation.",
    ],
    "slots": {"language": _LANGS, "task": _TASKS},
    "constraints": {"min_words": 4},
    "response": {
        "mode": "rule",
        "template": "Here's an approach in {language} to {task}: start by clarifying "
                    "the inputs and edge cases, then implement the core logic and test it.",
    },
}

BUILTIN_SEEDS = {"eval": DEMO_EVAL, "instruction": DEMO_INSTRUCTION}
