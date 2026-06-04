"""Convert generated records into common fine-tuning dataset formats.

  raw       what the generator emits (alpaca-ish: instruction/input/output, or prompt)
  alpaca    {instruction, input, output}
  sharegpt  {conversations: [{from: human, value}, {from: gpt, value}]}
  openai    {messages: [{role: system?}, {role: user}, {role: assistant}]}

Eval (prompt-only) records keep their prompt as the human/user turn with an
empty completion, so the same dataset can drive evaluation or be completed later.
"""
from __future__ import annotations

from typing import Any, Dict, List

FORMATS = ("raw", "alpaca", "sharegpt", "openai")


def _parts(rec: Dict[str, Any]):
    system = rec.get("system", "")
    instruction = rec.get("instruction", rec.get("prompt", ""))
    user_input = rec.get("input", "")
    output = rec.get("output", rec.get("response", ""))
    user = instruction if not user_input else f"{instruction}\n\n{user_input}"
    return system, instruction, user_input, user, output


def to_format(records: List[Dict[str, Any]], fmt: str) -> List[Dict[str, Any]]:
    if fmt not in FORMATS:
        raise SystemExit(f"error: unknown --format {fmt!r} (choose from {', '.join(FORMATS)})")
    if fmt == "raw":
        return records

    out: List[Dict[str, Any]] = []
    for rec in records:
        system, instruction, user_input, user, output = _parts(rec)
        if fmt == "alpaca":
            out.append({"instruction": instruction, "input": user_input, "output": output})
        elif fmt == "sharegpt":
            convo = []
            if system:
                convo.append({"from": "system", "value": system})
            convo.append({"from": "human", "value": user})
            convo.append({"from": "gpt", "value": output})
            out.append({"conversations": convo})
        elif fmt == "openai":
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": output})
            out.append({"messages": msgs})
    return out
