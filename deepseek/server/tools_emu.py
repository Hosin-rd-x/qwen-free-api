"""
Tool-calling emulation (agent mode).

DeepSeek's web endpoint has no native function-calling channel, so we emulate
the OpenAI `tools` protocol with prompt engineering:

  1. When the request carries `tools`, a machine-readable contract is appended
     to the prompt: the model must answer EITHER with normal text OR with a
     single JSON object: {"name": "...", "arguments": {...}}.
  2. When the reply parses as such a JSON tool call, it is translated into a
     real OpenAI `tool_calls` response (finish_reason="tool_calls"), including
     the streaming shape, so Hermes / OpenAI function-calling clients work
     without any change.
  3. `tool` role messages (execution results) are serialised back into the
     prompt as "Function result for <name>: <content>".

Graceful degradation: if the reply is not a parseable tool call, it is passed
through as normal text — the agent sees plain content instead of a broken call.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import List, Optional

from .schemas import ChatMessage

_CALL_ID_PREFIX = "call_"


def _fmt_arguments(value) -> str:
    """Normalise the arguments field to a JSON string."""
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def tools_contract(tools: List[dict], tool_choice=None) -> str:
    """Build the system-side contract appended to the prompt when tools exist."""
    if not tools:
        return ""
    names = []
    lines = []
    for t in tools:
        fn = (t or {}).get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        names.append(name)
        desc = fn.get("description", "")
        params = fn.get("parameters")
        lines.append(f'- "{name}": {desc}')
        if params:
            lines.append(
                "  parameters: " + json.dumps(params, ensure_ascii=False)
            )
    if not names:
        return ""

    forced = ""
    if isinstance(tool_choice, dict) and tool_choice.get("function", {}).get("name"):
        forced = (
            f'\nYou MUST call the tool "{tool_choice["function"]["name"]}" '
            f"on this turn."
        )

    return (
        "\n\n---\n"
        "[TOOL PROTOCOL] You have access to these tools:\n"
        + "\n".join(lines)
        + "\n\nIf answering the user requires a tool, reply with ONLY one JSON "
        "object (no prose, no markdown fences) in exactly this shape:\n"
        '{"name": "<tool name>", "arguments": {<json arguments>}}\n'
        "If no tool is needed, answer in plain text as usual. "
        "Never mix JSON tool calls with explanatory text in the same reply."
        + forced
    )


def _balanced_objects(text: str, limit: int = 5):
    """Yield balanced {...} substrings (string-aware), outermost-first."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start:i + 1]
                    start = None
                    limit -= 1
                    if limit <= 0:
                        return


def extract_tool_call(text: str) -> Optional[dict]:
    """Return an OpenAI tool_call dict if `text` is a model-issued tool call.

    Accepted shapes: a bare JSON object, a ```json fenced block, or a balanced
    {...} object embedded in prose. Requires a string "name" key plus an
    "arguments"/"args" key (exactly the shape the tool contract demands) so
    ordinary JSON answers with a "name" field are not mistaken for tool calls.
    Everything else returns None (plain text pass-through).
    """
    if not text or "{" not in text:
        return None

    candidates: list = []
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*)\s*```", stripped, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    if stripped.startswith("{"):
        candidates.append(stripped)
    candidates.extend(_balanced_objects(text))

    for raw in candidates:
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        has_args = "arguments" in obj or "args" in obj
        if isinstance(name, str) and name and has_args:
            args = obj.get("arguments", obj.get("args"))
            return {
                "id": _CALL_ID_PREFIX + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _fmt_arguments(args),
                },
            }
    return None


def format_tool_result(msg: ChatMessage) -> str:
    """Serialise a `tool` role message back into the conversation prompt."""
    name = getattr(msg, "name", None) or ""
    label = f"Function result for {name}" if name else "Function result"
    return f"{label}: {_text_of(msg.content)}"


def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def build_prompt(messages: List[ChatMessage], tools: Optional[List[dict]] = None,
                 tool_choice=None) -> str:
    """Flatten messages into a DeepSeek prompt, tool-aware.

    Accepts ChatMessage objects or raw dicts (OpenAI shape). Extends the base
    flattening with:
      - `tool` role results rendered as function results,
      - assistant messages containing `tool_calls` shown as calls already made,
      - the tool contract appended when tools are present.
    """
    norm = []
    for m in messages:
        if isinstance(m, dict):
            m = ChatMessage(**{k: v for k, v in m.items()
                               if k in ChatMessage.model_fields})
        norm.append(m)
    messages = norm

    body_lines = []
    for m in messages:
        calls = getattr(m, "tool_calls", None)
        if m.role == "tool":
            body_lines.append(format_tool_result(m))
            continue
        text = _text_of(m.content)
        if calls:
            rendered = []
            for c in calls:
                fn = (c or {}).get("function") or {}
                rendered.append(
                    json.dumps(
                        {"name": fn.get("name", ""),
                         "arguments": fn.get("arguments", "{}")},
                        ensure_ascii=False,
                    )
                )
            body_lines.append(f"Assistant (called tool): {'; '.join(rendered)}")
            continue
        label = {"system": "System", "user": "User",
                 "assistant": "Assistant"}.get(m.role, m.role.capitalize())
        body_lines.append(f"{label}: {text}")

    if len(messages) == 1 and messages[0].role == "user" and not tools:
        return _text_of(messages[0].content)

    prompt = "\n\n".join(body_lines)
    if body_lines and not body_lines[-1].startswith("Assistant"):
        prompt += "\n\nAssistant:"
    contract = tools_contract(tools or [], tool_choice)
    if contract:
        prompt += contract
    return prompt


# --- OpenAI response shaping --------------------------------------------------

def tool_calls_message(call: dict) -> dict:
    """assistant message carrying one tool_call (OpenAI shape)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [call],
    }


def tool_call_delta_chunks(base: dict, call: dict) -> List[str]:
    """SSE frames announcing a tool call, OpenAI streaming shape.

    Delivered as: role frame -> full arguments delta -> finish frame. Agents
    accumulate deltas exactly like native OpenAI streams, so nothing changes
    on their side.
    """
    import time as _time

    frames = []

    def frame(delta: dict, finish=None):
        obj = dict(base)
        obj["created"] = int(_time.time())
        obj["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
        frames.append(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n")

    frame({"role": "assistant", "content": None})
    frame({"tool_calls": [
        {"index": 0, "id": call["id"], "type": "function",
         "function": {"name": call["function"]["name"], "arguments": ""}},
    ]})
    frame({"tool_calls": [
        {"index": 0, "function": {"arguments": call["function"]["arguments"]}},
    ]})
    frame({}, finish="tool_calls")
    return frames


def content_replay_chunks(base: dict, text: str) -> List[str]:
    """SSE frames replaying a buffered plain-text reply (tools were requested
    but the model answered with normal text). Split into a few pieces so
    clients still render a progressive feel."""
    import time as _time

    frames = []

    def frame(delta: dict, finish=None):
        obj = dict(base)
        obj["created"] = int(_time.time())
        obj["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
        frames.append(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n")

    frame({"role": "assistant", "content": ""})
    if text:
        piece = max(1, len(text) // 8)
        for i in range(0, len(text), piece):
            frame({"content": text[i:i + piece]})
    frame({}, finish="stop")
    return frames
