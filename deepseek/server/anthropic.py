"""Anthropic-compatible endpoint (`POST /v1/messages`).

Lets Anthropic-native agents (Claude Code, Cline-Anthropic, ...) point at
this bridge with ANTHROPIC_BASE_URL — the same "easy agent connection"
glm-free-api offers. Internally everything is translated to the OpenAI-ish
message list and served by the same DeepSeek client + agent shim, so tool
calling works here too: Anthropic `tools` in, `tool_use` blocks out
(stop_reason="tool_use").

Supported subset (deliberately small — local bridge, no extras):
  system (string or blocks), messages with text / tool_use / tool_result
  blocks, tools (function definitions), max_tokens, stream, metadata ignored,
  temperature/top_p accepted-but-forwarded-nowhere (DeepSeek web has no knobs).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ── request translation ─────────────────────────────────────────────────────

def _blocks_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(str(b.get("text", "")))
    return "\n".join(parts)


def anthropic_to_openai(body: dict) -> Tuple[str, List[dict], List[dict], bool, bool, bool]:
    """-> (system, openai_messages, tools, stream, thinking, search)"""
    system = _blocks_text(body.get("system"))
    tools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {"type": "object",
                                                        "properties": {}},
            },
        })

    th = body.get("thinking")
    thinking = isinstance(th, dict) and th.get("type") == "enabled"

    messages: List[dict] = []
    for m in body.get("messages") or []:
        role = m.get("role", "user")
        content = m.get("content")

        if isinstance(content, str):
            if role == "assistant":
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": content})
            continue

        # Block-list content: text / tool_use / tool_result.
        text_bits: List[str] = []
        pending_tool_results: List[dict] = []
        for b in content or []:
            if not isinstance(b, dict):
                continue
            btype = b.get("type")
            if btype == "text":
                text_bits.append(str(b.get("text", "")))
            elif btype == "tool_result":
                pending_tool_results.append({
                    "role": "tool",
                    "tool_call_id": b.get("tool_use_id") or "",
                    "content": _blocks_text(b.get("content")),
                })
            elif btype == "tool_use":
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(text_bits).strip(),
                    "tool_calls": [{
                        "id": b.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                        "type": "function",
                        "function": {
                            "name": b.get("name") or "",
                            "arguments": json.dumps(b.get("input") or {},
                                                    ensure_ascii=False),
                        },
                    }],
                })
                text_bits = []
        if text_bits:
            messages.append({
                "role": "assistant" if role == "assistant" else "user",
                "content": "\n".join(text_bits),
            })
        elif role != "assistant" and not pending_tool_results:
            messages.append({"role": "user", "content": ""})
        messages.extend(pending_tool_results)

    stream = bool(body.get("stream"))
    return system, messages, tools, stream, thinking, bool(body.get("search"))


# ── response shapes ─────────────────────────────────────────────────────────

def _now() -> int:
    return int(time.time())


def message_response(model: str, text: str, tool_calls: List[dict],
                     input_tokens: int = 0) -> dict:
    """Non-streaming Anthropic message (text + optional tool_use blocks)."""
    content: List[dict] = []
    if text.strip():
        content.append({"type": "text", "text": text})
    for tc in tool_calls:
        try:
            input_obj = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, ValueError, KeyError):
            input_obj = {}
        content.append({"type": "tool_use", "id": tc["id"],
                        "name": tc["function"]["name"], "input": input_obj})
    stop_reason = "tool_use" if tool_calls else "end_turn"
    out_tokens = max(1, len(text) // 4)
    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content or [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": max(0, input_tokens),
                  "output_tokens": out_tokens},
    }


def error_payload(message: str, err_type: str = "api_error") -> dict:
    return {"type": "error", "error": {"type": err_type, "message": message}}


# ── streaming ───────────────────────────────────────────────────────────────

def stream_message_events(model: str, events: Iterable[Tuple[str, str]],
                          agent_shim: Any) -> Iterable[str]:
    """Yield Anthropic SSE blocks for a streamed completion.

    `events` yields ("reasoning"|"content", text) pairs from the DeepSeek
    stream. Reasoning is surfaced as a distinct thinking block (Anthropic
    clients tolerate it; plain text clients simply skip it). When the agent
    shim captures tool-call spans they are emitted as tool_use blocks with
    stop_reason="tool_use".
    """

    def ev(name: str, data: dict) -> str:
        return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    msg_id = "msg_" + uuid.uuid4().hex[:24]
    yield ev("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield ev("ping", {"type": "ping"})

    block_index = -1
    opened_text = False
    opened_think = False

    def open_block(btype: str) -> int:
        nonlocal block_index, opened_text, opened_think
        block_index += 1
        if btype == "text":
            opened_text = True
            payload = {"type": "text", "text": ""}
        else:
            opened_think = True
            payload = {"type": "thinking", "thinking": ""}
        return block_index

    stop_reason = "end_turn"
    text_emitted = False

    for kind, chunk in events:
        if not chunk:
            continue
        if kind == "reasoning":
            idx = open_block("thinking") if not opened_think else block_index
            yield ev("content_block_delta", {
                "type": "content_block_delta", "index": idx,
                "delta": {"type": "thinking_delta", "thinking": chunk},
            })
            continue

        # Content text: run through the agent shim when active.
        if agent_shim is not None:
            safe = agent_shim.feed(chunk)
        else:
            safe = chunk
        if safe:
            if opened_think:
                yield ev("content_block_stop", {"type": "content_block_stop",
                                                "index": block_index})
                opened_think = False
            if not opened_text:
                idx = open_block("text")
                yield ev("content_block_start", {
                    "type": "content_block_start", "index": idx,
                    "content_block": {"type": "text", "text": ""},
                })
            text_emitted = True
            yield ev("content_block_delta", {
                "type": "content_block_delta", "index": block_index,
                "delta": {"type": "text_delta", "text": safe},
            })

    if agent_shim is not None:
        tail = agent_shim.finish()
        if tail:
            if not opened_text:
                idx = open_block("text")
                yield ev("content_block_start", {
                    "type": "content_block_start", "index": idx,
                    "content_block": {"type": "text", "text": ""},
                })
            text_emitted = True
            yield ev("content_block_delta", {
                "type": "content_block_delta", "index": block_index,
                "delta": {"type": "text_delta", "text": tail},
            })
        calls = agent_shim.drain_tool_calls()
        if calls:
            stop_reason = "tool_use"
            if opened_text or opened_think:
                yield ev("content_block_stop", {"type": "content_block_stop",
                                                "index": block_index})
                opened_text = opened_think = False
            for tc in calls:
                block_index += 1
                idx = block_index
                yield ev("content_block_start", {
                    "type": "content_block_start", "index": idx,
                    "content_block": {"type": "tool_use", "id": tc["id"],
                                      "name": tc["function"]["name"], "input": {}},
                })
                yield ev("content_block_delta", {
                    "type": "content_block_delta", "index": idx,
                    "delta": {"type": "input_json_delta",
                              "partial_json": tc["function"]["arguments"]},
                })
                yield ev("content_block_stop", {"type": "content_block_stop",
                                                "index": idx})

    if opened_text or opened_think:
        yield ev("content_block_stop", {"type": "content_block_stop",
                                        "index": block_index})
    if not text_emitted and stop_reason == "end_turn" and block_index < 0:
        # Guarantee at least one (empty) text block so clients have content.
        yield ev("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield ev("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": 1},
    })
    yield ev("message_stop", {"type": "message_stop"})
