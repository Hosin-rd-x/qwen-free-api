"""Agent mode: protocol-level tool calling for DeepSeek (Hermes-ready).

chat.deepseek.com's web API has NO native function-calling channel — only a
single prompt string in, text out. So, exactly like the modern shim in
Godde3s/glm-free-api (internal/zbridge/agent.go), we:

  1. TRANSFORM the OpenAI `messages` + `tools` into one XML-sectioned prompt
     (<system>, <tools>, <history>, <current_task>, <output_rules>) that
     teaches the model an exact textual tool-call protocol:

         <<<TOOL_CALL>>>{"name":"<tool>","arguments":{...}}<<<END_TOOL_CALL>>>

  2. PARSE the model's reply — tolerantly: markers wrapped in code fences,
     flat payloads ({"tool": name, ...params}) and alternate key spellings
     are all accepted — and convert spans back into native OpenAI
     `tool_calls` (finish_reason="tool_calls").

  3. REPLAY prior tool exchanges (assistant tool_calls + role:"tool"
     results) inside <history> in the same protocol, so multi-round agent
     loops (Hermes, Cursor, Cline, ...) work turn after turn.

The stream interceptor holds back text that might be a forming marker so
clients only ever see clean content deltas and clean tool_calls deltas.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ── protocol constants ──────────────────────────────────────────────────────

START_MARKER = "<<<TOOL_CALL>>>"
END_MARKER = "<<<END_TOOL_CALL>>>"
CALL_SCHEMA = '{"name":"<tool_name>","arguments":{<parameter JSON>}}'

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. Follow these rules strictly:\n"
    "(A) TOOL CALL: when a tool is needed, output EXACTLY this block and nothing else:\n"
    "    " + START_MARKER + CALL_SCHEMA + END_MARKER + "\n"
    "    The JSON object has EXACTLY two keys: \"name\" (the tool to call, spelled "
    "exactly as in <tools>) and \"arguments\" (an object with ONLY that tool's parameters).\n"
    "(B) FINAL ANSWER: plain text, only when no tool applies.\n"
    "- Never mix a tool call and a final answer in one reply.\n"
    "- Never print code fences (```bash, ```json). Only the runtime executes tools.\n"
    "- Never wrap tool-call markers in code fences.\n"
    "- Never invent results. Stop at " + END_MARKER + " and wait for tool output.\n"
    "- Never call a tool not listed in <tools>."
)

_OUTPUT_RULES = (
    "<output_rules>\n"
    "Reply with EXACTLY ONE of:\n"
    "1. " + START_MARKER + '{"name":"<tool_name>","arguments":{...}}' + END_MARKER +
    " (no fences, no other text)\n"
    "2. Plain text final answer (only if no tool applies to this step)\n"
    "The tool-call JSON uses EXACTLY the keys \"name\" and \"arguments\" — never a "
    "\"tool\" key, never bare top-level parameters.\n"
    "</output_rules>"
)


def _text_of(content: Any) -> str:
    """Extract plain text from an OpenAI content value (string or parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
        return "\n".join(parts)
    return str(content)


# ── tools rendering ─────────────────────────────────────────────────────────

def _tool_fn(tool: dict) -> dict:
    """Accept both the nested {type:function,function:{...}} and flat {...} forms."""
    fn = tool.get("function") if isinstance(tool, dict) else None
    return fn if isinstance(fn, dict) else (tool or {})


def render_agent_tools(tools: Optional[List[dict]]) -> str:
    if not tools:
        return "(no tools provided)"
    blocks = []
    for i, tool in enumerate(tools, 1):
        fn = _tool_fn(tool)
        name = fn.get("name") or "(unnamed)"
        lines = [f"[TOOL {i}] {name}"]
        if fn.get("description"):
            lines.append(f"desc: {fn['description']}")
        params = fn.get("parameters")
        if isinstance(params, dict) and params.get("properties"):
            lines.append("params JSON schema: " +
                         json.dumps(params, ensure_ascii=False, separators=(",", ":")))
        blocks.append("\n".join(lines))
    return "[TOOL CONTRACT]\n" + "\n\n".join(blocks) + \
        "\nTo call a tool, use the exact tool-call block from the rules below."


# ── message transform (modern shim) ────────────────────────────────────────

def _render_tool_call_block(name: str, arguments: Any) -> str:
    """Canonical wire form of one assistant tool call (what the model emits)."""
    if isinstance(arguments, str):
        args_raw = arguments
    else:
        args_raw = json.dumps(arguments or {}, ensure_ascii=False)
    return f"{START_MARKER}{json.dumps({'name': name, 'arguments': args_raw}, ensure_ascii=False)}{END_MARKER}"


def _render_assistant_turn(msg: dict) -> str:
    """Assistant text plus its tool_calls re-rendered as protocol blocks."""
    out = _text_of(msg.get("content")).strip()
    for tc in msg.get("tool_calls") or []:
        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
        if fn.get("name"):
            out = (out + "\n" if out else "") + \
                _render_tool_call_block(fn["name"], fn.get("arguments") or {})
    return out


def _render_tool_result(msg: dict) -> str:
    tcid = msg.get("tool_call_id") or ""
    attr = f' tool_call_id="{tcid}"' if tcid else ""
    return f"<tool_result{attr}>\n{_text_of(msg.get('content'))}\n</tool_result>"


def agent_transform_messages(messages: List[dict], tools: Optional[List[dict]]) -> List[dict]:
    """Flatten an OpenAI conversation into ONE user message the shim protocol
    can serve. Returns the new messages list (a single user message).

    Non-destructive: the input list is not modified.
    """
    if not tools:
        return messages

    system_txt = "\n\n".join(
        _text_of(m.get("content")) for m in messages
        if m.get("role") == "system" and _text_of(m.get("content")).strip()
    ).strip()

    # Conversation body: everything except leading system turns.
    body = [m for m in messages if m.get("role") != "system"]
    if not body:
        body = [{"role": "user", "content": ""}]

    # The last user message is the current task; everything before goes to history.
    last_user_idx = None
    for i in range(len(body) - 1, -1, -1):
        if body[i].get("role") == "user":
            last_user_idx = i
            break

    history: List[str] = []
    for i, m in enumerate(body):
        if i == last_user_idx:
            continue
        role = m.get("role")
        if role == "user":
            txt = _text_of(m.get("content")).strip()
            if txt:
                history.append(f"<u>{txt}</u>")
        elif role == "assistant":
            turn = _render_assistant_turn(m)
            if turn.strip():
                history.append(f"<a>{turn}</a>")
        elif role == "tool":
            history.append(_render_tool_result(m))

    current_task = _text_of(body[last_user_idx].get("content")).strip() \
        if last_user_idx is not None else "(no user message)"

    prompt_parts = ["<system>\n" + (system_txt or AGENT_SYSTEM_PROMPT) + "\n</system>"]
    if system_txt:
        prompt_parts.append("<agent_rules>\n" + AGENT_SYSTEM_PROMPT + "\n</agent_rules>")
    prompt_parts.append("<tools>\n" + render_agent_tools(tools) + "\n</tools>")
    if history:
        prompt_parts.append("<history>\n" + "\n".join(history) + "\n</history>")
    prompt_parts.append("<current_task>\n" + current_task + "\n</current_task>")
    prompt_parts.append(_OUTPUT_RULES)

    return [{"role": "user", "content": "\n\n".join(prompt_parts)}]


# ── tolerant span parsing ───────────────────────────────────────────────────

# Markers — canonical triple-bracket form with optional internal whitespace.
# (A single-bracket alternative would also match INSIDE a forming canonical
# marker — e.g. `<TOOL_CALL>` within `<<<TOOL_CALL>>` — corrupting the
# stream interceptor, so only the canonical form is accepted.)
_START_RE = re.compile(r"<<<\s*TOOL_CALL\s*>>>")
_END_RE = re.compile(r"<<<\s*END_TOOL_CALL\s*>>>")


def _coerce_arguments(raw: Any) -> str:
    """Normalize an arguments value into a JSON string (OpenAI wire format)."""
    if raw is None or raw == "":
        return "{}"
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return "{}"
        try:
            json.loads(s)
            return s
        except (json.JSONDecodeError, ValueError):
            return json.dumps({"_raw": s}, ensure_ascii=False)
    try:
        return json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _parse_payload(text: str) -> Optional[Tuple[str, str]]:
    """Parse the JSON between the markers, tolerating payload variants.

    Accepted shapes:
      {"name": "...", "arguments": {...}}        (canonical)
      {"name": "...", ...flat params}            (name + bare params)
      {"tool": "...", ...flat params}            (upstream JS-style flat call)
      {"tool": "...", "args"/"params"/"arguments": {...}}
    Returns (name, arguments_json_string) or None.
    """
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Tolerate trailing commas / single quotes the model may add.
        relaxed = text.replace("\n", " ")
        relaxed = re.sub(r",\s*([}\]])", r"\1", relaxed)
        try:
            obj = json.loads(relaxed)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(obj, dict):
        return None

    name = obj.get("name") or obj.get("tool") or obj.get("tool_name") or obj.get("function")
    if isinstance(name, dict):  # {"function":{"name":..,"arguments":..}}
        name = name.get("name")
    if not name or not isinstance(name, str):
        return None

    for key in ("arguments", "args", "params", "parameters", "payload", "input"):
        if key in obj:
            return name, _coerce_arguments(obj[key])

    # Flat style: every other top-level key is a parameter.
    params = {k: v for k, v in obj.items()
              if k not in ("name", "tool", "tool_name", "function")}
    return name, _coerce_arguments(params)


class AgentSpan:
    __slots__ = ("start", "end", "name", "arguments")

    def __init__(self, start: int, end: int, name: str, arguments: str):
        self.start, self.end = start, end
        self.name, self.arguments = name, arguments


def find_agent_spans(text: str) -> List[AgentSpan]:
    """Locate every complete, tolerant tool-call span in `text`."""
    spans: List[AgentSpan] = []
    pos = 0
    while True:
        m_start = _START_RE.search(text, pos)
        if not m_start:
            break
        m_end = _END_RE.search(text, m_start.end())
        if not m_end:
            break
        parsed = _parse_payload(text[m_start.end():m_end.start()])
        if parsed:
            spans.append(AgentSpan(m_start.start(), m_end.end(), parsed[0], parsed[1]))
            pos = m_end.end()
        else:
            # Unparseable payload: skip past this start marker, keep scanning.
            pos = m_start.end()
    return spans


def parse_agent_tool_calls(text: str) -> List[dict]:
    """OpenAI `tool_calls` array for every complete span (fresh call ids)."""
    calls = []
    for span in find_agent_spans(text):
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:8],
            "type": "function",
            "function": {"name": span.name, "arguments": span.arguments},
        })
    return calls


def strip_agent_tool_calls(text: str) -> str:
    """Remove every complete span (and its wrapping fence) from `text`."""
    out = text
    for span in reversed(find_agent_spans(text)):
        out = out[:span.start] + out[span.end:]
    # Drop now-orphaned code fences the model may have wrapped spans in.
    out = re.sub(r"```[a-zA-Z]*\s*```", "", out)
    return out.strip()


# ── streaming interceptor ───────────────────────────────────────────────────

class AgentInterceptor:
    """Streaming text -> (clean content deltas | tool_calls) converter.

    Feed each upstream text delta through feed(); it returns the text that is
    SAFE to emit as content now (markers are held back), or None. When a
    complete span has been captured, drain_tool_calls() returns it and the
    caller should emit OpenAI tool_calls deltas with finish_reason="tool_calls".

    finish() flushes any held-back text as plain content (the model wrote a
    partial/absent marker — better to show it than to swallow it).
    """

    # Longest marker-ish prefix we may hold back while waiting for more text.
    _MAX_HELD = 4096

    def __init__(self) -> None:
        self._buf = ""
        self._calls: List[dict] = []

    def feed(self, delta: str) -> Optional[str]:
        self._buf += delta
        out: List[str] = []
        while True:
            m = _START_RE.search(self._buf)
            if not m:
                # No start marker: emit everything except a suffix that could
                # be the beginning of one.
                keep = self._buf
                tail = self._possible_marker_prefix(keep)
                emit = keep[:len(keep) - len(tail)] if tail else keep
                if emit:
                    out.append(emit)
                    self._buf = keep[len(emit):]
                return "".join(out) if out else None

            # Emit text before the marker.
            pre = self._buf[:m.start()]
            if pre:
                out.append(pre)

            e = _END_RE.search(self._buf, m.end())
            if not e:
                # Still forming: hold back from marker start.
                if len(self._buf) - m.start() > self._MAX_HELD:
                    # Never completes — treat as literal text.
                    self._buf = ""
                    return "".join(out) if out else None
                self._buf = self._buf[m.start():]
                return "".join(out) if out else None

            parsed = _parse_payload(self._buf[m.end():e.start()])
            if parsed:
                self._calls.append({
                    "id": "call_" + uuid.uuid4().hex[:8],
                    "type": "function",
                    "function": {"name": parsed[0], "arguments": parsed[1]},
                })
                self._buf = self._buf[e.end():]
                continue  # keep scanning for further spans
            # Bad payload: drop markers, keep the payload text as content.
            inner = self._buf[m.end():e.start()]
            out.append(inner)
            self._buf = self._buf[e.end():]

    def has_calls(self) -> bool:
        return bool(self._calls)

    def drain_tool_calls(self) -> List[dict]:
        calls, self._calls = self._calls, []
        return calls

    def finish(self) -> str:
        """Flush held-back text as plain content (incomplete/no markers)."""
        rest, self._buf = self._buf, ""
        return rest

    @staticmethod
    def _possible_marker_prefix(text: str) -> str:
        """Longest suffix of `text` that is a strict prefix of a start marker
        (or a code fence opening) — held back until we know more."""
        candidates = ["<<<TOOL_CALL>>>", "```"]
        best = ""
        for cand in candidates:
            for k in range(1, len(cand)):
                if text.endswith(cand[:k]) and len(cand[:k]) > len(best):
                    best = cand[:k]
        return best
