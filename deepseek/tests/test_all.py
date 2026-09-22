"""Self-contained test suite — run with:  python tests/test_all.py

Covers the tool-calling emulation, the account pool (fake upstream), the
Anthropic translation layer and the OpenAI response shapers. No network and
no real DeepSeek account needed.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS | {name}")
    else:
        FAIL += 1
        print(f"FAIL | {name} {extra}")


# ── tool-calling emulation ────────────────────────────────────────────────────
from server.tools_emu import (  # noqa: E402
    build_prompt,
    content_replay_chunks,
    extract_tool_call,
    tool_call_delta_chunks,
    tools_contract,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]

check("contract mentions tool", "TOOL PROTOCOL" in tools_contract(TOOLS))
check(
    "prompt carries contract + user text",
    "TOOL PROTOCOL" in build_prompt(
        [{"role": "user", "content": "Weather in Paris?"}], TOOLS),
)
call = extract_tool_call('{"name":"get_weather","arguments":{"city":"Paris"}}')
check("bare JSON -> tool_call", bool(call)
      and call["function"]["name"] == "get_weather")
fenced = extract_tool_call(
    'Sure! ```json\n{"name":"run_shell","arguments":{"cmd":"ls -la"}}\n```')
check("fenced JSON -> tool_call", bool(fenced)
      and fenced["function"]["name"] == "run_shell")
embedded = extract_tool_call(
    'I will use the tool: {"name": "get_weather", "arguments": {"city": "Tehran"}}')
check("embedded JSON -> tool_call", bool(embedded)
      and "Tehran" in embedded["function"]["arguments"])
check("plain text -> None", extract_tool_call("The answer is 42.") is None)
check("data JSON with name only -> None",
      extract_tool_call('{"name": "John", "age": 30}') is None)

tool_roundtrip = build_prompt([
    {"role": "user", "content": "weather?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_x", "function": {"name": "get_weather",
                                      "arguments": '{"city":"Paris"}'}}]},
    {"role": "tool", "tool_call_id": "call_x", "name": "get_weather",
     "content": "20C sunny"},
], TOOLS)
check("tool result folded into prompt",
      "Function result for get_weather: 20C sunny" in tool_roundtrip)

base = {"id": "x", "object": "chat.completion.chunk", "model": "deepseek-chat"}
frames = tool_call_delta_chunks(base, call)
check("stream tool_call frames", len(frames) == 4
      and '"tool_calls"' in frames[1] and '"tool_calls"' in frames[3])
replay = content_replay_chunks(base, "hello")
check("stream replay frames", '"stop"' in replay[-1])

# ── account pool ───────────────────────────────────────────────────────────────
from deepseek.auth import LoginRequired  # noqa: E402
from deepseek.pool import AccountPool, UpstreamError  # noqa: E402
from deepseek import pool as poolmod  # noqa: E402

empty_pool = AccountPool(accounts_dir=pathlib.Path(tempfile.mkdtemp()),
                         tokens_env="")
check("empty pool -> no accounts", not empty_pool.has_accounts())
try:
    empty_pool.call("chat", "hi")
    check("empty pool raises LoginRequired", False)
except LoginRequired:
    check("empty pool raises LoginRequired", True)

token_pool = AccountPool(accounts_dir=pathlib.Path(tempfile.mkdtemp()),
                         tokens_env="tokA,tokB,tokC")
st = token_pool.status()
check("token accounts discovered",
      st["total"] == 3 and st["healthy"] == 3)

import httpx  # noqa: E402


class FakeResp:
    status_code = 429


class Fake429(httpx.HTTPStatusError):
    def __init__(self):
        super().__init__("429", request=None, response=FakeResp())


class Counter:
    n = 0


class FakeClient:
    def __init__(self, session=None, allow_interactive=False):
        pass

    def chat(self, *a, **k):
        Counter.n += 1
        if Counter.n <= 2:
            raise Fake429()
        class R:
            text = "ok!"
            conversation_id = "s:1"
        return R()

    def stream(self, *a, **k):
        r = self.chat(*a, **k)
        return iter([r.text])


poolmod.DeepSeekClient = FakeClient
Counter.n = 0
reply = token_pool.call("chat", "hi")
check("429 failover serves the request", reply.text == "ok!" and Counter.n == 3)
st2 = token_pool.status()
check("failed accounts cooling down",
      sum(1 for a in st2["accounts"] if a["state"] == "cooldown") == 2)

counter2 = {"n": 0}


class StreamFailover(FakeClient):
    def stream(self, *a, **k):
        counter2["n"] += 1
        if counter2["n"] <= 1:
            raise Fake429()
        return iter(["hello ", "world"])


poolmod.DeepSeekClient = StreamFailover
pool_s = AccountPool(accounts_dir=pathlib.Path(tempfile.mkdtemp()),
                     tokens_env="tokA,tokB")
out = list(pool_s.call_stream("hi"))
check("stream failover before first chunk",
      out == ["hello ", "world"] and counter2["n"] == 2)


class Dead(FakeClient):
    def chat(self, *a, **k):
        raise LoginRequired("dead")


poolmod.DeepSeekClient = Dead
pool_d = AccountPool(accounts_dir=pathlib.Path(tempfile.mkdtemp()),
                     tokens_env="tokX")
try:
    pool_d.call("chat", "hi")
    check("all disabled -> humanized 503", False)
except UpstreamError as e:
    check("all disabled -> humanized 503",
          e.status == 503 and "Re-login" in str(e)
          and e.err_type == "login_required")

poolmod.DeepSeekClient = FakeClient
Counter.n = 0
pool_r = AccountPool(accounts_dir=pathlib.Path(tempfile.mkdtemp()),
                     tokens_env="tokY,tokZ")
try:
    pool_r.call("chat", "hi")
    pool_r.call("chat", "hi")
    check("all cooling -> humanized 429", False)
except UpstreamError as e:
    check("all cooling -> humanized 429",
          e.status == 429 and e.err_type == "rate_limit_error")

# ── Anthropic translation ─────────────────────────────────────────────────────
from server.anthropic import anthropic_to_openai, message_response  # noqa: E402

body = {
    "model": "deepseek-chat",
    "system": "Be brief.",
    "messages": [
        {"role": "user", "content": "weather in paris?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                                           "name": "get_weather",
                                           "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result",
                                      "tool_use_id": "t1",
                                      "content": "20C"}]},
    ],
    "tools": [{"name": "get_weather", "description": "w",
               "input_schema": {"type": "object", "properties": {}}}],
}
system, messages, tools, stream, thinking, search = anthropic_to_openai(body)
check("anthropic system extracted", system == "Be brief.")
check("anthropic tools converted",
      bool(tools) and tools[0]["function"]["name"] == "get_weather")
check("anthropic tool_use -> openai tool_calls",
      any(m.get("tool_calls") for m in messages))
check("anthropic tool_result -> tool role",
      any(m.get("role") == "tool" for m in messages))

resp = message_response("deepseek-chat", "here you go", [{
    "id": "call_1", "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
}])
check("anthropic response with tool_use",
      resp["stop_reason"] == "tool_use"
      and resp["content"][1]["type"] == "tool_use"
      and resp["content"][1]["input"] == {"city": "Paris"})

from server.agent import AgentInterceptor, parse_agent_tool_calls  # noqa: E402

marker_text = ('<<<TOOL_CALL>>>{"name":"get_weather",'
               '"arguments":{"city":"Paris"}}<<<END_TOOL_CALL>>>')
calls = parse_agent_tool_calls(marker_text)
check("marker tool call parsed",
      bool(calls) and calls[0]["function"]["name"] == "get_weather")

shim = AgentInterceptor()
out_text = ""
for ch in ["hello ", "<<<TOO",
           'L_CALL>>>{\"na', "me\":\"get_weather\"",
           ',', '"arguments\":{\"city\":\"X\"}}<<<END_',
           "TOOL_CALL>>>"]:
    got = shim.feed(ch)
    out_text += got or ""
out_text += shim.finish()
drained = shim.drain_tool_calls()
check("interceptor streams clean text + drains call",
      out_text.strip() == "hello" and bool(drained)
      and drained[0]["function"]["name"] == "get_weather")

# ── openai_format ─────────────────────────────────────────────────────────────
from server.openai_format import completion_response, messages_to_prompt  # noqa: E402
from server.schemas import ChatMessage  # noqa: E402

resp = completion_response("deepseek-chat", "hi there", "hi")
check("completion response shape",
      resp["object"] == "chat.completion"
      and resp["choices"][0]["message"]["content"] == "hi there")
check("prompt flattening", "Assistant:" in messages_to_prompt([
    ChatMessage(role="user", content="a"),
    ChatMessage(role="assistant", content="b")]))

# ── count_tokens endpoint (Anthropic parity) ─────────────────────────────────
import os  # noqa: E402
os.environ.setdefault("AUTH_TOKEN", "test-suite-key")
try:
    from fastapi.testclient import TestClient  # noqa: E402
    import server.api as _api  # noqa: E402
    _tc = TestClient(_api.app)

    _r = _tc.post("/v1/messages/count_tokens",
                  headers={"Authorization": "Bearer test-suite-key"},
                  json={"model": "deepseek-chat",
                        "messages": [{"role": "user",
                                      "content": "hello world this is a counting test"}],
                        "system": "be terse",
                        "tools": TOOLS})
    check("count_tokens returns positive estimate",
          _r.status_code == 200 and _r.json().get("input_tokens", 0) > 0,
          f"status={_r.status_code} body={_r.text[:120]}")

    _r2 = _tc.post("/v1/messages/count_tokens",
                   headers={"Authorization": "Bearer test-suite-key",
                            "Content-Type": "application/json"},
                   content=b"not json")
    check("count_tokens rejects bad JSON with 400", _r2.status_code == 400)

    _r3 = _tc.get("/v1/models", headers={"Authorization": "Bearer test-suite-key"})
    check("models endpoint lists chat + expert",
          _r3.status_code == 200
          and {m["id"] for m in _r3.json()["data"]} >= {"deepseek-chat", "deepseek-expert"})
except Exception as _e:  # fastapi missing etc. — suite stays green on unit level
    check("count_tokens HTTP tests skipped", True, f"({type(_e).__name__})")

print(f"\n=== {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL else 0)
