"""
OpenAI-compatible FastAPI server for DeepSeek — multi-account edition.

Point any OpenAI client at http://localhost:8000/v1 :

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Endpoints:
    GET  /                    Persian RTL dashboard (also /dashboard)
    GET  /api/status          live pool + config JSON
    GET  /v1/models
    POST /v1/chat/completions (stream=true, tools/function-calling supported)
    GET  /healthz

Requests under /v1 are rate limited per client IP (default 30/min, set via
RATE_LIMIT_PER_MINUTE); /healthz and the dashboard are exempt.
"""

from __future__ import annotations

import json
import threading
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from deepseek.auth import LoginRequired
from deepseek.pool import AccountPool, UpstreamError

from .config import (
    MODEL_DESCRIPTIONS,
    MODEL_MAP,
    POOL_COOLDOWN_BASE,
    RATE_LIMIT_PER_MINUTE,
    SERVER_INTERACTIVE_LOGIN,
    DEEPSEEK_TOKENS,
    ACCOUNTS_DIR,
    is_known_model,
    resolve_model_type,
)
from .dashboard import DASHBOARD_HTML
from .openai_format import completion_response, messages_to_prompt, stream_chunks, _est_tokens
from .ratelimit import RateLimiter, install_rate_limit
from .schemas import ChatCompletionRequest, ChatMessage
from .tools_emu import build_prompt, content_replay_chunks, extract_tool_call, tool_call_delta_chunks
from .anthropic import (
    anthropic_to_openai,
    error_payload as anthropic_error,
    message_response,
    stream_message_events,
)
from .agent import (
    AgentInterceptor,
    agent_transform_messages,
    parse_agent_tool_calls,
    strip_agent_tool_calls,
)

load_dotenv()

app = FastAPI(title="DeepSeek Free API", version="1.1.0")
install_rate_limit(app, RateLimiter(limit=RATE_LIMIT_PER_MINUTE, window=60.0))

# Shared account pool; account clients are built lazily off the event loop.
_pool: AccountPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> AccountPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = AccountPool(
                    accounts_dir=ACCOUNTS_DIR,
                    tokens_env=DEEPSEEK_TOKENS,
                    cooldown_base=POOL_COOLDOWN_BASE,
                )
    return _pool


def _error(message: str, status: int = 500, err_type: str = "server_error"):
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type}},
    )


# --- dashboard / status --------------------------------------------------------

@app.get("/")
@app.get("/dashboard")
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/status")
def api_status():
    try:
        pool = get_pool()
        pool_state = pool.status()
    except Exception as e:  # dashboard must never crash
        pool_state = {"accounts": [], "total": 0, "healthy": 0, "error": str(e)}
    return {
        "service": "deepseek-free-api",
        "version": "1.1.0",
        "models": list(MODEL_MAP),
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "pool": pool_state,
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "created": created,
                "owned_by": "deepseek",
                "description": MODEL_DESCRIPTIONS.get(name, ""),
            }
            for name in MODEL_MAP
        ],
    }


# --- Anthropic parity: count_tokens (local estimate, no upstream) -----------

@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: Request):
    """Anthropic-parity token count for client-side budgeting.

    Claude Code / Cline call this before sending messages. DeepSeek's web
    API exposes no token count, so — same as the rest of this bridge — we
    return the bridge-standard rough estimate (~4 chars/token) over
    messages + system + tools, with zero upstream cost.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content=anthropic_error(
            "Invalid JSON body", "invalid_request_error"))
    payload = json.dumps(body.get("messages", []), ensure_ascii=False)
    if body.get("system"):
        payload += json.dumps(body["system"], ensure_ascii=False)
    if body.get("tools"):
        payload += json.dumps(body["tools"], ensure_ascii=False)
    return {"input_tokens": _est_tokens(payload)}


# --- chat -------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        return _error("`messages` must not be empty", status=400, err_type="invalid_request_error")

    if not is_known_model(req.model):
        return _error(
            f"The model `{req.model}` does not exist. Available models: "
            f"{', '.join(MODEL_MAP)}",
            status=404, err_type="model_not_found",
        )

    # A thread's model is fixed when it's created, so on resume we ignore `model`
    # (the OpenAI SDK always sends one) and let the existing thread's model stand.
    model_type = None if req.conversation_id else resolve_model_type(req.model)
    prompt = build_prompt(req.messages, req.tools, req.tool_choice)

    try:
        # Off the event loop: pool/client construction uses Playwright's sync
        # API when reading saved sessions, which errors inside the asyncio loop.
        pool = await run_in_threadpool(get_pool)
    except Exception as e:
        return _error(f"Failed to initialise account pool: {e}")

    if not pool.has_accounts():
        return _error(
            "No DeepSeek account is registered yet. Log in once to create one:\n"
            "    python -m deepseek.auth --account main\n"
            "(این دستور یک‌بار مرورگر باز می‌کند؛ بعد از ورود، سشن ذخیره می‌شود.)",
            status=503, err_type="login_required",
        )

    if req.stream:
        return await _run_stream(pool, req, prompt, model_type)

    try:
        reply = await run_in_threadpool(
            pool.call, "chat", prompt, req.conversation_id, model_type,
            req.thinking, req.search,
        )
    except LoginRequired as e:
        return _error(str(e), status=503, err_type="login_required")
    except UpstreamError as e:
        return _error(str(e), status=e.status, err_type=e.err_type)
    except Exception as e:
        return _error(f"DeepSeek request failed: {e}")

    content = reply.text
    # Emulated function-calling: translate a JSON tool-call reply into the
    # OpenAI shape (only when the request actually carried tools).
    if req.tools:
        call = extract_tool_call(content)
        if call is not None:
            resp = completion_response(req.model, "", prompt, reply.conversation_id)
            resp["choices"][0]["message"] = {
                "role": "assistant", "content": None, "tool_calls": [call],
            }
            resp["choices"][0]["finish_reason"] = "tool_calls"
            return resp

    return completion_response(req.model, content, prompt, reply.conversation_id)


def _to_chat_message(m: dict):
    return ChatMessage(**{k: v for k, v in m.items() if k in ChatMessage.model_fields})


# --- Anthropic endpoint -------------------------------------------------------

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic-native endpoint (Claude Code, Cline, any ANTHROPIC_BASE_URL
    client). Same pool, same account failover; tools become marker-based
    tool_use blocks (stop_reason="tool_use")."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content=anthropic_error(
            "Invalid JSON body", "invalid_request_error"))

    system, messages, tools, stream, thinking, search = anthropic_to_openai(body)
    model_name = body.get("model") or "deepseek-chat"
    if not is_known_model(model_name):
        return JSONResponse(status_code=404, content=anthropic_error(
            f"The model `{model_name}` does not exist. Available models: "
            f"{', '.join(MODEL_MAP)}", "not_found_error"))

    agent_on = bool(tools)
    if agent_on:
        messages = agent_transform_messages(messages, tools)
    prompt = messages_to_prompt([_to_chat_message(m) for m in messages])
    if system and not agent_on:
        prompt = f"System: {system}\n\n{prompt}"

    try:
        pool = await run_in_threadpool(get_pool)
    except Exception as e:
        return JSONResponse(status_code=500, content=anthropic_error(
            f"Failed to initialise account pool: {e}"))

    if not pool.has_accounts():
        return JSONResponse(status_code=503, content=anthropic_error(
            "No DeepSeek account is registered yet. Log in once:\n"
            "    python -m deepseek.auth --account main",
            "authentication_error"))

    model_type = None if body.get("conversation_id") else resolve_model_type(model_name)
    try:
        # Off the event loop: may build clients / read saved sessions (sync API).
        chunk_iter = await run_in_threadpool(
            pool.call_stream, prompt, body.get("conversation_id"),
            model_type, thinking, search,
        )
    except LoginRequired as e:
        return JSONResponse(status_code=503, content=anthropic_error(
            str(e), "authentication_error"))
    except UpstreamError as e:
        atype = "rate_limit_error" if e.status == 429 else "api_error"
        if e.err_type == "login_required":
            atype = "authentication_error"
        return JSONResponse(status_code=e.status, content=anthropic_error(str(e), atype))
    except Exception as e:
        return JSONResponse(status_code=502, content=anthropic_error(
            f"DeepSeek request failed: {e}"))

    if not stream:
        parts = []
        try:
            for chunk in chunk_iter:
                parts.append(chunk)
        except Exception as e:
            return JSONResponse(status_code=502, content=anthropic_error(
                f"DeepSeek stream failed: {e}"))
        text = "".join(parts)
        calls = parse_agent_tool_calls(text) if agent_on else []
        if calls:
            text = strip_agent_tool_calls(text)
        return JSONResponse(message_response(model_name, text, calls,
                                             input_tokens=len(prompt) // 4))

    shim = AgentInterceptor() if agent_on else None

    def sse():
        yield from stream_message_events(model_name,
                                         (("content", c) for c in chunk_iter),
                                         shim)

    return StreamingResponse(sse(), media_type="text/event-stream")


def _sse_error(message: str) -> str:
    """An SSE frame carrying a readable error, so streaming clients see it."""
    return "data: " + json.dumps(
        {"error": {"message": message, "type": "server_error",
                   "code": "upstream_error"}},
        ensure_ascii=False,
    ) + "\n\n"


def _stream_gen(pool: AccountPool, req: ChatCompletionRequest, prompt: str,
                model_type):
    """Streaming generator with pool failover before the first chunk.

    With tools present, the reply is buffered and replayed as either real
    tool_call chunks (emulated function-calling) or normal content chunks.
    Errors arriving before any chunk are retried inside the pool; errors after
    that are surfaced as one readable SSE error frame.
    """
    try:
        stream = pool.call_stream(
            prompt, req.conversation_id, model_type,
            req.thinking, req.search,
        )
    except LoginRequired as e:
        yield _sse_error(str(e))
        yield "data: [DONE]\n\n"
        return
    except UpstreamError as e:
        yield _sse_error(str(e))
        yield "data: [DONE]\n\n"
        return

    if not req.tools:
        yield from stream_chunks(req.model, stream)
        return

    # Emulated function-calling over a streamed request: buffer, then replay.
    buffer: list[str] = []
    try:
        for chunk in stream:
            buffer.append(chunk)
    except Exception as e:  # mid-stream failure — surface once, no retry
        yield _sse_error(f"DeepSeek stream failed: {e}")
        yield "data: [DONE]\n\n"
        return

    text = "".join(buffer)
    cid = getattr(stream, "conversation_id", None)
    base = {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion.chunk",
        "model": req.model,
        "conversation_id": cid,
    }
    call = extract_tool_call(text)
    if call is not None:
        yield from tool_call_delta_chunks(base, call)
    else:
        yield from content_replay_chunks(base, text)
    yield "data: [DONE]\n\n"


async def _run_stream(pool: AccountPool, req: ChatCompletionRequest,
                      prompt: str, model_type):
    return StreamingResponse(
        _stream_gen(pool, req, prompt, model_type),
        media_type="text/event-stream",
    )
