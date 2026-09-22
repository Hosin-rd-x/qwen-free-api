# DeepSeek Free API — OpenAI-compatible bridge for chat.deepseek.com

**English | [فارسی](#فارسی)**

Turn your free DeepSeek web account into a local OpenAI-compatible API — with a multi-account pool that bypasses rate limits, a Persian web dashboard, and emulated function-calling so agents like Hermes just work. Also speaks the **Anthropic protocol** (`/v1/messages`) for Claude Code & friends.

> Unofficial project. Not affiliated with or endorsed by DeepSeek. Automates the consumer web experience for personal use — use responsibly and within DeepSeek's terms.

---

## Why this version

| | Original idea | This version |
|---|---|---|
| Accounts | 1 session | **Multi-account pool** — round-robin, 429 cooldown with exponential backoff, automatic failover, dead-session detection |
| Dashboard | none | **Persian RTL web dashboard** at `/` — account states, live playground, copy-paste agent snippets |
| Tool calling | ✗ | **OpenAI function-calling emulation** — send `tools=`, get real `tool_calls` responses (streaming included) |
| Protocols | OpenAI only | **OpenAI + Anthropic** (`/v1/messages` with `tool_use` blocks) |
| Errors | raw tracebacks | **Humanized, actionable messages** (what happened + the exact command to fix it) |
| Tests | none | `python tests/test_all.py` — 27 checks, no account needed |
| Run | manual steps | **`./start.sh`** (Linux/macOS) or **`start.bat`** (Windows) |

## Requirements

- Python 3.9+
- A free [chat.deepseek.com](https://chat.deepseek.com) account
- Windows / macOS / Linux

## Quick start (2 commands)

```bash
./start.sh                        # 1) install + run → dashboard at http://localhost:8000   (Windows: start.bat)
python -m deepseek.auth --account main   # 2) one-time login (opens a browser)
```

The login step opens a real browser once, handles the human-check, and saves the session to `session/accounts/main.json`. After that the server picks it up automatically — no restart needed. Add more accounts to multiply your rate limits:

```bash
python -m deepseek.auth --account work
python -m deepseek.auth --account alt
python -m deepseek.auth --list
```

<details>
<summary>Manual setup (without start.sh)</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
python app.py                     # serves on http://localhost:8000
```
</details>

## Use from any OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")

r = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(r.choices[0].message.content)

# resume a thread (extra field returned by every response):
r2 = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "continue"}],
    extra_body={"conversation_id": r.conversation_id},
)
```

Streaming (`stream=True`) works exactly like OpenAI SSE.

### Anthropic-native clients (Claude Code, Cline, …)

Point `ANTHROPIC_BASE_URL` at the bridge — same accounts, same tools:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=anything
export ANTHROPIC_MODEL=deepseek-chat
```

## Models

| model id | DeepSeek wire value | notes |
|---|---|---|
| `deepseek-chat` | `default` | Instant — fast model, best default for agents |
| `deepseek-expert` | `expert` | Expert — stronger and slower |

Per-request extras (OpenAI `extra_body`): `"thinking": true` → DeepThink reasoning, `"search": true` → web search.

## Tool calling (agents / Hermes / function calling)

DeepSeek's web endpoint has no native function-calling, so this bridge emulates the OpenAI protocol: it appends a machine-readable tool contract to the prompt and translates the model's JSON answer into a **real `tool_calls` response** — including streaming deltas. Your agent code doesn't change:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]

r = client.chat.completions.create(model="deepseek-chat", messages=msgs, tools=tools)
msg = r.choices[0].message
if msg.tool_calls:
    print(msg.tool_calls[0].function.name, msg.tool_calls[0].function.arguments)
    # feed the result back as {"role": "tool", "tool_call_id": ..., "content": "..."}
```

## Multi-account pool

Every request round-robins across healthy accounts. If DeepSeek rate-limits one account (429), it is put on cooldown (base 120 s, doubling per consecutive hit, capped at 30 min) and the next account serves the request transparently. Expired sessions (401/403) are disabled with a clear message instead of breaking the server. Token-only accounts can be added via `.env`:

```env
DEEPSEEK_TOKENS=eyJhbGciOi... , eyJhbGciOi...
```

## Dashboard

Open `http://localhost:8000/` — Persian RTL panel with live account states (active / cooldown / disabled, request counters, last errors), setup guidance, copy-paste snippets for OpenAI SDK / curl / tool-calling, and a live playground to test chat without any client.

## Endpoints

| endpoint | description |
|---|---|
| `GET /` , `/dashboard` | web dashboard |
| `GET /api/status` | live pool + config JSON |
| `GET /v1/models` | model list |
| `POST /v1/chat/completions` | chat (stream, tools, conversation_id) |
| `POST /v1/messages` | Anthropic protocol (system, tool_use blocks, stream) |
| `POST /v1/messages/count_tokens` | Anthropic parity — local token estimate, zero upstream cost (v1.1.0) |
| `GET /healthz` | liveness (rate-limit exempt) |

## Tests

```bash
python tests/test_all.py     # 30 checks: tool-calling, pool failover, Anthropic translation, count_tokens
```

No real account or network needed — the upstream is faked inside the tests.

## Notes & limitations

- **Login is human-only by design**: the AWS WAF human-check must be solved once by you, in a real browser. The session is then refreshed headlessly and reused.
- Sessions are trusted for 6 hours and refreshed automatically from the saved browser profile.
- Rate limiting toward clients: 30 req/min per IP by default (`RATE_LIMIT_PER_MINUTE`).
- This project automates the free consumer web experience — it is not the paid official API and has no SLA.

## فارسی

**پل OpenAI-سازگار برای chat.deepseek.com** — اکانت رایگان وب دیپ‌سیک را به API محلی تبدیل کنید؛ با استخر چند-اکانته (دور زدن محدودیت نرخ)، داشبورد فارسی، و شبیه‌سازی tool calling برای عامل‌ها مثل Hermes.

### شروع سریع (۲ دستور)

```bash
./start.sh                                 # نصب + اجرا → داشبورد روی http://localhost:8000
python -m deepseek.auth --account main     # ورود یک‌بارمه (مرورگر باز می‌شود)
```

- مرحله ورود فقط یک‌بار است: کد انسانی (human-check) را در مرورگر واقعی حل می‌کنید و سشن ذخیره می‌شود؛ دفعات بعد خودکار تازه‌سازی می‌شود.
- برای محدودیت نرخ بالاتر، اکانت بیشتر اضافه کنید: `python -m deepseek.auth --account work` — استخر خودکار بین اکانت‌ها می‌چرخد؛ 429 روی یک اکانت یعنی cooldown موقت و سوئیچ شفاف به اکانت بعدی.
- مدل‌ها: `deepseek-chat` (سریع) و `deepseek-expert` (قوی‌تر). گزینه‌های هر درخواست: `thinking` (DeepThink) و `search` (جستجوی وب).
- اتصال عامل/برنامه: کافیست `base_url` را روی `http://localhost:8000/v1` بگذارید و `api_key` را هرچه بگذارید. برای tool calling همان کد استاندارد OpenAI را بزنید — پل خودش تبدیل می‌کند. کلاینت‌های Anthropic-محور هم `/v1/messages` دارند.
- داشبورد: `http://localhost:8000/` — وضعیت زنده اکانت‌ها، اسنیپت‌های آماده کپی، و تست زنده چت.
- نکته: این پروژه تجربه وب رایگان مصرف‌کننده را خودکار می‌کند، نه API رسمی پولی؛ مصرف شخصی و مسئولانه.
