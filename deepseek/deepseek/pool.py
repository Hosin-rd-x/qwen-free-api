"""
Account pool — multi-account support with cooldown and failover.

Why: chat.deepseek.com rate-limits and human-checks per account. With several
accounts the server rotates between them transparently, so a 429 on one
account never surfaces to the caller as long as another is healthy.

Account sources (merged, in this order):

  1. session/accounts/*.json — one saved Session per account. Create each with:
         python -m deepseek.auth --account NAME
  2. session/session.json    — the legacy single-session file (if present).
  3. DEEPSEEK_TOKENS (.env)  — bearer tokens separated by comma or newline.
     Token-only accounts carry no cookies; they work when the WAF is not
     challenging the IP and are otherwise cooled down like any other account.

Behaviour:

  - Round-robin across healthy accounts.
  - Upstream 429 / rate-limit text  -> cooldown (base 120 s, doubles per
    consecutive hit, capped at 30 min) and fail over to the next account.
  - Upstream 401 / session-expired  -> account disabled until re-login.
  - WAF / human-check               -> short cooldown (300 s) on that account.
  - call() fails over before the first chunk reaches the caller, so streaming
    callers never receive duplicated content from a retried attempt.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from .auth import LoginRequired, Session
from .client import BASE, DeepSeekClient

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

RATE_LIMIT_TEXTS = ("rate limit", "too many requests", "busy", "try again")
HUMAN_CHECK_TEXTS = ("human", "verify", "waf", "captcha", "robot")
AUTH_TEXTS = (
    "authorization failed", "invalid token", "unauthorized", "not logged in",
    "authentication", "login required", "session expired",
)


class UpstreamError(RuntimeError):
    """A classified upstream failure, humanised and carry an error type."""

    def __init__(self, message: str, status: int = 502,
                 err_type: str = "server_error"):
        super().__init__(message)
        self.status = status
        self.err_type = err_type


@dataclass
class _Account:
    name: str
    source: str                       # "file" | "token"
    path: Optional[Path] = None
    token: Optional[str] = None
    client: Optional[DeepSeekClient] = None
    client_built: bool = False
    requests: int = 0
    errors: int = 0
    last_error: str = ""
    cooldown_until: float = 0.0
    consecutive_429: int = 0
    disabled: bool = False
    disabled_reason: str = ""
    lock: field(default_factory=threading.Lock, repr=False) = None

    def __post_init__(self):
        if self.lock is None:
            self.lock = threading.Lock()

    # --- state helpers ------------------------------------------------------
    def in_cooldown(self, now: float) -> bool:
        return now < self.cooldown_until

    def healthy(self, now: float) -> bool:
        return not self.disabled and not self.in_cooldown(now)

    def mark_429(self, base: float, cap: float = 1800.0) -> float:
        self.consecutive_429 += 1
        delay = min(base * (2 ** (self.consecutive_429 - 1)), cap)
        self.cooldown_until = time.time() + delay
        self.last_error = "rate limited by DeepSeek"
        return delay

    def mark_ok(self) -> None:
        self.consecutive_429 = 0
        self.last_error = ""
        self.requests += 1

    def mark_disabled(self, reason: str) -> None:
        self.disabled = True
        self.disabled_reason = reason
        self.last_error = reason

    def mark_waf(self) -> None:
        self.cooldown_until = time.time() + 300.0
        self.last_error = "human-check (WAF) challenged this account"

    def drop_client(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None
        self.client_built = False


class AccountPool:
    """Thread-safe pool of DeepSeek accounts with round-robin + failover."""

    def __init__(
        self,
        accounts_dir: Path,
        tokens_env: str = "",
        cooldown_base: float = 120.0,
    ):
        self._accounts: List[_Account] = []
        self._rr = 0
        self._lock = threading.Lock()
        self.cooldown_base = cooldown_base
        self._scan(accounts_dir, tokens_env)

    # --- discovery -----------------------------------------------------------
    def _scan(self, accounts_dir: Path, tokens_env: str) -> None:
        found: List[_Account] = []
        if accounts_dir.is_dir():
            for p in sorted(accounts_dir.glob("*.json")):
                try:
                    sess = Session.load(p)
                    if sess and sess.token:
                        found.append(_Account(name=p.stem, source="file", path=p))
                except Exception:
                    continue
        legacy = accounts_dir.parent / "session.json"
        if legacy.is_file() and not any(a.source == "file" for a in found):
            try:
                sess = Session.load(legacy)
                if sess and sess.token:
                    found.append(_Account(name="default", source="file", path=legacy))
            except Exception:
                pass
        for i, raw in enumerate(self._split_tokens(tokens_env)):
            found.append(_Account(name=f"token-{i + 1}", source="token", token=raw))
        with self._lock:
            self._accounts = found

    @staticmethod
    def _split_tokens(env: str) -> List[str]:
        out = []
        for chunk in (env or "").replace("\n", ",").split(","):
            chunk = chunk.strip()
            if chunk:
                out.append(chunk)
        return out

    def rescan(self) -> None:
        for a in self._accounts:
            a.drop_client()
        self._scan(
            self._accounts_dir, os.getenv("DEEPSEEK_TOKENS", "")
        ) if hasattr(self, "_accounts_dir") else None

    # --- session / client ----------------------------------------------------
    def _session_for(self, acc: _Account) -> Session:
        if acc.source == "file":
            sess = Session.load(acc.path)
            if sess and sess.token:
                return sess
            raise LoginRequired(
                f"Account '{acc.name}' has no saved session. Re-login with:\n"
                f"    python -m deepseek.auth --account {acc.name}"
            )
        return Session(
            token=acc.token, cookies={}, user_agent=DEFAULT_UA, captured_at=time.time()
        )

    def _client_for(self, acc: _Account) -> DeepSeekClient:
        if acc.client is None:
            acc.client = DeepSeekClient(
                session=self._session_for(acc), allow_interactive=False
            )
            acc.client_built = True
        return acc.client

    # --- selection ------------------------------------------------------------
    def _healthy(self) -> List[_Account]:
        now = time.time()
        return [a for a in self._accounts if a.healthy(now)]

    def healthy_count(self) -> int:
        return len(self._healthy())

    def has_accounts(self) -> bool:
        return bool(self._accounts)

    # --- execution --------------------------------------------------------------
    def _classify(self, acc: _Account, exc: Exception) -> str:
        """Record a failure on `acc`. Returns 'retry' or 'raise'."""
        if isinstance(exc, LoginRequired):
            acc.mark_disabled("session expired — re-login required")
            acc.drop_client()
            return "retry"
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code if exc.response is not None else 0
            acc.last_error = f"HTTP {code} from DeepSeek"
            if code == 429:
                delay = acc.mark_429(self.cooldown_base)
                acc.last_error = f"429 rate limited — cooldown {int(delay)}s"
                return "retry"
            if code in (401, 403):
                acc.mark_disabled(f"HTTP {code} — session rejected, re-login required")
                acc.drop_client()
                return "retry"
            if code == 400:
                acc.last_error = "HTTP 400 — bad request"
                return "raise"
            if code >= 500:
                return "raise"
            return "retry"
        text = str(exc).lower()
        if any(t in text for t in AUTH_TEXTS):
            acc.mark_disabled("rejected by DeepSeek (invalid/expired token) — re-login")
            acc.drop_client()
            return "retry"
        if any(t in text for t in HUMAN_CHECK_TEXTS):
            acc.mark_waf()
            return "retry"
        if any(t in text for t in RATE_LIMIT_TEXTS):
            acc.mark_429(self.cooldown_base)
            return "retry"
        return "raise"

    def _exhausted(self, last_exc: Optional[Exception]) -> Exception:
        """Build the humanised error for 'no account could serve this'."""
        if not self._accounts:
            return LoginRequired()
        now = time.time()
        if all(a.disabled for a in self._accounts):
            return UpstreamError(
                "All accounts are disabled (expired sessions). Re-login:\n"
                "    python -m deepseek.auth --account main",
                status=503, err_type="login_required",
            )
        if all(a.in_cooldown(now) for a in self._accounts):
            soonest = min(a.cooldown_until for a in self._accounts)
            return UpstreamError(
                "All accounts are rate-limited right now. "
                f"Next account ready in {int(soonest - now)}s.",
                status=429, err_type="rate_limit_error",
            )
        return UpstreamError(
            f"No healthy DeepSeek account available. Last error: {last_exc}",
            status=503,
        )

    def _next_account(self) -> Optional[_Account]:
        healthy = self._healthy()
        if not healthy:
            return None
        with self._lock:
            acc = healthy[self._rr % len(healthy)]
            self._rr += 1
        return acc

    def call(self, method: str, *args, **kwargs):
        """Run client.<method>(*args) with round-robin and failover.

        Use this for blocking calls (client.chat) where the whole request
        completes inside. For streaming use call_stream(), which forces the
        upstream request before the first chunk leaves the pool so failures
        can still fail over cleanly.
        """
        if not self._accounts:
            raise LoginRequired()
        last_exc: Optional[Exception] = None
        tried = 0
        while tried < max(1, len(self._accounts)):
            acc = self._next_account()
            if acc is None:
                break
            tried += 1
            try:
                client = self._client_for(acc)
                result = getattr(client, method)(*args, **kwargs)
                acc.mark_ok()
                try:
                    result._pool_account = acc.name  # annotate for logging
                except Exception:
                    pass
                return result
            except Exception as e:
                if isinstance(e, (UpstreamError,)):
                    raise
                acc.errors += 1
                last_exc = e
                if self._classify(acc, e) == "raise":
                    raise UpstreamError(self._humanize(e), status=502) from e
                continue
        raise self._exhausted(last_exc)

    def call_stream(self, *args, **kwargs):
        """Return an iterator of reply-text chunks with pre-first-chunk failover.

        The first chunk is consumed inside the pool: any request-time error
        (429, WAF, dead session) is classified and retried on the next account
        BEFORE the caller receives anything, so streamed output is never
        duplicated. Errors after the first chunk propagate to the caller.
        """
        if not self._accounts:
            raise LoginRequired()
        last_exc: Optional[Exception] = None
        tried = 0
        while tried < max(1, len(self._accounts)):
            acc = self._next_account()
            if acc is None:
                break
            tried += 1
            try:
                client = self._client_for(acc)
                stream = client.stream(*args, **kwargs)
                first = next(iter(stream))
                acc.mark_ok()

                def _gen(acc=acc, stream=stream, first=first):
                    yield first
                    yield from stream

                return _gen()
            except StopIteration:
                acc.mark_ok()
                return iter(())
            except Exception as e:
                acc.errors += 1
                last_exc = e
                if self._classify(acc, e) == "raise":
                    raise UpstreamError(self._humanize(e), status=502) from e
                continue
        raise self._exhausted(last_exc)

    @staticmethod
    def _humanize(exc: Exception) -> str:
        """Turn raw upstream failures into readable, actionable messages."""
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            code = exc.response.status_code
            if code == 400:
                return ("DeepSeek rejected the request (HTTP 400) — the prompt "
                        "may be empty or too long.")
            if code >= 500:
                return (f"DeepSeek upstream error (HTTP {code}) — the site is "
                        "having trouble; try again shortly.")
            return f"DeepSeek returned HTTP {code}."
        text = str(exc)
        low = text.lower()
        if any(t in low for t in AUTH_TEXTS):
            return ("DeepSeek rejected the account (invalid or expired token). "
                    "Re-login:\n    python -m deepseek.auth --account main")
        if "pow" in low or "challenge" in low:
            return ("PoW challenge could not be solved — DeepSeek changed its "
                    "proof-of-work; update the project.")
        return f"DeepSeek request failed: {text}"

    # --- status for dashboard / API ---------------------------------------------
    def status(self) -> dict:
        now = time.time()
        accounts = []
        for a in self._accounts:
            state = "active"
            if a.disabled:
                state = "disabled"
            elif a.in_cooldown(now):
                state = "cooldown"
            accounts.append(
                {
                    "name": a.name,
                    "source": a.source,
                    "state": state,
                    "requests": a.requests,
                    "errors": a.errors,
                    "cooldown_s": max(0, int(a.cooldown_until - now)),
                    "last_error": a.last_error,
                }
            )
        return {
            "accounts": accounts,
            "total": len(accounts),
            "healthy": len([a for a in self._accounts if a.healthy(now)]),
        }
