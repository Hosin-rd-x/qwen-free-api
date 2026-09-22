"""Server configuration: models, account pool and rate-limit settings."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Requests per minute allowed per client IP (override with RATE_LIMIT_PER_MINUTE).
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

# When the server has no session, should it pop a visible browser window for
# interactive sign-in (the first request then blocks until you finish)?
# On by default for local single-user use. Set to "0"/"false" for headless
# deployments, where it instead returns a 503 telling the caller to run
# `python -m deepseek.auth`.
SERVER_INTERACTIVE_LOGIN = os.getenv("SERVER_INTERACTIVE_LOGIN", "1").lower() not in (
    "0", "false", "no", "off",
)

# --- Account pool -------------------------------------------------------------
# Directory holding one saved Session JSON per account
# (create with: python -m deepseek.auth --account NAME).
from deepseek.pool_accounts import accounts_dir as _accounts_dir

ACCOUNTS_DIR = _accounts_dir()

# Bearer tokens (comma / newline separated) from .env — token-only accounts
# used as a fallback when no saved browser session exists for them.
DEEPSEEK_TOKENS = os.getenv("DEEPSEEK_TOKENS", "")

# Base cooldown for a rate-limited account, seconds; doubles per consecutive
# 429 up to 30 minutes.
POOL_COOLDOWN_BASE = float(os.getenv("DEEPSEEK_COOLDOWN_BASE", "120"))

# Public model ids the server advertises (via /v1/models) and accepts, mapped to
# DeepSeek's `model_type` wire value. This is the MODEL axis ONLY — it picks
# which model answers. DeepThink and web Search are orthogonal tools requested
# per call via `tool_names` (see deepseek.client.KNOWN_TOOLS), never encoded in
# the model name.
#
# "vision" is deferred: it only does anything with an image attached, which needs
# ref_file_ids / file-upload plumbing we don't have yet.
MODEL_MAP = {
    "deepseek-chat":   "default",   # Instant — the fast default model
    "deepseek-expert": "expert",    # Expert  — the stronger, slower model
}

DEFAULT_MODEL = "deepseek-chat"

# Shown on the dashboard and in /v1/models metadata.
MODEL_DESCRIPTIONS = {
    "deepseek-chat":   "Instant (fast) model — great default for agents and chat",
    "deepseek-expert": "Expert (stronger, slower) model — hard reasoning",
}


def is_known_model(name: str) -> bool:
    """Whether `name` is a model id we accept (used to 404 unknown models)."""
    return name in MODEL_MAP


def resolve_model_type(name: str) -> str:
    """Translate a public model id to DeepSeek's `model_type` wire value.

    Caller must check `is_known_model` first; this raises KeyError otherwise.
    """
    return MODEL_MAP[name]
