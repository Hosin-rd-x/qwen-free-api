"""Shared path helper for the multi-account session directory.

One saved Session JSON per account lives in session/accounts/NAME.json
(create with: python -m deepseek.auth --account NAME). Override the location
with DEEPSEEK_ACCOUNTS_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def accounts_dir() -> Path:
    return Path(os.getenv(
        "DEEPSEEK_ACCOUNTS_DIR", str(ROOT / "session" / "accounts")
    ))
