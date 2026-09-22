#!/usr/bin/env bash
# DeepSeek Free API — one-command launcher (Linux / macOS / WSL)
#   ./start.sh
# Creates the venv, installs dependencies, checks the login session and starts
# the OpenAI-compatible server with the web dashboard at http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

CYAN='\033[1;36m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; RED='\033[1;31m'; NC='\033[0m'
say() { echo -e "${CYAN}==>${NC} $1"; }
ok()  { echo -e "${GREEN} ✓${NC} $1"; }
warn(){ echo -e "${YELLOW} ⚠${NC} $1"; }

echo -e "${CYAN}"
cat <<'BANNER'
┌──────────────────────────────────────────────────┐
│        DeepSeek Free API  ·  deepseek-chat / expert        │
│     OpenAI-compatible bridge for chat.deepseek.com        │
└──────────────────────────────────────────────────┘
BANNER
echo -e "${NC}"

# ---- 1) Python ------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo -e "${RED}✗ python3 not found — install Python 3.9+ first${NC}"; exit 1
fi
ok "python3 $(python3 -V | cut -d' ' -f2)"

# ---- 2) venv + dependencies -------------------------------------------------
if [ ! -d .venv ]; then
  say "creating virtual environment (.venv)…"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if ! python -c "import fastapi, httpx, wasmtime" >/dev/null 2>&1; then
  say "installing dependencies (first run takes a minute)…"
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
fi
ok "dependencies ready"

# ---- 3) Playwright browser (only needed for the login step) ------------------
if ! python -c "from playwright.sync_api import sync_playwright" >/dev/null 2>&1; then
  warn "playwright import failed — the login step needs it"
elif [ ! -d "$HOME/.cache/ms-playwright" ] && [ ! -d "$HOME/Library/Caches/ms-playwright" ]; then
  say "installing chromium for playwright (login browser)…"
  python -m playwright install chromium >/dev/null 2>&1 || true
fi

# ---- 4) account check --------------------------------------------------------
ACC_DIR="session/accounts"
ACC_COUNT=0
if [ -d "$ACC_DIR" ]; then
  ACC_COUNT=$(ls "$ACC_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
fi
if [ "$ACC_COUNT" -eq 0 ] && [ ! -f session/session.json ] && [ -z "${DEEPSEEK_TOKENS:-}" ]; then
  warn "no DeepSeek account found — chat endpoints will return 503 until you log in once:"
  echo "      ${YELLOW}python -m deepseek.auth --account main${NC}"
  echo "      (opens a browser once; handles the human-check; session is saved)"
  echo "      add more accounts to bypass rate limits:"
  echo "      ${YELLOW}python -m deepseek.auth --account work${NC}"
  echo
fi

# ---- 5) .env ------------------------------------------------------------------
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  ok "created .env from .env.example"
fi
if [ -f .env ]; then set -a; . ./.env; set +a; fi

HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-8000}"
say "starting server on http://${HOST}:${PORT}"
echo -e "    dashboard : ${GREEN}http://${HOST}:${PORT}/${NC}"
echo -e "    agent API : ${GREEN}http://${HOST}:${PORT}/v1${NC}   (base_url for any OpenAI client)"
echo
exec python app.py
