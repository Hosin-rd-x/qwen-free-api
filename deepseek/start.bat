@echo off
REM ==========================================================================
REM  DeepSeek-Free-API (Godde3s) — one-command launcher (Windows)
REM
REM  First run:  double-click start.bat   (or: start.bat in a terminal)
REM ==========================================================================
setlocal
cd /d "%~dp0"

if not exist .env (
    copy .env.example .env >nul
    echo.
    echo  !  First-run setup: a starter .env was created.
    echo     For multi-account power, paste your DeepSeek token(s):
    echo        notepad .env
    echo        DEEPSEEK_TOKENS=token1,token2,token3
    echo        ^(token: chat.deepseek.com ^> F12 ^> Console ^>
    echo          JSON.parse^(localStorage.userToken^).value^)
    echo.
)

REM pick python
set PY=python
where python >nul 2>nul || set PY=py
where %PY% >nul 2>nul || (
    echo X  Python not found - install Python 3.9+ from https://python.org
    pause & exit /b 1
)

if not exist .venv (
    echo ^> Creating virtual environment ^(one-time^)...
    %PY% -m venv .venv || (echo X  venv failed & pause & exit /b 1)
)
call .venv\Scripts\activate.bat

if not exist .venv\.deps-installed (
    echo ^> Installing dependencies ^(one-time, ~1 min^)...
    pip install -q -r requirements.txt || (echo X  pip failed & pause & exit /b 1)
    type nul > .venv\.deps-installed
)

if "%DEEPSEEK_TOKENS%"=="" if not exist session\session.json (
    echo  !  No tokens / session yet. Run ONCE and sign in in the window:
    echo        python -m deepseek.auth
    echo.
)

echo ^> Starting DeepSeek-Free-API ...
echo    Dashboard:  http://localhost:%PORT%/          ^<- open this!
echo    OpenAI:     http://localhost:%PORT%/v1/chat/completions
echo    Anthropic:  http://localhost:%PORT%/v1/messages
echo    Stop:       CTRL+C
echo.
python app.py %*
pause
