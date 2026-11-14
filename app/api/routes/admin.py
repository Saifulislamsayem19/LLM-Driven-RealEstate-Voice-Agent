"""
Admin endpoints — API key management panel and requirements viewer.
Both require HTTP Basic auth (ADMIN_PASSWORD) and are hidden until it is set.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from app.core.config import settings
from app.core.security import require_admin, require_same_origin
from app.utils.data_manager import convert_numpy_types

router = APIRouter(tags=["admin"], dependencies=[Depends(require_admin)])

# OpenAI keys are letters, digits, "-" and "_" — anything else (e.g. a newline) could inject .env lines
_API_KEY_PATTERN = re.compile(r"^sk-[A-Za-z0-9_\-]{20,300}$")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _restart_application() -> None:
    print("🔄 Restarting application with new API key…")
    time.sleep(2)
    python = sys.executable
    os.execl(python, python, *sys.argv)


def _read_env_vars() -> dict[str, str]:
    env_path = settings.BASE_DIR / ".env"
    env_vars: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()
    return env_vars


def _update_env_vars(updates: dict[str, str]) -> None:
    """Set keys in .env in place, keeping comments and other settings."""
    env_path = settings.BASE_DIR / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    pending = dict(updates)
    for i, line in enumerate(lines):
        key = line.partition("=")[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in pending:
            lines[i] = f"{key}={pending.pop(key)}"
    lines += [f"{k}={v}" for k, v in pending.items()]
    env_path.write_text("\n".join(lines) + "\n")


def _mask(key: str) -> str:
    return f"…{key[-4:]}" if len(key) > 8 else ""


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/requirements")
async def view_requirements():
    """Admin endpoint to view saved user requirements."""
    req_file = str(settings.USER_REQUIREMENTS_FILE)
    if not os.path.exists(req_file):
        raise HTTPException(status_code=404, detail="No requirements data available")
    try:
        df = pd.read_csv(req_file)
        return {"requirements": convert_numpy_types(df.to_dict(orient="records"))}
    except Exception as exc:
        print(f"/requirements error: {exc!r}")
        raise HTTPException(status_code=500, detail="Error reading requirements data")


@router.api_route("/api-@dmin", methods=["GET", "POST"])
async def admin_panel(
    request: Request,
    success: bool = Query(False),
    restarting: bool = Query(False),
    api_key: str = Form(None),
):
    templates = request.app.state.templates

    if request.method == "POST":
        require_same_origin(request)
        try:
            api_key = (api_key or "").strip()
            if not _API_KEY_PATTERN.match(api_key):
                raise ValueError("That doesn't look like a valid OpenAI API key (it should start with sk-).")

            _update_env_vars({"OPENAI_API_KEY": api_key, "APP_RESTART_PENDING": "True"})

            os.environ["OPENAI_API_KEY"] = api_key
            print("✅ OpenAI API key updated via admin panel")

            threading.Thread(target=_restart_application, daemon=True).start()

            return templates.TemplateResponse(
                request,
                "admin.html",
                {"request": request, "success": True, "restarting": True, "current_api_key": _mask(api_key)},
            )

        except Exception as exc:
            print(f"Admin key update failed: {exc!r}")
            message = str(exc) if isinstance(exc, ValueError) else "Could not update the API key."
            return templates.TemplateResponse(
                request,
                "admin.html",
                {"request": request, "success": False, "error": message,
                 "current_api_key": _mask(os.getenv("OPENAI_API_KEY", ""))},
            )

    # GET
    current_api_key = _mask(os.getenv("OPENAI_API_KEY", ""))
    just_restarted = False
    if not restarting:
        env_vars = _read_env_vars()
        just_restarted = env_vars.get("APP_RESTART_PENDING") == "True"

    return templates.TemplateResponse(
        request,
        "admin.html",
        {
            "request": request,
            "success": success or just_restarted,
            "restarting": restarting,
            "current_api_key": current_api_key,
        },
    )
