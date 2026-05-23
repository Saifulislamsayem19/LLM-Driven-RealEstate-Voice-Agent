"""
Admin endpoints — API key management panel and requirements viewer.
Logic preserved exactly from the original app.py.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from app.core.config import settings
from app.utils.data_manager import convert_numpy_types

router = APIRouter(tags=["admin"])


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


def _write_env_vars(env_vars: dict[str, str]) -> None:
    env_path = settings.BASE_DIR / ".env"
    env_path.write_text("\n".join(f"{k}={v}" for k, v in env_vars.items()) + "\n")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/requirements")
async def view_requirements():
    """Admin endpoint to view saved user requirements."""
    req_file = str(settings.USER_REQUIREMENTS_FILE)
    if not os.path.exists(req_file):
        raise HTTPException(status_code=404, detail="No requirements data available")
    try:
        df = pd.read_csv(req_file)
        return {"requirements": df.to_dict(orient="records")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading requirements data: {exc}")


@router.api_route("/api-@dmin", methods=["GET", "POST"])
async def admin_panel(
    request: Request,
    success: bool = Query(False),
    restarting: bool = Query(False),
    api_key: str = Form(None),
):
    templates = request.app.state.templates

    if request.method == "POST":
        try:
            if not api_key:
                raise ValueError("API key cannot be empty")

            env_vars = _read_env_vars()
            old_key = env_vars.get("OPENAI_API_KEY", "")
            env_vars["OPENAI_API_KEY"] = api_key
            env_vars["APP_RESTART_PENDING"] = "True"
            _write_env_vars(env_vars)

            os.environ["OPENAI_API_KEY"] = api_key
            print(f"✅ OpenAI API key updated from {old_key[:10]}… to {api_key[:10]}…")

            threading.Thread(target=_restart_application, daemon=True).start()

            return templates.TemplateResponse(
                "admin.html",
                {"request": request, "success": True, "restarting": True, "current_api_key": api_key},
            )

        except Exception as exc:
            current_api_key = os.getenv("OPENAI_API_KEY", "")
            return templates.TemplateResponse(
                "admin.html",
                {"request": request, "success": False, "error": str(exc), "current_api_key": current_api_key},
            )

    # GET
    current_api_key = os.getenv("OPENAI_API_KEY", "")
    just_restarted = False
    if not restarting:
        env_vars = _read_env_vars()
        just_restarted = env_vars.get("APP_RESTART_PENDING") == "True"

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "success": success or just_restarted,
            "restarting": restarting,
            "current_api_key": current_api_key,
        },
    )
