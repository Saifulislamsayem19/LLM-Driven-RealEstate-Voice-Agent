"""
Application factory — wires middleware, routers, and lifespan together.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import audio, chat, admin, health
from app.services.agent import agent
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    _clear_restart_flag()

    if not agent.initialize():
        raise RuntimeError("Failed to initialise data or vector database.")
    if not agent.setup_chain():
        raise RuntimeError("Failed to set up LangChain workflow.")

    yield

    print("Shutting down HomeFind AI…")


def _clear_restart_flag() -> None:
    """Remove APP_RESTART_PENDING from .env on clean startup."""
    env_path = settings.BASE_DIR / ".env"
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text().splitlines(keepends=True)
        env_path.write_text(
            "".join(l for l in lines if not l.strip().startswith("APP_RESTART_PENDING="))
        )
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not clear restart flag: {exc}")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="AI-powered real estate assistant",
        version=settings.VERSION,
        lifespan=lifespan,
    )

    # ── Middleware ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files & templates ────────────────────────────────────────────
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")
    app.state.templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

    # ── Routers ─────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(audio.router, prefix="/audio", tags=["audio"])
    app.include_router(admin.router)

    return app
