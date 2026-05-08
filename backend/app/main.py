"""FastAPI application entry point for EPAM VERITAS v1.0."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging config. Must run BEFORE our modules import and create their own
# loggers, otherwise Python's default (WARNING-only to stderr) swallows
# every INFO line from backend.*. Uvicorn's own logger is configured
# separately by uvicorn itself and is unaffected by this call.
# Format shows timestamp, level, logger name, message — critical for
# pipeline diagnostics ("Language mix: ...", "final ASR engine = ...",
# "Raw LLM output: ..." etc.).
# EPAM_LOG_LEVEL env var overrides (DEBUG/INFO/WARNING/ERROR).
# ---------------------------------------------------------------------------
_log_level_name = os.environ.get("EPAM_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # override any handlers Uvicorn or other libs added first
)

# ---------------------------------------------------------------------------
# Windows DLL search path for FFmpeg shared libraries.
# Required by torchcodec (pyannote 4.0 audio loader) on Windows because
# Python 3.8+ no longer resolves dependency DLLs through PATH. The directory
# must contain avcodec-*.dll / avformat-*.dll etc. (FFmpeg 7 shared build).
# Override via EPAM_FFMPEG_DLL_DIR env var; otherwise probe common locations.
# ---------------------------------------------------------------------------
if os.name == "nt":
    _candidates = [
        os.environ.get("EPAM_FFMPEG_DLL_DIR"),
        r"C:\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin",
        r"C:\ffmpeg-7-shared\ffmpeg-n7.1-latest-win64-gpl-shared\bin",
    ]
    for _p in _candidates:
        if _p and os.path.isdir(_p):
            try:
                os.add_dll_directory(_p)
                logging.getLogger(__name__).info(
                    "Registered FFmpeg DLL directory for torchcodec: %s", _p
                )
                break
            except Exception:
                pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import AppConfig
from backend.app.routes import auth, developer, meetings, speakers, system, users
from backend.core.audit import AuditAction, audit_log
from backend.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)
APP_VERSION = "1.0.0"

# Global orchestrator instance
orchestrator: Orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.

    Loads configuration and creates orchestrator on startup.
    Releases resources on shutdown.
    """
    global orchestrator, _cors_origins

    # Startup
    logger.info("Starting EPAM VERITAS v%s...", APP_VERSION)

    # Load configuration from environment and optional YAML
    # Try config/settings.yaml first, then settings.yaml in cwd
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        config_path = Path("settings.yaml")
    if config_path.exists():
        config = AppConfig.load_from_yaml(config_path)
        logger.info(f"Configuration loaded from {config_path}")
    else:
        config = AppConfig()
        # Even when no YAML is present we still want per-user paths
        # applied to the hardcoded defaults — load_from_yaml normally
        # does this, but the fallback path needs the same treatment.
        config._apply_user_data_root()
        logger.warning("No settings.yaml found — using hardcoded defaults")

    # Per-user data isolation (Sprint 2026-04-30, task #34): create
    # the user data root + standard subdirs on first run so audit/
    # encryption/cleanup don't fail on a missing directory.
    from backend.app.paths import (
        ensure_user_data_dirs,
        get_user_data_root,
    )
    user_data_root = ensure_user_data_dirs(
        "data/meetings",
        "data/backups",
        "data/audit",
        "data/temp",
    )
    logger.info(f"User data root: {user_data_root}")

    # Update CORS origins from config
    if config.security.cors_production_origins:
        _cors_origins.clear()
        _cors_origins.extend(config.security.cors_production_origins)
        logger.info(f"CORS production origins: {_cors_origins}")
    elif config.server.cors_origins:
        _cors_origins.clear()
        _cors_origins.extend(config.server.cors_origins)
        logger.info(f"CORS origins from config: {_cors_origins}")

    # Create orchestrator
    orchestrator = Orchestrator(config)
    logger.info("Orchestrator initialized")

    # Ensure output directory exists
    Path(config.output.output_dir).mkdir(parents=True, exist_ok=True)

    audit_log(action=AuditAction.SYSTEM_STARTUP, details={"version": APP_VERSION})

    yield

    # Shutdown
    audit_log(action=AuditAction.SYSTEM_SHUTDOWN)
    logger.info("Shutting down EPAM VERITAS...")
    if orchestrator:
        # Release any loaded models
        try:
            orchestrator._ensure_engines_loaded()
            if orchestrator._asr and orchestrator._asr.is_loaded:
                orchestrator._asr.unload()
            if orchestrator._diarization and orchestrator._diarization.is_loaded:
                orchestrator._diarization.unload()
            if orchestrator._summarization and orchestrator._summarization.is_loaded:
                orchestrator._summarization.unload()
        except Exception as e:
            logger.warning(f"Error releasing models during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="EPAM VERITAS",
    version=APP_VERSION,
    description="Verbal Intelligence, Transcription and Summarization",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


# Configure CORS middleware — origins are set dynamically from config
# In production: set EPAM_SECURITY_CORS_PRODUCTION_ORIGINS or restrict
# server.cors_origins in settings.yaml.
# The lifespan handler updates CORS origins from loaded config.
_cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


def _get_cors_origins() -> list[str]:
    """Return current CORS origins (updated from config at startup)."""
    return _cors_origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
    expose_headers=["Content-Disposition"],
    max_age=3600,
)


# Health check endpoint
@app.get("/api/health", tags=["system"])
async def health():
    """
    Health check endpoint.

    Returns:
        Dictionary with status and version
    """
    return {
        "status": "ok",
        "version": APP_VERSION,
        "service": "epam-veritas",
    }


# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(meetings.router)
app.include_router(speakers.router)
app.include_router(system.router)
app.include_router(developer.router)

# Include WebSocket for real-time updates
from backend.app.websocket import app as ws_app

app.include_router(ws_app)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API documentation link."""
    return {
        "message": f"EPAM VERITAS v{APP_VERSION}",
        "docs": "/api/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    # Backend port. The launcher passes EPAM_BACKEND_PORT (default
    # 8765) and uvicorn's own --port arg; this fallback only kicks in
    # if someone runs `python -m backend.app.main` directly without
    # the launcher.
    _port = int(os.environ.get("EPAM_BACKEND_PORT", "8765"))

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=_port,
        reload=False,
        workers=1,
        log_level="info",
    )
