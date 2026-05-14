"""API routes for system information and health checks."""

import logging
import psutil
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from backend.app.auth import TokenData, get_current_user, require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])


def get_orchestrator():
    """Get the global orchestrator instance."""
    from backend.app.main import orchestrator

    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


def get_vram_manager():
    """Get VRAM manager singleton."""
    from backend.core.vram_manager import VRAMManager

    return VRAMManager()


@router.get("/health")
async def system_health(user: TokenData = Depends(get_current_user)) -> dict:
    """
    Get system health status.

    Returns:
        Dictionary with system health metrics
    """
    try:
        # Get CPU and memory usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Get orchestrator status
        orchestrator = get_orchestrator()

        return {
            "status": "healthy",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "active_jobs": len(orchestrator.list_jobs()),
        }

    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")


@router.get("/gpu")
async def gpu_status(user: TokenData = Depends(get_current_user)) -> dict:
    """
    Get GPU status and VRAM usage.

    Returns:
        Dictionary with GPU metrics:
        - vram_used_gb: float
        - vram_total_gb: float
        - vram_free_gb: float
        - gpu_utilization: float (0-100%)
        - temperature: float (°C)
        - device_name: str
        - current_model: Optional[str]
        - is_available: bool
    """
    try:
        vram = get_vram_manager()
        status = vram.get_status()

        return {
            **status,
            "is_available": vram.is_gpu_available,
        }

    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get GPU status")


@router.post("/gpu/release")
async def release_gpu_memory(user: TokenData = Depends(require_admin)) -> dict:
    """
    Force release all GPU memory.

    Use with caution - will unload any loaded models.

    Returns:
        Confirmation dictionary
    """
    try:
        vram = get_vram_manager()
        vram.release_all()

        status = vram.get_status()

        return {
            "status": "released",
            "vram_free_gb": status["vram_free_gb"],
            "vram_total_gb": status["vram_total_gb"],
        }

    except Exception as e:
        logger.error(f"Failed to release GPU memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to release GPU memory")


@router.get("/features")
async def system_features(
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Report which optional features are available + currently default-on.

    Used by the upload UI to show/hide opt-in toggles. Each feature
    has:
      - `available` — can it be turned on at all (deps installed)?
      - `default_enabled` — is the global config default true?
    """
    orchestrator = get_orchestrator()
    config = orchestrator._config

    # WhisperX availability — try-import without keeping the module
    # loaded.
    try:
        import whisperx  # noqa: F401  type: ignore[import-not-found]
        whisperx_available = True
    except Exception:
        whisperx_available = False

    # text2num is required for normalize_numbers (Tier 1 leftover).
    try:
        import text_to_num  # noqa: F401  type: ignore[import-not-found]
        text2num_available = True
    except Exception:
        text2num_available = False

    pp = getattr(config, "postprocessing", None)
    summ = config.summarization

    return {
        "llm_correction": {
            "available": True,  # uses Ollama which is required anyway
            "default_enabled": bool(getattr(pp, "llm_correction", False))
            if pp else False,
            "label_ru": "Улучшенное распознавание (LLM-коррекция)",
            "description_ru": (
                "Исправляет очевидные ошибки распознавания через LLM. "
                "Замедляет на ~1 мин."
            ),
        },
        "polish_pass": {
            "available": True,
            "default_enabled": bool(getattr(summ, "polish_pass", False)),
            "label_ru": "Полировка форматирования протокола",
            "description_ru": (
                "Только для административных встреч. "
                "Замедляет на ~30 сек."
            ),
        },
        "whisperx": {
            "available": whisperx_available,
            "default_enabled": False,
            "label_ru": "WhisperX (точные таймкоды слов)",
            "description_ru": (
                "Дополнительный движок ASR с фонетическим выравниванием. "
                "Лучше для записей с перебиваниями."
            ),
            "install_hint_ru": (
                "Запустите INSTALL_WHISPERX.bat в корне проекта."
            ),
        },
        "normalize_numbers": {
            "available": text2num_available,
            "default_enabled": bool(getattr(pp, "normalize_numbers", False))
            if pp else False,
            "label_ru": "Нормализация чисел (отключено)",
            "description_ru": (
                "Преобразует слова в цифры. Отключено по умолчанию из-за "
                "ошибок на порядковых ('двадцать пятого' -> '20 пятого')."
            ),
        },
        "cloud_compare": {
            "available": bool(
                getattr(config.security, "cloud_compare_enabled", False)
            ),
            "default_enabled": bool(
                getattr(config.security, "cloud_compare_enabled", False)
            ),
            "label_ru": "Cloud comparison",
            "description_ru": (
                "Disabled by default. Sends transcript content to an "
                "external API, so use only for explicit testing."
            ),
        },
    }


@router.get("/info")
async def system_info(user: TokenData = Depends(get_current_user)) -> dict:
    """
    Get detailed system information.

    Returns:
        Dictionary with system configuration
    """
    try:
        orchestrator = get_orchestrator()
        config = orchestrator._config

        return {
            "version": "2.0.0",
            "service": "epam-transcriber",
            "config": {
                "asr_model": config.asr.model,
                "diarization_model": config.diarization.embedding_model,
                "summarization_provider": getattr(
                    config.summarization, "provider", "ollama"
                ),
                "summarization_base_url": (
                    config.summarization.openai_base_url
                    if getattr(config.summarization, "provider", "ollama")
                    == "openai_compatible"
                    else config.summarization.ollama_base_url
                ),
                "summarization_model": config.summarization.ollama_model,
                "output_dir": config.output.output_dir,
            },
            "environment": {
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            },
        }

    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system info")


@router.get("/stats")
async def system_stats(user: TokenData = Depends(get_current_user)) -> dict:
    """
    Get system statistics.

    Returns:
        Dictionary with system statistics
    """
    try:
        orchestrator = get_orchestrator()
        jobs = orchestrator.list_jobs()

        # Calculate job statistics
        completed = sum(1 for j in jobs if j.state.value == "completed")
        failed = sum(1 for j in jobs if j.state.value == "error")
        processing = sum(1 for j in jobs if j.state.value in ["transcribing", "diarizing", "summarizing"])

        return {
            "total_jobs": len(jobs),
            "completed_jobs": completed,
            "failed_jobs": failed,
            "processing_jobs": processing,
            "pending_jobs": len(jobs) - completed - failed - processing,
        }

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system stats")


@router.get("/audit")
async def list_audit_entries(
    limit: int = 200,
    offset: int = 0,
    action: str = "",
    user_filter: str = "",
    since: str = "",
    user: TokenData = Depends(get_current_user),
) -> dict:
    """List recent audit-log entries (admin-only).

    Sprint 2026-04-30 (task #42). Reads JSONL files from the user's
    audit log directory (per-user under %APPDATA%\\VERITAS\\<user>\\
    after task #34) and returns the most recent entries first. Filters
    are applied server-side so the page can paginate large logs
    without round-tripping the whole history.

    Args:
        limit: Max entries to return (1..500).
        offset: Skip the first N matching entries.
        action: Filter by exact action name (e.g. "auth.login_success").
        user_filter: Substring match on the entry's username field.
        since: ISO timestamp; only entries with timestamp >= this.
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")

    limit = max(1, min(500, int(limit)))
    offset = max(0, int(offset))

    try:
        from backend.core.audit import get_audit_logger
        logger_instance = get_audit_logger()
        log_dir = logger_instance._log_dir  # noqa: SLF001
    except Exception as e:
        logger.error(f"Failed to access audit logger: {e}")
        raise HTTPException(
            status_code=500, detail="Audit logger unavailable",
        ) from e

    if not log_dir.exists():
        return {"total": 0, "entries": []}

    import json
    import re

    entries: list[dict] = []
    # Iterate newest file → oldest, lines newest at bottom. We read
    # all matching files but cap the total entries at a sane upper
    # bound so a multi-GB log doesn't OOM the request.
    files = sorted(log_dir.glob("audit_*.jsonl"), reverse=True)
    HARD_LINE_CAP = 10_000
    lines_seen = 0
    for f in files:
        if lines_seen >= HARD_LINE_CAP:
            break
        try:
            with open(f, "r", encoding="utf-8") as fp:
                file_lines = fp.readlines()
        except OSError:
            continue
        # Reverse within-file so newest-first ordering is preserved.
        for line in reversed(file_lines):
            lines_seen += 1
            if lines_seen >= HARD_LINE_CAP:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if action and obj.get("action") != action:
                continue
            if user_filter:
                username = (obj.get("username") or "").lower()
                if user_filter.lower() not in username:
                    continue
            if since:
                ts = obj.get("timestamp", "")
                if ts < since:
                    continue
            entries.append(obj)

    total = len(entries)
    paginated = entries[offset : offset + limit]
    return {"total": total, "entries": paginated}
