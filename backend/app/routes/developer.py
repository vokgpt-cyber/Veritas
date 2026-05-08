"""Developer settings API for controlled VERITAS pipeline tuning."""
from __future__ import annotations

import json
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.app.auth import TokenData, auth_config, get_current_user
from backend.core.audit import AuditAction, audit_log
from backend.core.developer_settings import (
    find_catalog_entry,
    load_settings,
    save_settings,
)

router = APIRouter(prefix="/api/developer", tags=["developer"])


def require_developer(user: TokenData = Depends(get_current_user)) -> TokenData:
    """Allow admins and developer-role users into the developer panel."""
    if user.role not in {"admin", "developer"} and user.username != auth_config.default_username:
        raise HTTPException(status_code=403, detail="Developer access required")
    return user


class ActivePipelineRequest(BaseModel):
    profile: Optional[str] = None
    asr: Optional[str] = None
    diarization: Optional[str] = None
    llm: Optional[str] = None


class PromptUpdateRequest(BaseModel):
    enabled: bool
    system_template: str = ""
    user_template: str = ""
    notes: str = ""


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _get_orchestrator():
    from backend.app.main import orchestrator

    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


def _apply_runtime_model_settings(settings: dict[str, Any]) -> None:
    """Apply selected model manifest entries to the in-memory config."""
    orchestrator = _get_orchestrator()
    config = orchestrator._config
    active = settings.get("active", {})

    asr = find_catalog_entry(settings, "asr", active.get("asr", ""))
    if asr:
        config.asr.engine = str(asr.get("engine") or config.asr.engine)
        if asr.get("whisper_model"):
            config.asr.whisper_model = str(asr["whisper_model"])

    diar = find_catalog_entry(
        settings, "diarization", active.get("diarization", "")
    )
    if diar:
        config.diarization.engine = "pyannote"
        if diar.get("model"):
            config.diarization.pyannote_model = str(diar["model"])

    llm = find_catalog_entry(settings, "llm", active.get("llm", ""))
    if llm:
        provider = str(llm.get("provider") or "ollama")
        if provider not in {"ollama", "openai_compatible"}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported LLM provider: {provider}",
            )
        config.summarization.provider = provider
        if llm.get("model"):
            config.summarization.ollama_model = str(llm["model"])
        if provider == "openai_compatible":
            config.summarization.openai_base_url = str(
                llm.get("base_url") or config.summarization.openai_base_url
            )
        elif llm.get("base_url"):
            config.summarization.ollama_base_url = str(llm["base_url"])
        if llm.get("structured_outputs") is not None:
            config.summarization.structured_outputs = bool(
                llm.get("structured_outputs")
            )

    # Existing engine instances captured config-derived provider/model
    # fields at construction time. Drop them so the next job instantiates
    # fresh engines with the selected manifest.
    if getattr(orchestrator, "_summarization", None) is not None:
        try:
            if orchestrator._summarization.is_loaded:
                orchestrator._summarization.unload()
        except Exception:
            pass
        orchestrator._summarization = None
    if getattr(orchestrator, "_asr", None) is not None:
        try:
            if orchestrator._asr.is_loaded:
                orchestrator._asr.unload()
        except Exception:
            pass
        orchestrator._asr = None
        orchestrator._asr_engine_name = None
    if getattr(orchestrator, "_diarization", None) is not None:
        try:
            if orchestrator._diarization.is_loaded:
                orchestrator._diarization.unload()
        except Exception:
            pass
        orchestrator._diarization = None


@router.get("/settings")
async def get_developer_settings(
    user: TokenData = Depends(require_developer),
) -> dict[str, Any]:
    """Return model catalog, active selections, and prompt overrides."""
    return load_settings()


@router.put("/settings/active")
async def update_active_pipeline(
    payload: ActivePipelineRequest,
    request: Request,
    user: TokenData = Depends(require_developer),
) -> dict[str, Any]:
    """Persist and apply selected pipeline model entries."""
    settings = load_settings()
    active = dict(settings.get("active", {}))
    updates = payload.model_dump(exclude_none=True)

    for category in ("asr", "diarization", "llm"):
        if category in updates:
            model_id = updates[category]
            if find_catalog_entry(settings, category, model_id) is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown {category} model id: {model_id}",
                )
            active[category] = model_id
    if "profile" in updates:
        profile_ids = {p.get("id") for p in settings.get("profiles", [])}
        if updates["profile"] not in profile_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown profile id: {updates['profile']}",
            )
        active["profile"] = updates["profile"]

    settings["active"] = active
    save_settings(settings)
    _apply_runtime_model_settings(settings)
    audit_log(
        action=AuditAction.SYSTEM_CONFIG_CHANGE,
        username=user.username,
        ip_address=_client_ip(request),
        resource_type="developer_settings",
        details={"active": active},
    )
    return settings


@router.put("/prompts/{prompt_id}")
async def update_prompt_override(
    prompt_id: str,
    payload: PromptUpdateRequest,
    request: Request,
    user: TokenData = Depends(require_developer),
) -> dict[str, Any]:
    """Update one prompt override. Built-in prompt is used when disabled."""
    settings = load_settings()
    prompts = settings.get("prompts", [])
    for prompt in prompts:
        if prompt.get("id") == prompt_id:
            prompt.update(payload.model_dump())
            prompt["version"] = int(prompt.get("version", 1)) + 1
            prompt["updated_at"] = datetime.utcnow().isoformat()
            prompt["updated_by"] = user.username
            save_settings(settings)
            audit_log(
                action=AuditAction.SYSTEM_CONFIG_CHANGE,
                username=user.username,
                ip_address=_client_ip(request),
                resource_type="prompt",
                resource_id=prompt_id,
                details={"enabled": payload.enabled},
            )
            return prompt
    raise HTTPException(status_code=404, detail=f"Prompt not found: {prompt_id}")


def _safe_job_summary(job_id: str) -> dict[str, Any]:
    orchestrator = _get_orchestrator()
    job = orchestrator.get_job(job_id)
    if job is None:
        meeting_dir = orchestrator.get_meeting_dir(job_id)
        if not meeting_dir.exists():
            raise HTTPException(status_code=404, detail="Meeting not found")
        job_data: dict[str, Any] = {"id": job_id, "state": "unknown"}
    else:
        job_data = job.model_dump(mode="json")
        # Remove document-specific context from the technical export.
        job_data["context"] = "[redacted]"
        job_data["court_participants"] = "[redacted]"
        job_data["court_dictionary"] = "[redacted]"

    config = orchestrator._config
    return {
        "job": job_data,
        "pipeline": {
            "asr_engine": config.asr.engine,
            "gigaam_batch_size": config.asr.gigaam_batch_size,
            "diarization_engine": config.diarization.engine,
            "pyannote_model": config.diarization.pyannote_model,
            "diarization_min_speakers": config.diarization.min_speakers,
            "diarization_max_speakers": config.diarization.max_speakers,
            "llm_provider": getattr(config.summarization, "provider", "ollama"),
            "llm_model": config.summarization.ollama_model,
            "llm_num_ctx": config.summarization.ollama_num_ctx,
            "llm_max_new_tokens": config.summarization.max_new_tokens,
            "admin_max_new_tokens": config.summarization.admin_max_new_tokens,
            "temperature": config.summarization.temperature,
            "top_p": config.summarization.top_p,
            "structured_outputs": config.summarization.structured_outputs,
            "admin_recall_pass": config.summarization.admin_recall_pass,
        },
        "developer_settings": load_settings().get("active", {}),
        "generated_at": datetime.utcnow().isoformat(),
        "privacy": (
            "Transcript text, context documents, dictionary terms, and protocol "
            "content are intentionally omitted from this technical log bundle."
        ),
    }


def _audit_events_for_job(job_id: str) -> list[dict[str, Any]]:
    """Return sanitized audit events for one job, without document content."""
    try:
        from backend.app.paths import resolve_user_path

        audit_dir = resolve_user_path("data/audit")
    except Exception:
        audit_dir = Path("data/audit")

    events: list[dict[str, Any]] = []
    for path in sorted(audit_dir.glob("audit_*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("resource_id") != job_id:
                continue
            event.pop("ip_address", None)
            details = event.get("details")
            if isinstance(details, dict):
                for key in (
                    "context",
                    "court_participants",
                    "court_dictionary",
                    "transcript",
                    "text",
                    "summary",
                ):
                    if key in details:
                        details[key] = "[redacted]"
            events.append(event)
    return events[-200:]


@router.get("/meetings/{job_id}/technical-log")
async def export_technical_log(
    job_id: str,
    user: TokenData = Depends(require_developer),
) -> FileResponse:
    """Download a redacted technical ZIP for one processing session."""
    summary = _safe_job_summary(job_id)
    temp_dir = Path(tempfile.mkdtemp(prefix="veritas_techlog_"))
    zip_path = temp_dir / f"veritas_technical_log_{job_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "run_summary.json",
            json.dumps(summary, indent=2, ensure_ascii=False),
        )
        zf.writestr(
            "developer_settings_active.json",
            json.dumps(
                load_settings().get("active", {}),
                indent=2,
                ensure_ascii=False,
            ),
        )
        zf.writestr(
            "audit_events_redacted.json",
            json.dumps(_audit_events_for_job(job_id), indent=2, ensure_ascii=False),
        )
        zf.writestr(
            "README.txt",
            (
                "VERITAS technical log bundle.\n"
                "No transcript text, uploaded document text, court dictionary, "
                "or protocol content is included.\n"
            ),
        )
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_path.name,
    )
