"""API routes for meeting processing and management."""

import asyncio
import json as json_lib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from backend.app.auth import TokenData, get_current_user
from backend.app.models import (
    AlignedSegment,
    MeetingJob,
    MeetingProtocol,
    MeetingType,
    PipelineState,
)
from backend.core.audit import AuditAction, audit_log

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/meetings", tags=["meetings"])

# Default maximum extracted text from each context file. The runtime
# value comes from summarization.context_file_max_chars when available.
_DEFAULT_CONTEXT_FILE_MAX_CHARS = 20000
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_DEFAULT_MAX_MEDIA_UPLOAD_BYTES = 5 * 1024 * 1024 * 1024
_DEFAULT_MAX_CONTEXT_UPLOAD_BYTES = 50 * 1024 * 1024
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid integer env %s=%r; using %s", name, raw, default)
        return default


def _display_upload_filename(filename: str | None, default_name: str) -> str:
    raw = (filename or "").replace("\\", "/")
    name = Path(raw).name.strip()
    return name or default_name


def _safe_upload_filename(filename: str | None, default_name: str) -> str:
    name = _display_upload_filename(filename, default_name)
    safe_name = _SAFE_FILENAME_RE.sub("_", name).strip(" .")
    if not safe_name:
        safe_name = default_name
    if len(safe_name) > 180:
        suffix = Path(safe_name).suffix[:20]
        stem = Path(safe_name).stem[:140].strip(" .") or "upload"
        safe_name = f"{stem}{suffix}"
    return safe_name


async def _write_upload_to_path(
    upload: UploadFile,
    path: Path,
    *,
    max_bytes: int,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with open(path, "wb") as target:
            while True:
                chunk = await upload.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    target.close()
                    path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded file is too large",
                    )
                target.write(chunk)
    except HTTPException:
        raise
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return total


def _extract_text_from_file(
    file_path: Path,
    max_chars: int = _DEFAULT_CONTEXT_FILE_MAX_CHARS,
) -> str:
    """Extract text content from uploaded context files (DOCX, XLSX, PDF, TXT).

    Returns truncated text suitable for injection into LLM prompts.
    All processing is CPU-only - no VRAM needed.
    """
    ext = file_path.suffix.lower()
    text = ""

    try:
        if ext == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="replace")

        elif ext == ".docx":
            from docx import Document
            doc = Document(str(file_path))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif ext in (".xlsx", ".xls"):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(str(file_path), read_only=True, data_only=True)
                rows = []
                for sheet in wb.worksheets:
                    for row in sheet.iter_rows(values_only=True):
                        cells = [str(c) if c is not None else "" for c in row]
                        line = " | ".join(c for c in cells if c)
                        if line:
                            rows.append(line)
                text = "\n".join(rows)
                wb.close()
            except ImportError:
                logger.warning("openpyxl not installed - cannot read Excel files")

        elif ext == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(str(file_path))
                pages = []
                for page in reader.pages[:10]:  # First 10 pages max
                    pages.append(page.extract_text() or "")
                text = "\n".join(pages)
            except ImportError:
                logger.warning("PyPDF2 not installed - cannot read PDF files")

    except Exception as e:
        logger.warning(f"Failed to extract text from {file_path.name}: {e}")

    # Truncate to keep within prompt budget
    max_chars = max(1000, int(max_chars))
    if len(text) > max_chars:
        text = text[:max_chars] + "..."

    return text.strip()


def _split_dictionary_terms(raw: str) -> list[str]:
    """Split operator-provided court dictionary text into unique terms."""
    import re

    terms: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;\n\r]+", raw or ""):
        term = part.strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def _build_court_context_block(
    *,
    participants: str,
    dictionary: str,
) -> str:
    """Build a structured context block for court-hearing processing."""
    blocks: list[str] = []
    participants = (participants or "").strip()
    dictionary = (dictionary or "").strip()
    if participants:
        blocks.append("COURT_PARTICIPANTS_AND_ROLES:\n" + participants)
    if dictionary:
        blocks.append("COURT_DICTIONARY_TERMS:\n" + dictionary)
    return "\n\n".join(blocks).strip()


def _review_state_path(job_id: str) -> Path:
    orchestrator = get_orchestrator()
    return orchestrator.get_meeting_dir(job_id) / "speaker_review.json"


def _load_review_state(job_id: str) -> dict:
    path = _review_state_path(job_id)
    if not path.exists():
        return {"confirmed": False}
    try:
        return json_lib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"confirmed": False}


def _save_review_state(job_id: str, state: dict) -> None:
    path = _review_state_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json_lib.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _invalidate_speaker_review(
    job_id: str,
    *,
    username: str,
    reason: str,
) -> None:
    state = _load_review_state(job_id)
    if not state.get("confirmed"):
        return
    state.update(
        {
            "confirmed": False,
            "invalidated_at": datetime.utcnow().isoformat(),
            "invalidated_by": username,
            "invalidated_reason": reason,
        }
    )
    _save_review_state(job_id, state)
    audit_log(
        action=AuditAction.MEETING_SPEAKER_REVIEW_INVALIDATE,
        username=username,
        resource_type="meeting",
        resource_id=job_id,
        details={"reason": reason},
    )


def _stored_meeting_type(job_id: str) -> Optional[str]:
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)
    if job is not None and job.meeting_type is not None:
        return (
            job.meeting_type.value
            if hasattr(job.meeting_type, "value")
            else str(job.meeting_type)
        )
    protocol_file = orchestrator.get_meeting_dir(job_id) / f"{job_id}.json"
    if not protocol_file.exists():
        return None
    try:
        data = json_lib.loads(protocol_file.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get("meeting_type")
    except Exception:
        return None
    return None


_UNRESOLVED_SPEAKER_RE = re.compile(r"^speaker[\s_-]?\d+$", re.IGNORECASE)


def _is_unresolved_speaker(label: str | None) -> bool:
    """Return True for machine labels like SPEAKER_01 / Speaker 1."""
    return bool(label and _UNRESOLVED_SPEAKER_RE.match(label.strip()))


def _load_aligned_segments(job_id: str) -> list[AlignedSegment]:
    """Load the current human-edited aligned transcript from disk."""
    aligned_data = get_orchestrator()._load_intermediate(job_id, "aligned")
    if not aligned_data:
        return []
    return [
        AlignedSegment(**item) if isinstance(item, dict) else item
        for item in aligned_data
    ]


def _segment_speaker_label(segment: AlignedSegment) -> str:
    """Display name that should be rendered in a final stenogram."""
    return (segment.speaker_name or segment.speaker_id or "").strip()


def _speaker_labels_in_first_seen_order(segments: list[AlignedSegment]) -> list[str]:
    """Collect non-empty speaker labels in transcript order."""
    labels: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        label = _segment_speaker_label(segment)
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def _refresh_court_protocol_from_aligned(job_id: str) -> dict:
    """Sync court protocol participants and stenogram turns from edited transcript.

    The court DOCX must reflect the human speaker review, not stale
    upload-time or LLM-produced labels. This function rebuilds the
    wrapped court protocol payload from the current aligned transcript
    and regenerates the DOCX when the payload changed.
    """
    segments = _load_aligned_segments(job_id)
    speakers = _speaker_labels_in_first_seen_order(segments)
    unresolved = sorted(
        {
            label
            for label in (_segment_speaker_label(segment) for segment in segments)
            if _is_unresolved_speaker(label)
        }
    )
    turns = [
        {
            "speaker": _segment_speaker_label(segment),
            "text": (segment.text or "").strip(),
            "start_s": segment.start,
        }
        for segment in segments
        if (segment.text or "").strip()
    ]

    orchestrator = get_orchestrator()
    meeting_dir = orchestrator.get_meeting_dir(job_id)
    protocol_file = meeting_dir / f"{job_id}.json"
    docx_file = meeting_dir / f"{job_id}.docx"

    if protocol_file.exists():
        protocol_json = json_lib.loads(protocol_file.read_text(encoding="utf-8"))
        if (
            isinstance(protocol_json, dict)
            and protocol_json.get("meeting_type") == MeetingType.COURT_HEARING.value
            and isinstance(protocol_json.get("payload"), dict)
        ):
            payload = protocol_json["payload"]
            changed = False
            if speakers and payload.get("participants") != speakers:
                payload["participants"] = speakers
                changed = True
            if turns and payload.get("turns") != turns:
                payload["turns"] = turns
                changed = True

            if changed:
                protocol_file.write_text(
                    json_lib.dumps(protocol_json, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                from backend.core.formatter import ProtocolFormatter
                from backend.engine.protocols.schemas import CourtHearingProtocol

                ProtocolFormatter().format_docx(
                    CourtHearingProtocol(**payload),
                    str(docx_file),
                )

    return {
        "speakers": speakers,
        "segments_count": len(segments),
        "turns_count": len(turns),
        "unresolved": unresolved,
    }


def get_orchestrator():
    """Get the global orchestrator instance."""
    from backend.app.main import orchestrator

    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orchestrator


def _can_access_job(job: MeetingJob, user: TokenData) -> bool:
    if user.role == "admin":
        return True
    if not job.owner_username:
        return True
    return job.owner_username == user.username


def _require_meeting_access(job_id: str, user: TokenData) -> Optional[MeetingJob]:
    """Return job when visible to user; raise 403 for another user's job."""
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)
    if job is not None and not _can_access_job(job, user):
        raise HTTPException(status_code=403, detail="Meeting access denied")
    return job


def background_process_meeting(
    job_id: str,
    expected_speakers_min: Optional[int] = None,
    expected_speakers_max: Optional[int] = None,
) -> None:
    """
    Process meeting in background.

    Args:
        job_id: Job identifier
        expected_speakers_min: Lower bound for pyannote min_speakers,
            or None to use config default. Resolved by the upload
            handler from the speaker-bucket UI choice (T3.1).
        expected_speakers_max: Upper bound for pyannote max_speakers,
            or None to use config default.
    """
    import asyncio

    try:
        # Get event loop or create one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Run async processing
        loop.run_until_complete(
            _async_process_meeting(
                job_id, expected_speakers_min, expected_speakers_max
            )
        )

    except Exception as e:
        logger.error(f"Background processing failed for {job_id}: {e}", exc_info=True)


async def _async_process_meeting(
    job_id: str,
    expected_speakers_min: Optional[int] = None,
    expected_speakers_max: Optional[int] = None,
) -> None:
    """
    Async processing of meeting.

    Args:
        job_id: Job identifier
        expected_speakers_min: Lower bound for pyannote min_speakers.
        expected_speakers_max: Upper bound for pyannote max_speakers.
    """
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)

    if job is None:
        logger.error(f"Job not found: {job_id}")
        return

    # Get audio file path from audio_original directory (where upload saves it)
    meeting_dir = orchestrator.get_meeting_dir(job_id)
    audio_original_dir = meeting_dir / "audio_original"
    audio_path = None

    if audio_original_dir.exists():
        # Find the uploaded file in audio_original/
        audio_files = list(audio_original_dir.iterdir())
        if audio_files:
            audio_path = audio_files[0]

    # Fallback: check for audio.wav directly (backward compatibility)
    if audio_path is None or not audio_path.exists():
        fallback_path = meeting_dir / "audio.wav"
        if fallback_path.exists():
            audio_path = fallback_path

    if audio_path is None or not audio_path.exists():
        logger.error(f"Audio file not found in {meeting_dir}")
        job.state = PipelineState.ERROR
        job.error = "Audio file not found"
        return

    def on_progress(updated_job: MeetingJob) -> None:
        """Update job progress and broadcast via WebSocket."""
        logger.info(
            f"Progress: {updated_job.state.value} "
            f"{updated_job.progress:.0f}% - {updated_job.current_stage}"
        )
        # Schedule async WebSocket broadcast from sync callback
        import asyncio
        from backend.app.websocket import send_progress_update
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(send_progress_update(updated_job.id))
        except RuntimeError:
            pass  # No event loop - skip WebSocket update

    # Pull per-job overrides set by the upload handler (UI checkboxes).
    overrides = (
        getattr(orchestrator, "_job_overrides", {}) or {}
    ).get(job_id, {}) or {}

    # Process meeting
    try:
        protocol = await orchestrator.process_meeting(
            str(audio_path), job, on_progress=on_progress,
            expected_speakers_min=expected_speakers_min,
            expected_speakers_max=expected_speakers_max,
            enable_llm_correction=overrides.get("enable_llm_correction"),
            enable_polish_pass=overrides.get("enable_polish_pass"),
            asr_engine_override=overrides.get("asr_engine_override"),
        )
    finally:
        # Clean up the overrides dict for this job so it doesn't leak.
        if hasattr(orchestrator, "_job_overrides"):
            orchestrator._job_overrides.pop(job_id, None)

    logger.info(f"Meeting {job_id} processing {'completed' if protocol else 'failed'}")


@router.post("/upload", response_model=MeetingJob)
async def upload_audio(
    request: Request,
    file: UploadFile = File(...),
    expected_speakers: Optional[int] = Form(default=None),
    expected_speakers_min: Optional[int] = Form(default=None),
    expected_speakers_max: Optional[int] = Form(default=None),
    context: Optional[str] = Form(default=None),
    context_files: list[UploadFile] = File(default=[]),
    meeting_type: MeetingType = Form(default=MeetingType.GENERIC),
    court_participants: Optional[str] = Form(default=None),
    court_dictionary: Optional[str] = Form(default=None),
    # Per-meeting opt-in toggles (override config defaults).
    # None = use config default; True/False = override for this run.
    enable_llm_correction: Optional[bool] = Form(default=None),
    enable_polish_pass: Optional[bool] = Form(default=None),
    asr_engine_override: Optional[str] = Form(default=None),
    background_tasks: BackgroundTasks = None,
    user: TokenData = Depends(get_current_user),
) -> MeetingJob:
    """
    Upload audio file and create job.

    Automatically starts background processing. Accepts optional meeting
    context (free text) and supporting documents (DOCX, XLSX, PDF, TXT)
    whose content is extracted and fed to the summarization LLM.

    Args:
        request: FastAPI request for client IP extraction
        file: Audio or video file. Supported audio: MP3, WAV, M4A,
            FLAC, OGG, AAC. Supported video: MP4, MOV, AVI, MKV, WEBM
            - ffmpeg in the preprocessor strips the audio track.
        expected_speakers: Optional expected number of speakers (2-20)
        context: Optional meeting context/agenda text
        context_files: Optional supporting documents for context
        meeting_type: Meeting type (court_hearing, administrative,
            client_meeting, interview, generic). Selects the
            summarization prompt and DOCX template. Defaults to generic.
        background_tasks: FastAPI background tasks

    Returns:
        Created MeetingJob with initial state
    """
    orchestrator = get_orchestrator()

    # Pre-flight VRAM check (2026-04-28): prior failure was 1-2 min into
    # the pipeline because another GPU app was holding 22+ GB. Fail
    # fast at upload time so the user sees the issue BEFORE waiting
    # for preprocessing/language-detection. Min 4 GB free is enough
    # for the smallest engine (GigaAM ASR ~3 GB) to start; later
    # diarization needs ~9.5 GB for pyannote, but by then ASR has
    # unloaded so we don't need to reserve all 9.5 upfront.
    try:
        from backend.core.vram_manager import VRAMManager

        vram = VRAMManager()
        if vram.is_gpu_available:
            status = vram.get_status()
            free_gb = float(status.get("vram_free_gb", 0.0))
            total_gb = float(status.get("vram_total_gb", 0.0))
            if free_gb < 4.0:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Недостаточно свободной видеопамяти: доступно "
                        f"{free_gb:.1f} ГБ из {total_gb:.1f} ГБ. "
                        f"Закройте другие приложения, использующие GPU "
                        f"(LM Studio, Stable Diffusion, ChatGPT desktop, "
                        f"игры в фоне), и попробуйте снова. Минимум для "
                        f"запуска: 4 ГБ свободно."
                    ),
                )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # VRAM check itself failed - log but allow run to proceed.
        # The pipeline's own VRAMManager polling will catch it later
        # if memory really is exhausted.
        logger.warning(
            "Pre-flight VRAM check failed (%s); proceeding without it",
            exc,
        )

    # Single-GPU: reject if another job is already processing
    if orchestrator.has_active_job():
        raise HTTPException(
            status_code=409,
            detail="Another meeting is currently being processed. Please wait for it to finish.",
        )

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name")

    display_filename = _display_upload_filename(file.filename, "uploaded_media")

    # Audio extensions go through ffmpeg's resample + mono-down path;
    # video extensions go through the same ffmpeg call which extracts
    # the audio track natively. The preprocessor (backend/core/audio.py)
    # produces a 16kHz mono WAV regardless of input container.
    allowed_audio = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
    allowed_video = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    allowed_extensions = allowed_audio | allowed_video
    file_ext = Path(display_filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                f"Audio: {', '.join(sorted(allowed_audio))}. "
                f"Video: {', '.join(sorted(allowed_video))}."
            ),
        )

    # Create job with context + meeting type
    stored_filename = _safe_upload_filename(
        display_filename,
        f"uploaded_media{file_ext}",
    )
    job = MeetingJob(
        filename=display_filename,
        context=context or "",
        meeting_type=meeting_type,
        owner_username=user.username,
        court_participants=(court_participants or "").strip(),
        court_dictionary=(court_dictionary or "").strip(),
    )
    logger.info(
        f"Upload meeting_type={meeting_type.value}, "
        f"context ({len(job.context)} chars): {job.context[:200]!r}"
    )
    meeting_dir = orchestrator.get_meeting_dir(job.id)

    # Save uploaded audio file
    audio_path = meeting_dir / "audio_original" / stored_filename
    media_max_bytes = _env_int(
        "VERITAS_MAX_MEDIA_UPLOAD_BYTES",
        _DEFAULT_MAX_MEDIA_UPLOAD_BYTES,
    )

    try:
        saved_size = await _write_upload_to_path(
            file,
            audio_path,
            max_bytes=media_max_bytes,
        )

        logger.info(f"Uploaded audio file: {audio_path} ({saved_size} bytes)")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save audio file")

    # Process context files - extract text and append to job context
    context_doc_exts = {".docx", ".xlsx", ".xls", ".pdf", ".txt"}
    context_file_max_chars = int(
        getattr(
            orchestrator._config.summarization,
            "context_file_max_chars",
            _DEFAULT_CONTEXT_FILE_MAX_CHARS,
        )
    )
    for ctx_file in context_files:
        if not ctx_file.filename:
            continue
        ctx_display_name = _display_upload_filename(
            ctx_file.filename,
            "context_file",
        )
        ctx_ext = Path(ctx_display_name).suffix.lower()
        if ctx_ext not in context_doc_exts:
            continue
        try:
            ctx_safe_name = _safe_upload_filename(
                ctx_display_name,
                f"context_file{ctx_ext}",
            )
            ctx_path = meeting_dir / "context_files" / ctx_safe_name
            await _write_upload_to_path(
                ctx_file,
                ctx_path,
                max_bytes=_env_int(
                    "VERITAS_MAX_CONTEXT_UPLOAD_BYTES",
                    _DEFAULT_MAX_CONTEXT_UPLOAD_BYTES,
                ),
            )
            extracted = _extract_text_from_file(
                ctx_path,
                max_chars=context_file_max_chars,
            )
            if extracted:
                job.context += f"\n\n[{ctx_display_name}]:\n{extracted}"
                logger.info(
                    "Extracted %s chars from %s",
                    len(extracted),
                    ctx_display_name,
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Failed to process context file {ctx_display_name}: {e}")

    job.context = job.context.strip()

    if meeting_type == MeetingType.COURT_HEARING:
        court_context = _build_court_context_block(
            participants=job.court_participants,
            dictionary=job.court_dictionary,
        )
        if court_context:
            job.context = (
                f"{job.context}\n\n{court_context}".strip()
                if job.context
                else court_context
            )

    # Register job
    orchestrator._jobs[job.id] = job
    orchestrator._persist_job(job)

    # Audit log
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"
    audit_log(
        action=AuditAction.MEETING_UPLOAD,
        username=user.username,
        ip_address=client_ip,
        resource_type="meeting",
        resource_id=job.id,
        details={
            "filename": display_filename,
            "stored_filename": stored_filename,
            "size_bytes": saved_size,
            "has_context": bool(job.context),
            "meeting_type": meeting_type.value,
            "has_court_dictionary": bool(job.court_dictionary),
        },
    )

    # Resolve speaker constraints from the three possible sources, in
    # order of preference:
    #   1. explicit min+max (T3.1: speaker bucket UI sends these)
    #   2. expected_speakers (legacy single-value field, widened ±1)
    #   3. config defaults (no constraint, broad pyannote range)
    resolved_min: Optional[int] = None
    resolved_max: Optional[int] = None

    if expected_speakers_min is not None or expected_speakers_max is not None:
        # New bucket path. Validate range (cap upper bound at 40 - pyannote
        # struggles past that and it's well outside any realistic meeting).
        if expected_speakers_min is not None and expected_speakers_min < 1:
            raise HTTPException(
                status_code=400,
                detail="expected_speakers_min must be >= 1",
            )
        if expected_speakers_max is not None and expected_speakers_max > 40:
            raise HTTPException(
                status_code=400,
                detail="expected_speakers_max must be <= 40",
            )
        if (
            expected_speakers_min is not None
            and expected_speakers_max is not None
            and expected_speakers_min > expected_speakers_max
        ):
            raise HTTPException(
                status_code=400,
                detail="expected_speakers_min must be <= expected_speakers_max",
            )
        resolved_min = expected_speakers_min
        resolved_max = expected_speakers_max
    elif expected_speakers is not None:
        # Legacy single-value path. Validate per old contract.
        if expected_speakers < 2 or expected_speakers > 40:
            raise HTTPException(
                status_code=400,
                detail="expected_speakers must be between 2 and 40",
            )
        # Widen +/- 1 so pyannote has a tight range to converge in
        # without being pinned to an exact count.
        resolved_min = max(2, expected_speakers - 1)
        resolved_max = expected_speakers + 2

    job.expected_speakers_min = resolved_min
    job.expected_speakers_max = resolved_max
    orchestrator._persist_job(job)

    if meeting_type == MeetingType.COURT_HEARING:
        orchestrator._save_intermediate(
            job.id,
            "court_context",
            {
                "expected_speakers_min": resolved_min,
                "expected_speakers_max": resolved_max,
                "court_participants": job.court_participants,
                "court_dictionary": job.court_dictionary,
                "dictionary_terms": _split_dictionary_terms(
                    job.court_dictionary
                ),
            },
        )

    # Validate ASR engine override if provided.
    if asr_engine_override is not None:
        valid_engines = {
            "auto", "gigaam", "hf-whisper", "whisper", "whisperx",
            "qwen", "nemo",
        }
        if asr_engine_override not in valid_engines:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid asr_engine_override: {asr_engine_override!r}. "
                    f"Valid: {sorted(valid_engines)}"
                ),
            )

    # Stash per-job overrides on the orchestrator under the job ID so
    # the background processor can read them. Simple dict keyed by
    # job_id; cleared after processing finishes.
    orchestrator._job_overrides = getattr(orchestrator, "_job_overrides", {})
    orchestrator._job_overrides[job.id] = {
        "enable_llm_correction": enable_llm_correction,
        "enable_polish_pass": enable_polish_pass,
        "asr_engine_override": asr_engine_override,
    }

    # Start background processing
    if background_tasks:
        background_tasks.add_task(
            background_process_meeting,
            job.id,
            resolved_min,
            resolved_max,
        )
    else:
        logger.warning("No background tasks executor available, processing may be synchronous")

    return job


@router.get("/{job_id}/status", response_model=MeetingJob)
async def get_meeting_status(job_id: str, user: TokenData = Depends(get_current_user)) -> MeetingJob:
    """
    Get meeting processing status.

    Args:
        job_id: Job identifier

    Returns:
        MeetingJob with current state and progress
    """
    orchestrator = get_orchestrator()
    job = _require_meeting_access(job_id, user)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return job


@router.post("/{job_id}/retry", response_model=MeetingJob)
async def retry_meeting(
    job_id: str,
    background_tasks: BackgroundTasks = None,
    user: TokenData = Depends(get_current_user),
) -> MeetingJob:
    """Re-run a previously-failed job from the start.

    Use case: pipeline failed because of a transient external resource
    issue (out of VRAM because another app was holding the GPU,
    Ollama briefly down, network blip on a model fetch). User has
    fixed the situation and wants to retry without re-uploading the
    audio file.

    Refuses to retry if another job is currently processing - same
    single-GPU constraint as upload. Also pre-flight checks VRAM the
    same way upload does.
    """
    orchestrator = get_orchestrator()
    job = _require_meeting_access(job_id, user)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.state != PipelineState.ERROR:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Job is in state {job.state.value!r}, not 'error'. "
                f"Retry is only valid for failed jobs."
            ),
        )
    if orchestrator.has_active_job():
        raise HTTPException(
            status_code=409,
            detail="Another meeting is currently being processed.",
        )

    # Pre-flight VRAM check, same as upload.
    try:
        from backend.core.vram_manager import VRAMManager

        vram = VRAMManager()
        if vram.is_gpu_available:
            status = vram.get_status()
            free_gb = float(status.get("vram_free_gb", 0.0))
            total_gb = float(status.get("vram_total_gb", 0.0))
            if free_gb < 4.0:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Недостаточно свободной видеопамяти: "
                        f"{free_gb:.1f} ГБ из {total_gb:.1f} ГБ. Закройте "
                        f"другие GPU-приложения и попробуйте снова."
                    ),
                )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retry pre-flight VRAM check failed (%s)", exc)

    # Reset job state. Keep the job_id so previously-saved files
    # (audio_original/, intermediate JSON if any) stay associated.
    job.state = PipelineState.UPLOADED
    job.progress = 0.0
    job.current_stage = "Retrying"
    job.error = None
    job.retry_count = (job.retry_count or 0) + 1

    # Re-trigger background processing with the same per-job speaker
    # constraints the operator chose during upload.
    if background_tasks:
        background_tasks.add_task(
            background_process_meeting,
            job_id,
            job.expected_speakers_min,
            job.expected_speakers_max,
        )
    else:
        logger.warning(
            "No background_tasks available; retry job will not run async"
        )

    logger.info("Retrying job %s (attempt #%d)", job_id, job.retry_count)
    orchestrator._persist_job(job)
    return job


@router.get("/{job_id}/transcript", response_model=list[AlignedSegment])
async def get_transcript(job_id: str, user: TokenData = Depends(get_current_user)) -> list[AlignedSegment]:
    """
    Get aligned transcript for meeting.

    Args:
        job_id: Job identifier

    Returns:
        List of AlignedSegment objects
    """
    orchestrator = get_orchestrator()
    job = _require_meeting_access(job_id, user)

    # Allow transcript access if job is past alignment OR if files exist on disk
    # (handles backend restart case where job state is lost from memory)
    if job is not None:
        past_alignment = job.state in (
            PipelineState.QA_VALIDATING,
            PipelineState.SUMMARIZING,
            PipelineState.FORMATTING,
            PipelineState.COMPLETED,
        )
        if not past_alignment:
            raise HTTPException(
                status_code=400,
                detail=f"Transcript not ready yet (state: {job.state.value})",
            )

    # Load aligned transcript from intermediate results
    aligned_data = orchestrator._load_intermediate(job_id, "aligned")

    # Fallback: read transcript from protocol JSON if intermediate was cleaned up
    if not aligned_data:
        protocol_data = orchestrator._load_intermediate(job_id, job_id)
        if protocol_data and isinstance(protocol_data, dict):
            aligned_data = protocol_data.get("transcript")
        # Also try the standard protocol filename
        if not aligned_data:
            meeting_dir = orchestrator.get_meeting_dir(job_id)
            protocol_file = meeting_dir / f"{job_id}.json"
            if protocol_file.exists():
                import json as json_lib
                with open(protocol_file, "r", encoding="utf-8") as f:
                    protocol_json = json_lib.load(f)
                aligned_data = protocol_json.get("transcript")

    if not aligned_data:
        raise HTTPException(status_code=404, detail="Transcript not found")

    # Convert to AlignedSegment objects
    segments = []
    for item in aligned_data:
        if isinstance(item, dict):
            segments.append(AlignedSegment(**item))
        else:
            segments.append(item)

    return segments


@router.get("/{job_id}/review")
async def get_transcript_review(
    job_id: str,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Return speaker/dictionary review data for the transcript editor."""
    orchestrator = get_orchestrator()
    job = _require_meeting_access(job_id, user)

    aligned_data = orchestrator._load_intermediate(job_id, "aligned")
    if not aligned_data:
        raise HTTPException(status_code=404, detail="Transcript not found")

    segments = [
        AlignedSegment(**item) if isinstance(item, dict) else item
        for item in aligned_data
    ]

    court_context = orchestrator._load_intermediate(job_id, "court_context") or {}
    expected_min = (
        job.expected_speakers_min
        if job is not None and job.expected_speakers_min is not None
        else court_context.get("expected_speakers_min")
    )
    expected_max = (
        job.expected_speakers_max
        if job is not None and job.expected_speakers_max is not None
        else court_context.get("expected_speakers_max")
    )
    court_participants = (
        job.court_participants
        if job is not None and job.court_participants
        else court_context.get("court_participants", "")
    )
    court_dictionary = (
        job.court_dictionary
        if job is not None and job.court_dictionary
        else court_context.get("court_dictionary", "")
    )
    dictionary_terms = court_context.get("dictionary_terms")
    if not isinstance(dictionary_terms, list):
        dictionary_terms = _split_dictionary_terms(str(court_dictionary or ""))

    by_speaker: dict[str, dict] = {}
    for seg in segments:
        sid = seg.speaker_id
        item = by_speaker.setdefault(
            sid,
            {
                "speaker_id": sid,
                "speaker_name": seg.speaker_name,
                "suggested_speaker_name": seg.suggested_speaker_name,
                "suggested_speaker_confidence": seg.suggested_speaker_confidence,
                "turn_count": 0,
                "duration_s": 0.0,
                "low_confidence_turns": 0,
                "samples": [],
            },
        )
        item["turn_count"] += 1
        item["duration_s"] += max(0.0, float(seg.end) - float(seg.start))
        if (
            seg.attribution_confidence is not None
            and seg.attribution_confidence < 0.8
        ):
            item["low_confidence_turns"] += 1
        if not item.get("speaker_name") and seg.speaker_name:
            item["speaker_name"] = seg.speaker_name
        if (
            not item.get("suggested_speaker_name")
            and seg.suggested_speaker_name
        ):
            item["suggested_speaker_name"] = seg.suggested_speaker_name
            item["suggested_speaker_confidence"] = (
                seg.suggested_speaker_confidence
            )
        samples = item["samples"]
        text = (seg.text or "").strip()
        if text and len(samples) < 3:
            samples.append(text[:140] + ("..." if len(text) > 140 else ""))

    speaker_items = sorted(
        by_speaker.values(),
        key=lambda x: x["duration_s"],
        reverse=True,
    )
    for item in speaker_items:
        item["duration_s"] = round(float(item["duration_s"]), 1)

    full_text = "\n".join(seg.text or "" for seg in segments).casefold()
    dict_items = []
    for term in dictionary_terms:
        key = str(term).strip()
        if not key:
            continue
        count = full_text.count(key.casefold())
        dict_items.append({"term": key, "count": count, "found": count > 0})

    return {
        "job_id": job_id,
        "meeting_type": (
            job.meeting_type.value
            if job is not None and hasattr(job.meeting_type, "value")
            else str(job.meeting_type) if job is not None else None
        ),
        "expected_speakers_min": expected_min,
        "expected_speakers_max": expected_max,
        "detected_speakers": len(speaker_items),
        "speaker_count_matches": (
            None
            if expected_min is None or expected_max is None
            else expected_min <= len(speaker_items) <= expected_max
        ),
        "court_participants": court_participants,
        "dictionary_terms": dict_items,
        "dictionary_missing_count": sum(1 for item in dict_items if not item["found"]),
        "low_confidence_turns": sum(
            1
            for seg in segments
            if seg.attribution_confidence is not None
            and seg.attribution_confidence < 0.8
        ),
        "speaker_review": _load_review_state(job_id),
        "speakers": speaker_items,
    }


@router.post("/{job_id}/review/confirm")
async def confirm_speaker_review(
    job_id: str,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Mark court-hearing speaker review as confirmed by a human."""
    _require_meeting_access(job_id, user)
    if _stored_meeting_type(job_id) != MeetingType.COURT_HEARING.value:
        raise HTTPException(
            status_code=400,
            detail="Speaker review confirmation is required only for court hearings.",
        )

    segments = _load_aligned_segments(job_id)
    if not segments:
        raise HTTPException(status_code=404, detail="Transcript not found")

    refresh = _refresh_court_protocol_from_aligned(job_id)
    unresolved = refresh.get("unresolved", [])
    if unresolved:
        raise HTTPException(
            status_code=409,
            detail=(
                "Unresolved speaker labels remain: "
                + ", ".join(unresolved)
                + ". Rename them in the transcript editor before confirming."
            ),
        )

    speakers = refresh.get("speakers") or _speaker_labels_in_first_seen_order(segments)
    job = _require_meeting_access(job_id, user)
    court_context = get_orchestrator()._load_intermediate(job_id, "court_context") or {}
    expected_min = (
        job.expected_speakers_min
        if job is not None and job.expected_speakers_min is not None
        else court_context.get("expected_speakers_min")
    )
    expected_max = (
        job.expected_speakers_max
        if job is not None and job.expected_speakers_max is not None
        else court_context.get("expected_speakers_max")
    )
    if (
        expected_min is not None
        and expected_max is not None
        and expected_min == expected_max
        and len(speakers) != int(expected_min)
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Expected exactly {expected_min} speakers, but transcript "
                f"currently has {len(speakers)}. Add or reassign speakers "
                "in the transcript editor before confirming."
            ),
        )
    state = {
        "confirmed": True,
        "confirmed_at": datetime.utcnow().isoformat(),
        "confirmed_by": user.username,
        "speakers": speakers,
        "segments_count": len(segments),
    }
    _save_review_state(job_id, state)
    audit_log(
        action=AuditAction.MEETING_SPEAKER_REVIEW_CONFIRM,
        username=user.username,
        resource_type="meeting",
        resource_id=job_id,
        details={"speakers": speakers, "segments_count": len(segments)},
    )

    # Keep the protocol header aligned with the human-verified transcript.
    # Upload-time participant hints are useful context, but after review the
    # confirmed speaker names should win in the final court DOCX.
    try:
        orchestrator = get_orchestrator()
        meeting_dir = orchestrator.get_meeting_dir(job_id)
        protocol_file = meeting_dir / f"{job_id}.json"
        if protocol_file.exists():
            protocol_json = json_lib.loads(protocol_file.read_text(encoding="utf-8"))
            if (
                isinstance(protocol_json, dict)
                and protocol_json.get("meeting_type") == MeetingType.COURT_HEARING.value
                and isinstance(protocol_json.get("payload"), dict)
            ):
                protocol_json["payload"]["participants"] = speakers
                protocol_file.write_text(
                    json_lib.dumps(protocol_json, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                from backend.core.formatter import ProtocolFormatter
                from backend.engine.protocols.schemas import CourtHearingProtocol

                protocol_obj = CourtHearingProtocol(**protocol_json["payload"])
                ProtocolFormatter().format_docx(
                    protocol_obj,
                    str(meeting_dir / f"{job_id}.docx"),
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to refresh court participants after speaker review: %s",
            exc,
        )
    return {"status": "confirmed", "job_id": job_id, "speaker_review": state}


@router.put("/{job_id}/transcript")
async def save_transcript(
    job_id: str,
    segments: list[AlignedSegment],
    user: TokenData = Depends(get_current_user),
) -> dict:
    """
    Save edited transcript.

    Args:
        job_id: Job identifier
        segments: Updated transcript segments

    Returns:
        Confirmation dictionary
    """
    orchestrator = get_orchestrator()
    job = _require_meeting_access(job_id, user)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    try:
        # Save edited segments to main aligned file (overwrites original)
        orchestrator._save_intermediate(job_id, "aligned", segments)
        _invalidate_speaker_review(
            job_id,
            username=user.username,
            reason="transcript_edited",
        )

        return {
            "status": "ok",
            "job_id": job_id,
            "segments_count": len(segments),
        }

    except Exception as e:
        logger.error(f"Failed to save transcript: {e}")
        raise HTTPException(status_code=500, detail="Failed to save transcript")


@router.get("/{job_id}/protocol")
async def get_protocol(job_id: str, request: Request, format: str = "docx", user: TokenData = Depends(get_current_user)) -> FileResponse:
    """
    Download meeting protocol.

    Args:
        job_id: Job identifier
        format: Output format (docx, json)

    Returns:
        File response with protocol
    """
    orchestrator = get_orchestrator()
    job = _require_meeting_access(job_id, user)
    meeting_dir = orchestrator.get_meeting_dir(job_id)

    # Check job state if available, but also allow download if files exist on disk
    # (handles case where backend restarted and lost in-memory job state)
    if job is not None and job.state != PipelineState.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Meeting not yet processed (state: {job.state.value})",
        )

    requested_format = format.lower()

    if requested_format in {"actions_docx", "actions"}:
        protocol_file = meeting_dir / f"{job_id}.json"
        if not protocol_file.exists():
            raise HTTPException(status_code=404, detail="Protocol data not found")
        protocol_json = json_lib.loads(protocol_file.read_text(encoding="utf-8"))
        if (
            not isinstance(protocol_json, dict)
            or protocol_json.get("meeting_type") != MeetingType.ADMINISTRATIVE.value
            or not isinstance(protocol_json.get("payload"), dict)
        ):
            raise HTTPException(
                status_code=400,
                detail="Action list export is available only for administrative meetings",
            )
        from backend.core.formatter import ProtocolFormatter
        from backend.engine.protocols.schemas import AdministrativeProtocol

        actions_path = meeting_dir / f"{job_id}_actions.docx"
        payload = AdministrativeProtocol(**protocol_json["payload"])
        ProtocolFormatter.format_admin_actions_docx(payload, str(actions_path))
        file_path = actions_path
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif requested_format == "docx":
        if (
            _stored_meeting_type(job_id) == MeetingType.COURT_HEARING.value
            and not _load_review_state(job_id).get("confirmed", False)
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Speaker review is required before final DOCX download "
                    "for court hearings. Open the transcript editor and "
                    "confirm speakers first."
                ),
            )
        if _stored_meeting_type(job_id) == MeetingType.COURT_HEARING.value:
            try:
                refresh = _refresh_court_protocol_from_aligned(job_id)
                unresolved = refresh.get("unresolved", [])
                if unresolved:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "Unresolved speaker labels remain: "
                            + ", ".join(unresolved)
                            + ". Rename them in the transcript editor before downloading DOCX."
                        ),
                    )
                aligned_data = orchestrator._load_intermediate(job_id, "aligned")
                if aligned_data:
                    segments = [
                        AlignedSegment(**item) if isinstance(item, dict) else item
                        for item in aligned_data
                    ]
                    speakers = _speaker_labels_in_first_seen_order(segments)
                    protocol_file = meeting_dir / f"{job_id}.json"
                    if protocol_file.exists() and speakers:
                        protocol_json = json_lib.loads(
                            protocol_file.read_text(encoding="utf-8")
                        )
                        if (
                            isinstance(protocol_json, dict)
                            and protocol_json.get("meeting_type")
                            == MeetingType.COURT_HEARING.value
                            and isinstance(protocol_json.get("payload"), dict)
                            and protocol_json["payload"].get("participants")
                            != speakers
                        ):
                            protocol_json["payload"]["participants"] = speakers
                            protocol_file.write_text(
                                json_lib.dumps(
                                    protocol_json,
                                    indent=2,
                                    ensure_ascii=False,
                                ),
                                encoding="utf-8",
                            )
                            from backend.core.formatter import ProtocolFormatter
                            from backend.engine.protocols.schemas import (
                                CourtHearingProtocol,
                            )

                            protocol_obj = CourtHearingProtocol(
                                **protocol_json["payload"]
                            )
                            ProtocolFormatter().format_docx(
                                protocol_obj,
                                str(meeting_dir / f"{job_id}.docx"),
                            )
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to sync court protocol participants: %s",
                    exc,
                )
        file_path = meeting_dir / f"{job_id}.docx"
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif requested_format == "json":
        file_path = meeting_dir / f"{job_id}.json"
        media_type = "application/json"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Protocol file not found: {format}")

    # Audit log download
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"
    audit_log(
        action=AuditAction.MEETING_DOWNLOAD,
        username=user.username,
        ip_address=client_ip,
        resource_type="meeting",
        resource_id=job_id,
        details={"format": format},
    )

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name,
    )


@router.put("/{job_id}/speakers/rename")
async def rename_speaker(
    job_id: str,
    old_name: str,
    new_name: str,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Rename a speaker across all transcript segments for a meeting.

    Updates speaker_name in the aligned transcript (both in-memory intermediate
    and any saved JSON files). This allows the user to replace "Speaker_1" with
    a real name like "Ivanov A.V.".

    Args:
        job_id: Job identifier.
        old_name: Current speaker name or speaker_id to match.
        new_name: New speaker display name.

    Returns:
        Confirmation with count of renamed segments.
    """
    if not new_name or not new_name.strip():
        raise HTTPException(status_code=400, detail="new_name must not be empty")

    orchestrator = get_orchestrator()
    _require_meeting_access(job_id, user)
    meeting_dir = orchestrator.get_meeting_dir(job_id)

    # Load aligned transcript
    aligned_data = orchestrator._load_intermediate(job_id, "aligned")
    if not aligned_data:
        # Try edited version
        aligned_data = orchestrator._load_intermediate(job_id, "aligned_edited")
    if not aligned_data:
        raise HTTPException(status_code=404, detail="Transcript not found")

    # Rename matching speakers
    renamed_count = 0
    for item in aligned_data:
        if isinstance(item, dict):
            matches_id = item.get("speaker_id") == old_name
            matches_name = item.get("speaker_name") == old_name
            if matches_id or matches_name:
                item["speaker_name"] = new_name.strip()
                renamed_count += 1
        elif hasattr(item, "speaker_id"):
            if item.speaker_id == old_name or item.speaker_name == old_name:
                item.speaker_name = new_name.strip()
                renamed_count += 1

    if renamed_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No segments found for speaker '{old_name}'",
        )

    # Save updated transcript
    orchestrator._save_intermediate(job_id, "aligned", aligned_data)
    _invalidate_speaker_review(
        job_id,
        username=user.username,
        reason="speaker_renamed",
    )

    # Also update protocol JSON if it exists
    import json as json_lib

    protocol_file = meeting_dir / f"{job_id}.json"
    if protocol_file.exists():
        try:
            with open(protocol_file, "r", encoding="utf-8") as f:
                protocol_json = json_lib.load(f)

            # Protocol JSON can be in one of two shapes on disk:
            #   1. LEGACY: MeetingProtocol fields at the root
            #        {meeting_date, topic, participants, summary, ...,
            #         transcript: [...]}
            #   2. NEW: ProtocolResult wrapper (Block 7+)
            #        {"meeting_type": "...",
            #         "payload": {...type-specific schema...}}
            #
            # Detect and rename speakers in the right places per shape.
            is_wrapped = (
                "meeting_type" in protocol_json
                and "payload" in protocol_json
                and isinstance(protocol_json.get("payload"), dict)
            )
            new_name_clean = new_name.strip()

            if is_wrapped:
                payload = protocol_json["payload"]
                mtype = protocol_json.get("meeting_type", "")

                # Participants is a list[str] for admin/court/etc. -
                # replace string entries that equal the old_name.
                parts = payload.get("participants")
                if isinstance(parts, list):
                    payload["participants"] = [
                        new_name_clean if str(p) == old_name else p
                        for p in parts
                    ]

                # Stenogram turns for admin + court: list of
                # {"speaker": str, "text": str}. Rename speaker field.
                turns = payload.get("turns")
                if isinstance(turns, list):
                    for t in turns:
                        if isinstance(t, dict) and t.get("speaker") == old_name:
                            t["speaker"] = new_name_clean

                # Admin LEGACY: department-grouped decisions/tasks/
                # open_issues, each item has a .speaker string.
                for dept in payload.get("departments", []) or []:
                    if not isinstance(dept, dict):
                        continue
                    for bucket in ("decisions", "tasks", "open_issues"):
                        for item in dept.get(bucket, []) or []:
                            if (
                                isinstance(item, dict)
                                and item.get("speaker") == old_name
                            ):
                                item["speaker"] = new_name_clean

                # Admin FLAT (T3.3): top-level decisions/open_questions/
                # risks have a .speaker; tasks have an .owner. Rename
                # in all of them when the old speaker matches.
                for item in payload.get("items", []) or []:
                    if not isinstance(item, dict):
                        continue
                    if item.get("speaker") == old_name:
                        item["speaker"] = new_name_clean
                    if item.get("owner") == old_name:
                        item["owner"] = new_name_clean
                for bucket in ("decisions", "open_questions", "risks"):
                    for item in payload.get(bucket, []) or []:
                        if (
                            isinstance(item, dict)
                            and item.get("speaker") == old_name
                        ):
                            item["speaker"] = new_name_clean
                for task in payload.get("tasks", []) or []:
                    if isinstance(task, dict) and task.get("owner") == old_name:
                        task["owner"] = new_name_clean
            else:
                # Legacy shape: transcript segments + participants with
                # speaker_id/speaker_name fields.
                if "transcript" in protocol_json:
                    for seg in protocol_json["transcript"]:
                        if (
                            seg.get("speaker_id") == old_name
                            or seg.get("speaker_name") == old_name
                        ):
                            seg["speaker_name"] = new_name_clean
                if "participants" in protocol_json:
                    for p in protocol_json["participants"]:
                        if (
                            p.get("speaker_id") == old_name
                            or p.get("speaker_name") == old_name
                        ):
                            p["speaker_name"] = new_name_clean

            with open(protocol_file, "w", encoding="utf-8") as f:
                json_lib.dump(
                    protocol_json, f, indent=2, ensure_ascii=False
                )

            # Regenerate DOCX with updated speaker names.
            # Dispatch on meeting_type (if wrapped) so we construct the
            # RIGHT Pydantic class. Building a legacy MeetingProtocol
            # from admin-shaped JSON silently loses all admin content
            # because Pydantic discards unknown keys - that's the
            # "blank DOCX after rename" bug fixed 2026-04-23.
            try:
                from backend.core.formatter import ProtocolFormatter
                from backend.app.models import MeetingProtocol
                from backend.engine.protocols.schemas import (
                    AdministrativeProtocol,
                    ClientMeetingProtocol,
                    CourtHearingProtocol,
                    InterviewProtocol,
                )

                protocol_obj = None
                if is_wrapped:
                    payload_data = protocol_json["payload"]
                    mtype = str(protocol_json.get("meeting_type", "")).lower()
                    if mtype == "administrative":
                        protocol_obj = AdministrativeProtocol(**payload_data)
                    elif mtype == "court_hearing":
                        protocol_obj = CourtHearingProtocol(**payload_data)
                    elif mtype == "client_meeting":
                        protocol_obj = ClientMeetingProtocol(**payload_data)
                    elif mtype == "interview":
                        protocol_obj = InterviewProtocol(**payload_data)
                    else:
                        # Generic or unknown wrapped type - try legacy
                        # MeetingProtocol against the unwrapped payload.
                        protocol_obj = MeetingProtocol(**payload_data)
                else:
                    protocol_obj = MeetingProtocol(**protocol_json)

                formatter = ProtocolFormatter()
                docx_path = meeting_dir / f"{job_id}.docx"
                formatter.format_docx(protocol_obj, str(docx_path))
                logger.info(
                    f"Regenerated DOCX after speaker rename: {docx_path} "
                    f"(type={type(protocol_obj).__name__})"
                )
            except Exception as docx_err:
                logger.warning(
                    f"Failed to regenerate DOCX after rename: {docx_err}"
                )

        except Exception as e:
            logger.warning(f"Failed to update protocol JSON: {e}")

    logger.info(
        f"Renamed speaker '{old_name}' -> '{new_name}' in {renamed_count} segments "
        f"for job {job_id}"
    )

    return {
        "status": "ok",
        "job_id": job_id,
        "old_name": old_name,
        "new_name": new_name.strip(),
        "renamed_segments": renamed_count,
    }


@router.get("/{job_id}/protocol/data")
async def get_protocol_data(
    job_id: str,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Return protocol data as JSON for frontend rendering.

    Unlike GET /protocol (which returns a file download), this returns
    the MeetingProtocol object as JSON so the frontend can render
    summary, participants, topics, etc.
    """
    orchestrator = get_orchestrator()
    _require_meeting_access(job_id, user)
    meeting_dir = orchestrator.get_meeting_dir(job_id)

    # Try loading from protocol JSON file
    protocol_file = meeting_dir / f"{job_id}.json"
    if not protocol_file.exists():
        # Try intermediate
        data = orchestrator._load_intermediate(job_id, "protocol")
        if data:
            return data if isinstance(data, dict) else data.model_dump() if hasattr(data, "model_dump") else {}
        raise HTTPException(status_code=404, detail="Protocol data not found")

    with open(protocol_file, "r", encoding="utf-8") as f:
        return json_lib.load(f)


@router.post("/{job_id}/resummarize")
async def resummarize_meeting(
    job_id: str,
    background_tasks: BackgroundTasks,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Re-run summarization on the (edited) transcript.

    Loads the current aligned transcript, runs the LLM summarization
    engine, and regenerates the protocol DOCX/JSON. Useful after the
    user has edited the transcript or renamed speakers.
    """
    orchestrator = get_orchestrator()
    job = _require_meeting_access(job_id, user)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if orchestrator.has_active_job():
        raise HTTPException(
            status_code=409,
            detail="Another job is processing. Wait for it to finish.",
        )

    # Mark job as re-summarizing
    job.state = PipelineState.SUMMARIZING
    job.progress = 70.0
    job.current_stage = "Re-summarizing with updated transcript"
    job.error = None
    orchestrator._persist_job(job)

    background_tasks.add_task(_async_resummarize, job_id)

    return {"status": "started", "job_id": job_id}


async def _async_resummarize(job_id: str) -> None:
    """Background task: re-summarize and regenerate protocol."""
    orchestrator = get_orchestrator()
    job = orchestrator.get_job(job_id)
    if job is None:
        return

    try:
        # Load current transcript (includes user edits and renames)
        aligned_data = orchestrator._load_intermediate(job_id, "aligned")
        if not aligned_data:
            raise RuntimeError("No transcript found for re-summarization")

        aligned = [
            AlignedSegment(**item) if isinstance(item, dict) else item
            for item in aligned_data
        ]

        # Detect language from transcript segments
        from backend.core.postprocessor import detect_language
        detected_language = detect_language(aligned)

        # Load summarization engine
        config = orchestrator._config
        if not config.summarization.enabled:
            raise RuntimeError("Summarization is disabled in config")

        # Ollama manages its own VRAM - no VRAMManager needed
        orchestrator._summarization.load()

        # Get meeting context from job
        meeting_context = job.context or ""

        protocol = await orchestrator._summarization.process(
            aligned,
            language=detected_language,
            meeting_context=meeting_context,
            meeting_type=job.meeting_type,
        )

        # Ollama unload (sends keep_alive=0)
        orchestrator._summarization.unload()

        if protocol is None:
            raise RuntimeError("Re-summarization failed")

        # Save and reformat
        orchestrator._save_intermediate(job_id, "protocol", protocol)

        from backend.core.formatter import ProtocolFormatter
        formatter = ProtocolFormatter()
        meeting_dir = orchestrator.get_meeting_dir(job_id)
        formatter.format_docx(protocol, str(meeting_dir / f"{job_id}.docx"))
        formatter.format_json(protocol, str(meeting_dir / f"{job_id}.json"))

        job.state = PipelineState.COMPLETED
        job.progress = 100.0
        job.current_stage = "Re-summarization complete"
        orchestrator._persist_job(job)
        logger.info(f"Re-summarization complete for job {job_id}")

    except Exception as e:
        logger.error(f"Re-summarization failed for {job_id}: {e}", exc_info=True)
        job.state = PipelineState.ERROR
        job.error = f"Re-summarization failed: {e}"
        orchestrator._persist_job(job)


@router.post("/{job_id}/summarize-cloud")
async def summarize_with_claude(
    job_id: str,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Summarize transcript using Claude API (online, for comparison with local Qwen).

    Requires ANTHROPIC_API_KEY environment variable to be set.
    This is a testing/comparison endpoint - not for production use.
    """
    orchestrator = get_orchestrator()
    _require_meeting_access(job_id, user)
    if not bool(
        getattr(orchestrator._config.security, "cloud_compare_enabled", False)
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "Cloud comparison is disabled. Enable "
                "EPAM_SECURITY_CLOUD_COMPARE_ENABLED=true only for explicit "
                "testing, because it sends transcript content outside the "
                "on-premise environment."
            ),
        )

    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="ANTHROPIC_API_KEY not set. Add it to .env or environment.",
        )

    # Load transcript
    aligned_data = orchestrator._load_intermediate(job_id, "aligned")
    if not aligned_data:
        raise HTTPException(status_code=404, detail="Transcript not found")

    aligned = [
        AlignedSegment(**item) if isinstance(item, dict) else item
        for item in aligned_data
    ]

    # Get meeting context
    job = orchestrator.get_job(job_id)
    meeting_context = job.context if job else ""

    # Format transcript for Claude
    lines = []
    for seg in aligned:
        speaker = seg.speaker_name or seg.speaker_id
        m = int(seg.start // 60)
        s = int(seg.start % 60)
        lines.append(f"{speaker} [{m:02d}:{s:02d}]: {seg.text}")
    transcript_text = "\n".join(lines)

    # Build prompt - structured protocol, concise style
    system = (
        "Ты - секретарь юридической фирмы EPAM. "
        "Составь протокол совещания на русском языке по стенограмме.\n\n"
        "Стиль: КРАТКО, по существу, без воды. Каждый пункт - максимум 1-2 предложения.\n\n"
        "Формат:\n"
        "## ТЕМА СОВЕЩАНИЯ\nОдно предложение.\n\n"
        "## КРАТКОЕ СОДЕРЖАНИЕ\n1 абзац (3-5 предложений). Только факты и позиции сторон.\n\n"
        "## ОСНОВНЫЕ ТЕМЫ\n- тема (кратко)\n\n"
        "## ПРИНЯТЫЕ РЕШЕНИЯ\n- решение (только если ЯВНО принято). Если решений не было - пиши 'Нет'.\n\n"
        "## ЗАДАЧИ И ПОРУЧЕНИЯ\n- задача | ответственный | срок. Если задач не было - пиши 'Нет'.\n\n"
        "## ОТКРЫТЫЕ ВОПРОСЫ\n- вопрос\n\n"
        "ЗАПРЕЩЕНО выдумывать информацию, которой нет в стенограмме. "
        "Лучше пропустить раздел, чем заполнить его домыслами."
    )
    if meeting_context:
        system += f"\n\nMeeting context provided by user:\n{meeting_context}"

    user_msg = f"Transcript:\n\n{transcript_text}"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        cloud_summary = response.content[0].text

        logger.info(
            f"Claude summarization for {job_id}: {len(cloud_summary)} chars, "
            f"tokens in={response.usage.input_tokens} out={response.usage.output_tokens}"
        )

        return {
            "status": "ok",
            "job_id": job_id,
            "model": "claude-opus-4-20250514",
            "summary": cloud_summary,
            "tokens_in": response.usage.input_tokens,
            "tokens_out": response.usage.output_tokens,
        }

    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Claude API error: {e}")


@router.get("/", response_model=list[MeetingJob])
async def list_meetings(user: TokenData = Depends(get_current_user)) -> list[MeetingJob]:
    """
    List all meetings.

    Returns:
        List of MeetingJob objects
    """
    orchestrator = get_orchestrator()
    jobs = orchestrator.list_jobs()
    if user.role == "admin":
        return jobs
    return [job for job in jobs if _can_access_job(job, user)]


@router.delete("/{job_id}")
async def delete_meeting(job_id: str, request: Request, user: TokenData = Depends(get_current_user)) -> dict:
    """
    Delete meeting and its data.

    Args:
        job_id: Job identifier
        request: FastAPI request for client IP extraction

    Returns:
        Confirmation dictionary
    """
    orchestrator = get_orchestrator()
    job = _require_meeting_access(job_id, user)
    meeting_dir = Path(orchestrator._config.output.output_dir) / job_id
    if job is None and not meeting_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if not orchestrator.delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Audit log deletion
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"
    audit_log(
        action=AuditAction.MEETING_DELETE,
        username=user.username,
        ip_address=client_ip,
        resource_type="meeting",
        resource_id=job_id,
    )

    return {
        "status": "deleted",
        "job_id": job_id,
    }
