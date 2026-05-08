"""API routes for speaker management and voice enrollment (Block 7).

Sprint 2026-04-30 rewrite: replaces the JSON-file storage with the
encrypted SQLite-backed voice_enrollment module. Each Windows user
sees the same shared voice DB (employee voices are organisation-wide,
not per-account). Embedding extraction uses pyannote/embedding via
ECAPA-TDNN — separate from the diarization pipeline so we don't need
to load the full diarization graph during enrollment.

Endpoints:

* ``POST /api/speakers/`` — enrol new profile from audio sample
* ``GET  /api/speakers/`` — list all enrolled profiles
* ``GET  /api/speakers/{id}`` — get profile metadata
* ``PUT  /api/speakers/{id}`` — update name/department/position
* ``POST /api/speakers/{id}/samples`` — add additional voice sample
* ``DELETE /api/speakers/{id}`` — delete profile + all embeddings
* ``POST /api/speakers/register`` — legacy alias for ``POST /``,
  kept until the frontend migration completes

Audio samples are written to a temp directory, embedded, and then
deleted. We never persist the source audio — only the encrypted
embedding survives. This is the law-firm privacy contract: voices
identify employees but the recordings themselves are not stored.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from backend.app.auth import TokenData, get_current_user
from backend.app.models import SpeakerProfile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/speakers", tags=["speakers"])


# Audio formats accepted for enrollment. Matches the upload page's
# whitelist plus a couple of forms users send when recording on phones.
_ALLOWED_AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".aac",
}


def _profile_dict_to_model(d: dict) -> SpeakerProfile:
    """Convert a voice_enrollment dict into a SpeakerProfile model.

    voice_enrollment returns ISO timestamp strings; SpeakerProfile
    expects datetimes. Empty/None last_used_at falls through.
    """
    def _parse_dt(v):
        if v is None or v == "":
            return None
        try:
            return datetime.fromisoformat(str(v))
        except (TypeError, ValueError):
            return None

    return SpeakerProfile(
        id=str(d.get("id", "")),
        name=str(d.get("name", "")),
        department=str(d.get("department") or ""),
        position=(str(d["position"]) if d.get("position") else None),
        is_admin_default=bool(d.get("is_admin_default", False)),
        embedding_path="",  # Block 7: vectors live in SQLite, not files.
        registered_at=_parse_dt(d.get("created_at")) or datetime.utcnow(),
        last_used_at=_parse_dt(d.get("last_used_at")),
        sample_count=int(d.get("sample_count", 0) or 0),
        match_count=int(d.get("match_count", 0) or 0),
        meetings_count=int(d.get("match_count", 0) or 0),  # alias for legacy UI
    )


def _validate_audio(audio: UploadFile) -> None:
    """Raise HTTP 400 if the uploaded file isn't a recognisable audio."""
    if not audio.filename:
        raise HTTPException(
            status_code=400, detail="Audio file is required",
        )
    ext = Path(audio.filename).suffix.lower()
    if ext not in _ALLOWED_AUDIO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format '{ext}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_AUDIO_EXTS))}"
            ),
        )


async def _save_upload_to_temp(audio: UploadFile) -> Path:
    """Persist the uploaded audio to a temp file. Caller deletes it.

    voice_enrollment.extract_embedding wants a path on disk (pyannote
    Inference reads via torchaudio/librosa). We write under
    tempfile.gettempdir() so the file disappears with the OS reboot
    even if our cleanup misses it.
    """
    suffix = Path(audio.filename or ".wav").suffix.lower() or ".wav"
    fd, tmp_path = tempfile.mkstemp(prefix="veritas_voice_", suffix=suffix)
    import os
    os.close(fd)
    tmp = Path(tmp_path)
    contents = await audio.read()
    tmp.write_bytes(contents)
    return tmp


@router.post("/", response_model=SpeakerProfile)
async def enrol_speaker(
    name: str = Form(...),
    department: str = Form(""),
    position: str = Form(""),
    is_admin_default: bool = Form(False),
    audio: UploadFile | None = File(None),
    user: TokenData = Depends(get_current_user),
) -> SpeakerProfile:
    """Create an employee profile, with an optional voice sample.

    Form fields: name (required), department, position, is_admin_default.
    When ``audio`` is present, the file is embedded and discarded; only
    the encrypted embedding is stored.
    """
    if not name or not name.strip():
        raise HTTPException(
            status_code=400, detail="Speaker name is required"
        )
    if audio is not None:
        _validate_audio(audio)

    # Lazy import — voice_enrollment depends on pyannote and torch
    # which we don't want to pin at module load if they're missing.
    try:
        from backend.engine import voice_enrollment as ve
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                "Voice enrollment unavailable. "
                "Ensure pyannote-audio and torch are installed: "
                f"{e}"
            ),
        ) from e

    if audio is None:
        try:
            profile = ve.create_profile_metadata(
                name=name.strip(),
                department=department,
                position=position,
                is_admin_default=is_admin_default,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        logger.info(
            f"Created employee profile: {profile['name']} ({profile['id']})"
        )
        return _profile_dict_to_model(profile)

    tmp_path = await _save_upload_to_temp(audio)
    try:
        try:
            profile = ve.create_profile(
                name=name.strip(),
                department=department,
                position=position,
                audio_path=tmp_path,
                is_admin_default=is_admin_default,
            )
        except RuntimeError as e:
            # extract_embedding raises RuntimeError on model load fail
            # or shape mismatch; surface as 500 with a clear message.
            logger.exception("Voice enrollment failed")
            raise HTTPException(
                status_code=500, detail=str(e),
            ) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info(f"Enrolled voice profile: {profile['name']} ({profile['id']})")
    return _profile_dict_to_model(profile)


@router.get("/", response_model=list[SpeakerProfile])
async def list_speakers(
    user: TokenData = Depends(get_current_user),
) -> list[SpeakerProfile]:
    """List all enrolled voice profiles."""
    try:
        from backend.engine import voice_enrollment as ve
        profiles = ve.list_profiles()
    except ImportError:
        # Voice enrollment module unavailable — return empty list
        # so the UI can still render its empty state cleanly.
        return []
    return [_profile_dict_to_model(p) for p in profiles]


@router.get("/{speaker_id}", response_model=SpeakerProfile)
async def get_speaker(
    speaker_id: str,
    user: TokenData = Depends(get_current_user),
) -> SpeakerProfile:
    """Get a single voice profile by id."""
    try:
        from backend.engine import voice_enrollment as ve
        profile = ve._load_profile(speaker_id)
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Voice enrollment unavailable: {e}",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _profile_dict_to_model(profile)


@router.put("/{speaker_id}", response_model=SpeakerProfile)
async def update_speaker(
    speaker_id: str,
    name: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    position: Optional[str] = Form(None),
    is_admin_default: Optional[bool] = Form(None),
    user: TokenData = Depends(get_current_user),
) -> SpeakerProfile:
    """Update employee/profile metadata.

    Does not touch embeddings — to refresh a voice signature, POST a
    new sample to ``/{id}/samples``. To replace an existing voice,
    delete the profile and enrol fresh.
    """
    try:
        from backend.engine import voice_enrollment as ve
        profile = ve.update_profile(
            speaker_id,
            name=name,
            department=department,
            position=position,
            is_admin_default=is_admin_default,
        )
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Voice enrollment unavailable: {e}",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return _profile_dict_to_model(profile)


@router.post("/{speaker_id}/samples", response_model=SpeakerProfile)
async def add_speaker_sample(
    speaker_id: str,
    audio: UploadFile = File(...),
    user: TokenData = Depends(get_current_user),
) -> SpeakerProfile:
    """Add an additional voice sample to an existing profile.

    More samples = better averaged embedding = better match accuracy
    over time as the voice changes (cold, microphone difference,
    etc.).
    """
    _validate_audio(audio)
    try:
        from backend.engine import voice_enrollment as ve
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Voice enrollment unavailable: {e}",
        ) from e

    tmp_path = await _save_upload_to_temp(audio)
    try:
        try:
            profile = ve.add_sample(speaker_id, tmp_path)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    return _profile_dict_to_model(profile)


@router.delete("/{speaker_id}")
async def delete_speaker(
    speaker_id: str,
    user: TokenData = Depends(get_current_user),
) -> dict:
    """Delete a voice profile and all associated embeddings.

    This is irreversible — once deleted, the voice can only be
    recovered by re-enrolling from a fresh sample.
    """
    try:
        from backend.engine import voice_enrollment as ve
        ok = ve.delete_profile(speaker_id)
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Voice enrollment unavailable: {e}",
        ) from e
    if not ok:
        raise HTTPException(
            status_code=404, detail=f"Speaker not found: {speaker_id}",
        )
    logger.info(f"Deleted voice profile: {speaker_id}")
    return {"status": "deleted", "speaker_id": speaker_id}


# ---------------------------------------------------------------------
# Legacy alias — frontend still posts to /register; remove once UI
# migrates to the new POST / endpoint.
# ---------------------------------------------------------------------

@router.post("/register", response_model=SpeakerProfile)
async def register_speaker_legacy(
    name: str = Form(...),
    position: Optional[str] = Form(None),
    audio: UploadFile = File(...),
    user: TokenData = Depends(get_current_user),
) -> SpeakerProfile:
    """Legacy enrollment endpoint — same behaviour as ``POST /``.

    Kept until the frontend migrates. New clients should use
    ``POST /api/speakers/`` directly.
    """
    return await enrol_speaker(
        name=name,
        department="",
        position=position or "",
        is_admin_default=False,
        audio=audio,
        user=user,
    )
