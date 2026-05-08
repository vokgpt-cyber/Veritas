"""Pydantic data models for EPAM VERITAS v2.0."""
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class WordInfo(BaseModel):
    """Individual word with timing and confidence information."""
    start: float
    end: float
    word: str
    confidence: float

    model_config = ConfigDict(from_attributes=True)


class TranscriptionSegment(BaseModel):
    """Raw transcription segment from ASR engine."""
    start: float
    end: float
    text: str
    confidence: float
    words: list[WordInfo] = []

    model_config = ConfigDict(from_attributes=True)


class DiarizationSegment(BaseModel):
    """Speaker diarization segment."""
    start: float
    end: float
    speaker_id: str
    speaker_name: Optional[str] = None
    # True when this segment temporally overlaps with another speaker's
    # segment (interruption / talking-over). Set by pyannote 4.0
    # overlap detection in pyannote_diarization.py. Used by the aligner
    # to downgrade attribution_confidence so the editor UI can flag
    # these as "verify here". Added Tier 2 (2026-04-23) per stakeholder
    # feedback that interruptions are mis-attributed.
    is_overlap: bool = False

    model_config = ConfigDict(from_attributes=True)


class AlignedSegment(BaseModel):
    """Transcript segment with speaker alignment."""
    start: float
    end: float
    text: str
    speaker_id: str
    speaker_name: Optional[str] = None
    # Optional non-authoritative voice-profile match. Court hearings use
    # this as a review hint instead of silently replacing the speaker name.
    suggested_speaker_name: Optional[str] = None
    suggested_speaker_confidence: Optional[float] = None
    confidence: float = 1.0
    # Speaker-attribution confidence: winner_speaker_time / total_diarized_time
    # within this segment. 1.0 = single speaker owns the whole chunk,
    # 0.5 = contested between two speakers, None = no diarization overlap found.
    # Values below ~0.6 flag that the ASR chunk spans multiple speaker turns
    # and the attribution should be treated as uncertain.
    attribution_confidence: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class Participant(BaseModel):
    """Meeting participant information."""
    speaker_id: str
    speaker_name: Optional[str] = None
    speaking_time: float = 0.0
    speaking_share: float = 0.0

    model_config = ConfigDict(from_attributes=True)


class TopicItem(BaseModel):
    """Meeting topic/discussion item."""
    title: str
    content: str
    speakers: list[str] = []

    model_config = ConfigDict(from_attributes=True)


class Decision(BaseModel):
    """Meeting decision."""
    text: str
    responsible: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TaskItem(BaseModel):
    """Action item/task from meeting."""
    text: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class MeetingProtocol(BaseModel):
    """Complete meeting protocol/minutes."""
    meeting_date: Optional[datetime] = None
    topic: str = ""
    participants: list[Participant] = []
    summary: str = ""
    key_topics: list[TopicItem] = []
    decisions: list[Decision] = []
    tasks: list[TaskItem] = []
    open_questions: list[str] = []
    transcript: list[AlignedSegment] = []

    model_config = ConfigDict(from_attributes=True)


class QAIssue(BaseModel):
    """Quality assurance issue."""
    severity: Literal["critical", "warning", "info"]
    stage: str
    message: str
    details: Optional[dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class QAResult(BaseModel):
    """Quality assurance validation result."""
    passed: bool
    score: float = 0.0
    issues: list[QAIssue] = []
    warnings: list[str] = []
    suggestions: list[str] = []

    model_config = ConfigDict(from_attributes=True)


class PipelineState(str, Enum):
    """Pipeline processing state."""
    UPLOADED = "uploaded"
    PREPROCESSING = "preprocessing"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    ALIGNING = "aligning"
    SUMMARIZING = "summarizing"
    QA_VALIDATING = "qa_validating"
    FORMATTING = "formatting"
    COMPLETED = "completed"
    ERROR = "error"
    RETRYING = "retrying"


class MeetingType(str, Enum):
    """Meeting type — selects the prompt template, output schema, and
    DOCX formatter used during summarization.

    COURT_HEARING: courtroom transcripts, rendered as Q&A table matching
                   the gold-standard stenogram format.
    ADMINISTRATIVE: internal EPAM admin meetings, output grouped by
                   department (HR, Finance, IT, PR, IAO, AHO, etc.)
                   into Decisions / Tasks / Open Issues.
    CLIENT_MEETING: meetings with clients — identifies client, EPAM reps
                   and client reps, topics, commitments, follow-ups.
    INTERVIEW: job interviews — produces dual output (internal
                   confidential scorecard + external developmental feedback).
    GENERIC: fallback for uncategorized meetings — the original 7-section
                   protocol format kept for backward compatibility.
    """

    COURT_HEARING = "court_hearing"
    ADMINISTRATIVE = "administrative"
    CLIENT_MEETING = "client_meeting"
    INTERVIEW = "interview"
    GENERIC = "generic"


class MeetingJob(BaseModel):
    """Meeting processing job tracking."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    state: PipelineState = PipelineState.UPLOADED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    progress: float = 0.0
    current_stage: str = ""
    eta_seconds: Optional[float] = None
    error: Optional[str] = None
    retry_count: int = 0
    context: str = ""  # User-provided meeting context (agenda, notes, file text)
    meeting_type: MeetingType = MeetingType.GENERIC  # Selects summarization prompt + DOCX template
    owner_username: Optional[str] = None
    expected_speakers_min: Optional[int] = None
    expected_speakers_max: Optional[int] = None
    court_participants: str = ""
    court_dictionary: str = ""

    model_config = ConfigDict(from_attributes=True)


class SpeakerProfile(BaseModel):
    """Registered speaker profile for identification.

    Sprint 2026-04-30 (Block 7) extended this model with department,
    sample_count, match_count, and last_used_at to support the new
    voice enrollment workflow. Old fields (embedding_path,
    meetings_count) are kept for backward compatibility with the
    legacy JSON-file storage path; new code uses the
    backend.engine.voice_enrollment module which stores embeddings
    in encrypted SQLite at the shared data root.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    position: Optional[str] = None
    department: str = ""  # Block 7
    is_admin_default: bool = False
    embedding_path: str = ""  # legacy file-based storage
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None  # Block 7
    meetings_count: int = 0  # legacy
    sample_count: int = 0  # Block 7: number of voice samples enrolled
    match_count: int = 0  # Block 7: how often this profile matched

    model_config = ConfigDict(from_attributes=True)


class AudioMetadata(BaseModel):
    """Audio file metadata."""
    duration: float
    sample_rate: int
    channels: int
    codec: str
    bitrate: Optional[int] = None
    file_size: int = 0

    model_config = ConfigDict(from_attributes=True)
