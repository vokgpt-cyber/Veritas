"""Typed result schemas for meeting-type-specific protocols.

Each meeting type has its own result schema. A generic `ProtocolResult`
wraps them so the engine can return a single opaque value regardless of
type, and the formatter dispatches on the wrapped payload's class.

These schemas are the CONTRACT between prompt parsing and DOCX
formatting. If a field is optional, the formatter must tolerate None.
If a field is required, the parser must guarantee a value (even if
empty) before the engine returns.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from backend.app.models import MeetingProtocol


# =====================================================================
# Court hearing
# =====================================================================

class CourtHearingTurn(BaseModel):
    """One exchange in a court-hearing transcript: speaker + verbatim text.

    Court transcripts are verbatim — the LLM does NOT paraphrase. The
    speaker label is a normalised form ('Суд:', 'Д. Голубев:', etc.)
    matching the gold standard format.

    The optional start_s timestamp (seconds from audio start) is rendered
    in the DOCX as [HH:MM:SS] before the speaker label so lawyers can
    cross-reference the audio. Added 2026-04-23 per stakeholder
    feedback ("практически всегда есть такой запрос от юристов").
    """

    speaker: str
    text: str
    # Optional segment start time in seconds from audio start. None for
    # turns that don't have timestamp metadata (e.g. summary-mode runs
    # or older parsed protocols).
    start_s: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class CourtHearingProtocol(BaseModel):
    """Court hearing protocol — stenogram with summary header.

    The DOCX render is title page + summary + two-column Q&A table.
    Page numbers added by the formatter (user requirement 2026-04-21).
    """

    # Title: "Стенограмма судебного заседания № 123 от 13.04.2026"
    title: str = "Стенограмма судебного заседания"
    # Case/matter number if extractable from transcript context, else empty.
    case_number: str = ""
    # Hearing date (DD.MM.YYYY) if mentioned in transcript, else empty.
    hearing_date: str = ""
    # Brief summary — strictly factual, no inference. Names parties,
    # what was heard, what was decided. 3-7 sentences.
    summary: str = ""
    # Parties / participants identified in the transcript.
    # "Суд", "Истец", "Ответчик", plus individual names when stated.
    participants: list[str] = Field(default_factory=list)
    # Verbatim Q&A table. Speaker labels normalised to the gold format.
    turns: list[CourtHearingTurn] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# Administrative meeting
# =====================================================================

class DepartmentItem(BaseModel):
    """A single decision, task, or open issue for one department.

    The `text` is the substance. `speaker` attribution is required when
    the LLM can identify who raised the item — it's the anti-hallucination
    anchor (the verification pass drops items with no attributable
    speaker).
    """

    text: str
    # Speaker who raised/decided/assigned this item. Empty string if the
    # LLM couldn't attribute — verification pass will drop such items.
    speaker: str = ""
    # Optional deadline for tasks only. DD.MM.YYYY or free text.
    deadline: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class DepartmentBlock(BaseModel):
    """A department's contributions across all three sections.

    The formatter renders: Department heading, then three sub-sections
    (Decisions, Tasks, Open Issues). Empty sub-sections get a "Нет"
    placeholder so readers see the department was considered and had
    nothing for that bucket.
    """

    # Canonical slug from departments.DEPARTMENT_SLUGS_ORDERED.
    slug: str
    # Display name (Cyrillic) cached for the formatter.
    display_name: str
    decisions: list[DepartmentItem] = Field(default_factory=list)
    tasks: list[DepartmentItem] = Field(default_factory=list)
    open_issues: list[DepartmentItem] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

    def is_empty(self) -> bool:
        """True when the department has nothing across all three sections."""
        return (
            not self.decisions
            and not self.tasks
            and not self.open_issues
        )


class FlatProtocolItem(BaseModel):
    """One decision / task / open question / risk in the admin schema.

    Originally the flat 9-section format (T3.3, 2026-04-23) used a
    verification pass to DROP items without `evidence`. That defensive
    pattern produced sparse protocols (2-3 decisions on hour-long
    meetings). 2026-05-04 (task #45) the model dropped the second
    verification call and instead asks the first call to attach a
    `confidence` label per item. UI surfaces low-confidence items
    with an amber tag so the user curates manually — same pattern as
    Otter/Fireflies/Notion AI in production.

    speaker / owner / deadline / evidence remain on the schema but
    are now permissive: empty values are allowed, the user fills
    them post-hoc. This matches user feedback 2026-05-04: "людей и
    сроки мы найдём потом как проставить".
    """

    # Practical admin item type. Older protocols did not have this
    # field; defaulting to "thesis" keeps them renderable.
    kind: Literal["decision", "task", "open_question", "risk", "thesis"] = "thesis"
    text: str
    # Speaker who raised / decided / proposed the item. May be empty
    # when the LLM couldn't attribute and the user will fill in.
    speaker: str = ""
    # Short evidence phrase from the transcript. Helps the user
    # cross-reference the item against the audio. Empty allowed —
    # used to be required, gating dropped 2026-05-04.
    evidence: str = ""
    # Optional transcript timestamp close to the evidence quote.
    # Free text because the LLM may return "00:12:34" while legacy
    # protocols do not carry it at all.
    timestamp: str = ""
    # Department or person responsible for executing the task.
    # Tasks-only field. None for non-task items.
    owner: Optional[str] = None
    # Optional deadline. DD.MM.YYYY or short free-text. None when not
    # stated in transcript.
    deadline: Optional[str] = None
    # Confidence label set by the LLM during extraction (task #45).
    # "high" — explicit, unambiguous (e.g. clear "решили" marker)
    # "medium" — clear from context but no explicit marker
    # "low" — inferred, ambiguous, or evidence is weak
    # Defaults to "medium" so older protocols (which lacked this
    # field) render reasonably without a migration script.
    # UI shows low-confidence items with an amber tag for review.
    confidence: Literal["high", "medium", "low"] = "medium"

    model_config = ConfigDict(from_attributes=True)


class TopicSummary(BaseModel):
    """One topic block in the map-reduce admin protocol layout
    (Sprint 2026-04-30, task #35).

    Replaces the flat-9-section format's collapsed buckets ("all
    decisions across all topics in one list") with topic-grouped
    structure: each topic has its own discussion paragraph plus
    its own decisions, tasks, and open questions. Reflects how
    administrative meetings actually flow — sequential topic-by-topic
    rather than category-by-category.

    Source: EPAM IT team's original prompt design which the project
    initially abandoned in favour of the flat schema; user
    re-prioritised on 2026-04-30 after observing that the flat
    format produces "bare" protocols (e.g. 2 decisions / 4 tasks
    over a 50-minute meeting that covered HR, finance, IT, and
    compliance distinctly).

    Risks live OUTSIDE topics on the parent ``AdministrativeProtocol``
    because they typically cross-cut (compliance risk affects
    multiple departments simultaneously) — putting them under one
    topic understates their scope.
    """

    # Short Russian topic name, e.g. "Подбор персонала и HR" or
    # "Дебиторка по проектам Русал/БСФ/Фосагро". 3-7 words ideal.
    name: str
    # 1-3 sentences in Russian summarising what was discussed under
    # this topic. Strictly factual, no inference.
    discussion: str = ""
    # Topic-scoped buckets. Empty = "no decisions/tasks/questions
    # were raised under this topic" (vs the flat format where empty
    # meant "no decisions in the whole meeting"). The verifier still
    # drops items without `evidence`, same as flat format.
    decisions: list[FlatProtocolItem] = Field(default_factory=list)
    tasks: list[FlatProtocolItem] = Field(default_factory=list)
    open_questions: list[FlatProtocolItem] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class AdministrativeProtocol(BaseModel):
    """Admin meeting protocol.

    Two layouts coexist (formatter dispatches on which fields are
    populated):

    * NEW flat 9-section format (T3.3, 2026-04-23) — matches EPAM IT
      schema. Used for all new admin runs. Sections, in order:
        meeting_date, meeting_goal, participants, summary, topics,
        decisions, tasks, open_questions, risks, turns (stenogram).

    * LEGACY department-grouped format — produced by older code paths
      where decisions/tasks/open_issues were subdivided by HR / Finance
      / IT / PR / IAO / AHO / Other. Kept on the schema so existing
      protocols on disk still parse and render via the legacy formatter.

    The formatter selects the right path by checking which field set is
    populated (see _format_administrative_docx in formatter.py).
    """

    title: str = "Протокол административного совещания"
    meeting_date: str = ""  # DD.MM.YYYY
    # Stated goal of the meeting if the participants articulated one.
    # Empty when not explicitly stated. Added in flat schema (T3.3).
    meeting_goal: str = ""
    summary: str = ""  # 3-7 sentences: what the meeting covered.
    participants: list[str] = Field(default_factory=list)
    # Flat-schema sections — populated by the new flat parser, empty
    # for legacy department-grouped protocols.
    topics: list[str] = Field(default_factory=list)
    # Practical v1.0 admin layout: one unified list of decisions,
    # tasks, open questions, risks, and important working theses.
    # This is the user-facing source for new DOCX output. Legacy
    # flat buckets below remain populated for compatibility with
    # existing UI/API consumers and old protocol files.
    items: list[FlatProtocolItem] = Field(default_factory=list)
    decisions: list[FlatProtocolItem] = Field(default_factory=list)
    tasks: list[FlatProtocolItem] = Field(default_factory=list)
    open_questions: list[FlatProtocolItem] = Field(default_factory=list)
    risks: list[FlatProtocolItem] = Field(default_factory=list)
    # Map-reduce topic-segmented layout (Sprint 2026-04-30, task #35).
    # When non-empty, the formatter dispatches to the topic-grouped
    # rendering path — each topic gets its own block with discussion +
    # decisions + tasks + open_questions. Risks remain at the protocol
    # level because they typically cross-cut multiple topics. Backward
    # compatible: when this list is empty, the formatter falls back to
    # the flat 9-section layout from T3.3.
    topic_summaries: list[TopicSummary] = Field(default_factory=list)
    # Legacy department-grouped section — populated only by the older
    # code path. Empty for new flat-schema protocols. Formatter detects
    # which layout to render based on which is non-empty.
    departments: list[DepartmentBlock] = Field(default_factory=list)
    # Verbatim stenogram turns from the aligned transcript, populated by
    # the parser (not by the LLM). Rendered as the final СТЕНОГРАММА
    # section in the DOCX so the reader can audit every extracted claim.
    turns: list[CourtHearingTurn] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# Client meeting
# =====================================================================

class ClientCommitment(BaseModel):
    """A commitment made during a client meeting.

    Either side can commit — `party` identifies which. Speaker is the
    individual who made the commitment. Deadline is optional.
    """

    text: str
    # Who made the commitment: 'epam' or 'client'.
    party: Literal["epam", "client"] = "epam"
    speaker: str = ""
    deadline: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ClientMeetingProtocol(BaseModel):
    """Client meeting protocol — identifies client, reps, topics, commitments.

    Per user requirement 2026-04-21: must explicitly identify the
    client, EPAM representatives, and client representatives.
    """

    title: str = "Протокол встречи с клиентом"
    meeting_date: str = ""
    # Client organisation name.
    client_name: str = ""
    # EPAM representatives present (names + optional roles).
    epam_representatives: list[str] = Field(default_factory=list)
    # Client representatives present.
    client_representatives: list[str] = Field(default_factory=list)
    summary: str = ""
    # Topics discussed — brief bullets with speaker attribution.
    topics: list[DepartmentItem] = Field(default_factory=list)
    # Client requests / concerns raised during the meeting.
    client_requests: list[DepartmentItem] = Field(default_factory=list)
    # Commitments — either side — with deadline when set.
    commitments: list[ClientCommitment] = Field(default_factory=list)
    # Follow-up actions EPAM needs to take.
    follow_ups: list[DepartmentItem] = Field(default_factory=list)
    # Questions raised but not answered in the meeting.
    parked_questions: list[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# Interview
# =====================================================================

class InterviewObservation(BaseModel):
    """One observation with supporting evidence from the transcript.

    Evidence is a verbatim quote from the candidate. Without evidence,
    the observation is unsupported and must be dropped.
    """

    observation: str
    # Verbatim candidate quote supporting this observation, or empty.
    evidence: str = ""

    model_config = ConfigDict(from_attributes=True)


class InternalInterviewReport(BaseModel):
    """Confidential scorecard for the hiring team.

    Includes strengths, concerns, technical depth signals, communication
    signals. NOT shared with the candidate. Per Block 12 vision.
    """

    role: str = ""
    candidate_name: str = ""
    summary: str = ""
    strengths: list[InterviewObservation] = Field(default_factory=list)
    concerns: list[InterviewObservation] = Field(default_factory=list)
    technical_signals: list[InterviewObservation] = Field(default_factory=list)
    communication_signals: list[InterviewObservation] = Field(default_factory=list)
    open_questions_for_next_round: list[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ExternalInterviewReport(BaseModel):
    """Developmental feedback for the candidate (EPAM-branded).

    NO scores, NO pass/fail, NO internal assessments. Purely
    developmental. Per Block 12: "a gift to every candidate regardless
    of outcome".
    """

    candidate_name: str = ""
    speaking_style_notes: list[str] = Field(default_factory=list)
    strongest_moments: list[InterviewObservation] = Field(default_factory=list)
    areas_to_develop: list[str] = Field(default_factory=list)
    closing_note: str = ""

    model_config = ConfigDict(from_attributes=True)


class InterviewProtocol(BaseModel):
    """Interview protocol — wrapper holding both internal and external reports.

    The formatter renders two separate DOCX files from this single
    payload: {job_id}_internal.docx (confidential) and {job_id}_external.docx
    (for candidate).
    """

    meeting_date: str = ""
    internal: InternalInterviewReport = Field(default_factory=InternalInterviewReport)
    external: ExternalInterviewReport = Field(default_factory=ExternalInterviewReport)

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# Generic (backward compatibility)
# =====================================================================

class GenericProtocolPayload(BaseModel):
    """Wraps the legacy 7-section protocol for unified return type.

    The existing MeetingProtocol model already holds this content. This
    payload exists only so ProtocolResult can carry it under the same
    dispatch mechanism.
    """

    summary: str = ""
    topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    tasks: list[dict] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


# =====================================================================
# Dispatch wrapper
# =====================================================================

# Union of all meeting-type-specific payloads. The engine returns one
# ProtocolResult and the formatter dispatches on `.payload.__class__`.
# MeetingProtocol (legacy 7-section) is included so the existing formatter
# path stays the source of truth for GENERIC meetings.
ProtocolPayload = Union[
    CourtHearingProtocol,
    AdministrativeProtocol,
    ClientMeetingProtocol,
    InterviewProtocol,
    MeetingProtocol,
    GenericProtocolPayload,
]


class ProtocolResult(BaseModel):
    """Engine return value carrying the type-specific payload.

    The meeting_type field is stable — the formatter uses it to dispatch
    to the right DOCX renderer. Payload shape matches the type.
    """

    meeting_type: str  # Value of MeetingType enum.
    payload: ProtocolPayload

    model_config = ConfigDict(from_attributes=True)


__all__ = [
    "CourtHearingTurn",
    "CourtHearingProtocol",
    "DepartmentItem",
    "DepartmentBlock",
    "FlatProtocolItem",
    "TopicSummary",
    "AdministrativeProtocol",
    "ClientCommitment",
    "ClientMeetingProtocol",
    "InterviewObservation",
    "InternalInterviewReport",
    "ExternalInterviewReport",
    "InterviewProtocol",
    "GenericProtocolPayload",
    "ProtocolPayload",
    "ProtocolResult",
]
