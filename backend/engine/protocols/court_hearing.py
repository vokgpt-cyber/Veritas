"""Court hearing protocol builder.

Output matches the gold-standard stenogram format:
    - Title: "Стенограмма судебного заседания № X от DD.MM.YYYY"
    - Brief summary (what was heard, who was present, what was decided)
    - Verbatim Q&A table (two columns: speaker | utterance)

Critical design decision (2026-04-21): the LLM does NOT produce the
verbatim turns. It only produces the summary, participant list, and
metadata (case number, date). The turns are taken DIRECTLY from the
aligned transcript — zero risk of LLM paraphrasing, reordering, or
dropping utterances.

This matters because a court stenogram is a legal artefact. Any LLM
paraphrase invalidates its evidentiary value.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from backend.app.models import AlignedSegment
from backend.engine.protocols.schemas import (
    CourtHearingProtocol,
    CourtHearingTurn,
    ProtocolResult,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Prompt construction
# =====================================================================

def build_prompt(
    transcript_text: str,
    language: str = "ru",
    meeting_context: str = "",
) -> list[dict]:
    """Build the Ollama chat prompt for court-hearing summary extraction.

    The LLM is instructed to output ONLY structured JSON. The turns
    field is NOT requested — turns come from the aligned transcript.

    Args:
        transcript_text: Formatted transcript (speaker [timestamp]: text).
        language: "ru" or "en". Court hearings are effectively ru-only
            for VERITAS but we preserve the parameter.
        meeting_context: User-provided context attached to the job.

    Returns:
        List of Ollama chat messages.
    """
    system = (
        "You are a court stenographer's assistant preparing the summary "
        "header for a verbatim court hearing transcript. The Russian "
        "transcript will follow.\n\n"
        "YOUR TASK: produce a strict JSON object with the following schema "
        "describing WHAT happened in the hearing. Do NOT reproduce the "
        "turns — the verbatim transcript is already captured.\n\n"
        "JSON SCHEMA (all string values must be Russian):\n"
        "{\n"
        '  "case_number": "<case/claim number if stated in transcript, else empty>",\n'
        '  "hearing_date": "<DD.MM.YYYY if stated, else empty>",\n'
        '  "summary": "<3-7 sentences in Russian: the matter heard, '
        'what the parties argued, what the court decided or postponed. '
        'Strictly factual, no inference.>",\n'
        '  "participants": ["<role or name>", ...],\n'
        '  "speaker_roles": {"<speaker id like SPEAKER_01>": "<role label '
        'like Суд or Истец or Ответчик or individual name>", ...}\n'
        "}\n\n"
        "STRICT ANTI-HALLUCINATION RULES:\n"
        "- Extract ONLY what the transcript says explicitly.\n"
        "- If a field has no support in the transcript, use an empty "
        'string ("") or empty array ([]). NEVER invent.\n'
        "- summary: cite the actual case matter, the parties' positions, "
        "and the actual ruling if the court stated one. Use neutral "
        "language; do not characterise or editorialise.\n"
        "- participants: list the roles present (e.g., 'Суд', 'Истец', "
        "'Ответчик', 'Представитель истца') plus individual names when "
        "stated. Do NOT invent names.\n"
        "- speaker_roles: map the diarization labels (SPEAKER_01, "
        "SPEAKER_02 etc.) to their role in the hearing. If you cannot "
        "tell which speaker is which, leave the mapping empty — do NOT "
        "guess.\n"
        "- OUTPUT ONLY the JSON object. No markdown code fences, no "
        "commentary before or after."
    )

    if meeting_context:
        system += (
            f"\n\nAdditional case context provided by the operator "
            f"(use only if consistent with the transcript):\n{meeting_context}"
        )

    user = f"Court hearing transcript (Russian):\n\n{transcript_text}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_verification_prompt(
    protocol_text: str,
    transcript_text: str,
    language: str = "ru",
) -> list[dict]:
    """Verification pass for the court hearing summary.

    The verifier checks the summary narrative against the transcript
    and removes any claim that isn't supported. Returns the same JSON
    shape (possibly with fields blanked out).
    """
    system = (
        "You are a strict fact-checker for a court hearing protocol. "
        "Your only job is to remove unsupported claims from the JSON.\n\n"
        "RULES:\n"
        "1. For every fact in the summary, find a literal word-level "
        "match in the transcript that supports it.\n"
        "2. If a claim cannot be verified, DELETE that sentence from the "
        "summary. Do not annotate. Do not explain.\n"
        "3. If speaker_roles misattributes a speaker (the transcript "
        "evidence contradicts the role), FIX the mapping or remove it.\n"
        "4. If case_number or hearing_date is not found in the transcript, "
        'blank the field (empty string "").\n'
        "5. Never add new information. You can only delete or correct.\n"
        "6. OUTPUT ONLY the corrected JSON object. Same schema as input."
    )

    max_transcript = 20000
    truncated = transcript_text[:max_transcript]
    user = (
        f"PROTOCOL JSON TO VERIFY:\n\n{protocol_text}\n\n"
        f"---\n\nTRANSCRIPT:\n\n{truncated}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# =====================================================================
# Parsing
# =====================================================================

# Speaker-label normalisation for the gold-standard Q&A table.
# Maps common role strings (case-insensitive) to the gold display form.
_ROLE_DISPLAY: dict[str, str] = {
    "суд": "Суд:",
    "court": "Суд:",
    "judge": "Суд:",
    "истец": "Истец:",
    "claimant": "Истец:",
    "ответчик": "Ответчик:",
    "defendant": "Ответчик:",
    "представитель истца": "Представитель истца:",
    "представитель ответчика": "Представитель ответчика:",
    "третье лицо": "Третье лицо:",
    "мужской голос": "М. Голос:",
    "male voice": "М. Голос:",
    "женский голос": "Ж. Голос:",
    "female voice": "Ж. Голос:",
}


def _normalise_role_label(raw: str) -> str:
    """Convert an LLM-emitted role label to the gold-standard display form.

    Falls back to adding a trailing colon if the label doesn't match any
    known role (e.g., an individual name like "Голубев" becomes "Голубев:").
    """
    if not raw:
        return ""
    needle = raw.strip().rstrip(":").strip().lower()
    if needle in _ROLE_DISPLAY:
        return _ROLE_DISPLAY[needle]
    # Individual name or unknown role — preserve as-is with trailing colon.
    label = raw.strip().rstrip(":").strip()
    return f"{label}:" if label else ""


def parse_response(
    raw_text: str,
    aligned: list[AlignedSegment],
) -> ProtocolResult:
    """Parse the LLM JSON response and combine with aligned transcript.

    The LLM provides the summary header; the aligned transcript provides
    the verbatim turns. Output is a ProtocolResult wrapping a
    CourtHearingProtocol.

    Args:
        raw_text: Raw LLM output — expected to be a JSON object.
        aligned: Aligned transcript segments (source of truth for turns).

    Returns:
        ProtocolResult with CourtHearingProtocol payload.
    """
    parsed = _extract_json(raw_text) or {}
    speaker_roles: dict[str, str] = {}
    roles_raw = parsed.get("speaker_roles", {}) or {}
    if isinstance(roles_raw, dict):
        for spk_id, role in roles_raw.items():
            if isinstance(spk_id, str) and isinstance(role, str):
                speaker_roles[spk_id.strip()] = role.strip()

    # Build the verbatim Q&A table from aligned segments.
    turns: list[CourtHearingTurn] = []
    for seg in aligned:
        text = (seg.text or "").strip()
        if not text:
            continue
        # Prefer an explicit speaker_name (post-rename), then LLM-supplied
        # role mapping, then raw speaker_id.
        label_source = (
            seg.speaker_name
            or speaker_roles.get(seg.speaker_id, "")
            or seg.speaker_id
            or ""
        )
        label = _normalise_role_label(label_source)
        # Capture segment start time for [HH:MM:SS] rendering in DOCX.
        start_s = getattr(seg, "start", None)
        try:
            start_s = float(start_s) if start_s is not None else None
        except (TypeError, ValueError):
            start_s = None
        turns.append(
            CourtHearingTurn(speaker=label, text=text, start_s=start_s)
        )

    # Participants: prefer explicit list from LLM, else derive from roles map,
    # else derive from segments directly.
    participants_raw = parsed.get("participants") or []
    participants: list[str] = []
    if isinstance(participants_raw, list):
        participants = [str(p).strip() for p in participants_raw if str(p).strip()]
    if not participants:
        # Derive from unique speaker labels in the Q&A table.
        seen: set[str] = set()
        for turn in turns:
            if turn.speaker and turn.speaker not in seen:
                seen.add(turn.speaker)
                participants.append(turn.speaker.rstrip(":"))

    title_base = "Стенограмма судебного заседания"
    case_number = str(parsed.get("case_number") or "").strip()
    hearing_date = str(parsed.get("hearing_date") or "").strip()
    if case_number:
        title = f"{title_base} № {case_number}"
    else:
        title = title_base
    if hearing_date:
        title = f"{title} от {hearing_date}"

    summary = str(parsed.get("summary") or "").strip()

    payload = CourtHearingProtocol(
        title=title,
        case_number=case_number,
        hearing_date=hearing_date,
        summary=summary,
        participants=participants,
        turns=turns,
    )
    return ProtocolResult(meeting_type="court_hearing", payload=payload)


# =====================================================================
# JSON extraction helper (shared style across protocol builders)
# =====================================================================

def _extract_json(raw: str) -> Optional[dict[str, Any]]:
    """Extract the first top-level JSON object from raw LLM output.

    Ollama's format:"json" option produces clean JSON, but we still guard
    against markdown code fences and stray prose.
    """
    if not raw:
        return None

    text = raw.strip()
    # Strip markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    # Try direct parse first.
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: find the first {...} block.
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse fallback JSON block: %s", exc)

    logger.warning("Court hearing LLM output was not valid JSON; returning empty")
    return None


__all__ = [
    "build_prompt",
    "build_verification_prompt",
    "parse_response",
]
