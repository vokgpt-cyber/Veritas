"""Client meeting protocol builder.

Per user requirement 2026-04-21: client meetings must explicitly identify
    - WHO the meeting is with (client organisation name),
    - WHO attended from EPAM (names + roles),
    - WHO attended from the client side (names + roles).

Structure:
    - Title, date, client identification header
    - Brief summary (what was discussed, overall outcome)
    - Topics discussed (with speaker attribution)
    - Client requests / concerns raised
    - Commitments (EPAM-side and client-side, with deadlines)
    - Follow-up actions EPAM must take
    - Parked questions (raised but not answered)

Commitments carry a 'party' field (epam|client) — critical for client
meetings because both sides make commitments and it matters who owes whom.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from backend.engine.protocols.schemas import (
    ClientCommitment,
    ClientMeetingProtocol,
    DepartmentItem,
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
    """Build the Ollama chat prompt for a client meeting.

    The LLM is instructed to emit strict JSON identifying the client,
    participants (split by side), topics, requests, commitments, follow-ups,
    and parked questions.

    Args:
        transcript_text: Formatted transcript.
        language: "ru" or "en" — client meetings are usually Russian at EPAM.
        meeting_context: User-provided context (expected to contain client
            name if the transcript alone isn't sufficient).
    """
    system = (
        "You are a meeting secretary for EPAM law firm client meetings. "
        "The Russian transcript will follow. Your task: extract the client "
        "identification, participants split by side (EPAM vs client), "
        "topics discussed, client requests, commitments, follow-ups, and "
        "unresolved questions.\n\n"
        "JSON SCHEMA (all string values in Russian unless stated):\n"
        "{\n"
        '  "meeting_date": "<DD.MM.YYYY if stated, else empty>",\n'
        '  "client_name": "<the client organisation name — REQUIRED. '
        'Use the context if the transcript does not state it directly>",\n'
        '  "epam_representatives": ["<name and role, e.g. \'Иванов А.В., '
        'партнёр\'>", ...],\n'
        '  "client_representatives": ["<name and role from the client '
        'side>", ...],\n'
        '  "summary": "<3-7 sentences: the matter discussed, the overall '
        'tone, what was agreed or left open>",\n'
        '  "topics": [\n'
        '    {"text": "<what was discussed>", "speaker": "<who raised or '
        'drove this topic>"}\n'
        '  ],\n'
        '  "client_requests": [\n'
        '    {"text": "<what the client asked for or is concerned about>", '
        '"speaker": "<which client representative raised it>"}\n'
        '  ],\n'
        '  "commitments": [\n'
        '    {"party": "epam|client", "text": "<what was promised>", '
        '"speaker": "<the person who made the commitment>", '
        '"deadline": "<DD.MM.YYYY if stated, else empty>"}\n'
        '  ],\n'
        '  "follow_ups": [\n'
        '    {"text": "<action EPAM must take after the meeting>", '
        '"speaker": "<EPAM person responsible>"}\n'
        '  ],\n'
        '  "parked_questions": ["<question raised but not answered>", ...]\n'
        "}\n\n"
        "STRICT ANTI-HALLUCINATION RULES:\n"
        "- Extract ONLY what the transcript (or the provided context) says "
        "explicitly. Do NOT invent client names, representative names, or "
        "commitments.\n"
        "- Every topic, request, commitment, and follow-up MUST have a "
        "speaker attribution. If you cannot attribute it, OMIT the item.\n"
        "- Split participants correctly: EPAM representatives go in "
        "'epam_representatives'; client representatives go in "
        "'client_representatives'. If you cannot tell which side a "
        "person is on, put them in EPAM (the transcript is from EPAM's "
        "side) and note this is a guess in the summary.\n"
        "- For commitments, 'party' MUST be 'epam' or 'client' — the side "
        "making the promise. Lawyers on our side commit for EPAM; client "
        "personnel commit for the client.\n"
        "- Deadlines must be literally stated in the transcript (e.g. 'до "
        "пятницы', 'к 15 числа', 'в течение недели'). If absent, leave "
        "empty.\n"
        "- Parked questions are things that were asked but NOT answered in "
        "the meeting. Do not list answered questions as parked.\n"
        "- Empty arrays are valid. Do NOT invent items to fill sections.\n"
        "- OUTPUT ONLY the JSON object. No markdown fences, no commentary."
    )

    if meeting_context:
        system += (
            f"\n\nMeeting context provided by the operator "
            f"(authoritative for client name and participants if "
            f"stated): \n{meeting_context}"
        )

    user = f"Client meeting transcript (Russian):\n\n{transcript_text}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_verification_prompt(
    protocol_text: str,
    transcript_text: str,
    language: str = "ru",
) -> list[dict]:
    """Verification pass for client meeting JSON.

    Drops items without transcript support. Never adds new items.
    Preserves client_name and representative lists (these often come
    from meeting context rather than transcript text).
    """
    system = (
        "You are a strict fact-checker for a client meeting protocol in "
        "JSON form.\n\n"
        "RULES:\n"
        "1. For every topic, request, commitment, and follow-up, find a "
        "literal word-level match in the transcript. If you cannot, DELETE "
        "that item.\n"
        "2. If a commitment has no party, no speaker, or no text, DELETE it.\n"
        "3. If a commitment claims a deadline not in the transcript, blank "
        "the deadline field.\n"
        "4. Preserve 'client_name', 'epam_representatives', and "
        "'client_representatives' AS-IS unless the transcript clearly "
        "contradicts them — they often come from operator-provided context, "
        "not the transcript.\n"
        "5. Preserve 'parked_questions' that were literally raised in the "
        "transcript and clearly not answered; delete those that were "
        "answered.\n"
        "6. Never add new items. You can only delete or correct fields.\n"
        "7. Preserve the JSON shape exactly. Keep the same top-level keys.\n"
        "8. OUTPUT ONLY the corrected JSON object. No markdown, no commentary."
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

def parse_response(
    raw_text: str,
    aligned: Optional[list] = None,
) -> ProtocolResult:
    """Parse the LLM JSON and build a ClientMeetingProtocol.

    Args:
        raw_text: Raw LLM output.
        aligned: Unused for client meetings (present for interface uniformity).

    Returns:
        ProtocolResult with ClientMeetingProtocol payload.
    """
    parsed = _extract_json(raw_text) or {}

    meeting_date = str(parsed.get("meeting_date") or "").strip()
    client_name = str(parsed.get("client_name") or "").strip()
    summary = str(parsed.get("summary") or "").strip()

    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip()]

    epam_reps = _string_list(parsed.get("epam_representatives"))
    client_reps = _string_list(parsed.get("client_representatives"))
    parked_questions = _string_list(parsed.get("parked_questions"))

    def _items(value: Any) -> list[DepartmentItem]:
        if not isinstance(value, list):
            return []
        result: list[DepartmentItem] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text") or "").strip()
            if not text:
                continue
            speaker = str(entry.get("speaker") or "").strip()
            deadline_raw = entry.get("deadline")
            deadline: Optional[str] = None
            if deadline_raw:
                deadline = str(deadline_raw).strip() or None
                if deadline and deadline.lower() in ("none", "null", "-", "нет"):
                    deadline = None
            result.append(DepartmentItem(
                text=text,
                speaker=speaker,
                deadline=deadline,
            ))
        return result

    topics = _items(parsed.get("topics"))
    client_requests = _items(parsed.get("client_requests"))
    follow_ups = _items(parsed.get("follow_ups"))

    # Commitments need special handling because of the party field.
    commitments: list[ClientCommitment] = []
    raw_commitments = parsed.get("commitments") or []
    if isinstance(raw_commitments, list):
        for entry in raw_commitments:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text") or "").strip()
            if not text:
                continue
            party_raw = str(entry.get("party") or "").strip().lower()
            # Canonicalise party: anything matching "client" goes to client,
            # everything else defaults to EPAM (we're from EPAM's side).
            if party_raw in ("client", "клиент", "клиента", "client side"):
                party = "client"
            else:
                party = "epam"
            speaker = str(entry.get("speaker") or "").strip()
            deadline_raw = entry.get("deadline")
            deadline: Optional[str] = None
            if deadline_raw:
                deadline = str(deadline_raw).strip() or None
                if deadline and deadline.lower() in ("none", "null", "-", "нет"):
                    deadline = None
            commitments.append(ClientCommitment(
                text=text,
                party=party,
                speaker=speaker,
                deadline=deadline,
            ))

    # Build title: "Протокол встречи с клиентом — <client_name> — DD.MM.YYYY"
    title_base = "Протокол встречи с клиентом"
    title_parts = [title_base]
    if client_name:
        title_parts.append(client_name)
    if meeting_date:
        title_parts.append(meeting_date)
    title = " — ".join(title_parts)

    payload = ClientMeetingProtocol(
        title=title,
        meeting_date=meeting_date,
        client_name=client_name,
        epam_representatives=epam_reps,
        client_representatives=client_reps,
        summary=summary,
        topics=topics,
        client_requests=client_requests,
        commitments=commitments,
        follow_ups=follow_ups,
        parked_questions=parked_questions,
    )
    return ProtocolResult(meeting_type="client_meeting", payload=payload)


# =====================================================================
# Shared JSON extractor
# =====================================================================

def _extract_json(raw: str) -> Optional[dict[str, Any]]:
    """Extract the first top-level JSON object from raw LLM output."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse fallback JSON block: %s", exc)
    logger.warning("Client meeting LLM output was not valid JSON; returning empty")
    return None


__all__ = [
    "build_prompt",
    "build_verification_prompt",
    "parse_response",
]
