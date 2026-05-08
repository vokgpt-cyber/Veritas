"""Interview protocol builder — dual-output design.

Per Block 12 vision and user confirmation 2026-04-21:

    INTERNAL report (confidential, for interviewers):
      - summary of the interview
      - strengths (with transcript evidence)
      - concerns (with transcript evidence)
      - technical depth signals
      - communication signals
      - open questions for the next round

    EXTERNAL report (for the candidate, EPAM-branded):
      - speaking style notes (neutral, observational)
      - strongest moments (positive reinforcement)
      - areas to develop (constructive, non-judgemental)
      - closing note
      - NO scores, NO pass/fail, NO internal assessments

Both reports are produced from ONE LLM pass (the internal prompt is richer
and the external report is derived from a filtered subset of the same
analysis). This keeps the two reports consistent — the external report
never mentions something the internal report hasn't already reasoned
about.

Every observation carries a 'evidence' field: a verbatim candidate quote
from the transcript. The verification pass drops any observation whose
evidence isn't found in the transcript.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from backend.engine.protocols.schemas import (
    ExternalInterviewReport,
    InternalInterviewReport,
    InterviewObservation,
    InterviewProtocol,
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
    """Build the Ollama chat prompt for an interview.

    The LLM produces BOTH reports in one pass for consistency. The
    internal section is confidential; the external section is shared
    with the candidate.

    Args:
        transcript_text: Formatted transcript (interviewer + candidate).
        language: "ru" or "en". EPAM interviews are mostly Russian.
        meeting_context: Role being interviewed for, seniority expected,
            any case-specific guidance.
    """
    system = (
        "You are an interview analyst for EPAM. The Russian transcript of "
        "a job interview will follow. You will produce TWO reports in one "
        "JSON object:\n"
        "  1. INTERNAL — confidential, for the hiring team. Honest, "
        "specific, evidence-backed.\n"
        "  2. EXTERNAL — developmental feedback for the candidate. "
        "Constructive, non-judgemental, no scores, no pass/fail.\n\n"
        "JSON SCHEMA (all string values in Russian):\n"
        "{\n"
        '  "meeting_date": "<DD.MM.YYYY if stated, else empty>",\n'
        '  "internal": {\n'
        '    "role": "<role the candidate is interviewing for>",\n'
        '    "candidate_name": "<candidate full name if stated>",\n'
        '    "summary": "<3-5 sentences: overall impression, headline '
        'strengths and concerns>",\n'
        '    "strengths": [\n'
        '      {"observation": "<what the candidate did well>", '
        '"evidence": "<verbatim candidate quote from the transcript>"}\n'
        '    ],\n'
        '    "concerns": [\n'
        '      {"observation": "<concern or gap>", '
        '"evidence": "<verbatim candidate quote>"}\n'
        '    ],\n'
        '    "technical_signals": [\n'
        '      {"observation": "<technical depth signal — '
        'e.g. understands vs. buzzwords-only>", '
        '"evidence": "<verbatim quote>"}\n'
        '    ],\n'
        '    "communication_signals": [\n'
        '      {"observation": "<clarity, structure, confidence, '
        'engagement>", "evidence": "<verbatim quote>"}\n'
        '    ],\n'
        '    "open_questions_for_next_round": ["<topic or question to '
        'probe in the next round>", ...]\n'
        '  },\n'
        '  "external": {\n'
        '    "candidate_name": "<candidate name>",\n'
        '    "speaking_style_notes": ["<neutral observation about the '
        'candidate\\\'s speaking style — pace, filler words, structure>"'
        ", ...],\n"
        '    "strongest_moments": [\n'
        '      {"observation": "<what the candidate did best>", '
        '"evidence": "<verbatim quote>"}\n'
        '    ],\n'
        '    "areas_to_develop": ["<constructive suggestion, phrased '
        'developmentally>", ...],\n'
        '    "closing_note": "<1-2 warm sentences thanking the candidate '
        'and wishing them well>"\n'
        '  }\n'
        "}\n\n"
        "STRICT RULES:\n"
        "- Every 'observation' MUST have 'evidence' — a verbatim candidate "
        "quote from the transcript. If you cannot quote the candidate, "
        "OMIT the observation.\n"
        "- Use ONLY what the transcript says. Do NOT speculate about "
        "prior experience or attitudes not shown in the interview.\n"
        "- The internal 'concerns' section must stay PROFESSIONAL — focus "
        "on skill/fit gaps with evidence, never personality judgements.\n"
        "- The external report MUST NOT include: scores, pass/fail, "
        "hire/no-hire, comparison to other candidates, salary, confidential "
        "assessments, any internal 'concerns'.\n"
        "- 'areas_to_develop' in the external report should be phrased "
        "developmentally: 'You might practise structuring answers with "
        "examples' — NOT 'You lack structure'.\n"
        "- 'closing_note' must be warm and wish the candidate well, "
        "regardless of hiring outcome.\n"
        "- Empty arrays are valid. Do NOT invent items.\n"
        "- OUTPUT ONLY the JSON object. No markdown fences, no commentary."
    )

    if meeting_context:
        system += (
            f"\n\nInterview context provided by the operator "
            f"(role, seniority, specific areas to probe):\n{meeting_context}"
        )

    user = f"Interview transcript (Russian):\n\n{transcript_text}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_verification_prompt(
    protocol_text: str,
    transcript_text: str,
    language: str = "ru",
) -> list[dict]:
    """Verification pass for interview JSON.

    For every observation (internal AND external), verify the evidence
    quote exists in the transcript. Drop unsupported observations.
    """
    system = (
        "You are a strict fact-checker for an interview report in JSON "
        "form. Your job is to remove unsupported observations.\n\n"
        "RULES:\n"
        "1. For every 'observation' in any of the arrays (internal.strengths, "
        "internal.concerns, internal.technical_signals, "
        "internal.communication_signals, external.strongest_moments), check "
        "that the 'evidence' field is a verbatim or near-verbatim quote "
        "from the transcript. If it is not, DELETE that observation.\n"
        "2. If an observation has empty or trivial evidence (e.g. a single "
        "word), DELETE it.\n"
        "3. Preserve candidate_name and role — these often come from "
        "context rather than transcript.\n"
        "4. Preserve 'speaking_style_notes', 'areas_to_develop', "
        "'closing_note', 'open_questions_for_next_round' AS-IS unless they "
        "explicitly contradict the transcript — these are analytical "
        "summaries, not quotable events.\n"
        "5. Enforce the external-report firewall: if the external section "
        "contains anything score-like, pass/fail, or any content that "
        "looks copied from internal.concerns, DELETE those fields.\n"
        "6. Never add new content. Only delete or blank.\n"
        "7. Preserve JSON shape exactly. Keep top-level keys.\n"
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
    """Parse the LLM JSON and build an InterviewProtocol.

    Args:
        raw_text: Raw LLM output.
        aligned: Unused for interviews (present for interface uniformity).

    Returns:
        ProtocolResult with InterviewProtocol payload containing both
        internal and external reports.
    """
    parsed = _extract_json(raw_text) or {}

    meeting_date = str(parsed.get("meeting_date") or "").strip()

    internal_raw = parsed.get("internal") or {}
    external_raw = parsed.get("external") or {}
    if not isinstance(internal_raw, dict):
        internal_raw = {}
    if not isinstance(external_raw, dict):
        external_raw = {}

    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip()]

    def _observations(value: Any) -> list[InterviewObservation]:
        if not isinstance(value, list):
            return []
        result: list[InterviewObservation] = []
        for entry in value:
            if not isinstance(entry, dict):
                continue
            observation = str(entry.get("observation") or "").strip()
            if not observation:
                continue
            evidence = str(entry.get("evidence") or "").strip()
            result.append(InterviewObservation(
                observation=observation,
                evidence=evidence,
            ))
        return result

    internal = InternalInterviewReport(
        role=str(internal_raw.get("role") or "").strip(),
        candidate_name=str(internal_raw.get("candidate_name") or "").strip(),
        summary=str(internal_raw.get("summary") or "").strip(),
        strengths=_observations(internal_raw.get("strengths")),
        concerns=_observations(internal_raw.get("concerns")),
        technical_signals=_observations(internal_raw.get("technical_signals")),
        communication_signals=_observations(
            internal_raw.get("communication_signals")
        ),
        open_questions_for_next_round=_string_list(
            internal_raw.get("open_questions_for_next_round")
        ),
    )

    external = ExternalInterviewReport(
        candidate_name=str(external_raw.get("candidate_name") or "").strip(),
        speaking_style_notes=_string_list(
            external_raw.get("speaking_style_notes")
        ),
        strongest_moments=_observations(external_raw.get("strongest_moments")),
        areas_to_develop=_string_list(external_raw.get("areas_to_develop")),
        closing_note=str(external_raw.get("closing_note") or "").strip(),
    )

    # Fall back: if external lacks a name, use internal's.
    if not external.candidate_name and internal.candidate_name:
        external = ExternalInterviewReport(
            candidate_name=internal.candidate_name,
            speaking_style_notes=external.speaking_style_notes,
            strongest_moments=external.strongest_moments,
            areas_to_develop=external.areas_to_develop,
            closing_note=external.closing_note,
        )

    payload = InterviewProtocol(
        meeting_date=meeting_date,
        internal=internal,
        external=external,
    )
    return ProtocolResult(meeting_type="interview", payload=payload)


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
    logger.warning("Interview LLM output was not valid JSON; returning empty")
    return None


__all__ = [
    "build_prompt",
    "build_verification_prompt",
    "parse_response",
]
