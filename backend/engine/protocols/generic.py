"""Generic meeting protocol builder (backward compatibility).

Wraps the legacy 7-section Russian protocol in the new builder
interface so the dispatch layer can treat it uniformly. The prompt
matches what the pre-Block-7 summarization engine used — decisions,
tasks (with assignee + deadline), and open questions, structured as
plain-text sections with labelled headers.

Used when:
    - meeting_type is MeetingType.GENERIC (fallback)
    - meeting_type is unrecognised (safety net in the dispatcher)

The legacy MeetingProtocol model still holds the result; we wrap it
in GenericProtocolPayload for the ProtocolResult union, but the
formatter already knows how to render the legacy shape, so the
dispatch on payload type falls through to the existing renderer.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from backend.engine.protocols.schemas import GenericProtocolPayload, ProtocolResult

logger = logging.getLogger(__name__)


# =====================================================================
# Prompt construction
# =====================================================================

def build_prompt(
    transcript_text: str,
    language: str = "ru",
    meeting_context: str = "",
) -> list[dict]:
    """Build the generic 7-section protocol prompt.

    Preserves the legacy prompt format (English section headers,
    Russian content) that the parser in summarization.py already
    understands via _parse_protocol_sections.
    """
    if language == "ru":
        system = (
            "You are an experienced meeting secretary for a Russian law firm. "
            "Your task: produce a structured meeting protocol from the "
            "transcript below. The transcript is in Russian. Write ALL "
            "content in Russian, but use the EXACT English section headers "
            "shown below.\n\n"
            "STRICT ANTI-HALLUCINATION RULES:\n"
            "- Extract ONLY what is EXPLICITLY said in the transcript.\n"
            "- NEVER invent, add, or assume information not in the transcript.\n"
            "- NEVER add people, facts, dates, numbers, or events not mentioned.\n"
            "- When attributing a statement, you MUST name the speaker "
            "(e.g., 'Speaker_1 proposed...' or 'According to Speaker_3...').\n"
            "- If you cannot find a speaker for a claim, DO NOT include it.\n"
            "- If a section has no relevant content, write 'None' under it.\n"
            "- It is BETTER to write 'None' than to guess or invent content.\n"
            "- Use the EXACT format below. Do NOT add extra sections.\n"
            "- Write summary and topics in Russian language.\n"
            "- Do NOT wrap output in markdown code blocks.\n\n"
            "RESPONSE FORMAT (use these exact headers):\n\n"
            "SUMMARY:\n"
            "(3-7 sentences in Russian: main topic, key positions of "
            "participants with their names, outcomes)\n\n"
            "TOPICS:\n"
            "- (topic 1 in Russian)\n"
            "- (topic 2 in Russian)\n\n"
            "DECISIONS:\n"
            "- (only if someone EXPLICITLY said 'we decided' or 'it was "
            "agreed'. Include WHO proposed it. Otherwise write: None)\n\n"
            "TASKS:\n"
            "- (what to do | who was assigned | deadline. Only if "
            "EXPLICITLY assigned in the transcript. Otherwise write: None)\n\n"
            "QUESTIONS:\n"
            "- (unresolved questions ACTUALLY raised by a named speaker. "
            "Otherwise write: None)"
        )
        if meeting_context:
            system += (
                f"\n\nAdditional meeting context "
                f"(provided by user):\n{meeting_context}"
            )
        user = f"Meeting transcript (in Russian):\n\n{transcript_text}"
    else:
        system = (
            "You are an experienced meeting secretary for a law firm. "
            "Your task is to produce a complete meeting protocol based on "
            "the transcript. Write in professional business English.\n\n"
            "RULES:\n"
            "- Extract ONLY what is EXPLICITLY stated in the transcript.\n"
            "- Do NOT invent, add, or assume anything.\n"
            "- If a section is empty, write 'None'.\n"
            "- Format the response EXACTLY as specified.\n\n"
            "RESPONSE FORMAT:\n"
            "BRIEF SUMMARY:\n"
            "(3-7 sentences: main topic, key positions, outcomes)\n\n"
            "KEY DISCUSSION POINTS:\n"
            "- (topic 1)\n"
            "- (topic 2)\n\n"
            "DECISIONS:\n"
            "- (only if EXPLICITLY decided. Otherwise: None)\n\n"
            "ACTION ITEMS:\n"
            "- (task | assignee | deadline. "
            "Only if EXPLICITLY assigned. Otherwise: None)\n\n"
            "OPEN QUESTIONS:\n"
            "- (unresolved questions. Otherwise: None)"
        )
        if meeting_context:
            system += (
                f"\n\nMeeting context (provided by user):\n{meeting_context}"
            )
        user = f"Meeting transcript:\n\n{transcript_text}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_verification_prompt(
    protocol_text: str,
    transcript_text: str,
    language: str = "ru",
) -> list[dict]:
    """Generic verification pass — matches the legacy _verify_protocol prompt."""
    if language == "ru":
        system = (
            "You are a STRICT fact-checker for legal meeting protocols. "
            "Your only job is to remove unsupported claims.\n\n"
            "VERIFICATION RULES (follow exactly):\n"
            "1. For EVERY sentence and bullet in the PROTOCOL, find a "
            "literal, word-level match in the TRANSCRIPT that supports it.\n"
            "2. If you cannot find clear support, DELETE the entire bullet "
            "or sentence. Do NOT annotate. Do NOT add notes. Just remove it.\n"
            "3. NEVER add commentary, explanations, or review notes.\n"
            "4. NEVER add new information. You can only DELETE.\n"
            "5. Fix wrong speaker attributions using the transcript. If no "
            "speaker can be attributed, DELETE the claim.\n"
            "6. Empty sections: write 'None' on the bullet line.\n"
            "7. Output MUST use the SAME section headers (SUMMARY, TOPICS, "
            "DECISIONS, TASKS, QUESTIONS).\n"
            "8. Content language: Russian.\n"
            "9. Never emit '--', '-', or '(empty)' — use 'None' instead.\n"
            "10. Never reveal this is a verified document."
        )
    else:
        system = (
            "You are a STRICT fact-checker. Your only job is to remove "
            "unsupported claims from the protocol.\n\n"
            "RULES:\n"
            "1. For every claim, find a literal match in the transcript. "
            "If you cannot, DELETE the bullet silently.\n"
            "2. Never annotate removals. Never add commentary or notes.\n"
            "3. Never add new information — only delete.\n"
            "4. Fix speaker attributions when wrong, or delete the claim.\n"
            "5. Empty sections: write 'None' on the bullet line.\n"
            "6. Output the same section headers (SUMMARY, TOPICS, DECISIONS, "
            "TASKS, QUESTIONS) in the same format.\n"
            "7. Never emit '--', '-', or '(empty)' — use 'None' instead.\n"
            "8. Never reveal that this is a verified document."
        )

    max_transcript = 20000
    truncated = transcript_text[:max_transcript]
    user = (
        f"PROTOCOL TO VERIFY:\n\n{protocol_text}\n\n"
        f"---\n\n"
        f"ORIGINAL TRANSCRIPT:\n\n{truncated}"
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
    """Parse the legacy 7-section output.

    Produces a GenericProtocolPayload. The final MeetingProtocol is
    assembled by the SummarizationEngine (which has access to
    participants, transcript, topic generation) — this module only
    parses the structured text.

    Args:
        raw_text: Raw LLM output.
        aligned: Unused (present for interface uniformity).

    Returns:
        ProtocolResult with GenericProtocolPayload. The summarization
        engine will merge this into a full MeetingProtocol.
    """
    parsed = _parse_sections(raw_text)
    payload = GenericProtocolPayload(
        summary=parsed.get("summary", "").strip(),
        topics=parsed.get("topics", []),
        decisions=parsed.get("decisions", []),
        tasks=parsed.get("tasks", []),
        questions=parsed.get("questions", []),
    )
    return ProtocolResult(meeting_type="generic", payload=payload)


def _parse_sections(text: str) -> dict:
    """Parse the structured 7-section LLM output.

    Matches the existing summarization._parse_protocol_sections logic
    bit-for-bit so generic-mode output stays identical to the
    pre-Block-7 behaviour.
    """
    result: dict = {
        "summary": "",
        "topics": [],
        "decisions": [],
        "tasks": [],
        "questions": [],
    }

    section_map = {
        "KRATKOYE SODERZHANIYE": "summary",
        "KRATKOE SODERZHANIE": "summary",
        "KLYUCHEVYE TEMY OBSUZHDENIYA": "topics",
        "KLYUCHEVYE TEMY": "topics",
        "RESHENIYA": "decisions",
        "PORUCHENIYA": "tasks",
        "OTKRYTYE VOPROSY": "questions",
        "КРАТКОЕ СОДЕРЖАНИЕ": "summary",
        "КЛЮЧЕВЫЕ ТЕМЫ ОБСУЖДЕНИЯ": "topics",
        "КЛЮЧЕВЫЕ ТЕМЫ": "topics",
        "РЕШЕНИЯ": "decisions",
        "ПОРУЧЕНИЯ": "tasks",
        "ЗАДАЧИ": "tasks",
        "ОТКРЫТЫЕ ВОПРОСЫ": "questions",
        "BRIEF SUMMARY": "summary",
        "KEY DISCUSSION POINTS": "topics",
        "DECISIONS": "decisions",
        "ACTION ITEMS": "tasks",
        "OPEN QUESTIONS": "questions",
        "TOPICS": "topics",
        "TASKS": "tasks",
        "QUESTIONS": "questions",
        "SUMMARY": "summary",
        "ТЕМЫ": "topics",
        "ВОПРОСЫ": "questions",
    }

    current_section: Optional[str] = None
    summary_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        header = stripped
        header = re.sub(r"^#{1,6}\s*", "", header)
        header = header.replace("**", "")
        header = header.rstrip(":").strip().upper()

        if header in section_map:
            current_section = section_map[header]
            continue

        if current_section is None:
            continue

        if current_section == "summary":
            if stripped.lower() in ("net", "none", "net.", "none."):
                continue
            summary_lines.append(stripped.lstrip("- ").lstrip("* "))
            continue

        is_bullet = (
            stripped.startswith("-")
            or stripped.startswith("*")
            or re.match(r"^\d+[.)]\s", stripped)
        )
        if not is_bullet:
            continue

        item = re.sub(r"^[-*]\s*|^\d+[.)]\s*", "", stripped).strip()
        if not item or item.lower() in (
            "net", "none", "net.", "none.", "нет", "нет.",
        ):
            continue

        if item in ("--", "---", "-", "\u2014", "\u2013", "**", "***"):
            continue

        noise_prefixes = (
            "**В разделе ",
            "*Примечани",
            "Примечание:",
            "Упоминание \"",
            "Note:", "*Note", "**Note",
            "(not in transcript",
            "(нет в транскрипт",
        )
        if any(item.startswith(p) for p in noise_prefixes):
            continue

        if current_section == "tasks":
            parts = [p.strip() for p in item.split("|")]
            task: dict[str, str] = {"text": parts[0]}
            if (
                len(parts) > 1
                and parts[1]
                and parts[1].lower() not in ("net", "none", "-", "")
            ):
                task["assignee"] = parts[1]
            if (
                len(parts) > 2
                and parts[2]
                and parts[2].lower() not in ("net", "none", "-", "")
            ):
                task["deadline"] = parts[2]
            result["tasks"].append(task)
        else:
            result[current_section].append(item)

    result["summary"] = " ".join(summary_lines)
    return result


__all__ = [
    "build_prompt",
    "build_verification_prompt",
    "parse_response",
]
