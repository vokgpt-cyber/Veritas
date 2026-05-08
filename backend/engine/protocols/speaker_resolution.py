"""Speaker resolution post-pass: SPEAKER_X → attendee names.

After protocol extraction, this module makes ONE focused LLM call:
"given this list of attendees and these speaker examples, map each
SPEAKER_X to the most likely attendee name". Result is applied
across the protocol's `speaker` and `owner` fields plus the verbatim
stenogram turns.

**Why this is a separate pass and not part of extraction:**

We learned the hard way (sprint 2026-04-30) that asking the
extraction LLM to do everything at once produces inconsistent
mappings — it correctly resolves names in some topics, leaves
SPEAKER_X in others. A focused single-task call gets it right.
This is the same pattern Notion AI and Otter use: extract first,
attribute later.

**SPEAKER_X labels we leave unchanged:**

* When the LLM is unsure → returns the original label
* When the speaker isn't in the attendee list (external
  participant, IT support, junior lawyer not on the picker) →
  we relabel as "Спикер #N" (more readable than "SPEAKER_06")
  but don't fabricate an identity

The renamer never invents names. It only consolidates known
attendees from the operator's pre-meeting list.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Recognise SPEAKER_X labels in any case / underscore variant.
_SPEAKER_LABEL_RE = re.compile(r"^SPEAKER[_\s-]?(\d+)$", re.IGNORECASE)
_SPEAKER_REF_RE = re.compile(r"SPEAKER[_\s-]?(\d+)", re.IGNORECASE)


def _looks_like_speaker_label(value: str) -> bool:
    """Is this string a raw SPEAKER_X label vs a real name?"""
    if not value:
        return False
    return bool(_SPEAKER_LABEL_RE.match(value.strip()))


def _readable_speaker_label(value: str) -> str:
    """Convert 'SPEAKER_06' → 'Спикер #6' for display.

    Used for SPEAKER_X labels that the LLM couldn't map to an
    attendee. The technical label SPEAKER_06 looks alien in a
    Russian protocol; "Спикер #6" reads naturally.
    """
    m = _SPEAKER_LABEL_RE.match(value.strip())
    if not m:
        return value
    return f"Спикер #{int(m.group(1))}"


def _collect_speaker_samples(
    aligned: list,
    *,
    max_per_speaker: int = 3,
    max_chars_each: int = 120,
) -> dict[str, list[str]]:
    """Collect short sample turns per detected SPEAKER_X.

    Picks the first `max_per_speaker` turns of each speaker that
    are at least 30 characters long (filters out "Да", "Угу",
    "Хорошо" which give the LLM no signal to attribute on).
    """
    samples: dict[str, list[str]] = {}
    for seg in aligned or []:
        sid = getattr(seg, "speaker_id", None)
        if not sid or not _looks_like_speaker_label(sid):
            continue
        text = (getattr(seg, "text", "") or "").strip()
        if len(text) < 30:
            continue
        bucket = samples.setdefault(sid, [])
        if len(bucket) >= max_per_speaker:
            continue
        if len(text) > max_chars_each:
            text = text[: max_chars_each - 1] + "…"
        bucket.append(text)
    return samples


def build_speaker_mapping_prompt(
    attendees: list[str],
    samples: dict[str, list[str]],
) -> list[dict]:
    """Build the LLM prompt asking for SPEAKER_X → attendee mapping.

    Provides attendees + 3 sample turns per detected speaker. The
    model returns a JSON dict mapping. Strict instructions: don't
    invent names not in the list, return the original SPEAKER_X if
    unsure.
    """
    attendees_block = "\n".join(f"- {name}" for name in attendees)

    sample_lines: list[str] = []
    for sid in sorted(samples.keys()):
        sample_lines.append(f"\n{sid}:")
        for turn in samples[sid]:
            sample_lines.append(f"  • {turn}")
    samples_block = "\n".join(sample_lines)

    system = (
        "Ты сопоставляешь технические лейблы спикеров с реальными "
        "именами участников встречи.\n"
        "\n"
        "Дано: список присутствовавших участников и по 1-3 типичных "
        "фразы для каждого технического лейбла SPEAKER_X из стенограммы.\n"
        "\n"
        "Задача: для каждого SPEAKER_X определи, кому из участников "
        "наиболее вероятно принадлежит этот голос. Используй "
        "контекст: тема реплик, тон, обращения, упоминания имён в "
        "других репликах.\n"
        "\n"
        "ПРАВИЛА:\n"
        "• Можно сопоставить с именем ТОЛЬКО из списка участников\n"
        "• Если уверенности нет — возвращай null (НЕ выдумывай и "
        "НЕ перебирай по порядку)\n"
        "• Один и тот же участник может соответствовать нескольким "
        "SPEAKER_X (но это редко — только при сбое диаризации)\n"
        "• Имя в выходе должно ТОЧНО совпадать со строкой из списка "
        "участников\n"
        "\n"
        "ФОРМАТ ВЫВОДА — JSON со словарём mapping:\n"
        "{\n"
        '  "mapping": {\n'
        '    "SPEAKER_00": "Иванов А.В." | null,\n'
        '    "SPEAKER_01": "Петрова О.С." | null,\n'
        '    ...\n'
        "  }\n"
        "}\n"
        "\n"
        "Никакого текста вне JSON, никаких markdown-блоков."
    )
    user = (
        f"УЧАСТНИКИ ВСТРЕЧИ:\n{attendees_block}\n\n"
        f"ОБРАЗЦЫ РЕПЛИК ПО СПИКЕРАМ:{samples_block}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_speaker_mapping(raw: str, valid_attendees: list[str]) -> dict[str, str]:
    """Parse the LLM mapping response into validated SPEAKER_X → name dict.

    Drops mappings to names not in `valid_attendees` (defensive
    against hallucinated names). Drops mappings where the value is
    null/empty/garbage.
    """
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    if not isinstance(data, dict):
        return {}
    raw_mapping = data.get("mapping") if isinstance(data.get("mapping"), dict) else data
    if not isinstance(raw_mapping, dict):
        return {}

    valid_lower = {n.lower(): n for n in valid_attendees}
    result: dict[str, str] = {}
    for sid, name in raw_mapping.items():
        if not isinstance(sid, str) or not _looks_like_speaker_label(sid):
            continue
        if not isinstance(name, str):
            continue
        clean_name = name.strip()
        if not clean_name or clean_name.lower() in ("null", "none", "нет"):
            continue
        # Match case-insensitively against the attendee list to defend
        # against minor case/spacing differences from the LLM output.
        canonical = valid_lower.get(clean_name.lower())
        if canonical is None:
            logger.debug(
                "Dropping mapping %s -> %s (not in attendee list)",
                sid, clean_name,
            )
            continue
        result[sid.upper()] = canonical
    return result


def resolve_speaker_label(
    raw_value: str,
    mapping: dict[str, str],
) -> str:
    """Apply the mapping to a single speaker-label string.

    1. If raw_value is a SPEAKER_X label and we have a mapping → use the name.
    2. If raw_value is a SPEAKER_X label without mapping → "Спикер #N".
    3. Otherwise (already a real name) → unchanged.
    """
    if not raw_value:
        return raw_value
    cleaned = raw_value.strip()
    if not _looks_like_speaker_label(cleaned):
        if not _SPEAKER_REF_RE.search(cleaned):
            return cleaned

        def _replace(match: re.Match[str]) -> str:
            sid = f"SPEAKER_{int(match.group(1)):02d}"
            if sid in mapping:
                return mapping[sid]
            return _readable_speaker_label(sid)

        return _SPEAKER_REF_RE.sub(_replace, cleaned)
    sid_upper = cleaned.upper().replace(" ", "_").replace("-", "_")
    # Normalise SPEAKER 6 / SPEAKER-6 / Speaker_6 → SPEAKER_6
    m = _SPEAKER_LABEL_RE.match(sid_upper)
    if m:
        sid_upper = f"SPEAKER_{int(m.group(1)):02d}"
    if sid_upper in mapping:
        return mapping[sid_upper]
    return _readable_speaker_label(cleaned)


def apply_mapping_to_protocol(
    payload: Any,
    mapping: dict[str, str],
) -> None:
    """Apply the mapping in place across all speaker/owner fields.

    Handles both AdministrativeProtocol and CourtHearingProtocol
    payloads. Walks every speaker-bearing field and rewrites in
    place. Stenogram turns get the same treatment.
    """
    if mapping is None:
        return

    # AdministrativeProtocol-shaped fields
    practical_items = getattr(payload, "items", None) or []
    for item in practical_items:
        spk = getattr(item, "speaker", "")
        if spk:
            item.speaker = resolve_speaker_label(spk, mapping)
        owner = getattr(item, "owner", None)
        if owner:
            item.owner = resolve_speaker_label(owner, mapping)

    for bucket_name in ("decisions", "open_questions", "risks"):
        bucket = getattr(payload, bucket_name, None) or []
        for item in bucket:
            spk = getattr(item, "speaker", "")
            if spk:
                item.speaker = resolve_speaker_label(spk, mapping)

    tasks = getattr(payload, "tasks", None) or []
    for task in tasks:
        owner = getattr(task, "owner", None)
        if owner:
            task.owner = resolve_speaker_label(owner, mapping)
        spk = getattr(task, "speaker", "")
        if spk:
            task.speaker = resolve_speaker_label(spk, mapping)

    # Topic-segmented payload (kept behind flag but still possible).
    topic_summaries = getattr(payload, "topic_summaries", None) or []
    for topic in topic_summaries:
        for bucket_name in ("decisions", "open_questions"):
            bucket = getattr(topic, bucket_name, None) or []
            for item in bucket:
                spk = getattr(item, "speaker", "")
                if spk:
                    item.speaker = resolve_speaker_label(spk, mapping)
        for task in getattr(topic, "tasks", None) or []:
            owner = getattr(task, "owner", None)
            if owner:
                task.owner = resolve_speaker_label(owner, mapping)
            spk = getattr(task, "speaker", "")
            if spk:
                task.speaker = resolve_speaker_label(spk, mapping)

    # Stenogram (CourtHearingTurn list) — admin and court both have
    # this. Rewrite every turn's speaker label.
    turns = getattr(payload, "turns", None) or []
    for turn in turns:
        spk = getattr(turn, "speaker", "")
        if spk:
            turn.speaker = resolve_speaker_label(spk, mapping)

    # Court-hearing top-level participants list (already real names
    # like "Суд", "Истец" — _looks_like_speaker_label returns False
    # for those, so the rewrite is a no-op for known roles).
    participants = getattr(payload, "participants", None) or []
    if isinstance(participants, list):
        new_participants = []
        for name in participants:
            if isinstance(name, str):
                new_participants.append(resolve_speaker_label(name, mapping))
            else:
                new_participants.append(name)
        # Replace contents in place to keep the same list object
        participants[:] = new_participants


__all__ = [
    "build_speaker_mapping_prompt",
    "parse_speaker_mapping",
    "resolve_speaker_label",
    "apply_mapping_to_protocol",
    "_collect_speaker_samples",
    "_looks_like_speaker_label",
    "_readable_speaker_label",
]
