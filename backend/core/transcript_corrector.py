"""LLM post-correction pass over aligned transcript turns.

Tier 3 (T3.4, 2026-04-23). Optional pipeline stage that runs Gemma 4
over batches of post-aligned turns to fix word-level ASR errors that
no VAD/threshold/initial_prompt can address ("прощения" when the
speaker said "прошу", inflection mismatches, garbled fragments where
the correct word is unambiguous from context).

Strict design rules:
- Fix only obvious ASR errors. Never paraphrase. Never restructure
  sentences.
- Preserve every speaker label and timestamp exactly.
- Preserve turn count and turn order. Validation rejects the whole
  batch if either changes.
- Token-overlap floor: a corrected turn whose tokens overlap <70% with
  the original is rejected as paraphrase, original kept.

Default OFF (`postprocessing.llm_correction: false`). Enable per-run
via env var or config override. Single pass over the whole transcript
in batches of ~4000 chars.

Latency budget: ~30s for 60 min audio at 70-char/turn average. Adds
one Ollama call per batch. Reuses the SummarizationEngine's HTTP
client so no extra connection setup.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from backend.app.models import AlignedSegment

logger = logging.getLogger(__name__)


@dataclass
class CorrectionStats:
    """Summary of a correction pass for logging / observability."""
    total_turns: int = 0
    corrected_turns: int = 0
    rejected_batches: int = 0
    rejected_turns: int = 0
    failed_batches: int = 0
    chars_before: int = 0
    chars_after: int = 0


def _tokenize(text: str) -> list[str]:
    """Lowercase, punct-stripped word list — for overlap comparison.

    We don't need linguistic tokenisation; we need a stable bag-of-words
    count that ignores ASR-style minor variations.
    """
    return re.findall(r"\w+", (text or "").lower())


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings, 0.0..1.0.

    Returns 1.0 when both empty (treated as identical), 0.0 when one
    side is empty.
    """
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    sa = set(ta)
    sb = set(tb)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def _build_correction_prompt(turns_batch: list[dict]) -> list[dict]:
    """Build the Ollama chat messages for one correction batch.

    The LLM sees a JSON array of turns and must return a JSON array of
    the SAME length with the SAME turn_id values in order. Only `text`
    may differ — and only for unambiguous ASR errors. The prompt is in
    Russian (T3.3 confirmed Gemma handles Russian instructions well
    when the schema is crisp) but uses English for structural keywords.
    """
    system = (
        "Ты редактор-корректор стенограммы. Твоя ЕДИНСТВЕННАЯ задача "
        "— исправить очевидные ошибки распознавания речи в поле "
        "`text` каждой реплики, когда правильное слово однозначно "
        "следует из ближайшего контекста.\n"
        "\n"
        "СТРОГО ЗАПРЕЩЕНО:\n"
        "- Перефразировать или менять структуру предложений.\n"
        "- Добавлять или удалять реплики (turn_id).\n"
        "- Менять порядок реплик.\n"
        "- Менять speaker.\n"
        "- Дополнять текст словами, которых нет в исходной речи.\n"
        "- Сокращать содержание (даже если кажется, что речь "
        "повторяется).\n"
        "- Изменять смысл, тон или регистр высказывания.\n"
        "\n"
        "РАЗРЕШЕНО только:\n"
        "- Заменить очевидно неправильно распознанное слово на верное "
        "(например, «прощения» → «прошу», когда из контекста ясно, "
        "что речь идёт о просьбе).\n"
        "- Исправить очевидные опечатки и склонения, если правильная "
        "форма однозначна.\n"
        "- Расставить точки/запятые в местах, где они явно "
        "пропущены.\n"
        "\n"
        "Если ты не уверен, что слово ошибочно — оставь его как есть.\n"
        "Если правильная замена неоднозначна — оставь оригинал.\n"
        "Лучше оставить ошибку, чем добавить новую.\n"
        "\n"
        "ФОРМАТ ВВОДА: JSON-массив объектов "
        "{turn_id: int, speaker: str, text: str}.\n"
        "ФОРМАТ ВЫВОДА: JSON-массив СТРОГО ТОЙ ЖЕ ДЛИНЫ, в том же "
        "порядке, с теми же turn_id и speaker. Только text может "
        "отличаться.\n"
        "\n"
        "Никаких markdown-блоков. Никакого текста вне JSON-массива. "
        "Никаких пояснений."
    )

    user = (
        "Исправь стенограмму. Верни JSON-массив той же длины:\n\n"
        f"{json.dumps(turns_batch, ensure_ascii=False, indent=1)}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _validate_corrected_batch(
    original: list[dict],
    corrected: Any,
    min_overlap: float = 0.70,
) -> Optional[list[dict]]:
    """Validate the LLM-corrected batch.

    Returns the corrected list when all invariants hold; None when the
    batch must be rejected (caller falls back to originals).

    Invariants:
      1. Same length.
      2. Same turn_id sequence (in order).
      3. Each turn's speaker unchanged.
      4. Token overlap with original >= min_overlap (catches paraphrase).
      5. Each turn has a non-empty text. Empty text means LLM dropped
         content — reject.
    """
    if not isinstance(corrected, list):
        logger.warning("Corrector returned non-list; rejecting batch")
        return None
    if len(corrected) != len(original):
        logger.warning(
            "Corrector returned %d turns, expected %d; rejecting batch",
            len(corrected), len(original),
        )
        return None
    out: list[dict] = []
    for orig, cur in zip(original, corrected):
        if not isinstance(cur, dict):
            logger.warning("Corrector returned non-dict turn; rejecting batch")
            return None
        if cur.get("turn_id") != orig.get("turn_id"):
            logger.warning(
                "turn_id mismatch (%r vs %r); rejecting batch",
                cur.get("turn_id"), orig.get("turn_id"),
            )
            return None
        if cur.get("speaker") != orig.get("speaker"):
            logger.warning(
                "speaker changed for turn %s; rejecting batch",
                orig.get("turn_id"),
            )
            return None
        new_text = (cur.get("text") or "").strip()
        old_text = (orig.get("text") or "").strip()
        if not new_text:
            logger.warning(
                "Corrector emptied text for turn %s; rejecting batch",
                orig.get("turn_id"),
            )
            return None
        overlap = _token_overlap(old_text, new_text)
        if overlap < min_overlap:
            logger.info(
                "Turn %s overlap %.2f below %.2f; reverting just this turn",
                orig.get("turn_id"), overlap, min_overlap,
            )
            new_text = old_text  # individual revert, batch still OK
        out.append(
            {
                "turn_id": orig["turn_id"],
                "speaker": orig["speaker"],
                "text": new_text,
            }
        )
    return out


def correct_transcript(
    segments: list[AlignedSegment],
    summarization_engine: Any,
    chunk_chars: int = 4000,
    min_overlap: float = 0.70,
) -> tuple[list[AlignedSegment], CorrectionStats]:
    """Run LLM correction over the aligned transcript.

    Args:
        segments: Aligned transcript segments (post-postprocessor).
        summarization_engine: Loaded SummarizationEngine instance —
            we reuse its `_generate(messages)` method so we share the
            same Ollama client + retry logic. Caller must have already
            called `.load()` on it.
        chunk_chars: Approximate batch size in characters. Smaller =
            more LLM calls but tighter context. ~4000 is conservative.
        min_overlap: Minimum Jaccard token overlap between original
            and corrected text per turn. Below this, the individual
            turn reverts to original (paraphrase guard).

    Returns:
        (corrected_segments, stats). Segment count and order
        guaranteed unchanged. start/end/speaker_id/speaker_name/
        confidence/attribution_confidence preserved per segment.
    """
    stats = CorrectionStats(total_turns=len(segments))
    if not segments:
        return segments, stats

    # Build batches of turns until chunk_chars is exceeded.
    batches: list[list[int]] = []  # list of segment indices per batch
    current: list[int] = []
    current_chars = 0
    for idx, seg in enumerate(segments):
        text_len = len((seg.text or "").strip())
        # Accept the first turn into a fresh batch even if it's huge.
        if current and current_chars + text_len > chunk_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(idx)
        current_chars += text_len
    if current:
        batches.append(current)

    logger.info(
        "Transcript corrector: %d turns in %d batches (chunk_chars=%d)",
        len(segments), len(batches), chunk_chars,
    )

    # Output starts as a copy of the original; we splice in corrections
    # only where the batch validation passed.
    corrected_segments = list(segments)

    for batch_no, indices in enumerate(batches, 1):
        batch_payload = [
            {
                "turn_id": i,
                "speaker": (
                    segments[i].speaker_name
                    or segments[i].speaker_id
                    or ""
                ),
                "text": (segments[i].text or "").strip(),
            }
            for i in indices
        ]
        stats.chars_before += sum(len(t["text"]) for t in batch_payload)

        messages = _build_correction_prompt(batch_payload)

        try:
            raw = summarization_engine._generate(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Corrector batch %d/%d LLM call failed: %s; keeping originals",
                batch_no, len(batches), exc,
            )
            stats.failed_batches += 1
            stats.chars_after += sum(len(t["text"]) for t in batch_payload)
            continue

        # The LLM should return raw JSON. Strip code fences if it
        # disobeyed and added them.
        raw_clean = raw.strip()
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean)
        raw_clean = re.sub(r"\s*```\s*$", "", raw_clean)
        try:
            parsed = json.loads(raw_clean)
        except json.JSONDecodeError:
            # Try to find the first array bracket in case the model
            # added prose around the JSON.
            match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", raw_clean)
            if not match:
                logger.warning(
                    "Corrector batch %d/%d JSON parse failed; keeping originals",
                    batch_no, len(batches),
                )
                stats.failed_batches += 1
                stats.chars_after += sum(len(t["text"]) for t in batch_payload)
                continue
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Corrector batch %d/%d JSON parse failed (fallback): %s",
                    batch_no, len(batches), exc,
                )
                stats.failed_batches += 1
                stats.chars_after += sum(len(t["text"]) for t in batch_payload)
                continue

        validated = _validate_corrected_batch(
            batch_payload, parsed, min_overlap=min_overlap
        )
        if validated is None:
            stats.rejected_batches += 1
            stats.rejected_turns += len(indices)
            stats.chars_after += sum(len(t["text"]) for t in batch_payload)
            continue

        # Splice corrected text back into segments.
        for entry in validated:
            seg_idx = int(entry["turn_id"])
            new_text = entry["text"]
            old_seg = corrected_segments[seg_idx]
            if new_text != (old_seg.text or "").strip():
                stats.corrected_turns += 1
            corrected_segments[seg_idx] = AlignedSegment(
                start=old_seg.start,
                end=old_seg.end,
                text=new_text,
                speaker_id=old_seg.speaker_id,
                speaker_name=old_seg.speaker_name,
                confidence=old_seg.confidence,
                attribution_confidence=old_seg.attribution_confidence,
            )
            stats.chars_after += len(new_text)

    logger.info(
        "Transcript correction: %d/%d turns changed, %d batches rejected, "
        "%d batches failed, chars %d -> %d",
        stats.corrected_turns, stats.total_turns, stats.rejected_batches,
        stats.failed_batches, stats.chars_before, stats.chars_after,
    )
    return corrected_segments, stats
