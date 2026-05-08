"""Transcript post-processing: filler removal, punctuation restoration, sentence cleanup.

Cleans up raw ASR output before summarization to improve protocol quality.
Punctuation restoration uses deepmultilingualpunctuation (CPU-based, ~300MB model).
All other processing is rule-based (no ML models, no VRAM needed).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from backend.app.models import AlignedSegment

# Lazy-load punctuation model -- CPU only, ~300MB, loads on first use
_punct_model = None
_punct_available: Optional[bool] = None

logger = logging.getLogger(__name__)

# Russian filler words and discourse markers to remove.
# Includes syllabic hesitation sounds (\u0430-\u0430, \u043c-\u043c, \u044d-\u044d) in all their spelling
# variants -- ASR engines transcribe these differently between runs.
RU_FILLERS = {
    # Hesitation / voice-fill sounds
    "\u044d", "\u044d\u043c", "\u044d\u044d", "\u044d\u044d\u044d",  # \u044d, \u044d\u043c, \u044d\u044d, \u044d\u044d\u044d
    "\u044d-\u044d",                                                # \u044d-\u044d
    "\u0430\u0430", "\u0430\u0430\u0430",                           # \u0430\u0430, \u0430\u0430\u0430
    "\u0430-\u0430",                                                # \u0430-\u0430
    "\u043c\u043c", "\u043c\u043c\u043c",                           # \u043c\u043c, \u043c\u043c\u043c
    "\u043c-\u043c",                                                # \u043c-\u043c
    "\u043d\u0443-\u0443",                                          # \u043d\u0443-\u0443
    # Discourse markers
    "\u043d\u0443", "\u0432\u043e\u0442", "\u043a\u0430\u043a \u0431\u044b", "\u0442\u0438\u043f\u0430", "\u043a\u043e\u0440\u043e\u0447\u0435",
    "\u0437\u043d\u0430\u0447\u0438\u0442", "\u0442\u0430\u043a \u0441\u043a\u0430\u0437\u0430\u0442\u044c", "\u0432 \u043e\u0431\u0449\u0435\u043c", "\u043d\u0443 \u0432\u043e\u0442", "\u043d\u0443 \u043a\u0430\u043a \u0431\u044b",
    "\u044d\u0442\u043e \u0441\u0430\u043c\u043e\u0435", "\u0442\u043e \u0435\u0441\u0442\u044c \u043d\u0443", "\u043d\u0443 \u0442\u0438\u043f\u0430", "\u0432 \u043f\u0440\u0438\u043d\u0446\u0438\u043f\u0435 \u043d\u0443",
    "\u0441\u043a\u0430\u0436\u0435\u043c \u0442\u0430\u043a", "\u0433\u0440\u0443\u0431\u043e \u0433\u043e\u0432\u043e\u0440\u044f", "\u0441\u043e\u0431\u0441\u0442\u0432\u0435\u043d\u043d\u043e \u0433\u043e\u0432\u043e\u0440\u044f",
    "\u043d\u0443 \u044d\u0442\u043e", "\u043d\u0443 \u0442\u0430\u043c", "\u043d\u0443 \u0442\u0430\u043a\u043e\u0435", "\u043a\u0430\u043a \u044d\u0442\u043e", "\u0432\u043e\u0442 \u044d\u0442\u043e",
}

# English filler words
EN_FILLERS = {
    "uh", "um", "uhm", "hmm", "like", "you know", "i mean",
    "basically", "actually", "literally", "right", "so yeah",
    "kind of", "sort of",
}

# Patterns for repeated words/phrases (Whisper hallucination artifacts)
REPEAT_PATTERN = re.compile(r"\b(\w+(?:\s+\w+)?)\s+(?:\1\s*){2,}", re.IGNORECASE)

# Patterns for meaningless very short segments
MIN_MEANINGFUL_LENGTH = 3  # characters after cleanup

# Seam-polish thresholds (2026-04-21). When merging two same-speaker segments,
# the gap between them determines whether the join represents a sentence break
# or a continuous utterance:
#   gap >= GAP_SENTENCE_BREAK_S -> speaker paused, treat as new sentence
#   gap <= GAP_CONTINUOUS_S     -> back-to-back speech, one utterance
#   between                     -> ambiguous, do not force a decision
GAP_SENTENCE_BREAK_S = 0.6
GAP_CONTINUOUS_S = 0.15

# Characters treated as sentence terminators vs mid-sentence continuation
_TERMINAL_PUNCT = set(".!?\u2026")
_CONTINUATION_PUNCT = set(",;:\u2014\u2013-")

# Leading noise at turn start. Source audio often carries direct-speech
# dash markers (\u2014) and diarization-boundary leftovers (stray commas) into
# the start of a merged turn. Once we have explicit speaker_id labels,
# these glyphs at position 0 are redundant and ugly.
#
# The first character must be a dash or comma, and it must be followed by
# at least one more noise character (whitespace / comma / dash). Requiring
# a trailing separator protects compound words like "IT-\u0441\u0438\u0441\u0442\u0435\u043c\u044b" or a
# mid-turn dash that bled to the front: "\u2014\u0414\u0430" (no space) is preserved.
#
# Examples:
#   "\u2014 \u0414\u0430."        -> stripped ("\u0414\u0430.")
#   ", \u0432 \u043f\u044f\u0442\u043d\u0438\u0446\u0443"  -> stripped ("\u0412 \u043f\u044f\u0442\u043d\u0438\u0446\u0443")
#   "\u2014, \u0442\u0430\u043c"       -> stripped ("\u0422\u0430\u043c")
#   ",, \u043d\u0443"        -> stripped ("\u041d\u0443")
#   "\u2014\u0414\u0430."         -> preserved (no separator after the dash)
_LEADING_SPEAKER_DASH_RE = re.compile(
    r"^[\u2014\u2013\u002D,][\s,\u2014\u2013\u002D]+"
)

# Internal direct-speech markers that survived a same-speaker merge.
# The diarization boundary was in the middle of the speaker's run, so the
# sub-segment that became "internal" text still carries a leading "\u2014 ".
# After the merge we see patterns like ". \u2014 \u0427\u0442\u043e" or "? \u2014, \u0422\u0430\u043c" mid-turn.
#
# Rule: strip a dash (and any surrounding commas/dashes/whitespace) ONLY
# when it appears RIGHT AFTER a sentence-terminating punctuation (.!?...).
# This preserves legitimate grammar dashes which follow words, not punct:
#   "\u0434\u0430\u043b\u044c\u0448\u0435 \u2014 \u043d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e"  -> preserved (dash after word)
#   ". \u2014 \u0427\u0442\u043e \u0431\u044b\u043b\u043e \u0434\u0430\u043b\u044c\u0448\u0435"  -> stripped ("." becomes sentence end,
#                             "\u0427\u0442\u043e" re-capitalized)
_INTERNAL_DASH_MARKER_RE = re.compile(
    r"([.!?\u2026])\s+[\u2014\u2013\u002D][\s,\u2014\u2013\u002D]*(\S)"
)

# Repeated adjacent commas: "foo,, bar" or "foo, , bar" -> "foo, bar".
# Cleanup artefact from filler removal and seam polish.
_REPEATED_COMMA_RE = re.compile(r",(?:\s*,)+")


def _cap_first(s: str) -> str:
    """Capitalize the first alphabetic character of a string (Unicode-safe)."""
    if s and s[0].isalpha() and s[0].islower():
        return s[0].upper() + s[1:]
    return s


def _lower_first(s: str) -> str:
    """Lowercase the first alphabetic character of a string (Unicode-safe)."""
    if s and s[0].isalpha() and s[0].isupper():
        return s[0].lower() + s[1:]
    return s


def _polish_seam(prev_text: str, next_text: str, gap: float) -> tuple[str, str]:
    """Polish the seam between two same-speaker segments before they are joined.

    ASR engines (GigaAM, Whisper) emit each chunk independently and tend to
    capitalize the first word of every chunk -- which is wrong when the chunk
    is mid-sentence. This function applies rules keyed on how the previous
    segment ends and the silence gap between the two segments:

      prev ends with .!?...     -> trust terminal punct, capitalize next
      prev ends with ,;:-       -> continuation, lowercase next
      prev ends with letter/num + gap >= GAP_SENTENCE_BREAK_S
                                -> insert `.` on prev, capitalize next
      prev ends with letter/num + gap <= GAP_CONTINUOUS_S
                                -> continuous utterance, lowercase next
      anything else             -> ambiguous, leave both sides unchanged

    Proper-noun false negatives are tolerated: occasional lowercasing of a
    name after a comma or dash is acceptable collateral; the alternative
    (leaving spurious mid-sentence capitals) pollutes the transcript more.

    Returns polished (prev, next) intended to be joined with `prev + " " + next`.
    """
    if not prev_text or not next_text:
        return prev_text, next_text

    prev_stripped = prev_text.rstrip()
    next_stripped = next_text.lstrip()
    if not prev_stripped or not next_stripped:
        return prev_stripped, next_stripped

    last_char = prev_stripped[-1]

    if last_char in _TERMINAL_PUNCT:
        return prev_stripped, _cap_first(next_stripped)

    if last_char in _CONTINUATION_PUNCT:
        return prev_stripped, _lower_first(next_stripped)

    # No punctuation on prev -- lean on the silence gap
    if gap >= GAP_SENTENCE_BREAK_S:
        if last_char.isalnum():
            prev_stripped = prev_stripped + "."
        return prev_stripped, _cap_first(next_stripped)

    if gap <= GAP_CONTINUOUS_S:
        return prev_stripped, _lower_first(next_stripped)

    # Ambiguous middle band -- do not impose a decision
    return prev_stripped, next_stripped


def _cleanup_joined(text: str) -> str:
    """Collapse whitespace and remove space-before-punctuation after a join."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def _strip_leading_speaker_dash(text: str) -> str:
    """Strip a leading dash/comma cluster + whitespace from a turn.

    Handles em-dash, en-dash, hyphen, and leading commas (or any combination
    like "\u2014, " or ",, "). Applied at turn-flush time, after same-speaker
    merges are finalized. See _LEADING_SPEAKER_DASH_RE for rationale.
    """
    if not text:
        return text
    # Strip once at the absolute start; preserve internal dashes
    stripped = _LEADING_SPEAKER_DASH_RE.sub("", text, count=1)
    # Re-capitalize: a lowercase first letter after the strip means the
    # dash/comma was absorbing what the speaker intended as a sentence start
    return _cap_first(stripped) if stripped else stripped


def _strip_internal_dash_markers(text: str) -> str:
    """Strip dash markers after sentence-end punct inside a merged turn.

    When same-speaker sub-segments merge, each sub-segment's leading dash
    (the diarization-boundary direct-speech marker) ends up mid-turn right
    after the previous sub-segment's terminating punctuation:

        "foo. - Bar baz. -, Quux."  ->  "foo. Bar baz. Quux."

    Legitimate grammar dashes that follow a WORD (not punct) are preserved:

        "\u0434\u0430\u043b\u044c\u0448\u0435 - \u043d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e"       ->  unchanged
    """
    if not text:
        return text

    def _repl(match: re.Match[str]) -> str:
        punct, first_char = match.group(1), match.group(2)
        return f"{punct} {first_char.upper() if first_char.isalpha() else first_char}"

    return _INTERNAL_DASH_MARKER_RE.sub(_repl, text)


def _collapse_repeated_commas(text: str) -> str:
    """Collapse repeated commas (,, or , , ) into a single comma."""
    if not text:
        return text
    return _REPEATED_COMMA_RE.sub(",", text)


def _normalize_turn_text(text: str) -> str:
    """Turn-level text normalization, applied once at flush time.

    Order matters:
      1. Strip internal dash markers after sentence-end punct -- this
         undoes the survival of leading-"- " on inner sub-segments.
      2. Collapse repeated commas produced by filler removal.
      3. Strip the leading dash/comma cluster at the absolute turn start
         (cap-first fix applied inside the helper).
    """
    text = _strip_internal_dash_markers(text)
    text = _collapse_repeated_commas(text)
    text = _strip_leading_speaker_dash(text)
    return text


def detect_language(segments: list[AlignedSegment]) -> str:
    """Detect majority language from transcript segments.

    Counts Cyrillic vs Latin characters across all segments.

    Args:
        segments: List of aligned transcript segments

    Returns:
        "ru" if majority Cyrillic, "en" otherwise
    """
    cyrillic_count = 0
    latin_count = 0

    for seg in segments:
        text = seg.text or ""
        for char in text:
            if "\u0400" <= char <= "\u04ff":
                cyrillic_count += 1
            elif "a" <= char.lower() <= "z":
                latin_count += 1

    total = cyrillic_count + latin_count
    if total == 0:
        return "ru"  # Default to Russian for EPAM

    ru_share = cyrillic_count / total
    logger.info(
        f"Language detection: {cyrillic_count} Cyrillic, {latin_count} Latin "
        f"({ru_share:.0%} Russian)"
    )

    return "ru" if ru_share >= 0.5 else "en"


def clean_filler_words(text: str, language: str = "ru") -> str:
    """Remove filler words and discourse markers from text."""
    fillers = RU_FILLERS if language == "ru" else EN_FILLERS

    # Sort fillers by length (longest first) to avoid partial matches
    sorted_fillers = sorted(fillers, key=len, reverse=True)

    cleaned = text
    for filler in sorted_fillers:
        pattern = r"(?<!\w)" + re.escape(filler) + r"(?!\w)"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"^[,.\s]+", "", cleaned)
    cleaned = re.sub(r"[,\s]+$", "", cleaned)
    cleaned = re.sub(r"([.,!?])\s*\1+", r"\1", cleaned)

    return cleaned.strip()


def remove_repeated_phrases(text: str) -> str:
    """Remove Whisper hallucination artifacts (repeated phrases)."""
    cleaned = REPEAT_PATTERN.sub(r"\1", text)
    return cleaned.strip()


# =====================================================================
# Hallucination-artifact filter
# =====================================================================
# Whisper-family models trained on YouTube/podcast data emit a small set
# of recognizable hallucination patterns when they encounter silence or
# unintelligible audio. They never appear in real speech but show up in
# transcripts — stakeholder review of court_hearing_129 flagged
# "22:14 вы оплачиваете?" as one example. The patterns are short
# (<30 chars typical) and high-confidence false positives, so we strip
# the segment entirely when it matches any of the patterns AND the
# total text is short enough to not be a real utterance that happens
# to contain the pattern as a substring.

# Stray timecode-like artifact at the very start of a segment.
# Whisper-family models trained on YouTube/podcast subtitles
# occasionally emit chunk-boundary glitches that look like full
# `HH:MM:SS` timestamps because the training data had them in
# subtitle headers and chapter markers.
#
# 2026-04-28: tightened from `H:MM(:SS)?` to `H:MM:SS` after
# court_hearing_129 review found a real witness utterance
# "22, 14, вы оплачиваете" got transcribed as "22:14 вы
# оплачиваете" and then DELETED by this filter as a presumed
# artifact. Real legal speakers say "22:14" / "3:30" all the time;
# they almost never say "HH:MM:SS" with seconds, but Whisper's
# YouTube-trained artifacts almost always include seconds. So we
# now require all three groups, with the optional middle case
# (no seconds) excluded.
_TIMECODE_ARTIFACT_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}(?:\s|[.,!?])"
)

# YouTube-leftover Russian phrases. Lower-cased exact-match (after
# normalisation). Only strip when the segment is short — these phrases
# in real speech would normally be embedded in longer context.
_YOUTUBE_LEFTOVERS_RU = {
    "спасибо за просмотр",
    "подпишитесь на канал",
    "подписывайтесь на канал",
    "ставьте лайк",
    "не забудьте подписаться",
    "до новых встреч",
    "редактор субтитров",
    "корректор",
}

# English-language artifacts (Whisper drops English in Russian audio
# when it hallucinates from training-data leakage).
_YOUTUBE_LEFTOVERS_EN = {
    "subscribe",
    "thanks for watching",
    "like and subscribe",
    "see you next time",
    "thanks for watching!",
    "thank you.",
}

# Threshold below which a "matching" segment is considered a pure
# hallucination instead of a real utterance that happens to contain
# the pattern. Tuned conservatively: real legal speech turns are
# almost always longer than 30 chars.
_HALLUCINATION_MAX_LEN = 30


def filter_hallucination_artifacts(text: str) -> str:
    """Drop known Whisper-style hallucination patterns.

    Returns "" when the text is identified as a pure hallucination so
    the caller can drop the segment. Returns the input unchanged
    otherwise. Conservative — only triggers on short text where the
    pattern is dominant, never strips a substring out of a longer real
    utterance.
    """
    if not text:
        return text
    stripped = text.strip()
    if not stripped:
        return text

    # Timecode-like artifact at the start of a segment.
    if _TIMECODE_ARTIFACT_RE.match(stripped):
        # Only strip if the WHOLE segment looks like a hallucination
        # (short, no Cyrillic-rich content past the timecode). Real
        # turns might begin with a quoted timestamp.
        if len(stripped) <= _HALLUCINATION_MAX_LEN:
            logger.info(
                "Hallucination filter dropped (timecode-like): %r",
                stripped,
            )
            return ""

    # YouTube-leftover phrases. Only strip very short segments to
    # avoid clipping real speech that happens to mention these.
    if len(stripped) <= _HALLUCINATION_MAX_LEN:
        normalised = stripped.lower().strip(".!?,;:- \t\n")
        if normalised in _YOUTUBE_LEFTOVERS_RU:
            logger.info(
                "Hallucination filter dropped (RU leftover): %r",
                stripped,
            )
            return ""
        if normalised in _YOUTUBE_LEFTOVERS_EN:
            logger.info(
                "Hallucination filter dropped (EN leftover): %r",
                stripped,
            )
            return ""
        # Russian "Спасибо" alone is real speech but "спасибо за просмотр"
        # is a hallucination — the set lookup catches the phrase form.

    return text


def merge_short_segments(
    segments: list[AlignedSegment],
    min_duration: float = 0.5,
    max_turn_gap_s: float = 2.0,
) -> list[AlignedSegment]:
    """Merge consecutive same-speaker segments into speaker turns.

    A new turn starts at any speaker change or at a gap > max_turn_gap_s.

    2026-04-28: lowered max_turn_gap_s from 3.0 → 2.0 after the
    GigaAM-vs-WhisperX comparison on court_hearing_129 found the
    3.0s threshold was over-merging WhisperX's micro-segments
    (635 raw → 190 turns, 70% reduction) and creating wall-of-text
    turns. Court speakers naturally pause 1-2s within a single
    argument; the 2.0s threshold preserves that intuition while
    still merging genuine same-speaker continuations.
    Within a turn, adjacent segments are joined via _polish_seam() which
    inserts/strips punctuation and fixes capitalization at the boundary
    based on the silence gap.

    Propagation on merge:
      - attribution_confidence: min of both sides (None-safe). Ensures the
        contested-segment warning survives and never gets diluted by
        absorption into a confident neighbour.
      - confidence (ASR): duration-weighted average.
      - start/end: buffer.start -> seg.end.
    """
    if not segments:
        return segments

    merged: list[AlignedSegment] = []
    buffer: Optional[AlignedSegment] = None

    for seg in segments:
        if buffer is None:
            buffer = seg
            continue

        gap = seg.start - buffer.end
        if seg.speaker_id == buffer.speaker_id and gap < max_turn_gap_s:
            polished_prev, polished_next = _polish_seam(
                buffer.text, seg.text, gap
            )
            joined = _cleanup_joined(f"{polished_prev} {polished_next}")

            prev_ac = buffer.attribution_confidence
            next_ac = seg.attribution_confidence
            if prev_ac is not None and next_ac is not None:
                attr_conf: Optional[float] = min(prev_ac, next_ac)
            elif prev_ac is not None:
                attr_conf = prev_ac
            elif next_ac is not None:
                attr_conf = next_ac
            else:
                attr_conf = None

            dur_a = max(0.001, buffer.end - buffer.start)
            dur_b = max(0.001, seg.end - seg.start)
            conf_a = buffer.confidence if buffer.confidence is not None else 1.0
            conf_b = seg.confidence if seg.confidence is not None else 1.0
            weighted_conf = (conf_a * dur_a + conf_b * dur_b) / (dur_a + dur_b)

            buffer = AlignedSegment(
                start=buffer.start,
                end=seg.end,
                text=joined,
                speaker_id=buffer.speaker_id,
                speaker_name=buffer.speaker_name,
                confidence=weighted_conf,
                attribution_confidence=attr_conf,
            )
        else:
            _flush_turn(merged, buffer)
            buffer = seg

    _flush_turn(merged, buffer)

    return merged


def _flush_turn(merged: list[AlignedSegment],
                buffer: Optional[AlignedSegment]) -> None:
    """Finalize a turn: normalize text, drop if too short, append.

    Applies the turn-level normalization pass (see _normalize_turn_text):
    internal dash markers, repeated commas, leading dash/comma strip.

    Called at every turn boundary in merge_short_segments. Centralizes
    the flush logic so both the mid-loop flush (speaker change) and the
    tail flush (last buffer) get identical treatment.
    """
    if buffer is None or not buffer.text:
        return
    normalized = _normalize_turn_text(buffer.text)
    if len(normalized.strip()) < MIN_MEANINGFUL_LENGTH:
        return
    if normalized == buffer.text:
        merged.append(buffer)
        return
    merged.append(
        AlignedSegment(
            start=buffer.start,
            end=buffer.end,
            text=normalized,
            speaker_id=buffer.speaker_id,
            speaker_name=buffer.speaker_name,
            confidence=buffer.confidence,
            attribution_confidence=buffer.attribution_confidence,
        )
    )


def capitalize_sentences(text: str) -> str:
    """Ensure proper sentence capitalization."""
    if not text:
        return text

    result = re.sub(
        r"([.!?])\s+([a-zA-Z\u0430-\u044f\u0410-\u042f\u0451\u0401])",
        lambda m: m.group(1) + " " + m.group(2).upper(),
        text,
    )

    if result and result[0].isalpha():
        result = result[0].upper() + result[1:]

    return result


def _get_punct_model():
    """Lazy-load the punctuation restoration model (CPU, ~300MB)."""
    global _punct_model, _punct_available

    if _punct_available is False:
        return None
    if _punct_model is not None:
        return _punct_model

    try:
        from deepmultilingualpunctuation import PunctuationModel

        _punct_model = PunctuationModel()
        _punct_available = True
        logger.info("Punctuation restoration model loaded (CPU)")
        return _punct_model
    except ImportError:
        logger.warning(
            "deepmultilingualpunctuation not installed -- "
            "skipping punctuation restoration."
        )
        _punct_available = False
        return None
    except Exception as e:
        logger.warning(f"Failed to load punctuation model: {e}")
        _punct_available = False
        return None


def restore_punctuation(text: str) -> str:
    """Restore punctuation via deepmultilingualpunctuation (CPU BERT, ~300MB)."""
    if not text or len(text.strip()) < 5:
        return text

    model = _get_punct_model()
    if model is None:
        return text

    try:
        result = model.restore_punctuation(text)
        return result
    except Exception as e:
        logger.warning(f"Punctuation restoration failed: {e}")
        return text


# =====================================================================
# Russian number-words → digits normalisation
# =====================================================================
# Stakeholder review of court_hearing_129 (2026-04-23) found 6 of 128
# edits were spelled-out amounts that the legal team rewrote as digits
# (e.g. "шесть миллионов четыреста семь тысяч рублей" → "6 407 000
# рублей"). text2num.alpha2digit handles Russian compound numerals
# in-place: it leaves non-numeric words alone and rewrites only the
# digit sequences. We further format with NBSP thousands-separators
# in a small post-pass for legal-style readability.

_ALPHA2DIGIT = None  # lazy-loaded reference to text2num.alpha2digit


def _get_alpha2digit():
    """Lazy-import text2num. Returns the alpha2digit fn or None on
    ImportError (graceful degradation when the package isn't installed)."""
    global _ALPHA2DIGIT
    if _ALPHA2DIGIT is not None:
        return _ALPHA2DIGIT
    try:
        from text_to_num import alpha2digit  # type: ignore[import-not-found]
        _ALPHA2DIGIT = alpha2digit
    except ImportError:
        logger.warning(
            "text2num not installed — Russian number normalisation will "
            "be skipped. pip install text2num to enable."
        )
        _ALPHA2DIGIT = False  # negative cache
    return _ALPHA2DIGIT or None


_LARGE_NUMBER_RE = re.compile(r"\b(\d{4,})\b")


def _add_nbsp_thousands(match: re.Match[str]) -> str:
    """Insert NBSP every 3 digits from the right: 6407000 → 6 407 000.
    Used as a re.sub callback so we only touch digit groups of 4+."""
    n = match.group(1)
    rev = n[::-1]
    grouped = " ".join(rev[i:i + 3] for i in range(0, len(rev), 3))
    return grouped[::-1]


def normalize_russian_numbers(text: str) -> str:
    """Convert spelled-out Russian numerals to digits.

    Applies text2num.alpha2digit("ru") to the input. Conservative —
    leaves non-numeric content untouched. Adds NBSP thousands
    separators to numbers >= 1000 so legal documents read like
    "6 407 000 рублей" not "6407000 рублей".

    Returns input unchanged if text2num isn't installed or if the
    library throws (graceful degradation, never blocks the pipeline).
    """
    if not text or len(text.strip()) < 3:
        return text
    fn = _get_alpha2digit()
    if fn is None:
        return text
    try:
        # `relaxed=True` lets the parser skip stray words between
        # numerals — important because Russian inflects ("одна тысяча",
        # "одной тысячи", "одну тысячу" all mean "one thousand").
        converted = fn(text, "ru", relaxed=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("alpha2digit failed on text (%d chars): %s", len(text), exc)
        return text

    if not isinstance(converted, str) or not converted:
        return text

    # Insert NBSP thousands-separators on numbers ≥ 1000.
    return _LARGE_NUMBER_RE.sub(_add_nbsp_thousands, converted)


def postprocess_transcript(
    segments: list[AlignedSegment],
    language: Optional[str] = None,
    restore_punctuation_enabled: bool = False,
    remove_fillers: bool = True,
    remove_repetitions: bool = True,
    capitalize_sentences_enabled: bool = True,
    normalize_numbers_enabled: bool = True,
    filter_hallucinations_enabled: bool = True,
) -> tuple[list[AlignedSegment], str]:
    """Full post-processing pipeline for transcript segments.

    IMPORTANT -- restore_punctuation_enabled:
        DMP is only net-positive when ASR produces no punctuation.
        GigaAM v3 and antony66/whisper-large-v3-russian emit punctuation
        with higher F1 than DMP (benchmark on court_hearing_129:
        GigaAM 73.2%, DMP 53.7%). Running DMP on their output is a
        regression. Enable only for engines that drop punct.
    """
    if not segments:
        return segments, language or "ru"

    if language is None:
        language = detect_language(segments)

    logger.info(
        f"Post-processing {len(segments)} segments "
        f"(language: {language}, punct_restore: {restore_punctuation_enabled}, "
        f"fillers: {remove_fillers}, repetitions: {remove_repetitions}, "
        f"capitalize: {capitalize_sentences_enabled})"
    )

    cleaned = []
    removed_count = 0

    punct_model = _get_punct_model() if restore_punctuation_enabled else None
    if restore_punctuation_enabled:
        if punct_model:
            logger.warning(
                "Punctuation restoration ENABLED -- note that GigaAM/HF-Whisper "
                "engines already emit Russian punctuation with higher F1 than "
                "DMP (73%% vs 54%% on court benchmark). Disable unless your ASR "
                "engine drops punctuation."
            )
        else:
            logger.info("Punctuation restoration requested but model unavailable")

    for seg in segments:
        text = seg.text or ""

        # Hallucination filter runs FIRST so we don't waste cycles
        # capitalising / number-normalising a segment we're about to
        # drop. Empty result means "drop this segment".
        if filter_hallucinations_enabled:
            text = filter_hallucination_artifacts(text)
            if not text:
                removed_count += 1
                continue

        if punct_model and len(text.strip()) >= 5:
            text = restore_punctuation(text)
        if remove_fillers:
            text = clean_filler_words(text, language)
        if remove_repetitions:
            text = remove_repeated_phrases(text)
        # Russian number normalisation runs BEFORE capitalisation so
        # the capitalisation rule sees digits ("6 000 000 рублей") not
        # word forms — preserves sentence-start logic intact.
        if normalize_numbers_enabled and language == "ru":
            text = normalize_russian_numbers(text)
        if capitalize_sentences_enabled:
            text = capitalize_sentences(text)

        if text and len(text.strip()) >= MIN_MEANINGFUL_LENGTH:
            cleaned.append(
                AlignedSegment(
                    start=seg.start,
                    end=seg.end,
                    text=text,
                    speaker_id=seg.speaker_id,
                    speaker_name=seg.speaker_name,
                    confidence=seg.confidence,
                    attribution_confidence=seg.attribution_confidence,
                )
            )
        else:
            removed_count += 1

    merged = merge_short_segments(cleaned)

    # Address + legal abbreviation normalization (Sprint 2026-04-30, #40).
    # Applies regex substitutions from config/address_corrections.yaml
    # to fix consistent ASR mis-spellings of Moscow street names,
    # legal references, and currency forms. Runs AFTER merge so the
    # corrections see the final reader-facing text.
    try:
        from backend.core.address_normalize import normalize_segments
        normalize_segments(merged)
    except Exception as e:  # noqa: BLE001
        logger.debug(
            f"Address normalization failed (non-fatal): {e}"
        )

    logger.info(
        f"Post-processing complete: {len(segments)} -> {len(merged)} segments "
        f"({removed_count} empty removed, {len(cleaned) - len(merged)} merged)"
    )

    return merged, language
