"""Tests for transcript post-processing: dash strip + seam polish + merge.

Focused on the gap-aware seam polish rules (Bug 61), the leading
speaker-dash strip added in Phase A (2026-04-21), and the extended
turn-normalization rules from Phase A+ (internal dash markers,
repeated-comma collapse, leading comma strip). Tests run without ML
dependencies -- pure Python / Pydantic only.
"""
from __future__ import annotations

import pytest

from backend.app.models import AlignedSegment
from backend.core.postprocessor import (
    _cap_first,
    _collapse_repeated_commas,
    _flush_turn,
    _lower_first,
    _normalize_turn_text,
    _polish_seam,
    _strip_internal_dash_markers,
    _strip_leading_speaker_dash,
    clean_filler_words,
    merge_short_segments,
)


# ---------------------------------------------------------------------------
# _strip_leading_speaker_dash
# ---------------------------------------------------------------------------


class TestStripLeadingSpeakerDash:
    """Leading dash/comma cluster strip at turn-flush time."""

    def test_em_dash_space_stripped(self):
        """U+2014 EM DASH followed by space is removed from turn start."""
        assert _strip_leading_speaker_dash("\u2014 \u0414\u0430.") == "\u0414\u0430."

    def test_en_dash_space_stripped(self):
        """U+2013 EN DASH followed by space is removed from turn start."""
        assert _strip_leading_speaker_dash("\u2013 \u0414\u0430.") == "\u0414\u0430."

    def test_hyphen_space_stripped(self):
        """U+002D HYPHEN-MINUS followed by space is removed from turn start."""
        assert _strip_leading_speaker_dash("- \u0414\u0430.") == "\u0414\u0430."

    def test_no_leading_dash_unchanged(self):
        """Text without leading dash passes through untouched."""
        assert _strip_leading_speaker_dash("\u0414\u0430.") == "\u0414\u0430."

    def test_dash_without_space_preserved(self):
        """Dash not followed by separator (e.g. compound word) is preserved."""
        # Hyphenated Russian word, no space after the dash -> must not strip
        assert _strip_leading_speaker_dash("\u2014\u0414\u0430.") == "\u2014\u0414\u0430."

    def test_empty_string_safe(self):
        """Empty or None-like string returns unchanged."""
        assert _strip_leading_speaker_dash("") == ""

    def test_dash_then_lowercase_is_recapitalized(self):
        """After strip, a lowercase first letter should be capitalized."""
        # Russian lowercase 'da' -> expect uppercase after strip
        assert _strip_leading_speaker_dash("\u2014 \u0434\u0430.") == "\u0414\u0430."

    def test_internal_dash_preserved(self):
        """Only the leading dash is stripped; internal dashes survive."""
        # em-dash as direct-speech inside the same turn
        src = "\u2014 \u0414\u0430. \u2014 \u041a\u0430\u043a?"
        exp = "\u0414\u0430. \u2014 \u041a\u0430\u043a?"
        assert _strip_leading_speaker_dash(src) == exp

    def test_leading_comma_stripped(self):
        """User case #6: turn starts with ', text' -> strip comma, cap-first."""
        # ", v pyatnitsu" -> "V pyatnitsu"
        src = ", \u0432 \u043f\u044f\u0442\u043d\u0438\u0446\u0443"
        exp = "\u0412 \u043f\u044f\u0442\u043d\u0438\u0446\u0443"
        assert _strip_leading_speaker_dash(src) == exp

    def test_dash_comma_combo_stripped(self):
        """User case #7: turn starts with '-, text' -> strip both."""
        # em-dash, comma, space, 'tam'
        src = "\u2014, \u0442\u0430\u043c"
        exp = "\u0422\u0430\u043c"
        assert _strip_leading_speaker_dash(src) == exp

    def test_double_comma_prefix_stripped(self):
        """Leading ',, word' -> strip both commas."""
        src = ",, \u043d\u0443"
        exp = "\u041d\u0443"
        assert _strip_leading_speaker_dash(src) == exp

    def test_double_dash_prefix_stripped(self):
        """Leading '- - word' collapses under new (aggressive) regex."""
        # Both dashes are noise at turn start -- strip them all.
        src = "\u2014 \u2014 \u0414\u0430."
        exp = "\u0414\u0430."
        assert _strip_leading_speaker_dash(src) == exp


# ---------------------------------------------------------------------------
# _strip_internal_dash_markers
# ---------------------------------------------------------------------------


class TestStripInternalDashMarkers:
    """Strip dash markers after sentence-end punct inside merged turns."""

    def test_period_dash_space_stripped_and_recapitalized(self):
        """'. - X' becomes '. X' with X uppercased."""
        # ". - что" -> ". Что"
        src = "\u0444\u0443. \u2014 \u0447\u0442\u043e \u0431\u044b\u043b\u043e"
        exp = "\u0444\u0443. \u0427\u0442\u043e \u0431\u044b\u043b\u043e"
        assert _strip_internal_dash_markers(src) == exp

    def test_question_mark_dash_stripped(self):
        """User case #3: '? - X' becomes '? X'."""
        # "ne slyshu? - ne slyshu? - Alyon?"
        src = ("\u041d\u0435 \u0441\u043b\u044b\u0448\u0443? "
               "\u2014 \u041d\u0435 \u0441\u043b\u044b\u0448\u0443? "
               "\u2014 \u0410\u043b\u0451\u043d?")
        exp = ("\u041d\u0435 \u0441\u043b\u044b\u0448\u0443? "
               "\u041d\u0435 \u0441\u043b\u044b\u0448\u0443? "
               "\u0410\u043b\u0451\u043d?")
        assert _strip_internal_dash_markers(src) == exp

    def test_dash_comma_combo_stripped(self):
        """User case #7: '? -, X' becomes '? X'."""
        src = "\u0443\u0433\u0443. \u2014, \u043d\u0430\u0434\u043e"
        exp = "\u0443\u0433\u0443. \u041d\u0430\u0434\u043e"
        assert _strip_internal_dash_markers(src) == exp

    def test_grammar_dash_after_word_preserved(self):
        """'word - word' (predicate dash) is legitimate, must not strip."""
        # "dal'she - neizvestno"
        src = ("\u0434\u0430\u043b\u044c\u0448\u0435 \u2014 "
               "\u043d\u0435\u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e")
        assert _strip_internal_dash_markers(src) == src

    def test_ellipsis_triggers_strip(self):
        """Unicode ellipsis ... also counts as sentence-end punct."""
        src = "\u043d\u0443\u2026 \u2014 \u0434\u0430"
        exp = "\u043d\u0443\u2026 \u0414\u0430"
        assert _strip_internal_dash_markers(src) == exp

    def test_empty_string_safe(self):
        assert _strip_internal_dash_markers("") == ""

    def test_no_dash_unchanged(self):
        src = "\u041f\u0440\u0438\u0432\u0435\u0442. \u041a\u0430\u043a \u0434\u0435\u043b\u0430?"
        assert _strip_internal_dash_markers(src) == src

    def test_multiple_markers_all_stripped(self):
        """Chain of '. - X. - Y. - Z' collapses to '. X. Y. Z'."""
        # "A. - B. - C."
        src = "A. \u2014 B. \u2014 C."
        exp = "A. B. C."
        assert _strip_internal_dash_markers(src) == exp


# ---------------------------------------------------------------------------
# _collapse_repeated_commas
# ---------------------------------------------------------------------------


class TestCollapseRepeatedCommas:
    """Repeated adjacent commas collapse to a single comma."""

    def test_double_comma_collapsed(self):
        assert _collapse_repeated_commas("foo,, bar") == "foo, bar"

    def test_spaced_double_comma_collapsed(self):
        assert _collapse_repeated_commas("foo, , bar") == "foo, bar"

    def test_triple_comma_collapsed(self):
        assert _collapse_repeated_commas("foo,,, bar") == "foo, bar"

    def test_single_comma_unchanged(self):
        assert _collapse_repeated_commas("foo, bar") == "foo, bar"

    def test_no_comma_unchanged(self):
        assert _collapse_repeated_commas("foo bar baz") == "foo bar baz"

    def test_empty_string_safe(self):
        assert _collapse_repeated_commas("") == ""

    def test_user_case_double_comma_after_filler(self):
        """User case #5: 'A-a,, koe-kak' -> 'A-a, koe-kak'."""
        src = "\u0410-\u0430,, \u043a\u043e\u0435-\u043a\u0430\u043a"
        exp = "\u0410-\u0430, \u043a\u043e\u0435-\u043a\u0430\u043a"
        assert _collapse_repeated_commas(src) == exp


# ---------------------------------------------------------------------------
# _normalize_turn_text (integration of all three rules)
# ---------------------------------------------------------------------------


class TestNormalizeTurnText:
    """The flush-time orchestrator that applies all three rules in order."""

    def test_internal_marker_plus_leading_dash(self):
        """Both internal '. - ' and leading '- ' in the same turn are handled."""
        # "- Da. - Kak?" -> "Da. Kak?"
        src = "\u2014 \u0414\u0430. \u2014 \u041a\u0430\u043a?"
        exp = "\u0414\u0430. \u041a\u0430\u043a?"
        assert _normalize_turn_text(src) == exp

    def test_double_comma_plus_leading_dash(self):
        """Leading '- ' plus mid-turn ',,' in one pass."""
        # "- Foo,, bar." -> "Foo, bar."
        src = "\u2014 Foo,, bar."
        exp = "Foo, bar."
        assert _normalize_turn_text(src) == exp

    def test_no_changes_passthrough(self):
        """Clean text is returned identical."""
        src = "\u0414\u0430, \u0445\u043e\u0440\u043e\u0448\u043e."
        assert _normalize_turn_text(src) == src

    def test_user_case_multiple_markers_in_turn(self):
        """User case #7 polished end-to-end."""
        # "Ugu. -, nado posmotret', da. - Da."
        src = ("\u0423\u0433\u0443. \u2014, \u043d\u0430\u0434\u043e "
               "\u043f\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u0442\u044c, "
               "\u0434\u0430. \u2014 \u0414\u0430.")
        exp = ("\u0423\u0433\u0443. \u041d\u0430\u0434\u043e "
               "\u043f\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u0442\u044c, "
               "\u0434\u0430. \u0414\u0430.")
        assert _normalize_turn_text(src) == exp


# ---------------------------------------------------------------------------
# _polish_seam
# ---------------------------------------------------------------------------


class TestPolishSeam:
    """Four-branch gap-aware rule for same-speaker merge seams."""

    def test_terminal_punct_capitalizes_next(self):
        prev, nxt = _polish_seam("Good evening.", "how are you", gap=0.3)
        assert prev == "Good evening."
        assert nxt == "How are you"

    def test_continuation_punct_lowercases_next(self):
        prev, nxt = _polish_seam("Hello,", "World", gap=0.1)
        assert nxt == "world"

    def test_long_gap_inserts_period_and_capitalizes(self):
        prev, nxt = _polish_seam("Hello", "world", gap=1.0)
        assert prev == "Hello."
        assert nxt == "World"

    def test_short_gap_lowercases_next(self):
        prev, nxt = _polish_seam("I was", "Going", gap=0.05)
        assert nxt == "going"

    def test_ambiguous_middle_band_unchanged(self):
        prev, nxt = _polish_seam("I was", "Going", gap=0.3)
        assert prev == "I was"
        assert nxt == "Going"

    def test_empty_inputs_return_as_is(self):
        assert _polish_seam("", "next", 0.1) == ("", "next")
        assert _polish_seam("prev", "", 0.1) == ("prev", "")


# ---------------------------------------------------------------------------
# Filler-removal — hesitation syllables added in Phase A+
# ---------------------------------------------------------------------------


class TestHesitationFillers:
    """RU_FILLERS now includes а-а/аа/м-м/мм/э-э/ээ/ну-у variants."""

    def test_a_dash_a_removed(self):
        """'A-a, horosho' -> ', horosho' (filler gone, comma survives)."""
        # Input: "а-а, хорошо"
        src = "\u0430-\u0430, \u0445\u043e\u0440\u043e\u0448\u043e"
        out = clean_filler_words(src, language="ru")
        # The filler 'а-а' is removed; leading comma/whitespace trimmed too.
        assert "\u0430-\u0430" not in out
        assert "\u0445\u043e\u0440\u043e\u0448\u043e" in out

    def test_mm_removed(self):
        """Plain 'мм' is a hesitation sound, should be stripped."""
        src = "\u043c\u043c, \u0434\u0430"
        out = clean_filler_words(src, language="ru")
        assert "\u043c\u043c" not in out
        assert "\u0434\u0430" in out

    def test_eh_eh_removed(self):
        """'э-э' is a hesitation sound."""
        src = "\u044d-\u044d, \u043f\u043e\u043d\u044f\u043b"
        out = clean_filler_words(src, language="ru")
        assert "\u044d-\u044d" not in out
        assert "\u043f\u043e\u043d\u044f\u043b" in out

    def test_nu_u_removed(self):
        """Drawn-out 'ну-у' is a hesitation, 'ну' base form is also a filler."""
        src = "\u043d\u0443-\u0443, \u0434\u0430"
        out = clean_filler_words(src, language="ru")
        assert "\u043d\u0443-\u0443" not in out
        assert "\u0434\u0430" in out

    def test_word_containing_filler_substring_preserved(self):
        """Don't strip 'мм' from inside a legitimate word."""
        # "коммерческий" contains 'мм' -- must not be damaged
        src = "\u043a\u043e\u043c\u043c\u0435\u0440\u0447\u0435\u0441\u043a\u0438\u0439 \u043f\u0440\u043e\u0435\u043a\u0442"
        out = clean_filler_words(src, language="ru")
        assert "\u043a\u043e\u043c\u043c\u0435\u0440\u0447\u0435\u0441\u043a\u0438\u0439" in out


# ---------------------------------------------------------------------------
# merge_short_segments + _flush_turn integration
# ---------------------------------------------------------------------------


def _seg(start: float, end: float, text: str, speaker: str = "S1",
         conf: float = 0.9, attr: float | None = 0.9) -> AlignedSegment:
    return AlignedSegment(
        start=start,
        end=end,
        text=text,
        speaker_id=speaker,
        speaker_name=None,
        confidence=conf,
        attribution_confidence=attr,
    )


class TestMergeShortSegmentsWithFlush:
    """Verifies turn-level normalization is applied when a turn is flushed."""

    def test_single_turn_leading_dash_stripped(self):
        segs = [_seg(0.0, 1.5, "\u2014 \u0414\u0430, \u043a\u043e\u043d\u0435\u0447\u043d\u043e.")]
        merged = merge_short_segments(segs)
        assert len(merged) == 1
        assert not merged[0].text.startswith("\u2014 ")
        assert merged[0].text.startswith("\u0414")  # Capital D

    def test_merged_same_speaker_then_dash_strip(self):
        """Sub-segments merge, and leading dash on the first sub is stripped."""
        segs = [
            _seg(0.0, 1.5, "\u2014 \u0414\u0430."),
            _seg(1.6, 3.0, "\u041a\u043e\u043d\u0435\u0447\u043d\u043e."),
        ]
        merged = merge_short_segments(segs)
        assert len(merged) == 1
        # Leading dash gone, content preserved
        assert not merged[0].text.startswith("\u2014 ")
        assert "\u041a\u043e\u043d\u0435\u0447\u043d\u043e" in merged[0].text

    def test_speaker_change_triggers_independent_dash_strips(self):
        """Each speaker's turn gets its own normalization pass at flush."""
        segs = [
            _seg(0.0, 1.0, "\u2014 \u041f\u0435\u0440\u0432\u044b\u0439.", speaker="S1"),
            _seg(1.2, 2.0, "\u2014 \u0412\u0442\u043e\u0440\u043e\u0439.", speaker="S2"),
        ]
        merged = merge_short_segments(segs)
        assert len(merged) == 2
        assert not merged[0].text.startswith("\u2014 ")
        assert not merged[1].text.startswith("\u2014 ")

    def test_attribution_confidence_minned_on_merge(self):
        """Contested slice marks whole turn as contested (None-safe min)."""
        segs = [
            _seg(0.0, 1.0, "One.", conf=0.9, attr=0.95),
            _seg(1.1, 2.0, "Two.", conf=0.9, attr=0.55),  # contested
        ]
        merged = merge_short_segments(segs)
        assert len(merged) == 1
        assert merged[0].attribution_confidence == pytest.approx(0.55)

    def test_duration_weighted_confidence(self):
        """Merged confidence is duration-weighted average of children."""
        segs = [
            _seg(0.0, 4.0, "A.", conf=1.0, attr=0.9),  # 4s
            _seg(4.1, 5.1, "B.", conf=0.5, attr=0.9),  # 1s
        ]
        merged = merge_short_segments(segs)
        # (1.0 * 4 + 0.5 * 1) / (4 + 1) = 4.5/5 = 0.9
        assert merged[0].confidence == pytest.approx(0.9, abs=1e-3)

    def test_leading_comma_turn_normalized(self):
        """User case #6: merged turn starting with ', word' -> 'Word'."""
        # ", в пятницу, наверное."
        segs = [_seg(0.0, 1.5,
                     ", \u0432 \u043f\u044f\u0442\u043d\u0438\u0446\u0443, "
                     "\u043d\u0430\u0432\u0435\u0440\u043d\u043e\u0435.")]
        merged = merge_short_segments(segs)
        assert len(merged) == 1
        assert merged[0].text.startswith("\u0412")  # Capital V
        assert not merged[0].text.startswith(",")

    def test_internal_dash_marker_normalized(self):
        """User case #3: '? - X? - Y' collapses at flush."""
        # Same speaker, merged into one turn
        segs = [_seg(0.0, 3.0,
                     "\u041d\u0435 \u0441\u043b\u044b\u0448\u0443? "
                     "\u2014 \u041d\u0435 \u0441\u043b\u044b\u0448\u0443?")]
        merged = merge_short_segments(segs)
        assert len(merged) == 1
        # Middle marker gone
        assert " \u2014 " not in merged[0].text


class TestFlushTurnNoOpPath:
    """_flush_turn has a fast-path when normalization is a no-op."""

    def test_flush_turn_without_dash_appends_original(self):
        merged: list[AlignedSegment] = []
        seg = _seg(0.0, 2.0, "Plain text.")
        _flush_turn(merged, seg)
        assert merged == [seg]

    def test_flush_turn_none_buffer_is_safe(self):
        merged: list[AlignedSegment] = []
        _flush_turn(merged, None)
        assert merged == []

    def test_flush_turn_too_short_dropped(self):
        """Segments below MIN_MEANINGFUL_LENGTH after strip are dropped."""
        merged: list[AlignedSegment] = []
        # Single Cyrillic char after dash -> too short
        seg = _seg(0.0, 0.3, "\u2014 \u0430")
        _flush_turn(merged, seg)
        assert merged == []
