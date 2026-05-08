"""Tests for transcript alignment engine."""
import pytest

from backend.app.models import (
    AlignedSegment,
    DiarizationSegment,
    TranscriptionSegment,
    WordInfo,
)
from backend.core.aligner import TranscriptAligner


@pytest.fixture
def aligner():
    """Create aligner instance."""
    return TranscriptAligner()


class TestTranscriptAligner:
    """Transcript alignment tests."""

    def test_simple_alignment(self, aligner):
        """Two speakers, non-overlapping segments."""
        transcription = [
            TranscriptionSegment(
                start=0.0, end=5.0, text="Hello everyone", confidence=0.9
            ),
            TranscriptionSegment(
                start=6.0, end=10.0, text="Welcome to the meeting", confidence=0.85
            ),
        ]
        diarization = [
            DiarizationSegment(start=0.0, end=5.5, speaker_id="speaker_0"),
            DiarizationSegment(start=5.5, end=10.0, speaker_id="speaker_1"),
        ]

        result = aligner.align(transcription, diarization)

        assert len(result) >= 2
        assert result[0].speaker_id == "speaker_0"
        assert result[0].text == "Hello everyone"

    def test_single_speaker(self, aligner):
        """All segments from one speaker."""
        transcription = [
            TranscriptionSegment(
                start=0.0, end=3.0, text="First point", confidence=0.8
            ),
            TranscriptionSegment(
                start=3.5, end=6.0, text="Second point", confidence=0.85
            ),
        ]
        diarization = [
            DiarizationSegment(start=0.0, end=6.0, speaker_id="speaker_0"),
        ]

        result = aligner.align(transcription, diarization)

        assert all(seg.speaker_id == "speaker_0" for seg in result)

    def test_empty_inputs(self, aligner):
        """Empty transcription should return empty result."""
        result = aligner.align([], [])
        assert result == []

    def test_empty_transcription(self, aligner):
        """Empty transcription with non-empty diarization."""
        diarization = [
            DiarizationSegment(start=0.0, end=5.0, speaker_id="speaker_0"),
        ]
        result = aligner.align([], diarization)
        assert result == []

    def test_result_chronological_order(self, aligner):
        """Results should be in chronological order."""
        transcription = [
            TranscriptionSegment(
                start=0.0, end=3.0, text="First", confidence=0.8
            ),
            TranscriptionSegment(
                start=4.0, end=7.0, text="Second", confidence=0.85
            ),
            TranscriptionSegment(
                start=8.0, end=11.0, text="Third", confidence=0.9
            ),
        ]
        diarization = [
            DiarizationSegment(start=0.0, end=3.5, speaker_id="speaker_0"),
            DiarizationSegment(start=3.5, end=7.5, speaker_id="speaker_1"),
            DiarizationSegment(start=7.5, end=11.0, speaker_id="speaker_0"),
        ]

        result = aligner.align(transcription, diarization)

        for i in range(len(result) - 1):
            assert result[i].start <= result[i + 1].start

    def test_participant_extraction(self, aligner):
        """Test that participants are correctly extracted."""
        transcription = [
            TranscriptionSegment(
                start=0.0, end=5.0, text="Hello", confidence=0.9
            ),
            TranscriptionSegment(
                start=5.0, end=10.0, text="Hi there", confidence=0.85
            ),
        ]
        diarization = [
            DiarizationSegment(start=0.0, end=5.0, speaker_id="speaker_0"),
            DiarizationSegment(start=5.0, end=10.0, speaker_id="speaker_1"),
        ]

        result = aligner.align(transcription, diarization)
        speaker_ids = set(seg.speaker_id for seg in result)
        assert len(speaker_ids) >= 2

    def test_court_diarization_split_preserves_short_turns(self, aligner):
        """Court mode splits one ASR chunk by speaker-change boundaries."""
        transcription = [
            TranscriptionSegment(
                start=0.0,
                end=4.0,
                text="Any questions yes proceed",
                confidence=0.9,
                words=[
                    WordInfo(start=0.0, end=0.5, word="Any", confidence=0.9),
                    WordInfo(start=0.5, end=1.0, word="questions", confidence=0.9),
                    WordInfo(start=1.1, end=1.5, word="yes", confidence=0.9),
                    WordInfo(start=2.1, end=3.0, word="proceed", confidence=0.9),
                ],
            )
        ]
        diarization = [
            DiarizationSegment(start=0.0, end=1.0, speaker_id="judge"),
            DiarizationSegment(start=1.0, end=2.0, speaker_id="lawyer_a"),
            DiarizationSegment(start=2.0, end=4.0, speaker_id="judge"),
        ]

        result = aligner.align(
            transcription,
            diarization,
            diarization_split=True,
        )

        assert [seg.speaker_id for seg in result] == [
            "judge",
            "lawyer_a",
            "judge",
        ]
        assert [seg.text for seg in result] == [
            "Any questions",
            "yes",
            "proceed",
        ]

    def test_sentence_split_never_emits_inverted_timestamps(self, aligner):
        """Bad word-level timestamps must not leak start > end segments."""
        transcription = [
            TranscriptionSegment(
                start=10.0,
                end=20.0,
                text="First sentence. Second sentence.",
                confidence=0.9,
                words=[
                    WordInfo(start=13.0, end=12.0, word="First", confidence=0.9),
                    WordInfo(start=12.1, end=12.2, word="sentence.", confidence=0.9),
                    WordInfo(start=17.0, end=16.0, word="Second", confidence=0.9),
                    WordInfo(start=16.1, end=16.2, word="sentence.", confidence=0.9),
                ],
            )
        ]

        result = aligner.split_transcription_by_sentences(transcription)

        assert result
        assert all(seg.end >= seg.start for seg in result)

    def test_align_normalizes_inverted_input_timestamps(self, aligner):
        """Alignment output should be time-valid even if ASR gives bad bounds."""
        transcription = [
            TranscriptionSegment(
                start=5.0, end=4.0, text="Inverted bounds", confidence=0.9
            )
        ]
        diarization = [
            DiarizationSegment(start=4.0, end=5.0, speaker_id="speaker_0"),
        ]

        result = aligner.align(transcription, diarization, sentence_split=False)

        assert len(result) == 1
        assert result[0].start == 4.0
        assert result[0].end == 5.0
        assert result[0].speaker_id == "speaker_0"
