"""Tests for quality assurance validation engine."""
import pytest

from backend.app.config import AppConfig, QualityConfig
from backend.app.models import (
    AlignedSegment,
    AudioMetadata,
    DiarizationSegment,
    MeetingProtocol,
    Participant,
    TranscriptionSegment,
    WordInfo,
)
from backend.core.qa import QualityAssurance


@pytest.fixture
def qa():
    """Create QA engine with default config."""
    config = AppConfig()
    return QualityAssurance(config)


@pytest.fixture
def strict_qa():
    """Create QA engine with strict thresholds."""
    config = AppConfig(quality=QualityConfig(min_confidence=0.9, min_audio_duration=30.0))
    return QualityAssurance(config)


class TestAudioValidation:
    """Audio metadata validation tests."""

    def test_valid_audio(self, qa):
        meta = AudioMetadata(
            duration=300.0, sample_rate=16000, channels=1, codec="pcm_s16le"
        )
        result = qa.validate_audio(meta)
        assert result.passed

    def test_audio_too_short(self, qa):
        meta = AudioMetadata(
            duration=5.0, sample_rate=16000, channels=1, codec="pcm_s16le"
        )
        result = qa.validate_audio(meta)
        assert not result.passed
        assert any("short" in i.message.lower() for i in result.issues)

    def test_audio_very_long(self, qa):
        meta = AudioMetadata(
            duration=8000.0, sample_rate=16000, channels=1, codec="pcm_s16le"
        )
        result = qa.validate_audio(meta)
        # Should warn but not necessarily fail
        assert any("long" in i.message.lower() for i in result.issues)


class TestTranscriptionValidation:
    """Transcription validation tests."""

    def test_valid_transcription(self, qa):
        segments = [
            TranscriptionSegment(
                start=0.0, end=5.0, text="Hello everyone", confidence=0.85
            ),
            TranscriptionSegment(
                start=5.0, end=10.0, text="Welcome to the meeting", confidence=0.90
            ),
        ]
        result = qa.validate_transcription(segments)
        assert result.passed

    def test_low_confidence_transcription(self, strict_qa):
        segments = [
            TranscriptionSegment(
                start=0.0, end=5.0, text="mumble mumble", confidence=0.3
            ),
        ]
        result = strict_qa.validate_transcription(segments)
        assert not result.passed

    def test_empty_transcription(self, qa):
        result = qa.validate_transcription([])
        assert not result.passed


class TestDiarizationValidation:
    """Diarization validation tests."""

    def test_valid_diarization(self, qa):
        segments = [
            DiarizationSegment(start=0.0, end=5.0, speaker_id="speaker_0"),
            DiarizationSegment(start=5.0, end=10.0, speaker_id="speaker_1"),
            DiarizationSegment(start=10.0, end=15.0, speaker_id="speaker_0"),
        ]
        result = qa.validate_diarization(segments)
        assert result.passed

    def test_empty_diarization(self, qa):
        result = qa.validate_diarization([])
        assert not result.passed


class TestAlignmentValidation:
    """Alignment validation tests."""

    def test_valid_alignment(self, qa):
        segments = [
            AlignedSegment(
                start=0.0, end=5.0, text="Hello", speaker_id="speaker_0"
            ),
            AlignedSegment(
                start=5.0, end=10.0, text="Hi there", speaker_id="speaker_1"
            ),
        ]
        result = qa.validate_alignment(segments)
        assert result.passed

    def test_empty_alignment(self, qa):
        result = qa.validate_alignment([])
        assert not result.passed


class TestProtocolValidation:
    """Protocol validation tests."""

    def test_valid_protocol(self, qa):
        protocol = MeetingProtocol(
            topic="Test Meeting",
            summary="Discussed important topics.",
            participants=[
                Participant(speaker_id="s1", speaker_name="Alice"),
            ],
            transcript=[
                AlignedSegment(
                    start=0.0, end=5.0, text="Hello", speaker_id="s1"
                ),
            ],
        )
        result = qa.validate_protocol(protocol)
        assert result.passed

    def test_empty_protocol(self, qa):
        protocol = MeetingProtocol()
        result = qa.validate_protocol(protocol)
        assert not result.passed
