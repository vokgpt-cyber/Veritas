"""Tests for Pydantic data models."""
from datetime import datetime

import pytest

from backend.app.models import (
    AlignedSegment,
    AudioMetadata,
    Decision,
    DiarizationSegment,
    MeetingJob,
    MeetingProtocol,
    Participant,
    PipelineState,
    QAIssue,
    QAResult,
    SpeakerProfile,
    TaskItem,
    TopicItem,
    TranscriptionSegment,
    WordInfo,
)


class TestWordInfo:
    """Word-level transcription model tests."""

    def test_basic_creation(self):
        word = WordInfo(start=0.0, end=0.5, word="hello", confidence=0.95)
        assert word.word == "hello"
        assert word.confidence == 0.95

    def test_serialization(self):
        word = WordInfo(start=0.0, end=0.5, word="test", confidence=0.8)
        data = word.model_dump()
        assert data["word"] == "test"
        restored = WordInfo(**data)
        assert restored == word


class TestTranscriptionSegment:
    """Transcription segment model tests."""

    def test_creation_with_words(self):
        words = [
            WordInfo(start=0.0, end=0.3, word="hello", confidence=0.9),
            WordInfo(start=0.3, end=0.6, word="world", confidence=0.85),
        ]
        seg = TranscriptionSegment(
            start=0.0, end=0.6, text="hello world", confidence=0.87, words=words
        )
        assert len(seg.words) == 2
        assert seg.text == "hello world"

    def test_empty_words_default(self):
        seg = TranscriptionSegment(start=0.0, end=1.0, text="test", confidence=0.5)
        assert seg.words == []


class TestDiarizationSegment:
    """Diarization segment model tests."""

    def test_anonymous_speaker(self):
        seg = DiarizationSegment(start=0.0, end=5.0, speaker_id="speaker_0")
        assert seg.speaker_name is None

    def test_named_speaker(self):
        seg = DiarizationSegment(
            start=0.0, end=5.0, speaker_id="speaker_0", speaker_name="Alice"
        )
        assert seg.speaker_name == "Alice"


class TestAlignedSegment:
    """Aligned segment model tests."""

    def test_creation(self):
        seg = AlignedSegment(
            start=0.0,
            end=5.0,
            text="Hello everyone",
            speaker_id="speaker_0",
            speaker_name="Alice",
            confidence=0.92,
        )
        assert seg.speaker_id == "speaker_0"
        assert seg.confidence == 0.92


class TestMeetingProtocol:
    """Meeting protocol model tests."""

    def test_empty_protocol(self):
        protocol = MeetingProtocol()
        assert protocol.topic == ""
        assert protocol.participants == []
        assert protocol.decisions == []
        assert protocol.tasks == []
        assert protocol.transcript == []

    def test_full_protocol(self):
        protocol = MeetingProtocol(
            topic="Budget Review Q1",
            participants=[
                Participant(speaker_id="s1", speaker_name="Alice", speaking_time=120.0),
                Participant(speaker_id="s2", speaker_name="Bob", speaking_time=80.0),
            ],
            summary="The team reviewed Q1 budget allocations.",
            key_topics=[TopicItem(title="Budget", content="Q1 budget review", speakers=["Alice"])],
            decisions=[Decision(text="Approved budget increase", responsible="Alice")],
            tasks=[TaskItem(text="Prepare Q2 forecast", assignee="Bob", deadline="2026-04-15")],
            open_questions=["How to handle overflow?"],
        )
        assert len(protocol.participants) == 2
        assert len(protocol.decisions) == 1
        assert protocol.tasks[0].assignee == "Bob"


class TestMeetingJob:
    """Meeting job tracking model tests."""

    def test_auto_id_generation(self):
        job1 = MeetingJob(filename="meeting.wav")
        job2 = MeetingJob(filename="meeting.wav")
        assert job1.id != job2.id  # UUIDs should be unique

    def test_initial_state(self):
        job = MeetingJob(filename="test.wav")
        assert job.state == PipelineState.UPLOADED
        assert job.progress == 0.0
        assert job.error is None
        assert job.retry_count == 0

    def test_state_transitions(self):
        job = MeetingJob(filename="test.wav")
        job.state = PipelineState.TRANSCRIBING
        job.progress = 25.0
        assert job.state == PipelineState.TRANSCRIBING
        assert job.progress == 25.0


class TestPipelineState:
    """Pipeline state enum tests."""

    def test_all_states(self):
        states = [
            PipelineState.UPLOADED,
            PipelineState.PREPROCESSING,
            PipelineState.TRANSCRIBING,
            PipelineState.DIARIZING,
            PipelineState.ALIGNING,
            PipelineState.SUMMARIZING,
            PipelineState.QA_VALIDATING,
            PipelineState.FORMATTING,
            PipelineState.COMPLETED,
            PipelineState.ERROR,
            PipelineState.RETRYING,
        ]
        # All 11 states should be unique
        assert len(states) == len(set(states)) == 11


class TestQAModels:
    """QA result model tests."""

    def test_qa_issue(self):
        issue = QAIssue(
            severity="warning",
            stage="transcription",
            message="Low confidence detected",
        )
        assert issue.severity == "warning"

    def test_qa_result_passing(self):
        result = QAResult(passed=True, score=0.95)
        assert result.passed
        assert result.issues == []

    def test_qa_result_failing(self):
        result = QAResult(
            passed=False,
            score=0.3,
            issues=[
                QAIssue(severity="critical", stage="audio", message="Too short")
            ],
        )
        assert not result.passed
        assert len(result.issues) == 1


class TestSpeakerProfile:
    """Speaker profile model tests."""

    def test_creation(self):
        profile = SpeakerProfile(name="Alice", position="Manager")
        assert profile.name == "Alice"
        assert profile.meetings_count == 0
        assert profile.id  # Auto-generated UUID

    def test_auto_timestamp(self):
        profile = SpeakerProfile(name="Bob")
        assert isinstance(profile.registered_at, datetime)


class TestAudioMetadata:
    """Audio metadata model tests."""

    def test_creation(self):
        meta = AudioMetadata(
            duration=120.5,
            sample_rate=16000,
            channels=1,
            codec="pcm_s16le",
            file_size=3840000,
        )
        assert meta.duration == 120.5
        assert meta.channels == 1
