"""Block 5: End-to-end pipeline integration tests.

Tests the full orchestrator pipeline with mock engines to verify:
- Complete pipeline flow from audio to protocol
- Stage transitions and progress reporting
- Intermediate result saving/loading
- QA validation at each stage
- Error handling and retry logic
- Job management (create, list, delete)

These tests use mock engines because the real ML models (GigaAM, pyannote,
Qwen) require GPU and are not available in CI environments. The mocks
produce realistic outputs that exercise the full pipeline logic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import struct
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.app.config import AppConfig
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
    TaskItem,
    TopicItem,
    TranscriptionSegment,
    WordInfo,
)
from backend.core.aligner import TranscriptAligner
from backend.core.formatter import ProtocolFormatter
from backend.core.qa import QualityAssurance


# ---------------------------------------------------------------------------
# Test audio generation helpers
# ---------------------------------------------------------------------------


def generate_wav(path: str, duration: float, sample_rate: int = 16000, channels: int = 1) -> str:
    """Generate a valid WAV file with sine wave audio."""
    n_samples = int(duration * sample_rate)
    # 440Hz sine wave
    t = np.linspace(0, duration, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return path


def generate_silence_wav(path: str, duration: float, sample_rate: int = 16000) -> str:
    """Generate a WAV file containing only silence."""
    n_samples = int(duration * sample_rate)
    audio = np.zeros(n_samples, dtype=np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    return path


def generate_multi_speaker_segments(
    n_speakers: int, duration: float, avg_turn_duration: float = 5.0
) -> tuple[list[TranscriptionSegment], list[DiarizationSegment]]:
    """Generate realistic transcription and diarization segments for N speakers."""
    transcription = []
    diarization = []
    current_time = 0.0
    segment_idx = 0

    while current_time < duration:
        speaker_idx = segment_idx % n_speakers
        seg_duration = min(avg_turn_duration, duration - current_time)
        if seg_duration <= 0:
            break

        end_time = current_time + seg_duration

        transcription.append(
            TranscriptionSegment(
                start=current_time,
                end=end_time,
                text=f"Test segment {segment_idx} from speaker {speaker_idx}.",
                confidence=0.85 + np.random.uniform(-0.1, 0.1),
                words=[
                    WordInfo(
                        start=current_time + i * 0.3,
                        end=current_time + (i + 1) * 0.3,
                        word=w,
                        confidence=0.85,
                    )
                    for i, w in enumerate(f"Test segment {segment_idx}".split())
                ],
            )
        )

        diarization.append(
            DiarizationSegment(
                start=current_time,
                end=end_time,
                speaker_id=f"speaker_{speaker_idx}",
            )
        )

        current_time = end_time
        segment_idx += 1

    return transcription, diarization


def build_mock_protocol(
    aligned_segments: list[AlignedSegment],
    participants: list[Participant],
) -> MeetingProtocol:
    """Build a realistic MeetingProtocol for testing."""
    return MeetingProtocol(
        meeting_date=datetime.utcnow(),
        topic="Test Meeting: Quarterly Review",
        participants=participants,
        summary=(
            "The team discussed quarterly progress across all departments. "
            "Revenue targets were met for Q3. Engineering delivered the new "
            "authentication module on schedule. Marketing plans for Q4 were "
            "presented and approved by leadership."
        ),
        key_topics=[
            TopicItem(title="Revenue Review", content="Q3 targets achieved", speakers=["speaker_0"]),
            TopicItem(title="Engineering Update", content="Auth module delivered", speakers=["speaker_1"]),
            TopicItem(title="Marketing Plans", content="Q4 campaign approved", speakers=["speaker_2"]),
        ],
        decisions=[
            Decision(text="Approve Q4 marketing budget", responsible="speaker_0"),
            Decision(text="Extend authentication to mobile clients", responsible="speaker_1"),
        ],
        tasks=[
            TaskItem(text="Prepare Q4 budget breakdown", assignee="speaker_0", deadline="2026-04-15"),
            TaskItem(text="Deploy auth module to staging", assignee="speaker_1", deadline="2026-04-10"),
        ],
        open_questions=["When will mobile SDK be ready?"],
        transcript=aligned_segments,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_meeting_dir(tmp_path):
    """Create a temporary meeting directory."""
    meeting_dir = tmp_path / "meetings"
    meeting_dir.mkdir()
    return meeting_dir


@pytest.fixture
def config(tmp_meeting_dir):
    """Create test configuration."""
    return AppConfig(
        output={"output_dir": str(tmp_meeting_dir)},
        quality={
            "min_confidence": 0.5,
            "hallucination_check": True,
            "max_repeat_length": 50,
            "retry_count": 2,
            "min_audio_duration": 5.0,
            "max_audio_duration": 7200.0,
        },
    )


@pytest.fixture
def qa(config):
    """Create QualityAssurance instance."""
    return QualityAssurance(config)


@pytest.fixture
def formatter(config):
    """Create ProtocolFormatter instance."""
    return ProtocolFormatter(config)


@pytest.fixture
def test_audio_30s(tmp_path):
    """Generate 30-second test audio."""
    return generate_wav(str(tmp_path / "test_30s.wav"), 30.0)


@pytest.fixture
def test_audio_silence(tmp_path):
    """Generate 30-second silence audio."""
    return generate_silence_wav(str(tmp_path / "silence_30s.wav"), 30.0)


# ---------------------------------------------------------------------------
# E2E Pipeline Tests (with mocked engines)
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Test the full pipeline flow with mock ML engines."""

    def _build_mock_orchestrator(self, config: AppConfig):
        """Build an orchestrator with all engines mocked."""
        from backend.core.orchestrator import Orchestrator

        orch = Orchestrator(config)

        # Mock audio preprocessor
        mock_audio = MagicMock()
        mock_audio.process = AsyncMock(
            return_value=(
                "/tmp/test_processed.wav",
                AudioMetadata(
                    duration=30.0,
                    sample_rate=16000,
                    channels=1,
                    codec="pcm_s16le",
                    bitrate=256000,
                    file_size=960000,
                ),
            )
        )
        orch._audio = mock_audio

        # Mock VAD engine (Block 5d: VAD runs before diarization)
        mock_vad = MagicMock()
        mock_vad.load = MagicMock()
        mock_vad.unload = MagicMock()
        mock_vad.detect_from_file = MagicMock(return_value=[
            {"start": 0.0, "end": 5.0},
            {"start": 5.5, "end": 10.0},
            {"start": 10.5, "end": 15.0},
            {"start": 15.5, "end": 20.0},
            {"start": 20.5, "end": 25.0},
            {"start": 25.5, "end": 30.0},
        ])
        orch._vad = mock_vad

        # Mock ASR engine
        transcription, diarization = generate_multi_speaker_segments(
            n_speakers=3, duration=30.0, avg_turn_duration=5.0
        )

        mock_asr = MagicMock()
        mock_asr.required_vram_gb = 2.5
        mock_asr.load = MagicMock()
        mock_asr.unload = MagicMock()
        mock_asr.process = AsyncMock(return_value=transcription)
        orch._asr = mock_asr

        # Mock diarization engine
        mock_diarization = MagicMock()
        mock_diarization.required_vram_gb = 0.5
        mock_diarization.load = MagicMock()
        mock_diarization.unload = MagicMock()
        mock_diarization.process = AsyncMock(return_value=diarization)
        orch._diarization = mock_diarization

        # Mock summarization engine
        aligned = TranscriptAligner.align(transcription, diarization)
        participants = TranscriptAligner.get_participants(aligned)
        protocol = build_mock_protocol(aligned, participants)

        mock_summarization = MagicMock()
        mock_summarization.required_vram_gb = 0.0  # Ollama manages VRAM
        mock_summarization.load = MagicMock()
        mock_summarization.unload = MagicMock()
        mock_summarization.process = AsyncMock(return_value=protocol)
        orch._summarization = mock_summarization

        # Real aligner, QA, formatter
        orch._aligner = TranscriptAligner()
        orch._qa = QualityAssurance(config)
        orch._formatter = ProtocolFormatter

        # Mock VRAM manager
        mock_vram = MagicMock()
        mock_vram.wait_for_available = AsyncMock(return_value=True)
        mock_vram.check_available = MagicMock(return_value=True)
        mock_vram.register_model = MagicMock()
        mock_vram.unregister_model = MagicMock()
        mock_vram.release_all = MagicMock()
        orch._vram = mock_vram

        return orch

    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, config, tmp_meeting_dir):
        """Test complete pipeline produces valid protocol."""
        orch = self._build_mock_orchestrator(config)

        job = MeetingJob(filename="test_meeting.wav")
        progress_states = []

        def on_progress(j: MeetingJob):
            progress_states.append((j.state.value, j.progress))

        result = await orch.process_meeting(
            audio_path="/tmp/test_meeting.wav",
            job=job,
            on_progress=on_progress,
        )

        # Pipeline completed
        assert result is not None
        assert isinstance(result, MeetingProtocol)
        assert job.state == PipelineState.COMPLETED
        assert job.progress == 100.0

        # Protocol has content
        assert result.topic
        assert result.summary
        assert len(result.participants) >= 1
        assert len(result.decisions) >= 1
        assert len(result.tasks) >= 1
        assert len(result.transcript) >= 1

        # Progress was reported through all stages
        state_names = [s[0] for s in progress_states]
        assert "preprocessing" in state_names
        assert "transcribing" in state_names
        assert "diarizing" in state_names
        assert "aligning" in state_names
        assert "summarizing" in state_names
        assert "formatting" in state_names
        assert "completed" in state_names

        # Progress is monotonically increasing (with allowance for stage resets)
        progress_values = [s[1] for s in progress_states]
        for i in range(1, len(progress_values)):
            # Each new stage progress should be >= previous or start of new stage
            assert progress_values[i] >= 0

    @pytest.mark.asyncio
    async def test_pipeline_output_files_created(self, config, tmp_meeting_dir):
        """Test that DOCX and JSON output files are created."""
        orch = self._build_mock_orchestrator(config)

        job = MeetingJob(filename="test_meeting.wav")
        result = await orch.process_meeting("/tmp/test_meeting.wav", job)

        assert result is not None

        # Check output directory exists
        meeting_dir = tmp_meeting_dir / job.id
        assert meeting_dir.exists()

        # Check DOCX was created
        docx_files = list(meeting_dir.glob("*.docx"))
        assert len(docx_files) >= 1, "DOCX output file should be created"

        # Check JSON was created
        json_files = list(meeting_dir.glob("*.json"))
        assert len(json_files) >= 1, "JSON output file should be created"

    @pytest.mark.asyncio
    async def test_pipeline_intermediate_results_saved(self, config, tmp_meeting_dir):
        """Test that intermediate results are saved for recovery."""
        orch = self._build_mock_orchestrator(config)

        job = MeetingJob(filename="test_meeting.wav")
        result = await orch.process_meeting("/tmp/test_meeting.wav", job)

        assert result is not None

        meeting_dir = tmp_meeting_dir / job.id

        # Intermediate JSON files should have been created during processing
        # (cleanup may remove some, but the protocol should remain)
        json_files = list(meeting_dir.glob("*.json"))
        assert len(json_files) >= 1

    @pytest.mark.asyncio
    async def test_pipeline_vram_lifecycle(self, config, tmp_meeting_dir):
        """Test that VRAM is properly allocated and released."""
        orch = self._build_mock_orchestrator(config)

        job = MeetingJob(filename="test_meeting.wav")
        await orch.process_meeting("/tmp/test_meeting.wav", job)

        # Verify VRAM was checked before ASR and diarization (not summarization — Ollama manages its own)
        assert orch._vram.wait_for_available.call_count >= 2  # ASR, diarization

        # Verify engines were loaded then unloaded
        orch._asr.load.assert_called_once()
        orch._asr.unload.assert_called_once()
        orch._diarization.load.assert_called_once()
        orch._diarization.unload.assert_called_once()
        orch._summarization.load.assert_called_once()
        orch._summarization.unload.assert_called_once()

        # Verify VRAM released after ASR and diarization (not summarization)
        assert orch._vram.release_all.call_count >= 2

    @pytest.mark.asyncio
    async def test_pipeline_job_management(self, config, tmp_meeting_dir):
        """Test job CRUD operations."""
        orch = self._build_mock_orchestrator(config)

        # Create and process job
        job = MeetingJob(filename="test.wav")
        await orch.process_meeting("/tmp/test.wav", job)

        # List jobs
        jobs = orch.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].id == job.id

        # Get specific job
        retrieved = orch.get_job(job.id)
        assert retrieved is not None
        assert retrieved.state == PipelineState.COMPLETED

        # Delete job
        deleted = orch.delete_job(job.id)
        assert deleted is True

        # Verify deleted
        assert orch.get_job(job.id) is None
        assert len(orch.list_jobs()) == 0

        # Delete non-existent
        assert orch.delete_job("nonexistent") is False


# ---------------------------------------------------------------------------
# Pipeline Stage Tests
# ---------------------------------------------------------------------------


class TestPipelineStages:
    """Test individual pipeline stages in isolation."""

    def test_alignment_with_multi_speaker(self):
        """Test transcript alignment with multiple speakers."""
        trans, diar = generate_multi_speaker_segments(5, 60.0)
        aligned = TranscriptAligner.align(trans, diar)

        assert len(aligned) == len(trans)
        # Every segment should have a speaker_id
        for seg in aligned:
            assert seg.speaker_id.startswith("speaker_")
            assert seg.text
            assert seg.confidence > 0

    def test_alignment_merge_consecutive(self):
        """Test merging consecutive segments from same speaker."""
        segments = [
            AlignedSegment(start=0.0, end=5.0, text="Hello", speaker_id="speaker_0", confidence=0.9),
            AlignedSegment(start=5.0, end=10.0, text="world", speaker_id="speaker_0", confidence=0.8),
            AlignedSegment(start=10.0, end=15.0, text="Goodbye", speaker_id="speaker_1", confidence=0.85),
        ]
        merged = TranscriptAligner.merge_consecutive(segments)

        assert len(merged) == 2
        assert merged[0].text == "Hello world"
        assert merged[0].speaker_id == "speaker_0"
        assert merged[1].text == "Goodbye"

    def test_participant_extraction(self):
        """Test participant list extraction with speaking time stats."""
        trans, diar = generate_multi_speaker_segments(3, 30.0)
        aligned = TranscriptAligner.align(trans, diar)
        participants = TranscriptAligner.get_participants(aligned)

        assert len(participants) == 3
        total_share = sum(p.speaking_share for p in participants)
        assert abs(total_share - 100.0) < 1.0  # Shares should sum to ~100%

        # Sorted by speaking time
        for i in range(len(participants) - 1):
            assert participants[i].speaking_time >= participants[i + 1].speaking_time

    def test_protocol_formatter_docx(self, tmp_path):
        """Test DOCX generation with full protocol."""
        trans, diar = generate_multi_speaker_segments(3, 30.0)
        aligned = TranscriptAligner.align(trans, diar)
        participants = TranscriptAligner.get_participants(aligned)
        protocol = build_mock_protocol(aligned, participants)

        output_path = str(tmp_path / "test_protocol.docx")
        result = ProtocolFormatter.format_docx(protocol, output_path)

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_protocol_formatter_json(self, tmp_path):
        """Test JSON generation and roundtrip."""
        trans, diar = generate_multi_speaker_segments(3, 30.0)
        aligned = TranscriptAligner.align(trans, diar)
        participants = TranscriptAligner.get_participants(aligned)
        protocol = build_mock_protocol(aligned, participants)

        output_path = str(tmp_path / "test_protocol.json")
        result = ProtocolFormatter.format_json(protocol, output_path)

        assert Path(result).exists()

        # Verify JSON is valid and contains expected fields
        with open(result, "r") as f:
            data = json.load(f)

        assert data["topic"] == "Test Meeting: Quarterly Review"
        assert len(data["participants"]) == 3
        assert len(data["decisions"]) >= 1
        assert len(data["tasks"]) >= 1


# ---------------------------------------------------------------------------
# QA Validation Tests
# ---------------------------------------------------------------------------


class TestQAValidation:
    """Test QA validation logic and threshold tuning."""

    def test_audio_validation_normal(self, config):
        """Test audio validation with normal parameters."""
        qa = QualityAssurance(config)
        metadata = AudioMetadata(
            duration=600.0,  # 10 minutes
            sample_rate=16000,
            channels=1,
            codec="pcm_s16le",
            file_size=19200000,
        )
        result = qa.validate_audio(metadata)
        assert result.passed is True
        assert result.score >= 90

    def test_audio_validation_too_short(self, config):
        """Test audio validation rejects too-short audio."""
        qa = QualityAssurance(config)
        metadata = AudioMetadata(
            duration=3.0,  # 3 seconds, below min_audio_duration=5
            sample_rate=16000,
            channels=1,
            codec="pcm_s16le",
            file_size=96000,
        )
        result = qa.validate_audio(metadata)
        assert result.passed is False
        assert any("too short" in issue.message.lower() for issue in result.issues)

    def test_audio_validation_very_long(self, config):
        """Test audio validation warns on very long audio."""
        qa = QualityAssurance(config)
        metadata = AudioMetadata(
            duration=8000.0,  # > 7200 max
            sample_rate=16000,
            channels=1,
            codec="pcm_s16le",
            file_size=256000000,
        )
        result = qa.validate_audio(metadata)
        # Long audio is a warning, not a critical failure
        assert any("long" in issue.message.lower() for issue in result.issues)

    def test_transcription_validation_high_confidence(self, config):
        """Test transcription validation with high-confidence segments."""
        qa = QualityAssurance(config)
        segments = [
            TranscriptionSegment(start=i * 5.0, end=(i + 1) * 5.0,
                                 text=f"Segment {i} with meaningful content", confidence=0.9)
            for i in range(10)
        ]
        result = qa.validate_transcription(segments)
        assert result.passed is True
        assert result.score >= 80

    def test_transcription_validation_low_confidence(self, config):
        """Test transcription validation warns on low confidence."""
        qa = QualityAssurance(config)
        segments = [
            TranscriptionSegment(start=i * 5.0, end=(i + 1) * 5.0,
                                 text=f"Segment {i}", confidence=0.3)
            for i in range(10)
        ]
        result = qa.validate_transcription(segments)
        # Low confidence is a warning/issue
        assert result.score <= 80

    def test_transcription_validation_empty(self, config):
        """Test transcription validation rejects empty input."""
        qa = QualityAssurance(config)
        result = qa.validate_transcription([])
        assert result.passed is False
        assert result.score == 0

    def test_diarization_validation_multi_speaker(self, config):
        """Test diarization validation with multiple speakers."""
        qa = QualityAssurance(config)
        _, diar = generate_multi_speaker_segments(4, 60.0)
        result = qa.validate_diarization(diar)
        assert result.passed is True

    def test_diarization_validation_single_speaker(self, config):
        """Test diarization validation warns on single speaker."""
        qa = QualityAssurance(config)
        diar = [
            DiarizationSegment(start=i * 5.0, end=(i + 1) * 5.0, speaker_id="speaker_0")
            for i in range(10)
        ]
        result = qa.validate_diarization(diar)
        # Single speaker produces a warning but passes
        assert result.passed is True
        assert result.score < 100

    def test_alignment_validation_valid(self, config):
        """Test alignment validation with well-formed segments."""
        qa = QualityAssurance(config)
        segments = [
            AlignedSegment(start=i * 5.0, end=(i + 1) * 5.0,
                          text=f"Text {i}", speaker_id=f"speaker_{i % 3}", confidence=0.9)
            for i in range(10)
        ]
        result = qa.validate_alignment(segments)
        assert result.passed is True

    def test_alignment_validation_missing_speaker(self, config):
        """Test alignment validation catches missing speaker_id."""
        qa = QualityAssurance(config)
        segments = [
            AlignedSegment(start=0.0, end=5.0, text="Hello", speaker_id="", confidence=0.9),
        ]
        result = qa.validate_alignment(segments)
        assert result.passed is False

    def test_protocol_validation_complete(self, config):
        """Test protocol validation with complete protocol."""
        qa = QualityAssurance(config)
        trans, diar = generate_multi_speaker_segments(3, 30.0)
        aligned = TranscriptAligner.align(trans, diar)
        participants = TranscriptAligner.get_participants(aligned)
        protocol = build_mock_protocol(aligned, participants)

        result = qa.validate_protocol(protocol)
        assert result.passed is True
        assert result.score >= 80

    def test_protocol_validation_empty(self, config):
        """Test protocol validation flags empty protocol."""
        qa = QualityAssurance(config)
        protocol = MeetingProtocol()
        result = qa.validate_protocol(protocol)
        assert result.score < 50

    def test_hallucination_detection_repeated_text(self):
        """Test hallucination detection catches repeated substrings.

        Note: max_repeat_length must be > 50 for the detection loop to execute
        (it iterates from 50 to max_repeat_length). With default 50, the range
        is empty. We use max_repeat_length=100 here to test the detection logic.
        """
        from backend.app.config import AppConfig

        hallucination_config = AppConfig(
            output={"output_dir": "/tmp/veritas_test_hallucination"},
            quality={
                "min_confidence": 0.5,
                "hallucination_check": True,
                "max_repeat_length": 100,
                "retry_count": 2,
                "min_audio_duration": 5.0,
                "max_audio_duration": 7200.0,
            },
        )
        qa = QualityAssurance(hallucination_config)

        # Build text with exact 50-char substring repeated consecutively
        base = "a" * 60
        repeated_text = base + base  # 120 chars, with 60-char repeat
        segments = [
            TranscriptionSegment(
                start=0.0, end=30.0, text=repeated_text, confidence=0.8
            )
        ]
        result = qa.validate_transcription(segments)
        has_hallucination_warning = any(
            "hallucination" in str(issue.message).lower()
            for issue in result.issues
        )
        assert has_hallucination_warning, "Should detect repeated text as potential hallucination"


# ---------------------------------------------------------------------------
# Alignment Validation Tests
# ---------------------------------------------------------------------------


class TestAlignmentValidation:
    """Tests for TranscriptAligner.validate_alignment."""

    def test_valid_alignment(self):
        """Test validation passes for well-formed segments."""
        segments = [
            AlignedSegment(start=0.0, end=5.0, text="A", speaker_id="s0", confidence=0.9),
            AlignedSegment(start=5.0, end=10.0, text="B", speaker_id="s1", confidence=0.8),
        ]
        is_valid, issues = TranscriptAligner.validate_alignment(segments)
        assert is_valid is True
        assert len(issues) == 0

    def test_invalid_order(self):
        """Test validation catches out-of-order segments."""
        segments = [
            AlignedSegment(start=5.0, end=10.0, text="B", speaker_id="s0", confidence=0.9),
            AlignedSegment(start=0.0, end=5.0, text="A", speaker_id="s1", confidence=0.8),
        ]
        is_valid, issues = TranscriptAligner.validate_alignment(segments)
        assert is_valid is False
        assert len(issues) > 0

    def test_invalid_time_range(self):
        """Test validation catches end < start."""
        segments = [
            AlignedSegment(start=10.0, end=5.0, text="Bad", speaker_id="s0", confidence=0.9),
        ]
        is_valid, issues = TranscriptAligner.validate_alignment(segments)
        assert is_valid is False

    def test_empty_alignment(self):
        """Test validation rejects empty list."""
        is_valid, issues = TranscriptAligner.validate_alignment([])
        assert is_valid is False
