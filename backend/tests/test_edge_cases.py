"""Block 5: Edge case and error recovery tests.

Tests cover:
- Very short audio (<10s)
- Silence-heavy recordings
- Single speaker meetings
- Maximum speakers (20)
- Corrupted audio handling
- GPU OOM simulation
- Network interruption during processing
- Malformed input data
- Boundary conditions in all pipeline stages
- Recovery from intermediate failures
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import tempfile
import wave
from datetime import datetime
from pathlib import Path
from typing import Any
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
# Helpers
# ---------------------------------------------------------------------------


def generate_wav(path: str, duration: float, sr: int = 16000) -> str:
    """Generate a valid WAV file."""
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return path


@pytest.fixture
def config(tmp_path):
    return AppConfig(
        output={"output_dir": str(tmp_path / "output")},
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
    return QualityAssurance(config)


# ---------------------------------------------------------------------------
# Very Short Audio
# ---------------------------------------------------------------------------


class TestShortAudio:
    """Test handling of very short audio files."""

    def test_qa_rejects_1_second_audio(self, qa):
        """Audio under min_audio_duration should fail validation."""
        metadata = AudioMetadata(
            duration=1.0, sample_rate=16000, channels=1,
            codec="pcm_s16le", file_size=32000,
        )
        result = qa.validate_audio(metadata)
        assert result.passed is False

    def test_qa_accepts_minimum_audio(self, qa):
        """Audio at exactly min_audio_duration should pass."""
        metadata = AudioMetadata(
            duration=5.0, sample_rate=16000, channels=1,
            codec="pcm_s16le", file_size=160000,
        )
        result = qa.validate_audio(metadata)
        assert result.passed is True

    def test_qa_rejects_zero_duration(self, qa):
        """Zero-duration audio should fail."""
        metadata = AudioMetadata(
            duration=0.0, sample_rate=16000, channels=1,
            codec="pcm_s16le", file_size=0,
        )
        result = qa.validate_audio(metadata)
        assert result.passed is False

    def test_alignment_with_single_segment(self):
        """Alignment should work with just one segment."""
        trans = [TranscriptionSegment(start=0.0, end=3.0, text="Short.", confidence=0.9)]
        diar = [DiarizationSegment(start=0.0, end=3.0, speaker_id="speaker_0")]

        aligned = TranscriptAligner.align(trans, diar)
        assert len(aligned) == 1
        assert aligned[0].speaker_id == "speaker_0"


# ---------------------------------------------------------------------------
# Silence-Heavy Recordings
# ---------------------------------------------------------------------------


class TestSilenceHeavy:
    """Test handling of audio with lots of silence."""

    def test_empty_transcription_from_silence(self, qa):
        """Empty transcription (all silence) should fail QA."""
        result = qa.validate_transcription([])
        assert result.passed is False
        assert result.score == 0

    def test_empty_diarization_from_silence(self, qa):
        """Empty diarization should fail QA."""
        result = qa.validate_diarization([])
        assert result.passed is False
        assert result.score == 0

    def test_alignment_empty_transcription(self):
        """Alignment with empty transcription returns empty."""
        diar = [DiarizationSegment(start=0.0, end=30.0, speaker_id="speaker_0")]
        aligned = TranscriptAligner.align([], diar)
        assert len(aligned) == 0

    def test_alignment_empty_diarization(self):
        """Alignment with empty diarization assigns to speaker_0."""
        trans = [
            TranscriptionSegment(start=0.0, end=5.0, text="Hello", confidence=0.9),
            TranscriptionSegment(start=10.0, end=15.0, text="World", confidence=0.85),
        ]
        aligned = TranscriptAligner.align(trans, [])
        assert len(aligned) == 2
        assert all(seg.speaker_id == "speaker_0" for seg in aligned)

    def test_sparse_transcription_with_gaps(self, qa):
        """Transcription with large gaps between segments."""
        segments = [
            TranscriptionSegment(start=0.0, end=5.0, text="Beginning", confidence=0.9),
            # 50 second gap of silence
            TranscriptionSegment(start=55.0, end=60.0, text="End", confidence=0.85),
        ]
        result = qa.validate_transcription(segments)
        # Should pass but with warnings about gaps/empty segments
        assert result.score > 0


# ---------------------------------------------------------------------------
# Single Speaker
# ---------------------------------------------------------------------------


class TestSingleSpeaker:
    """Test handling of single-speaker meetings."""

    def test_single_speaker_diarization_qa(self, qa):
        """Single speaker should produce a warning but pass."""
        diar = [
            DiarizationSegment(start=i * 5.0, end=(i + 1) * 5.0, speaker_id="speaker_0")
            for i in range(20)
        ]
        result = qa.validate_diarization(diar)
        assert result.passed is True
        assert result.score < 100  # Warning reduces score
        assert any("1 speaker" in w for w in result.warnings)

    def test_single_speaker_alignment(self):
        """All segments should get the same speaker."""
        trans = [
            TranscriptionSegment(start=i * 5.0, end=(i + 1) * 5.0,
                                 text=f"Segment {i}", confidence=0.9)
            for i in range(10)
        ]
        diar = [
            DiarizationSegment(start=i * 5.0, end=(i + 1) * 5.0, speaker_id="speaker_0")
            for i in range(10)
        ]
        aligned = TranscriptAligner.align(trans, diar)

        assert all(seg.speaker_id == "speaker_0" for seg in aligned)

    def test_single_speaker_participant_stats(self):
        """Single speaker should get 100% speaking share."""
        trans = [
            TranscriptionSegment(start=0.0, end=60.0, text="Monologue", confidence=0.9)
        ]
        diar = [
            DiarizationSegment(start=0.0, end=60.0, speaker_id="speaker_0")
        ]
        aligned = TranscriptAligner.align(trans, diar)
        participants = TranscriptAligner.get_participants(aligned)

        assert len(participants) == 1
        assert abs(participants[0].speaking_share - 100.0) < 0.1


# ---------------------------------------------------------------------------
# Maximum Speakers (20)
# ---------------------------------------------------------------------------


class TestMaxSpeakers:
    """Test handling of meetings with many speakers (up to 20)."""

    @pytest.mark.parametrize("n_speakers", [5, 10, 15, 20])
    def test_alignment_with_many_speakers(self, n_speakers):
        """Alignment should handle up to 20 speakers correctly."""
        trans = []
        diar = []
        for i in range(n_speakers * 3):  # 3 turns per speaker
            speaker = f"speaker_{i % n_speakers}"
            start = i * 5.0
            end = start + 5.0
            trans.append(
                TranscriptionSegment(start=start, end=end,
                                     text=f"Speaker {i % n_speakers} talking", confidence=0.85)
            )
            diar.append(
                DiarizationSegment(start=start, end=end, speaker_id=speaker)
            )

        aligned = TranscriptAligner.align(trans, diar)
        participants = TranscriptAligner.get_participants(aligned)

        assert len(participants) == n_speakers
        total_share = sum(p.speaking_share for p in participants)
        assert abs(total_share - 100.0) < 1.0

    def test_20_speakers_docx_generation(self, tmp_path):
        """DOCX should handle 20-speaker participant table."""
        aligned = []
        for i in range(60):
            speaker = f"speaker_{i % 20}"
            aligned.append(
                AlignedSegment(
                    start=i * 5.0, end=(i + 1) * 5.0,
                    text=f"Utterance {i}", speaker_id=speaker, confidence=0.85,
                )
            )
        participants = TranscriptAligner.get_participants(aligned)

        protocol = MeetingProtocol(
            topic="Large Meeting",
            participants=participants,
            summary="A meeting with 20 participants.",
            transcript=aligned,
        )

        output = str(tmp_path / "20_speakers.docx")
        ProtocolFormatter.format_docx(protocol, output)

        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_diarization_qa_20_speakers(self, qa):
        """QA should accept up to 20 speakers."""
        diar = [
            DiarizationSegment(
                start=i * 3.0, end=(i + 1) * 3.0,
                speaker_id=f"speaker_{i % 20}",
            )
            for i in range(60)
        ]
        result = qa.validate_diarization(diar)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Corrupted / Malformed Input
# ---------------------------------------------------------------------------


def _ffprobe_available() -> bool:
    """Check if ffprobe supports -print_json flag."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-print_json", "-show_format", "/dev/null"],
            capture_output=True, text=True, timeout=5,
        )
        return "Option not found" not in (result.stderr or "")
    except Exception:
        return False


requires_ffprobe = pytest.mark.skipif(
    not _ffprobe_available(),
    reason="ffprobe with -print_json support not available",
)


class TestCorruptedInput:
    """Test handling of corrupted or malformed inputs."""

    @requires_ffprobe
    @pytest.mark.asyncio
    async def test_corrupted_audio_file(self, tmp_path):
        """Processing a corrupted file should raise an error."""
        from backend.core.audio import AudioPreprocessor

        # Write random bytes as a "WAV" file
        corrupt_path = str(tmp_path / "corrupt.wav")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT A REAL WAV FILE " * 100)

        is_valid = await AudioPreprocessor.validate(corrupt_path)
        assert is_valid is False

    @requires_ffprobe
    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path):
        """Empty file should fail validation."""
        from backend.core.audio import AudioPreprocessor

        empty_path = str(tmp_path / "empty.wav")
        Path(empty_path).touch()

        is_valid = await AudioPreprocessor.validate(empty_path)
        assert is_valid is False

    @requires_ffprobe
    @pytest.mark.asyncio
    async def test_nonexistent_file(self, tmp_path):
        """Non-existent file should raise FileNotFoundError."""
        from backend.core.audio import AudioPreprocessor

        with pytest.raises(FileNotFoundError):
            await AudioPreprocessor.process(
                str(tmp_path / "does_not_exist.wav"),
                str(tmp_path / "output"),
            )

    def test_transcription_with_empty_text_segments(self, qa):
        """Transcription segments with empty text should be flagged."""
        segments = [
            TranscriptionSegment(start=0.0, end=5.0, text="", confidence=0.5),
            TranscriptionSegment(start=5.0, end=10.0, text="   ", confidence=0.5),
            TranscriptionSegment(start=10.0, end=15.0, text="Real text", confidence=0.9),
        ]
        result = qa.validate_transcription(segments)
        assert any("empty" in w.lower() for w in result.warnings)

    def test_alignment_with_zero_duration_segments(self):
        """Zero-duration segments should not crash alignment."""
        trans = [
            TranscriptionSegment(start=5.0, end=5.0, text="Zero", confidence=0.9),
            TranscriptionSegment(start=5.0, end=10.0, text="Normal", confidence=0.9),
        ]
        diar = [
            DiarizationSegment(start=0.0, end=10.0, speaker_id="speaker_0"),
        ]
        aligned = TranscriptAligner.align(trans, diar)
        assert len(aligned) == 2

    def test_alignment_with_overlapping_segments(self):
        """Overlapping transcription segments should not crash."""
        trans = [
            TranscriptionSegment(start=0.0, end=10.0, text="Overlap A", confidence=0.9),
            TranscriptionSegment(start=5.0, end=15.0, text="Overlap B", confidence=0.8),
        ]
        diar = [
            DiarizationSegment(start=0.0, end=7.5, speaker_id="speaker_0"),
            DiarizationSegment(start=7.5, end=15.0, speaker_id="speaker_1"),
        ]
        aligned = TranscriptAligner.align(trans, diar)
        assert len(aligned) == 2

    def test_protocol_with_unicode_content(self, tmp_path):
        """Protocol with Russian/Unicode text should serialize correctly."""
        protocol = MeetingProtocol(
            topic="Obsuzhdeniye kvartalnykh rezultatov",
            summary="Komanda obsudila progress za kvartal.",
            participants=[
                Participant(speaker_id="speaker_0", speaker_name="Ivan Ivanov",
                           speaking_time=120.0, speaking_share=60.0),
            ],
            decisions=[Decision(text="Utverdili byudzhet na Q4")],
            tasks=[TaskItem(text="Podgotovit otchyot", assignee="Ivanov")],
            transcript=[
                AlignedSegment(start=0.0, end=5.0, text="Dobryy den.",
                              speaker_id="speaker_0", confidence=0.9),
            ],
        )

        # JSON roundtrip
        json_path = str(tmp_path / "unicode.json")
        ProtocolFormatter.format_json(protocol, json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["topic"] == protocol.topic

        # DOCX generation
        docx_path = str(tmp_path / "unicode.docx")
        ProtocolFormatter.format_docx(protocol, docx_path)
        assert Path(docx_path).exists()


# ---------------------------------------------------------------------------
# Error Recovery and Retry Logic
# ---------------------------------------------------------------------------


class TestErrorRecovery:
    """Test pipeline error handling and retry mechanisms."""

    def _build_orchestrator_with_failing_engine(
        self, config, fail_stage: str, fail_count: int = 1
    ):
        """Build orchestrator where a specific engine fails N times then succeeds."""
        from backend.core.orchestrator import Orchestrator

        orch = Orchestrator(config)

        # Standard mock data
        metadata = AudioMetadata(
            duration=30.0, sample_rate=16000, channels=1,
            codec="pcm_s16le", file_size=960000,
        )
        trans = [
            TranscriptionSegment(start=i * 5.0, end=(i + 1) * 5.0,
                                 text=f"Segment {i}", confidence=0.85)
            for i in range(6)
        ]
        diar = [
            DiarizationSegment(start=i * 5.0, end=(i + 1) * 5.0,
                               speaker_id=f"speaker_{i % 2}")
            for i in range(6)
        ]
        aligned = TranscriptAligner.align(trans, diar)
        participants = TranscriptAligner.get_participants(aligned)
        protocol = MeetingProtocol(
            topic="Test", summary="Test summary " * 10, participants=participants,
            decisions=[Decision(text="Test decision")],
            tasks=[TaskItem(text="Test task")],
            transcript=aligned,
        )

        # Mock VRAM
        mock_vram = MagicMock()
        mock_vram.wait_for_available = AsyncMock(return_value=True)
        mock_vram.register_model = MagicMock()
        mock_vram.unregister_model = MagicMock()
        mock_vram.release_all = MagicMock()
        orch._vram = mock_vram

        # Audio preprocessor
        mock_audio = MagicMock()
        if fail_stage == "preprocessing":
            call_count = {"n": 0}

            async def failing_process(*args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] <= fail_count:
                    raise RuntimeError("Preprocessing failed")
                return ("/tmp/processed.wav", metadata)

            mock_audio.process = failing_process
        else:
            mock_audio.process = AsyncMock(return_value=("/tmp/processed.wav", metadata))
        orch._audio = mock_audio

        # VAD (Block 5d: runs before diarization)
        mock_vad = MagicMock()
        mock_vad.load = MagicMock()
        mock_vad.unload = MagicMock()
        mock_vad.detect_from_file = MagicMock(return_value=[
            {"start": i * 5.0, "end": (i + 1) * 5.0} for i in range(6)
        ])
        orch._vad = mock_vad

        # ASR
        mock_asr = MagicMock()
        mock_asr.required_vram_gb = 2.5
        mock_asr.load = MagicMock()
        mock_asr.unload = MagicMock()
        if fail_stage == "asr":
            call_count = {"n": 0}

            async def failing_asr(*args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] <= fail_count:
                    raise RuntimeError("ASR OOM: CUDA out of memory")
                return trans

            mock_asr.process = failing_asr
        else:
            mock_asr.process = AsyncMock(return_value=trans)
        orch._asr = mock_asr

        # Diarization
        mock_diar = MagicMock()
        mock_diar.required_vram_gb = 0.5
        mock_diar.load = MagicMock()
        mock_diar.unload = MagicMock()
        mock_diar.process = AsyncMock(return_value=diar)
        orch._diarization = mock_diar

        # Summarization
        mock_summ = MagicMock()
        mock_summ.required_vram_gb = 5.0
        mock_summ.load = MagicMock()
        mock_summ.unload = MagicMock()
        mock_summ.process = AsyncMock(return_value=protocol)
        orch._summarization = mock_summ

        # Real pipeline components
        orch._aligner = TranscriptAligner()
        orch._qa = QualityAssurance(config)
        orch._formatter = ProtocolFormatter

        return orch

    @pytest.mark.asyncio
    async def test_retry_on_preprocessing_failure(self, config):
        """Pipeline should retry on preprocessing failure."""
        orch = self._build_orchestrator_with_failing_engine(
            config, fail_stage="preprocessing", fail_count=1
        )
        job = MeetingJob(filename="test.wav")
        result = await orch.process_meeting("/tmp/test.wav", job)

        # With retry, should succeed after first failure
        assert result is not None
        assert job.state == PipelineState.COMPLETED

    @pytest.mark.asyncio
    async def test_retry_on_asr_failure(self, config):
        """Pipeline should retry on ASR failure."""
        orch = self._build_orchestrator_with_failing_engine(
            config, fail_stage="asr", fail_count=1
        )
        job = MeetingJob(filename="test.wav")
        result = await orch.process_meeting("/tmp/test.wav", job)

        # With retry, should succeed
        assert result is not None

    @pytest.mark.asyncio
    async def test_permanent_failure_after_max_retries(self, config):
        """Pipeline should fail after max retries exceeded."""
        orch = self._build_orchestrator_with_failing_engine(
            config, fail_stage="preprocessing", fail_count=10  # More than max retries
        )
        job = MeetingJob(filename="test.wav")
        result = await orch.process_meeting("/tmp/test.wav", job)

        # Should fail permanently
        assert result is None
        assert job.state in (PipelineState.ERROR, PipelineState.RETRYING)
        assert job.error is not None

    @pytest.mark.asyncio
    async def test_vram_unavailable(self, config):
        """Pipeline should handle VRAM being unavailable."""
        from backend.core.orchestrator import Orchestrator

        orch = Orchestrator(config)

        # Setup basic mocks
        metadata = AudioMetadata(
            duration=30.0, sample_rate=16000, channels=1,
            codec="pcm_s16le", file_size=960000,
        )
        mock_audio = MagicMock()
        mock_audio.process = AsyncMock(return_value=("/tmp/processed.wav", metadata))
        orch._audio = mock_audio

        # VAD (Block 5d)
        mock_vad = MagicMock()
        mock_vad.load = MagicMock()
        mock_vad.unload = MagicMock()
        mock_vad.detect_from_file = MagicMock(return_value=[
            {"start": 0.0, "end": 10.0}, {"start": 10.5, "end": 20.0},
        ])
        orch._vad = mock_vad

        # Diarization engine
        mock_diar = MagicMock()
        mock_diar.required_vram_gb = 0.5
        mock_diar.load = MagicMock()
        mock_diar.unload = MagicMock()
        mock_diar.process = AsyncMock(return_value=[])
        orch._diarization = mock_diar

        # ASR engine
        mock_asr = MagicMock()
        mock_asr.required_vram_gb = 2.5
        mock_asr.load = MagicMock()
        mock_asr.unload = MagicMock()
        mock_asr.process = AsyncMock(return_value=[])
        orch._asr = mock_asr

        # VRAM manager that says "no VRAM available"
        mock_vram = MagicMock()
        mock_vram.wait_for_available = AsyncMock(return_value=False)
        mock_vram.register_model = MagicMock()
        mock_vram.unregister_model = MagicMock()
        mock_vram.release_all = MagicMock()
        orch._vram = mock_vram

        orch._aligner = TranscriptAligner()
        orch._qa = QualityAssurance(config)
        orch._formatter = ProtocolFormatter

        job = MeetingJob(filename="test.wav")
        result = await orch.process_meeting("/tmp/test.wav", job)

        # Should fail with VRAM error
        assert result is None
        assert "VRAM" in (job.error or "")


# ---------------------------------------------------------------------------
# Intermediate Save/Load
# ---------------------------------------------------------------------------


class TestIntermediateSaveLoad:
    """Test saving and loading of intermediate results."""

    def test_save_and_load_transcription(self, config, tmp_path):
        """Test saving/loading transcription intermediate results."""
        from backend.core.orchestrator import Orchestrator

        orch = Orchestrator(config)
        job_id = "test-save-load"

        # Create meeting directory
        meeting_dir = orch.get_meeting_dir(job_id)

        trans = [
            TranscriptionSegment(start=0.0, end=5.0, text="Hello", confidence=0.9),
            TranscriptionSegment(start=5.0, end=10.0, text="World", confidence=0.85),
        ]

        # Save
        orch._save_intermediate(job_id, "transcription", trans)

        # Verify file exists
        intermediate_file = meeting_dir / "transcription.json"
        assert intermediate_file.exists()

        # Load
        loaded = orch._load_intermediate(job_id, "transcription")
        assert loaded is not None
        assert len(loaded) == 2

    def test_load_nonexistent_intermediate(self, config):
        """Loading non-existent intermediate should return None."""
        from backend.core.orchestrator import Orchestrator

        orch = Orchestrator(config)
        result = orch._load_intermediate("nonexistent-job", "transcription")
        assert result is None


# ---------------------------------------------------------------------------
# Boundary Conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    """Test various boundary conditions in the pipeline."""

    def test_confidence_at_threshold(self, config):
        """Test segments at exact confidence threshold."""
        qa = QualityAssurance(config)
        segments = [
            TranscriptionSegment(start=0.0, end=5.0, text="Exact threshold",
                                 confidence=config.quality.min_confidence),
        ]
        result = qa.validate_transcription(segments)
        # At exactly the threshold should pass
        assert result.passed is True

    def test_confidence_just_below_threshold(self, config):
        """Test segments just below confidence threshold."""
        qa = QualityAssurance(config)
        segments = [
            TranscriptionSegment(start=0.0, end=5.0, text="Below threshold",
                                 confidence=config.quality.min_confidence - 0.01),
        ]
        result = qa.validate_transcription(segments)
        # Just below should trigger warning
        assert result.score < 100

    def test_max_audio_duration_boundary(self, config):
        """Test audio at exactly max duration."""
        qa = QualityAssurance(config)
        metadata = AudioMetadata(
            duration=config.quality.max_audio_duration,
            sample_rate=16000, channels=1,
            codec="pcm_s16le", file_size=0,
        )
        result = qa.validate_audio(metadata)
        # At max should pass (max is inclusive)
        assert result.passed is True

    def test_very_long_text_in_segment(self):
        """Test alignment with very long text in a single segment."""
        long_text = "word " * 10000  # ~50,000 characters
        trans = [TranscriptionSegment(start=0.0, end=300.0, text=long_text, confidence=0.85)]
        diar = [DiarizationSegment(start=0.0, end=300.0, speaker_id="speaker_0")]

        aligned = TranscriptAligner.align(trans, diar)
        assert len(aligned) == 1
        assert len(aligned[0].text) > 40000

    def test_many_short_segments(self):
        """Test alignment with many very short segments."""
        n = 1000
        trans = [
            TranscriptionSegment(start=i * 0.3, end=(i + 1) * 0.3,
                                 text=f"w{i}", confidence=0.9)
            for i in range(n)
        ]
        diar = [
            DiarizationSegment(start=i * 0.3, end=(i + 1) * 0.3,
                               speaker_id=f"speaker_{i % 5}")
            for i in range(n)
        ]
        aligned = TranscriptAligner.align(trans, diar)
        assert len(aligned) == n

    def test_protocol_formatter_all_empty_sections(self, tmp_path):
        """DOCX formatter should handle protocol with all empty sections."""
        protocol = MeetingProtocol(topic="Empty Meeting", summary="Nothing happened.")

        output = str(tmp_path / "empty_protocol.docx")
        ProtocolFormatter.format_docx(protocol, output)
        assert Path(output).exists()

    def test_merge_consecutive_empty_list(self):
        """Merge consecutive with empty list returns empty."""
        result = TranscriptAligner.merge_consecutive([])
        assert result == []

    def test_merge_consecutive_single_segment(self):
        """Merge consecutive with single segment returns it unchanged."""
        seg = AlignedSegment(start=0.0, end=5.0, text="Solo", speaker_id="s0", confidence=0.9)
        result = TranscriptAligner.merge_consecutive([seg])
        assert len(result) == 1
        assert result[0].text == "Solo"
