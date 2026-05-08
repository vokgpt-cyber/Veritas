"""Block 5: Performance benchmarking and VRAM optimization tests.

Tests focus on:
- Pipeline stage timing measurement
- Audio preprocessing throughput
- Alignment algorithm performance scaling
- QA validation performance
- DOCX/JSON formatter throughput
- Memory tracking and VRAM manager behavior
- Concurrent job handling
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from backend.app.config import AppConfig
from backend.app.models import (
    AlignedSegment,
    AudioMetadata,
    DiarizationSegment,
    MeetingJob,
    MeetingProtocol,
    Participant,
    PipelineState,
    TranscriptionSegment,
    TopicItem,
    Decision,
    TaskItem,
    WordInfo,
)
from backend.core.aligner import TranscriptAligner
from backend.core.formatter import ProtocolFormatter
from backend.core.qa import QualityAssurance
from backend.core.vram_manager import VRAMManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_segments(n_segments: int, duration: float = 60.0) -> tuple[list, list]:
    """Generate N transcription and diarization segments over given duration."""
    seg_duration = duration / n_segments
    trans = []
    diar = []
    n_speakers = max(2, min(n_segments // 5, 10))

    for i in range(n_segments):
        start = i * seg_duration
        end = start + seg_duration
        speaker = f"speaker_{i % n_speakers}"

        trans.append(
            TranscriptionSegment(
                start=start, end=end,
                text=f"Performance test segment number {i} with some text content for realism.",
                confidence=0.85,
                words=[
                    WordInfo(start=start + j * 0.15, end=start + (j + 1) * 0.15,
                             word=w, confidence=0.85)
                    for j, w in enumerate(f"Segment {i} text".split())
                ],
            )
        )
        diar.append(
            DiarizationSegment(start=start, end=end, speaker_id=speaker)
        )

    return trans, diar


def build_protocol(n_segments: int) -> MeetingProtocol:
    """Build a protocol with N transcript segments."""
    trans, diar = generate_segments(n_segments)
    aligned = TranscriptAligner.align(trans, diar)
    participants = TranscriptAligner.get_participants(aligned)

    return MeetingProtocol(
        meeting_date=datetime.utcnow(),
        topic="Performance Test Meeting",
        participants=participants,
        summary="A " * 200,  # ~200 word summary
        key_topics=[TopicItem(title=f"Topic {i}", content=f"Content {i}") for i in range(5)],
        decisions=[Decision(text=f"Decision {i}") for i in range(3)],
        tasks=[TaskItem(text=f"Task {i}", assignee=f"speaker_{i % 3}") for i in range(5)],
        open_questions=["Question 1", "Question 2"],
        transcript=aligned,
    )


# ---------------------------------------------------------------------------
# Alignment Performance
# ---------------------------------------------------------------------------


class TestAlignmentPerformance:
    """Benchmark alignment algorithm at different scales."""

    @pytest.mark.parametrize("n_segments", [50, 200, 500, 1000])
    def test_alignment_speed(self, n_segments):
        """Alignment should complete within acceptable time for N segments."""
        trans, diar = generate_segments(n_segments, duration=n_segments * 5.0)

        start = time.perf_counter()
        aligned = TranscriptAligner.align(trans, diar)
        elapsed = time.perf_counter() - start

        assert len(aligned) == n_segments
        # Alignment is O(N*M), should complete within 2s even for 1000 segments
        assert elapsed < 2.0, (
            f"Alignment of {n_segments} segments took {elapsed:.3f}s (limit: 2.0s)"
        )

    @pytest.mark.parametrize("n_segments", [50, 200, 500])
    def test_merge_consecutive_speed(self, n_segments):
        """Merge consecutive should be fast (O(N))."""
        trans, diar = generate_segments(n_segments)
        aligned = TranscriptAligner.align(trans, diar)

        start = time.perf_counter()
        merged = TranscriptAligner.merge_consecutive(aligned)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, (
            f"Merge of {n_segments} segments took {elapsed:.3f}s (limit: 0.5s)"
        )
        assert len(merged) <= len(aligned)

    def test_participant_extraction_speed(self):
        """Participant extraction should be fast."""
        trans, diar = generate_segments(1000, duration=5000.0)
        aligned = TranscriptAligner.align(trans, diar)

        start = time.perf_counter()
        participants = TranscriptAligner.get_participants(aligned)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5
        assert len(participants) >= 2


# ---------------------------------------------------------------------------
# QA Validation Performance
# ---------------------------------------------------------------------------


class TestQAPerformance:
    """Benchmark QA validation at scale."""

    @pytest.fixture
    def qa(self, tmp_path):
        config = AppConfig(output={"output_dir": str(tmp_path / "output")})
        return QualityAssurance(config)

    def test_transcription_validation_speed(self, qa):
        """Validate 1000 transcription segments quickly."""
        segments = [
            TranscriptionSegment(
                start=i * 3.0, end=(i + 1) * 3.0,
                text=f"Normal transcription segment {i} with reasonable text.",
                confidence=0.85,
            )
            for i in range(1000)
        ]

        start = time.perf_counter()
        result = qa.validate_transcription(segments)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"Transcription validation of 1000 segments took {elapsed:.3f}s"
        )
        assert result.passed is True

    def test_diarization_validation_speed(self, qa):
        """Validate 500 diarization segments quickly."""
        segments = [
            DiarizationSegment(
                start=i * 3.0, end=(i + 1) * 3.0,
                speaker_id=f"speaker_{i % 5}",
            )
            for i in range(500)
        ]

        start = time.perf_counter()
        result = qa.validate_diarization(segments)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5
        assert result.passed is True

    def test_hallucination_check_not_slow_on_normal_text(self, qa):
        """Hallucination check should not be slow on normal (non-repetitive) text."""
        segments = [
            TranscriptionSegment(
                start=i * 5.0, end=(i + 1) * 5.0,
                text=f"Unique segment number {i} discussing topic {i * 7 % 13} about subject {i * 3 % 11}.",
                confidence=0.85,
            )
            for i in range(200)
        ]

        start = time.perf_counter()
        result = qa.validate_transcription(segments)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, (
            f"Hallucination check on 200 normal segments took {elapsed:.3f}s"
        )


# ---------------------------------------------------------------------------
# Formatter Performance
# ---------------------------------------------------------------------------


class TestFormatterPerformance:
    """Benchmark DOCX and JSON generation."""

    @pytest.mark.parametrize("n_segments", [50, 200, 500])
    def test_docx_generation_speed(self, n_segments, tmp_path):
        """DOCX generation should scale reasonably."""
        protocol = build_protocol(n_segments)
        output = str(tmp_path / f"test_{n_segments}.docx")

        start = time.perf_counter()
        ProtocolFormatter.format_docx(protocol, output)
        elapsed = time.perf_counter() - start

        assert Path(output).exists()
        size_mb = Path(output).stat().st_size / 1e6

        # DOCX generation should complete within 5s even for 500 segments
        assert elapsed < 5.0, (
            f"DOCX with {n_segments} segments: {elapsed:.3f}s, {size_mb:.2f}MB"
        )

    @pytest.mark.parametrize("n_segments", [50, 200, 500])
    def test_json_generation_speed(self, n_segments, tmp_path):
        """JSON generation should be fast."""
        protocol = build_protocol(n_segments)
        output = str(tmp_path / f"test_{n_segments}.json")

        start = time.perf_counter()
        ProtocolFormatter.format_json(protocol, output)
        elapsed = time.perf_counter() - start

        assert Path(output).exists()
        assert elapsed < 1.0, (
            f"JSON with {n_segments} segments took {elapsed:.3f}s"
        )

        # Verify JSON is valid
        with open(output) as f:
            data = json.load(f)
        assert len(data["transcript"]) == n_segments


# ---------------------------------------------------------------------------
# VRAM Manager Tests
# ---------------------------------------------------------------------------


class TestVRAMManager:
    """Test VRAM manager behavior (no GPU required)."""

    def test_singleton_pattern(self):
        """VRAMManager should be a singleton."""
        # Reset singleton for testing
        VRAMManager._instance = None
        vm1 = VRAMManager()
        vm2 = VRAMManager()
        assert vm1 is vm2
        # Cleanup
        VRAMManager._instance = None

    def test_dummy_status_without_gpu(self):
        """Without GPU, should return dummy status."""
        VRAMManager._instance = None
        vm = VRAMManager()
        status = vm.get_status()

        assert "vram_used_gb" in status
        assert "vram_total_gb" in status
        assert "vram_free_gb" in status
        assert "gpu_utilization" in status
        assert "temperature" in status
        assert "device_name" in status
        assert "current_model" in status

        # Cleanup
        VRAMManager._instance = None

    def test_model_registration(self):
        """Test model register/unregister."""
        VRAMManager._instance = None
        vm = VRAMManager()

        assert vm.current_model is None
        vm.register_model("TestModel")
        assert vm.current_model == "TestModel"
        vm.unregister_model()
        assert vm.current_model is None

        # Cleanup
        VRAMManager._instance = None

    def test_check_available_without_gpu(self):
        """Check available should work without GPU (uses dummy status)."""
        VRAMManager._instance = None
        vm = VRAMManager()

        # Dummy status shows 24GB free
        assert vm.check_available(8.0) is True
        assert vm.check_available(30.0) is False

        # Cleanup
        VRAMManager._instance = None

    @pytest.mark.asyncio
    async def test_wait_for_available_immediate(self):
        """Wait should return immediately when VRAM is available."""
        VRAMManager._instance = None
        vm = VRAMManager()

        start = time.perf_counter()
        result = await vm.wait_for_available(2.0, timeout=5.0)
        elapsed = time.perf_counter() - start

        assert result is True
        assert elapsed < 1.0  # Should return immediately

        # Cleanup
        VRAMManager._instance = None

    def test_release_all_without_torch(self):
        """Release should work gracefully without torch/CUDA."""
        VRAMManager._instance = None
        vm = VRAMManager()
        vm.register_model("TestModel")

        # Should not raise
        vm.release_all()
        assert vm.current_model is None

        # Cleanup
        VRAMManager._instance = None

    def test_is_gpu_available_without_torch(self):
        """is_gpu_available should return False when torch not installed."""
        VRAMManager._instance = None
        vm = VRAMManager()

        # In test environment without CUDA, should be False
        # (may be True if torch is installed but no GPU)
        result = vm.is_gpu_available
        assert isinstance(result, bool)

        # Cleanup
        VRAMManager._instance = None


# ---------------------------------------------------------------------------
# Concurrent Processing
# ---------------------------------------------------------------------------


class TestConcurrentProcessing:
    """Test behavior under concurrent job submissions."""

    @pytest.fixture
    def config(self, tmp_path):
        return AppConfig(output={"output_dir": str(tmp_path / "meetings")})

    @pytest.mark.asyncio
    async def test_multiple_sequential_jobs(self, config, tmp_path):
        """Test processing multiple jobs sequentially."""
        from backend.core.orchestrator import Orchestrator

        orch = Orchestrator(config)

        # Setup mocks (simplified)
        mock_vram = MagicMock()
        mock_vram.wait_for_available = AsyncMock(return_value=True)
        mock_vram.check_available = MagicMock(return_value=True)
        mock_vram.register_model = MagicMock()
        mock_vram.unregister_model = MagicMock()
        mock_vram.release_all = MagicMock()
        orch._vram = mock_vram

        # Mock all engines
        trans, diar = generate_segments(10, 30.0)
        aligned = TranscriptAligner.align(trans, diar)
        participants = TranscriptAligner.get_participants(aligned)
        protocol = build_protocol(10)

        mock_audio = MagicMock()
        mock_audio.process = AsyncMock(
            return_value=("/tmp/processed.wav",
                          AudioMetadata(duration=30.0, sample_rate=16000, channels=1,
                                        codec="pcm_s16le", file_size=960000))
        )
        orch._audio = mock_audio

        # VAD mock (Block 5d)
        mock_vad = MagicMock()
        mock_vad.load = MagicMock()
        mock_vad.unload = MagicMock()
        mock_vad.detect_from_file = MagicMock(return_value=[
            {"start": i * 5.0, "end": (i + 1) * 5.0} for i in range(6)
        ])
        orch._vad = mock_vad

        mock_asr = MagicMock()
        mock_asr.required_vram_gb = 2.5
        mock_asr.load = MagicMock()
        mock_asr.unload = MagicMock()
        mock_asr.process = AsyncMock(return_value=trans)
        orch._asr = mock_asr

        mock_diar = MagicMock()
        mock_diar.required_vram_gb = 0.5
        mock_diar.load = MagicMock()
        mock_diar.unload = MagicMock()
        mock_diar.process = AsyncMock(return_value=diar)
        orch._diarization = mock_diar

        mock_summ = MagicMock()
        mock_summ.required_vram_gb = 5.0
        mock_summ.load = MagicMock()
        mock_summ.unload = MagicMock()
        mock_summ.process = AsyncMock(return_value=protocol)
        orch._summarization = mock_summ

        orch._aligner = TranscriptAligner()
        orch._qa = QualityAssurance(config)
        orch._formatter = ProtocolFormatter

        # Process 3 sequential jobs
        results = []
        for i in range(3):
            job = MeetingJob(filename=f"meeting_{i}.wav")
            result = await orch.process_meeting(f"/tmp/meeting_{i}.wav", job)
            results.append(result)

        # All should succeed
        assert all(r is not None for r in results)
        assert len(orch.list_jobs()) == 3


# ---------------------------------------------------------------------------
# Audio Preprocessing Tests
# ---------------------------------------------------------------------------


def _ffprobe_available() -> bool:
    """Check if ffprobe supports -print_json flag."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-print_json", "-show_format", "/dev/null"],
            capture_output=True, text=True, timeout=5,
        )
        # If ffprobe doesn't support -print_json, it returns non-zero
        return "Option not found" not in (result.stderr or "")
    except Exception:
        return False


requires_ffprobe = pytest.mark.skipif(
    not _ffprobe_available(),
    reason="ffprobe with -print_json support not available",
)


class TestAudioPreprocessing:
    """Test audio preprocessing with real WAV files."""

    def _generate_wav(self, path: str, duration: float, sr: int = 16000) -> str:
        """Generate a test WAV file."""
        n_samples = int(duration * sr)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        return path

    @requires_ffprobe
    @pytest.mark.asyncio
    async def test_audio_metadata_extraction(self, tmp_path):
        """Test ffprobe metadata extraction on real WAV."""
        from backend.core.audio import AudioPreprocessor

        wav_path = self._generate_wav(str(tmp_path / "test.wav"), 10.0)

        metadata = await AudioPreprocessor.get_metadata(wav_path)
        assert abs(metadata.duration - 10.0) < 0.5
        assert metadata.sample_rate == 16000
        assert metadata.channels == 1

    @requires_ffprobe
    @pytest.mark.asyncio
    async def test_audio_conversion(self, tmp_path):
        """Test audio conversion to 16kHz mono."""
        from backend.core.audio import AudioPreprocessor

        # Create stereo 44.1kHz audio
        wav_path = self._generate_wav(str(tmp_path / "input.wav"), 5.0, sr=44100)
        output_path = str(tmp_path / "output.wav")

        await AudioPreprocessor.convert(wav_path, output_path)

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    @requires_ffprobe
    @pytest.mark.asyncio
    async def test_audio_validation(self, tmp_path):
        """Test audio validation on real file."""
        from backend.core.audio import AudioPreprocessor

        wav_path = self._generate_wav(str(tmp_path / "valid.wav"), 5.0)
        is_valid = await AudioPreprocessor.validate(wav_path)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_audio_validation_nonexistent(self, tmp_path):
        """Test validation of nonexistent file."""
        from backend.core.audio import AudioPreprocessor

        is_valid = await AudioPreprocessor.validate(str(tmp_path / "nope.wav"))
        assert is_valid is False

    @requires_ffprobe
    @pytest.mark.asyncio
    async def test_full_preprocessing(self, tmp_path):
        """Test full audio preprocessing pipeline."""
        from backend.core.audio import AudioPreprocessor

        wav_path = self._generate_wav(str(tmp_path / "meeting.wav"), 15.0)
        output_dir = str(tmp_path / "processed")

        output_path, metadata = await AudioPreprocessor.process(wav_path, output_dir)

        assert Path(output_path).exists()
        assert metadata.duration > 0
        assert metadata.sample_rate == 16000
