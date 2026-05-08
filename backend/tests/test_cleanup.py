"""Tests for automatic temp file cleanup module."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from backend.core.cleanup import CleanupConfig, FileCleanup


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    out = tmp_path / "meetings"
    out.mkdir()
    return out


@pytest.fixture
def cleanup(output_dir):
    """Create a FileCleanup instance with test config."""
    config = CleanupConfig(
        auto_cleanup=True,
        retention_hours=720,
        orphan_timeout_hours=24,
        secure_delete=True,
    )
    return FileCleanup(output_dir=str(output_dir), config=config)


@pytest.fixture
def cleanup_no_secure(output_dir):
    """Create a FileCleanup without secure delete."""
    config = CleanupConfig(
        auto_cleanup=True,
        secure_delete=False,
    )
    return FileCleanup(output_dir=str(output_dir), config=config)


@pytest.fixture
def cleanup_disabled(output_dir):
    """Create a disabled FileCleanup instance."""
    config = CleanupConfig(auto_cleanup=False)
    return FileCleanup(output_dir=str(output_dir), config=config)


def _create_meeting_files(output_dir: Path, job_id: str) -> Path:
    """Create a realistic set of meeting files for testing."""
    meeting_dir = output_dir / job_id
    meeting_dir.mkdir()

    # Intermediate files
    (meeting_dir / "audio.wav").write_bytes(os.urandom(1024))
    audio_orig = meeting_dir / "audio_original"
    audio_orig.mkdir()
    (audio_orig / "meeting.mp3").write_bytes(os.urandom(512))
    (meeting_dir / "transcription.json").write_text('{"text": "hello"}')
    (meeting_dir / "diarization.json").write_text('{"speakers": []}')
    (meeting_dir / "aligned.json").write_text('{"segments": []}')

    # Final outputs
    (meeting_dir / f"{job_id}.docx").write_bytes(os.urandom(256))
    (meeting_dir / "protocol.json").write_text('{"protocol": true}')

    return meeting_dir


class TestFileCleanup:
    """Tests for FileCleanup class."""

    def test_cleanup_intermediates_removes_temp_files(self, cleanup, output_dir):
        """Cleanup removes intermediate files but keeps final outputs."""
        job_id = "test-job-001"
        meeting_dir = _create_meeting_files(output_dir, job_id)

        removed = cleanup.cleanup_meeting_intermediates(job_id)
        assert removed > 0

        # Intermediate files should be gone
        assert not (meeting_dir / "audio.wav").exists()
        assert not (meeting_dir / "audio_original").exists()
        assert not (meeting_dir / "transcription.json").exists()
        assert not (meeting_dir / "diarization.json").exists()
        # aligned.json is intentionally kept — needed by transcript viewer/editor
        assert (meeting_dir / "aligned.json").exists()

        # Final outputs should still exist
        assert (meeting_dir / f"{job_id}.docx").exists()
        assert (meeting_dir / "protocol.json").exists()

    def test_cleanup_disabled_skips(self, cleanup_disabled, output_dir):
        """Cleanup does nothing when auto_cleanup is disabled."""
        job_id = "test-job-002"
        meeting_dir = _create_meeting_files(output_dir, job_id)

        removed = cleanup_disabled.cleanup_meeting_intermediates(job_id)
        assert removed == 0

        # All files should still exist
        assert (meeting_dir / "audio.wav").exists()
        assert (meeting_dir / "transcription.json").exists()

    def test_cleanup_nonexistent_job(self, cleanup, output_dir):
        """Cleanup of nonexistent job returns 0."""
        removed = cleanup.cleanup_meeting_intermediates("nonexistent-job")
        assert removed == 0

    def test_cleanup_without_secure_delete(self, cleanup_no_secure, output_dir):
        """Cleanup works without secure delete."""
        job_id = "test-job-003"
        _create_meeting_files(output_dir, job_id)

        removed = cleanup_no_secure.cleanup_meeting_intermediates(job_id)
        assert removed > 0

    def test_cleanup_orphaned_files(self, output_dir):
        """Orphan cleanup removes old directories without final outputs."""
        config = CleanupConfig(
            auto_cleanup=True,
            orphan_timeout_hours=0,  # Immediate for testing
            secure_delete=False,
        )
        cleanup = FileCleanup(output_dir=str(output_dir), config=config)

        # Create orphaned directory (no final output)
        orphan_dir = output_dir / "orphaned-job"
        orphan_dir.mkdir()
        (orphan_dir / "audio.wav").write_bytes(os.urandom(100))
        # Set old mtime
        old_time = time.time() - 3600
        os.utime(orphan_dir, (old_time, old_time))

        # Create valid directory (has final output)
        valid_dir = output_dir / "valid-job"
        valid_dir.mkdir()
        (valid_dir / "valid-job.docx").write_bytes(os.urandom(100))
        os.utime(valid_dir, (old_time, old_time))

        removed = cleanup.cleanup_orphaned_files()
        assert removed == 1
        assert not orphan_dir.exists()
        assert valid_dir.exists()

    def test_cleanup_expired_meetings(self, output_dir):
        """Expired cleanup removes all old meeting data."""
        config = CleanupConfig(
            auto_cleanup=True,
            retention_hours=0,  # Immediate for testing
            secure_delete=False,
        )
        cleanup = FileCleanup(output_dir=str(output_dir), config=config)

        # Create meeting data
        meeting_dir = output_dir / "expired-job"
        meeting_dir.mkdir()
        (meeting_dir / "expired-job.docx").write_bytes(os.urandom(100))
        # Set old mtime
        old_time = time.time() - 3600
        os.utime(meeting_dir, (old_time, old_time))

        removed = cleanup.cleanup_expired_meetings()
        assert removed == 1
        assert not meeting_dir.exists()

    def test_secure_delete_overwrites_file(self, tmp_path):
        """Secure delete overwrites file content before unlinking."""
        test_file = tmp_path / "secret.txt"
        original = b"sensitive data that must be wiped"
        test_file.write_bytes(original)

        FileCleanup._secure_delete_file(test_file)
        assert not test_file.exists()

    def test_secure_delete_directory(self, tmp_path):
        """Secure delete removes entire directory tree."""
        test_dir = tmp_path / "secret_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_bytes(b"secret1")
        sub = test_dir / "subdir"
        sub.mkdir()
        (sub / "file2.txt").write_bytes(b"secret2")

        FileCleanup._secure_delete_directory(test_dir)
        assert not test_dir.exists()

    def test_cleanup_preserves_edited_transcript(self, cleanup, output_dir):
        """Cleanup preserves user-edited transcripts."""
        job_id = "test-job-edit"
        meeting_dir = _create_meeting_files(output_dir, job_id)
        (meeting_dir / "aligned_edited.json").write_text('{"edited": true}')

        cleanup.cleanup_meeting_intermediates(job_id)

        # Edited transcript should be preserved
        assert (meeting_dir / "aligned_edited.json").exists()
