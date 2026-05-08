"""Automatic temporary file cleanup after processing.

Manages secure deletion of intermediate files (preprocessed audio, raw
transcription JSON, diarization data) once a meeting is fully processed
or has permanently failed. Keeps only final outputs (DOCX, JSON protocol).

Also provides scheduled cleanup for orphaned temp files older than a
configurable retention period.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class CleanupConfig(BaseSettings):
    """Cleanup configuration."""

    # Whether to auto-cleanup after processing completes
    auto_cleanup: bool = True
    # Hours to retain completed meeting data before full cleanup
    retention_hours: int = 720  # 30 days
    # Hours before orphaned temp files are cleaned up
    orphan_timeout_hours: int = 24
    # Whether to use secure delete (overwrite before unlink)
    secure_delete: bool = True

    model_config = SettingsConfigDict(env_prefix="EPAM_CLEANUP_")


# File patterns that are safe to remove after processing
INTERMEDIATE_PATTERNS = [
    "audio.wav",  # Preprocessed audio
    "audio_original",  # Original uploaded audio directory
    "transcription.json",  # Raw ASR output
    "diarization.json",  # Raw diarization output
    # NOTE: aligned.json is kept — needed by transcript viewer/editor endpoint
]

# File patterns to always keep
KEEP_PATTERNS = [
    "*.docx",  # Final protocol
    "protocol.json",  # Final protocol JSON
    "aligned_edited.json",  # User-edited transcript
]


class FileCleanup:
    """Manages secure cleanup of temporary and intermediate files.

    After a meeting is fully processed, removes intermediate pipeline
    outputs (raw audio, transcription segments, diarization data) while
    preserving final deliverables (DOCX protocol, JSON protocol).

    Attributes:
        _config: Cleanup configuration.
        _output_dir: Base output directory for meeting data.
    """

    def __init__(
        self,
        output_dir: str = "data/meetings",
        config: Optional[CleanupConfig] = None,
    ) -> None:
        """Initialize file cleanup manager.

        Args:
            output_dir: Base directory containing meeting subdirectories.
            config: Cleanup configuration. If None, loads from env vars.
        """
        if config is None:
            config = CleanupConfig()

        self._config = config
        self._output_dir = Path(output_dir)
        logger.info(
            f"File cleanup initialized: auto={config.auto_cleanup}, "
            f"retention={config.retention_hours}h, "
            f"secure_delete={config.secure_delete}"
        )

    def cleanup_meeting_intermediates(self, job_id: str) -> int:
        """Remove intermediate files for a completed meeting.

        Keeps only final outputs (DOCX, protocol JSON, edited transcript).

        Args:
            job_id: Meeting job identifier.

        Returns:
            Number of files/directories removed.
        """
        if not self._config.auto_cleanup:
            logger.debug(f"Auto-cleanup disabled, skipping for {job_id}")
            return 0

        meeting_dir = self._output_dir / job_id
        if not meeting_dir.exists():
            return 0

        removed_count = 0

        for pattern in INTERMEDIATE_PATTERNS:
            target = meeting_dir / pattern
            if target.exists():
                try:
                    if target.is_dir():
                        if self._config.secure_delete:
                            self._secure_delete_directory(target)
                        else:
                            shutil.rmtree(target)
                    else:
                        if self._config.secure_delete:
                            self._secure_delete_file(target)
                        else:
                            target.unlink()
                    removed_count += 1
                    logger.debug(f"Cleaned up: {target}")
                except OSError as e:
                    logger.warning(f"Failed to cleanup {target}: {e}")

        if removed_count > 0:
            logger.info(
                f"Cleaned up {removed_count} intermediate files for meeting {job_id}"
            )

        return removed_count

    def cleanup_orphaned_files(self) -> int:
        """Remove orphaned temporary files older than the timeout.

        Scans all meeting directories for files that appear to be from
        incomplete or abandoned processing jobs.

        Returns:
            Total number of files/directories removed.
        """
        if not self._output_dir.exists():
            return 0

        cutoff_time = time.time() - (self._config.orphan_timeout_hours * 3600)
        removed_count = 0

        for meeting_dir in self._output_dir.iterdir():
            if not meeting_dir.is_dir():
                continue

            # Check if directory has any recent modifications
            try:
                dir_mtime = meeting_dir.stat().st_mtime
            except OSError:
                continue

            if dir_mtime > cutoff_time:
                continue  # Directory was recently modified, skip

            # Check if there are any final outputs
            has_final_output = any(
                meeting_dir.glob("*.docx")
            ) or (meeting_dir / "protocol.json").exists()

            if not has_final_output:
                # No final output and old enough: this is orphaned
                try:
                    if self._config.secure_delete:
                        self._secure_delete_directory(meeting_dir)
                    else:
                        shutil.rmtree(meeting_dir)
                    removed_count += 1
                    logger.info(f"Removed orphaned meeting directory: {meeting_dir}")
                except OSError as e:
                    logger.warning(f"Failed to remove orphaned dir {meeting_dir}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} orphaned meeting directories")

        return removed_count

    def cleanup_expired_meetings(self) -> int:
        """Remove meeting data older than the retention period.

        Removes all files (including final outputs) for meetings that
        have exceeded the configured retention period.

        Returns:
            Number of meeting directories removed.
        """
        if not self._output_dir.exists():
            return 0

        cutoff_time = time.time() - (self._config.retention_hours * 3600)
        removed_count = 0

        for meeting_dir in self._output_dir.iterdir():
            if not meeting_dir.is_dir():
                continue

            try:
                dir_mtime = meeting_dir.stat().st_mtime
            except OSError:
                continue

            if dir_mtime < cutoff_time:
                try:
                    if self._config.secure_delete:
                        self._secure_delete_directory(meeting_dir)
                    else:
                        shutil.rmtree(meeting_dir)
                    removed_count += 1
                    logger.info(f"Removed expired meeting: {meeting_dir.name}")
                except OSError as e:
                    logger.warning(f"Failed to remove expired dir {meeting_dir}: {e}")

        if removed_count > 0:
            logger.info(f"Removed {removed_count} expired meeting directories")

        return removed_count

    @staticmethod
    def _secure_delete_file(file_path: Path) -> None:
        """Securely delete a file by overwriting with random data.

        Args:
            file_path: Path to the file to delete.
        """
        try:
            file_size = file_path.stat().st_size
            if file_size > 0:
                with open(file_path, "wb") as f:
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            file_path.unlink()
        except OSError as e:
            logger.warning(f"Secure delete failed for {file_path}: {e}")
            try:
                file_path.unlink()
            except OSError:
                pass

    @classmethod
    def _secure_delete_directory(cls, dir_path: Path) -> None:
        """Securely delete all files in a directory, then remove it.

        Args:
            dir_path: Path to the directory to delete.
        """
        try:
            for root, dirs, files in os.walk(dir_path, topdown=False):
                for name in files:
                    cls._secure_delete_file(Path(root) / name)
                for name in dirs:
                    try:
                        (Path(root) / name).rmdir()
                    except OSError:
                        pass
            dir_path.rmdir()
        except OSError as e:
            # Fall back to shutil
            logger.warning(f"Secure directory delete failed, using shutil: {e}")
            shutil.rmtree(dir_path, ignore_errors=True)
