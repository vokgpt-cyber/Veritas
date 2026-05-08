"""Audit logging for EPAM VERITAS.

Records security-relevant events (who processed what, when) to a
structured JSON log file. Each line is a self-contained JSON object
for easy parsing by SIEM tools.

Audit events include:
- Authentication (login success/failure, token refresh)
- Meeting operations (upload, process, download, delete)
- System operations (GPU release, config change)
- Security events (invalid tokens, unauthorized access attempts)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Enumeration of auditable actions."""

    # Authentication
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    TOKEN_VERIFY = "auth.token.verify"
    TOKEN_EXPIRED = "auth.token.expired"

    # Meeting operations
    MEETING_UPLOAD = "meeting.upload"
    MEETING_PROCESS_START = "meeting.process.start"
    MEETING_PROCESS_COMPLETE = "meeting.process.complete"
    MEETING_PROCESS_FAIL = "meeting.process.fail"
    MEETING_DOWNLOAD = "meeting.download"
    MEETING_DELETE = "meeting.delete"
    MEETING_TRANSCRIPT_VIEW = "meeting.transcript.view"
    MEETING_TRANSCRIPT_EDIT = "meeting.transcript.edit"
    MEETING_SPEAKER_REVIEW_CONFIRM = "meeting.speaker_review.confirm"
    MEETING_SPEAKER_REVIEW_INVALIDATE = "meeting.speaker_review.invalidate"

    # User/session operations
    USER_CREATE = "user.create"
    USER_UPDATE = "user.update"
    USER_DISABLE = "user.disable"
    USER_DELETE = "user.delete"
    USER_PASSWORD_RESET = "user.password_reset"
    USER_PASSWORD_CHANGE = "user.password_change"
    SESSION_LOGOUT = "auth.session.logout"

    # System operations
    SYSTEM_GPU_RELEASE = "system.gpu.release"
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_CONFIG_CHANGE = "system.config.change"

    # Security events
    SECURITY_UNAUTHORIZED = "security.unauthorized"
    SECURITY_FORBIDDEN = "security.forbidden"
    SECURITY_INVALID_TOKEN = "security.invalid_token"


class AuditEvent(BaseModel):
    """Structured audit event."""

    timestamp: str
    action: str
    username: Optional[str] = None
    ip_address: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None


class AuditConfig(BaseSettings):
    """Audit logging configuration."""

    enabled: bool = True
    log_dir: str = "data/audit"
    max_file_size_mb: int = 100
    max_files: int = 10

    model_config = SettingsConfigDict(env_prefix="EPAM_AUDIT_")


class AuditLogger:
    """Thread-safe audit logger writing structured JSON events.

    Writes one JSON object per line to rotating audit log files.
    Rotation occurs when the file exceeds max_file_size_mb.

    Attributes:
        _config: Audit configuration.
        _log_dir: Directory for audit log files.
        _lock: Thread lock for safe concurrent writes.
        _current_file: Currently active log file path.
    """

    def __init__(self, config: Optional[AuditConfig] = None) -> None:
        """Initialize audit logger.

        Args:
            config: Audit configuration. If None, loads from env vars.
        """
        if config is None:
            config = AuditConfig()

        self._config = config
        self._enabled = config.enabled
        # Per-user data isolation (Sprint 2026-04-30, task #34): if the
        # configured log_dir is relative, resolve it under the active
        # user's data root so different Windows accounts don't share an
        # audit log. Absolute paths pass through unchanged.
        try:
            from backend.app.paths import resolve_user_path
            self._log_dir = resolve_user_path(config.log_dir)
        except ImportError:
            # Allow audit to work even if paths.py is unavailable (e.g.
            # standalone test that imports audit before app package).
            self._log_dir = Path(config.log_dir)
        self._lock = threading.Lock()
        self._current_file: Optional[Path] = None
        self._current_size: int = 0

        if self._enabled:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._current_file = self._get_current_log_file()
            if self._current_file.exists():
                self._current_size = self._current_file.stat().st_size
            logger.info(f"Audit logging enabled: {self._log_dir}")
        else:
            logger.info("Audit logging disabled")

    def _get_current_log_file(self) -> Path:
        """Get the current audit log file path.

        Returns:
            Path to the current audit log file, named by date.
        """
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return self._log_dir / f"audit_{date_str}.jsonl"

    def _rotate_if_needed(self) -> None:
        """Rotate log file if it exceeds the size limit."""
        max_bytes = self._config.max_file_size_mb * 1024 * 1024
        if self._current_size >= max_bytes:
            # Rotate: rename current file with timestamp suffix
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            rotated = self._current_file.with_suffix(f".{timestamp}.jsonl")
            try:
                self._current_file.rename(rotated)
            except OSError as e:
                logger.warning(f"Failed to rotate audit log: {e}")

            self._current_file = self._get_current_log_file()
            self._current_size = 0

            # Cleanup old files beyond max_files
            self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        """Remove oldest audit log files beyond max_files limit."""
        try:
            log_files = sorted(
                self._log_dir.glob("audit_*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_file in log_files[self._config.max_files :]:
                old_file.unlink()
                logger.debug(f"Removed old audit log: {old_file}")
        except OSError as e:
            logger.warning(f"Failed to cleanup old audit logs: {e}")

    def log(
        self,
        action: AuditAction,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record an audit event.

        Args:
            action: The action being audited.
            username: User who performed the action.
            ip_address: Client IP address.
            resource_type: Type of resource affected (meeting, system, etc).
            resource_id: Identifier of the affected resource.
            details: Additional context (file sizes, formats, etc).
            success: Whether the action succeeded.
            error: Error message if the action failed.
        """
        if not self._enabled:
            return

        event = AuditEvent(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            action=action.value,
            username=username,
            ip_address=ip_address,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            success=success,
            error=error,
        )

        event_json = event.model_dump_json(exclude_none=True)

        with self._lock:
            try:
                # Check if we need a new date-based file
                expected_file = self._get_current_log_file()
                if expected_file != self._current_file:
                    self._current_file = expected_file
                    self._current_size = 0
                    if self._current_file.exists():
                        self._current_size = self._current_file.stat().st_size

                self._rotate_if_needed()

                with open(self._current_file, "a", encoding="utf-8") as f:
                    f.write(event_json + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                self._current_size += len(event_json) + 1

            except OSError as e:
                logger.error(f"Failed to write audit event: {e}")

        # Also log to standard logger at INFO level for operational visibility
        log_msg = (
            f"AUDIT: {action.value} user={username or 'anonymous'} "
            f"resource={resource_type}:{resource_id or 'N/A'} "
            f"success={success}"
        )
        if error:
            log_msg += f" error={error}"

        logger.info(log_msg)


# Global audit logger instance (initialized on first import)
_audit_logger: Optional[AuditLogger] = None
_audit_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Get or create the global audit logger instance.

    Returns:
        The singleton AuditLogger instance.
    """
    global _audit_logger
    if _audit_logger is None:
        with _audit_lock:
            if _audit_logger is None:
                _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(
    action: AuditAction,
    username: Optional[str] = None,
    ip_address: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    """Convenience function to log an audit event.

    Args:
        action: The action being audited.
        username: User who performed the action.
        ip_address: Client IP address.
        resource_type: Type of resource affected.
        resource_id: Identifier of the affected resource.
        details: Additional context.
        success: Whether the action succeeded.
        error: Error message if failed.
    """
    get_audit_logger().log(
        action=action,
        username=username,
        ip_address=ip_address,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        success=success,
        error=error,
    )
