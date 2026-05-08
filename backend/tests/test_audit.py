"""Tests for audit logging module."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.core.audit import (
    AuditAction,
    AuditConfig,
    AuditEvent,
    AuditLogger,
    audit_log,
)


@pytest.fixture
def audit_dir(tmp_path):
    """Create a temporary audit log directory."""
    audit_path = tmp_path / "audit"
    audit_path.mkdir()
    return audit_path


@pytest.fixture
def audit_logger(audit_dir):
    """Create an audit logger with temp directory."""
    config = AuditConfig(enabled=True, log_dir=str(audit_dir))
    return AuditLogger(config)


@pytest.fixture
def disabled_logger(audit_dir):
    """Create a disabled audit logger."""
    config = AuditConfig(enabled=False, log_dir=str(audit_dir))
    return AuditLogger(config)


class TestAuditEvent:
    """Tests for AuditEvent model."""

    def test_event_serialization(self):
        """AuditEvent serializes to JSON correctly."""
        event = AuditEvent(
            timestamp="2026-04-06T10:00:00+00:00",
            action="auth.login.success",
            username="admin",
            ip_address="192.168.1.1",
            success=True,
        )
        data = event.model_dump()
        assert data["action"] == "auth.login.success"
        assert data["username"] == "admin"
        assert data["success"] is True

    def test_event_excludes_none(self):
        """AuditEvent JSON excludes None fields."""
        event = AuditEvent(
            timestamp="2026-04-06T10:00:00+00:00",
            action="auth.login.success",
        )
        json_str = event.model_dump_json(exclude_none=True)
        data = json.loads(json_str)
        assert "username" not in data
        assert "ip_address" not in data


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_log_creates_file(self, audit_logger, audit_dir):
        """Logging an event creates a JSONL audit file."""
        audit_logger.log(
            action=AuditAction.LOGIN_SUCCESS,
            username="admin",
            ip_address="127.0.0.1",
        )

        log_files = list(audit_dir.glob("audit_*.jsonl"))
        assert len(log_files) == 1
        assert log_files[0].stat().st_size > 0

    def test_log_writes_valid_json(self, audit_logger, audit_dir):
        """Each line in audit log is valid JSON."""
        audit_logger.log(
            action=AuditAction.LOGIN_SUCCESS,
            username="admin",
        )
        audit_logger.log(
            action=AuditAction.MEETING_UPLOAD,
            username="admin",
            resource_type="meeting",
            resource_id="job-123",
        )

        log_files = list(audit_dir.glob("audit_*.jsonl"))
        assert len(log_files) == 1

        lines = log_files[0].read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            data = json.loads(line)
            assert "timestamp" in data
            assert "action" in data

    def test_log_records_all_fields(self, audit_logger, audit_dir):
        """Audit log records all provided fields."""
        audit_logger.log(
            action=AuditAction.MEETING_UPLOAD,
            username="admin",
            ip_address="10.0.0.5",
            resource_type="meeting",
            resource_id="abc-123",
            details={"filename": "meeting.wav", "size_bytes": 1048576},
            success=True,
        )

        log_files = list(audit_dir.glob("audit_*.jsonl"))
        data = json.loads(log_files[0].read_text().strip())
        assert data["action"] == "auth.login.success" or data["action"] == "meeting.upload"
        assert data["username"] == "admin"
        assert data["ip_address"] == "10.0.0.5"
        assert data["resource_type"] == "meeting"
        assert data["resource_id"] == "abc-123"
        assert data["details"]["filename"] == "meeting.wav"
        assert data["success"] is True

    def test_log_failure_event(self, audit_logger, audit_dir):
        """Audit log records failure events with error messages."""
        audit_logger.log(
            action=AuditAction.LOGIN_FAILURE,
            username="attacker",
            ip_address="10.0.0.99",
            success=False,
            error="Invalid credentials",
        )

        log_files = list(audit_dir.glob("audit_*.jsonl"))
        data = json.loads(log_files[0].read_text().strip())
        assert data["success"] is False
        assert data["error"] == "Invalid credentials"

    def test_disabled_logger_writes_nothing(self, disabled_logger, audit_dir):
        """Disabled logger does not create any log files."""
        disabled_logger.log(
            action=AuditAction.LOGIN_SUCCESS,
            username="admin",
        )

        log_files = list(audit_dir.glob("audit_*.jsonl"))
        assert len(log_files) == 0

    def test_multiple_events_append(self, audit_logger, audit_dir):
        """Multiple events append to the same log file."""
        for i in range(5):
            audit_logger.log(
                action=AuditAction.MEETING_UPLOAD,
                username="admin",
                resource_id=f"job-{i}",
            )

        log_files = list(audit_dir.glob("audit_*.jsonl"))
        assert len(log_files) == 1
        lines = log_files[0].read_text().strip().split("\n")
        assert len(lines) == 5

    def test_log_rotation(self, audit_dir):
        """Log rotates when file exceeds max size."""
        # Set tiny max size to trigger rotation quickly
        config = AuditConfig(
            enabled=True,
            log_dir=str(audit_dir),
            max_file_size_mb=0,  # 0 MB = rotate immediately
        )
        logger = AuditLogger(config)

        logger.log(action=AuditAction.LOGIN_SUCCESS, username="admin")
        logger.log(action=AuditAction.MEETING_UPLOAD, username="admin")

        log_files = list(audit_dir.glob("audit_*"))
        assert len(log_files) >= 2  # At least one rotated + current

    def test_all_actions_are_valid(self):
        """All AuditAction enum values are valid strings."""
        for action in AuditAction:
            assert isinstance(action.value, str)
            assert "." in action.value  # All actions use dot notation


class TestAuditLogConvenience:
    """Tests for the audit_log convenience function."""

    def test_audit_log_function_works(self, monkeypatch, audit_dir):
        """The audit_log convenience function creates events."""
        import backend.core.audit as audit_module

        config = AuditConfig(enabled=True, log_dir=str(audit_dir))
        test_logger = AuditLogger(config)

        monkeypatch.setattr(audit_module, "_audit_logger", test_logger)

        audit_log(
            action=AuditAction.SYSTEM_STARTUP,
            details={"version": "2.0.0"},
        )

        log_files = list(audit_dir.glob("audit_*.jsonl"))
        assert len(log_files) == 1
