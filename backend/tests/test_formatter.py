"""Tests for protocol formatter."""
import json
from pathlib import Path

import pytest

from backend.app.models import (
    AlignedSegment,
    Decision,
    MeetingProtocol,
    Participant,
    TaskItem,
    TopicItem,
)
from backend.core.formatter import ProtocolFormatter


@pytest.fixture
def sample_protocol():
    """Create a sample meeting protocol for testing."""
    return MeetingProtocol(
        topic="Quarterly Budget Review",
        participants=[
            Participant(
                speaker_id="s1",
                speaker_name="Alice",
                speaking_time=180.0,
                speaking_share=60.0,
            ),
            Participant(
                speaker_id="s2",
                speaker_name="Bob",
                speaking_time=120.0,
                speaking_share=40.0,
            ),
        ],
        summary="The team reviewed Q1 budget and approved allocations for Q2.",
        key_topics=[
            TopicItem(title="Q1 Results", content="Revenue exceeded targets by 12%", speakers=["Alice"]),
            TopicItem(title="Q2 Planning", content="New budget allocation proposed", speakers=["Bob"]),
        ],
        decisions=[
            Decision(text="Approved Q2 budget of $2M", responsible="Alice"),
            Decision(text="Deferred marketing spend review to next quarter"),
        ],
        tasks=[
            TaskItem(text="Prepare Q2 forecast document", assignee="Bob", deadline="2026-04-15"),
            TaskItem(text="Schedule follow-up meeting", assignee="Alice"),
        ],
        open_questions=["How to handle budget overflow?", "Team expansion timeline?"],
        transcript=[
            AlignedSegment(
                start=0.0,
                end=10.0,
                text="Welcome to the quarterly budget review.",
                speaker_id="s1",
                speaker_name="Alice",
            ),
            AlignedSegment(
                start=10.0,
                end=25.0,
                text="Let me present the Q1 results.",
                speaker_id="s2",
                speaker_name="Bob",
            ),
        ],
    )


class TestProtocolFormatterDOCX:
    """DOCX output tests."""

    def test_generate_docx(self, sample_protocol, tmp_path):
        """Test DOCX file generation."""
        output_path = str(tmp_path / "protocol.docx")
        ProtocolFormatter.format_docx(sample_protocol, output_path)

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_docx_is_valid(self, sample_protocol, tmp_path):
        """Generated DOCX should be a valid Office Open XML file."""
        from docx import Document

        output_path = str(tmp_path / "protocol.docx")
        ProtocolFormatter.format_docx(sample_protocol, output_path)

        doc = Document(output_path)
        assert len(doc.paragraphs) > 0

    def test_docx_empty_protocol(self, tmp_path):
        """Should handle empty protocol gracefully."""
        output_path = str(tmp_path / "empty.docx")
        empty = MeetingProtocol()
        ProtocolFormatter.format_docx(empty, output_path)
        assert Path(output_path).exists()


class TestProtocolFormatterJSON:
    """JSON output tests."""

    def test_generate_json(self, sample_protocol, tmp_path):
        """Test JSON file generation."""
        output_path = str(tmp_path / "protocol.json")
        ProtocolFormatter.format_json(sample_protocol, output_path)

        assert Path(output_path).exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["topic"] == "Quarterly Budget Review"
        assert len(data["participants"]) == 2
        assert len(data["decisions"]) == 2

    def test_json_roundtrip(self, sample_protocol, tmp_path):
        """Protocol serialized to JSON should be deserializable."""
        output_path = str(tmp_path / "protocol.json")
        ProtocolFormatter.format_json(sample_protocol, output_path)

        with open(output_path) as f:
            data = json.load(f)

        restored = MeetingProtocol(**data)
        assert restored.topic == sample_protocol.topic
        assert len(restored.participants) == len(sample_protocol.participants)
