"""Meeting protocol formatter with EPAM branding support.

Dispatches by payload type: MeetingProtocol (legacy 7-section) keeps its
existing renderer, and new Block-7 meeting types (court hearing,
administrative, client meeting, interview) get type-specific DOCX
templates that match the structure the LLM produces for each.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from backend.app.models import MeetingProtocol
from backend.engine.protocols.departments import (
    DEPARTMENT_BY_SLUG,
    DEPARTMENT_SLUGS_ORDERED,
)
from backend.engine.protocols.schemas import (
    AdministrativeProtocol,
    ClientMeetingProtocol,
    CourtHearingProtocol,
    DepartmentBlock,
    DepartmentItem,
    InterviewProtocol,
    ProtocolResult,
)

logger = logging.getLogger(__name__)


class ProtocolFormatter:
    """Generate branded meeting protocol documents."""

    # EPAM branding colors
    EPAM_RED = RGBColor(0xB2, 0x00, 0x1F)
    EPAM_DARK_GRAY = RGBColor(0x33, 0x33, 0x33)

    # Font settings
    HEADING_FONT = "Georgia"
    BODY_FONT = "Arial Narrow"

    @classmethod
    def _detect_protocol_language(cls, protocol: MeetingProtocol) -> str:
        """Detect language from protocol content (Cyrillic vs Latin)."""
        text = (protocol.summary or "") + (protocol.topic or "")
        cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
        latin = sum(1 for c in text if "a" <= c.lower() <= "z")
        return "ru" if cyrillic >= latin else "en"

    @classmethod
    def format_docx(
        cls,
        protocol: Union[MeetingProtocol, ProtocolResult, Any],
        output_path: str,
        language: str = "",
    ) -> str:
        """Generate DOCX, dispatching on payload type.

        Accepts either a MeetingProtocol (legacy 7-section), a
        ProtocolResult wrapping a type-specific payload, or any of the
        type-specific payloads directly. Each type has its own template
        designed for its structure.

        Args:
            protocol: MeetingProtocol, ProtocolResult, or type-specific
                payload (CourtHearingProtocol, AdministrativeProtocol,
                ClientMeetingProtocol, InterviewProtocol).
            output_path: Output file path. For InterviewProtocol this
                path is used for the internal report; the external
                report is saved alongside with "_external" suffix.
            language: Ignored — always uses Russian labels.

        Returns:
            Output path of the primary DOCX. For interviews this is the
            internal report path.
        """
        # Unwrap ProtocolResult if present.
        payload = protocol.payload if isinstance(protocol, ProtocolResult) else protocol

        # Dispatch by payload type.
        if isinstance(payload, CourtHearingProtocol):
            return cls._format_court_hearing_docx(payload, output_path)
        if isinstance(payload, AdministrativeProtocol):
            return cls._format_administrative_docx(payload, output_path)
        if isinstance(payload, ClientMeetingProtocol):
            return cls._format_client_meeting_docx(payload, output_path)
        if isinstance(payload, InterviewProtocol):
            return cls._format_interview_docx(payload, output_path)
        if isinstance(payload, MeetingProtocol):
            return cls._format_generic_docx(payload, output_path)

        raise TypeError(
            f"Unsupported payload type for DOCX formatting: {type(payload).__name__}"
        )

    @classmethod
    def _format_generic_docx(
        cls, protocol: MeetingProtocol, output_path: str
    ) -> str:
        """Legacy 7-section Russian protocol (pre-Block-7 format)."""
        logger.info(f"Generating DOCX (generic): {output_path}")

        labels = cls._get_labels("ru")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = Document()

        # Set margins
        for section in doc.sections:
            section.top_margin = Inches(0.75)
            section.bottom_margin = Inches(0.75)
            section.left_margin = Inches(0.75)
            section.right_margin = Inches(0.75)

        # Title page
        cls._add_title_page(doc, protocol, labels)

        # All 7 protocol sections (always rendered, empty shows placeholder)
        cls._add_participants_section(doc, protocol, labels)
        cls._add_summary_section(doc, protocol, labels)
        cls._add_key_topics_section(doc, protocol, labels)
        cls._add_decisions_section(doc, protocol, labels)
        cls._add_tasks_section(doc, protocol, labels)
        cls._add_open_questions_section(doc, protocol, labels)

        # Transcript appendix
        cls._add_transcript_section(doc, protocol, labels)

        doc.save(str(output_path))
        logger.info(f"DOCX saved: {output_path}")

        return str(output_path)

    @staticmethod
    def _get_labels(language: str) -> dict[str, str]:
        """Return section header labels for the given language."""
        if language == "ru":
            return {
                "protocol": "ПРОТОКОЛ СОВЕЩАНИЯ",
                "date": "Дата",
                "participants": "УЧАСТНИКИ",
                "name_id": "ФИО / ID",
                "speaking_time": "Время речи",
                "speaking_share": "Доля",
                "summary": "КРАТКОЕ СОДЕРЖАНИЕ",
                "key_topics": "ОСНОВНЫЕ ТЕМЫ",
                "decisions": "ПРИНЯТЫЕ РЕШЕНИЯ",
                "tasks": "ЗАДАЧИ И ПОРУЧЕНИЯ",
                "task": "Задача",
                "assignee": "Исполнитель",
                "deadline": "Срок",
                "open_questions": "ОТКРЫТЫЕ ВОПРОСЫ",
                "transcript": "СТЕНОГРАММА",
            }
        return {
            "protocol": "MEETING PROTOCOL",
            "date": "Date",
            "participants": "PARTICIPANTS",
            "name_id": "Name/ID",
            "speaking_time": "Speaking Time",
            "speaking_share": "Speaking Share",
            "summary": "SUMMARY",
            "key_topics": "KEY TOPICS",
            "decisions": "DECISIONS",
            "tasks": "ACTION ITEMS",
            "task": "Task",
            "assignee": "Assignee",
            "deadline": "Deadline",
            "open_questions": "OPEN QUESTIONS",
            "transcript": "TRANSCRIPT",
        }

    @classmethod
    def format_json(
        cls,
        protocol: Union[MeetingProtocol, ProtocolResult, Any],
        output_path: str,
    ) -> str:
        """Export protocol as JSON (wraps the payload with a type tag).

        Works for any pydantic payload — MeetingProtocol or any of the
        Block-7 type-specific schemas. The output includes a
        ``meeting_type`` discriminator so downstream consumers can parse
        the payload without inspection.

        Args:
            protocol: MeetingProtocol, ProtocolResult, or type-specific
                payload.
            output_path: Output file path.

        Returns:
            Output path.
        """
        logger.info(f"Generating JSON: {output_path}")

        if isinstance(protocol, ProtocolResult):
            meeting_type = protocol.meeting_type
            payload = protocol.payload
        else:
            payload = protocol
            meeting_type = cls._infer_meeting_type(payload)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(payload, MeetingProtocol):
            # Keep the legacy generic JSON shape flat for backwards
            # compatibility with older tests/tools that read `topic`,
            # `participants`, etc. at the top level.
            data = payload.model_dump(mode="json")
            data.setdefault("meeting_type", meeting_type)
        else:
            data = {
                "meeting_type": meeting_type,
                "payload": payload.model_dump(mode="json"),
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON saved: {output_path}")
        return str(output_path)

    @staticmethod
    def _infer_meeting_type(payload: Any) -> str:
        """Map a payload class to its meeting_type discriminator."""
        cls_map = {
            "CourtHearingProtocol": "court_hearing",
            "AdministrativeProtocol": "administrative",
            "ClientMeetingProtocol": "client_meeting",
            "InterviewProtocol": "interview",
            "MeetingProtocol": "generic",
        }
        return cls_map.get(payload.__class__.__name__, "generic")

    @classmethod
    def format_txt(cls, protocol: MeetingProtocol, output_path: str) -> str:
        """
        Export protocol as plain text.

        Args:
            protocol: MeetingProtocol object
            output_path: Output file path

        Returns:
            Output path
        """
        logger.info(f"Generating TXT: {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("MEETING PROTOCOL")
        lines.append("=" * 80)
        lines.append("")

        # Meeting info
        if protocol.meeting_date:
            lines.append(f"Date: {protocol.meeting_date.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

        if protocol.topic:
            lines.append(f"Topic: {protocol.topic}")

        lines.append("")

        # Participants
        if protocol.participants:
            lines.append("PARTICIPANTS:")
            for p in protocol.participants:
                name = p.speaker_name or p.speaker_id
                lines.append(
                    f"  - {name}: {p.speaking_time:.1f}s ({p.speaking_share:.1f}%)"
                )
            lines.append("")

        # Summary
        if protocol.summary:
            lines.append("SUMMARY:")
            lines.append(protocol.summary)
            lines.append("")

        # Key topics
        if protocol.key_topics:
            lines.append("KEY TOPICS:")
            for topic in protocol.key_topics:
                lines.append(f"  - {topic.content or topic.title}")
            lines.append("")

        # Decisions
        if protocol.decisions:
            lines.append("DECISIONS:")
            for decision in protocol.decisions:
                responsible = f" ({decision.responsible})" if decision.responsible else ""
                lines.append(f"  - {decision.text}{responsible}")
            lines.append("")

        # Tasks
        if protocol.tasks:
            lines.append("ACTION ITEMS:")
            for task in protocol.tasks:
                assignee = f" [{task.assignee}]" if task.assignee else ""
                deadline = f" - Due: {task.deadline}" if task.deadline else ""
                lines.append(f"  - {task.text}{assignee}{deadline}")
            lines.append("")

        # Open questions
        if protocol.open_questions:
            lines.append("OPEN QUESTIONS:")
            for question in protocol.open_questions:
                lines.append(f"  - {question}")
            lines.append("")

        # Transcript
        if protocol.transcript:
            lines.append("=" * 80)
            lines.append("TRANSCRIPT")
            lines.append("=" * 80)
            for seg in protocol.transcript:
                timestamp = cls._format_timestamp(seg.start)
                speaker = seg.speaker_name or seg.speaker_id
                lines.append(f"{speaker} [{timestamp}]: {seg.text}")

        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"TXT saved: {output_path}")
        return str(output_path)

    @classmethod
    def _format_timestamp(cls, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @classmethod
    def _add_title_page(
        cls, doc: Document, protocol: MeetingProtocol, labels: dict[str, str]
    ) -> None:
        """Add title page to document."""
        # EPAM header
        header = doc.add_paragraph()
        header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = header.add_run("EPAM")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(24)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_RED

        # Title
        title = doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.space_before = Pt(24)
        run = title.add_run(labels["protocol"])
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(20)
        run.font.bold = True

        # Topic
        if protocol.topic:
            topic_para = doc.add_paragraph()
            topic_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            topic_para.space_before = Pt(12)
            run = topic_para.add_run(protocol.topic)
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(14)
            run.font.color.rgb = cls.EPAM_RED

        # Date
        doc.add_paragraph()  # Spacing
        date_para = doc.add_paragraph()
        date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if protocol.meeting_date:
            date_str = protocol.meeting_date.strftime("%d.%m.%Y")
        else:
            date_str = datetime.now().strftime("%d.%m.%Y")
        run = date_para.add_run(f"{labels['date']}: {date_str}")
        run.font.name = cls.BODY_FONT
        run.font.size = Pt(11)

        # Page break
        doc.add_page_break()

    @staticmethod
    def _format_timestamp(seconds: float, hms: Optional[bool] = None) -> str:
        """Format an audio offset for display next to a transcript turn.

        Args:
            seconds: Offset from audio start.
            hms: If True, render `[HH:MM:SS]`. If False, render `[MM:SS]`.
                If None, render legacy `HH:MM:SS` without brackets.
                Pick HMS when any turn in the document is >= 1h so the
                format is consistent across the whole transcript.
        """
        try:
            total = max(0, int(round(float(seconds))))
        except (TypeError, ValueError):
            return ""
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if hms is None:
            return f"{h:02d}:{m:02d}:{s:02d}"
        if hms:
            return f"[{h:02d}:{m:02d}:{s:02d}]"
        return f"[{(h * 60 + m):02d}:{s:02d}]"

    @classmethod
    def _add_heading(cls, doc: Document, text: str) -> None:
        """Add styled heading to document."""
        heading = doc.add_paragraph(text, style="Heading 1")
        heading_format = heading.paragraph_format
        heading_format.space_before = Pt(12)
        heading_format.space_after = Pt(6)

        # Style the heading
        for run in heading.runs:
            run.font.name = cls.HEADING_FONT
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = cls.EPAM_RED

        # Add underline
        pPr = heading._element.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "12")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), "B2001F")
        pBdr.append(bottom)
        pPr.append(pBdr)

    @classmethod
    def _set_body_font(cls, paragraph) -> None:
        """Apply body font styling to a paragraph."""
        for run in paragraph.runs:
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)
            run.font.color.rgb = cls.EPAM_DARK_GRAY

    @classmethod
    def _add_participants_section(
        cls, doc: Document, protocol: MeetingProtocol, labels: dict[str, str]
    ) -> None:
        """Add participants section."""
        if not protocol.participants:
            return

        cls._add_heading(doc, labels["participants"])

        # Create table
        table = doc.add_table(rows=len(protocol.participants) + 1, cols=4)
        table.style = "Light Grid"  # neutral grey grid; "Accent 1" is theme-blue

        # Header row
        header_cells = table.rows[0].cells
        header_texts = [labels["name_id"], labels["speaking_time"], labels["speaking_share"], ""]
        for i, text in enumerate(header_texts):
            cell = header_cells[i]
            cell.text = text
            # Style header
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
            # Set background color
            shading_elm = OxmlElement("w:shd")
            shading_elm.set(qn("w:fill"), "B2001F")
            cell._element.get_or_add_tcPr().append(shading_elm)

        # Data rows
        for i, participant in enumerate(protocol.participants):
            row = table.rows[i + 1]
            row.cells[0].text = participant.speaker_name or participant.speaker_id
            row.cells[1].text = f"{participant.speaking_time:.1f}s"
            row.cells[2].text = f"{participant.speaking_share:.1f}%"

        doc.add_paragraph()

    @classmethod
    def _add_summary_section(
        cls, doc: Document, protocol: MeetingProtocol, labels: dict[str, str]
    ) -> None:
        """Add summary section. Shows 'None' placeholder if empty."""
        cls._add_heading(doc, labels["summary"])
        if not protocol.summary:
            para = doc.add_paragraph(cls._get_empty_label(labels))
            cls._set_body_font(para)
            return

        para = doc.add_paragraph(protocol.summary)
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_after = Pt(12)
        for run in para.runs:
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)

    @classmethod
    def _get_empty_label(cls, labels: dict[str, str]) -> str:
        """Return localized 'None' label based on language context."""
        # Russian labels use Cyrillic section names
        is_russian = any(
            ord(c) > 0x400 for c in labels.get("key_topics", "")
        )
        return "Нет" if is_russian else "None"

    @classmethod
    def _add_key_topics_section(
        cls, doc: Document, protocol: MeetingProtocol, labels: dict[str, str]
    ) -> None:
        """Add key topics section. Shows 'None' placeholder if empty."""
        cls._add_heading(doc, labels["key_topics"])
        if not protocol.key_topics:
            para = doc.add_paragraph(cls._get_empty_label(labels))
            cls._set_body_font(para)
            return

        for topic in protocol.key_topics:
            # Use content (full text) as single bullet item
            text = topic.content or topic.title
            para = doc.add_paragraph(text, style="List Bullet")
            para.paragraph_format.left_indent = Inches(0.25)

    @classmethod
    def _add_decisions_section(
        cls, doc: Document, protocol: MeetingProtocol, labels: dict[str, str]
    ) -> None:
        """Add decisions section. Shows 'None' placeholder if empty."""
        cls._add_heading(doc, labels["decisions"])
        if not protocol.decisions:
            para = doc.add_paragraph(cls._get_empty_label(labels))
            cls._set_body_font(para)
            return

        for decision in protocol.decisions:
            text = decision.text
            if decision.responsible:
                text += f" ({decision.responsible})"
            para = doc.add_paragraph(text, style="List Bullet")
            para.paragraph_format.left_indent = Inches(0.25)

    @classmethod
    def _add_tasks_section(
        cls, doc: Document, protocol: MeetingProtocol, labels: dict[str, str]
    ) -> None:
        """Add tasks/action items section. Shows 'None' placeholder if empty."""
        cls._add_heading(doc, labels["tasks"])
        if not protocol.tasks:
            para = doc.add_paragraph(cls._get_empty_label(labels))
            cls._set_body_font(para)
            return

        # Create table for tasks
        table = doc.add_table(rows=len(protocol.tasks) + 1, cols=3)
        table.style = "Light Grid"  # neutral grey grid; "Accent 1" is theme-blue

        # Header row
        header_cells = table.rows[0].cells
        header_texts = [labels["task"], labels["assignee"], labels["deadline"]]
        for i, text in enumerate(header_texts):
            cell = header_cells[i]
            cell.text = text
            # Style header
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
            # Set background color
            shading_elm = OxmlElement("w:shd")
            shading_elm.set(qn("w:fill"), "B2001F")
            cell._element.get_or_add_tcPr().append(shading_elm)

        # Data rows
        for i, task in enumerate(protocol.tasks):
            row = table.rows[i + 1]
            row.cells[0].text = task.text
            row.cells[1].text = task.assignee or ""
            row.cells[2].text = task.deadline or ""

        doc.add_paragraph()

    @classmethod
    def _add_open_questions_section(
        cls, doc: Document, protocol: MeetingProtocol, labels: dict[str, str]
    ) -> None:
        """Add open questions section. Shows 'None' placeholder if empty."""
        cls._add_heading(doc, labels["open_questions"])
        if not protocol.open_questions:
            para = doc.add_paragraph(cls._get_empty_label(labels))
            cls._set_body_font(para)
            return
        for question in protocol.open_questions:
            para = doc.add_paragraph(question, style="List Bullet")
            para.paragraph_format.left_indent = Inches(0.25)

    @classmethod
    def _add_transcript_section(
        cls, doc: Document, protocol: MeetingProtocol, labels: dict[str, str]
    ) -> None:
        """Add full transcript appendix."""
        if not protocol.transcript:
            return

        doc.add_page_break()
        cls._add_heading(doc, labels["transcript"])

        for segment in protocol.transcript:
            timestamp = cls._format_timestamp(segment.start)
            speaker = segment.speaker_name or segment.speaker_id

            # Speaker line (bold, colored)
            speaker_para = doc.add_paragraph()
            speaker_para.paragraph_format.left_indent = Inches(0)
            speaker_para.paragraph_format.space_before = Pt(6)

            run = speaker_para.add_run(f"{speaker} [{timestamp}]")
            run.font.name = cls.BODY_FONT
            run.font.bold = True
            run.font.color.rgb = cls.EPAM_RED
            run.font.size = Pt(10)

            run = speaker_para.add_run(": ")
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(10)

            # Text
            run = speaker_para.add_run(segment.text)
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(10)

            if segment.confidence < 0.8:
                # Add confidence indicator if low
                run = speaker_para.add_run(f" [confidence: {segment.confidence:.2f}]")
                run.font.italic = True
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)

    # =================================================================
    # Shared helpers for Block-7 templates
    # =================================================================

    @classmethod
    def _add_page_numbers_footer(cls, doc: Document) -> None:
        """Inject page numbers in the centered footer of every section.

        python-docx has no high-level API for page numbers, so we build
        the `PAGE` field via OxmlElement. Format: "Стр. N из M".
        """
        for section in doc.sections:
            footer = section.footer
            # Reuse or create the first paragraph.
            if footer.paragraphs:
                para = footer.paragraphs[0]
                for run in list(para.runs):
                    run.text = ""
            else:
                para = footer.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER

            # "Стр. " label
            label_run = para.add_run("Стр. ")
            label_run.font.name = cls.BODY_FONT
            label_run.font.size = Pt(9)
            label_run.font.color.rgb = cls.EPAM_DARK_GRAY

            # PAGE field
            run = para.add_run()
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(9)
            fldChar_begin = OxmlElement("w:fldChar")
            fldChar_begin.set(qn("w:fldCharType"), "begin")
            instrText = OxmlElement("w:instrText")
            instrText.set(qn("xml:space"), "preserve")
            instrText.text = "PAGE"
            fldChar_end = OxmlElement("w:fldChar")
            fldChar_end.set(qn("w:fldCharType"), "end")
            run._r.append(fldChar_begin)
            run._r.append(instrText)
            run._r.append(fldChar_end)

            # " из " separator
            sep_run = para.add_run(" из ")
            sep_run.font.name = cls.BODY_FONT
            sep_run.font.size = Pt(9)
            sep_run.font.color.rgb = cls.EPAM_DARK_GRAY

            # NUMPAGES field
            run = para.add_run()
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(9)
            fldChar_begin = OxmlElement("w:fldChar")
            fldChar_begin.set(qn("w:fldCharType"), "begin")
            instrText = OxmlElement("w:instrText")
            instrText.set(qn("xml:space"), "preserve")
            instrText.text = "NUMPAGES"
            fldChar_end = OxmlElement("w:fldChar")
            fldChar_end.set(qn("w:fldCharType"), "end")
            run._r.append(fldChar_begin)
            run._r.append(instrText)
            run._r.append(fldChar_end)

    @classmethod
    def _shade_cell(cls, cell, hex_color: str) -> None:
        """Apply a solid background color to a table cell."""
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), hex_color)
        cell._element.get_or_add_tcPr().append(shading)

    @classmethod
    def _clear_cell_shading(cls, cell) -> None:
        """Force a table cell to render with no background fill.

        Word table styles (especially those with conditional row
        banding) apply shading via the style's ``w:tblPr`` block.
        Setting an inline ``<w:shd w:val="clear" w:fill="auto"/>``
        on the cell overrides any banding from the style. Used by
        the court-hearing table renderer to guarantee white/transparent
        body cells per user request 2026-05-04 ("везде белый /
        прозрачный фон нужен"). Removes any pre-existing ``w:shd``
        children to avoid stacking.
        """
        tcPr = cell._element.get_or_add_tcPr()
        # Drop any existing shading element so we don't end up with
        # multiple `<w:shd>` siblings (rendering is undefined when
        # there are duplicates).
        for existing in tcPr.findall(qn("w:shd")):
            tcPr.remove(existing)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:val"), "clear")
        shading.set(qn("w:color"), "auto")
        shading.set(qn("w:fill"), "auto")
        tcPr.append(shading)

    @classmethod
    def _set_default_margins(cls, doc: Document) -> None:
        """Set consistent 0.75in margins across every section."""
        for section in doc.sections:
            section.top_margin = Inches(0.75)
            section.bottom_margin = Inches(0.75)
            section.left_margin = Inches(0.75)
            section.right_margin = Inches(0.75)

    @classmethod
    def _add_plain_paragraph(
        cls,
        doc: Document,
        text: str,
        *,
        italic: bool = False,
        bold: bool = False,
        size: int = 11,
        color: Optional[RGBColor] = None,
    ) -> None:
        """Add a simple body paragraph with EPAM body font."""
        para = doc.add_paragraph()
        run = para.add_run(text)
        run.font.name = cls.BODY_FONT
        run.font.size = Pt(size)
        run.font.italic = italic
        run.font.bold = bold
        run.font.color.rgb = color or cls.EPAM_DARK_GRAY

    # =================================================================
    # Court hearing
    # =================================================================

    @classmethod
    def _format_court_hearing_docx(
        cls, payload: CourtHearingProtocol, output_path: str
    ) -> str:
        """Court hearing stenogram: title page + summary + 2-column Q&A.

        Title format: "Стенограмма судебного заседания № X от DD.MM.YYYY"
        (per user requirement 2026-04-21). Page numbers in footer.
        The Q&A table is verbatim — turns come from the aligned transcript,
        the LLM does not paraphrase.
        """
        logger.info(f"Generating DOCX (court hearing): {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = Document()
        cls._set_default_margins(doc)

        # ----- Title block -----
        epam_hdr = doc.add_paragraph()
        epam_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = epam_hdr.add_run("EPAM")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_RED

        # Title: "Стенограмма судебного заседания [№ X] [от DD.MM.YYYY]"
        title_parts = [payload.title or "Стенограмма судебного заседания"]
        if payload.case_number:
            title_parts.append(f"№ {payload.case_number}")
        if payload.hearing_date:
            title_parts.append(f"от {payload.hearing_date}")
        title_text = " ".join(title_parts)

        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_before = Pt(18)
        title_para.paragraph_format.space_after = Pt(18)
        run = title_para.add_run(title_text)
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY

        # ----- Participants -----
        if payload.participants:
            cls._add_heading(doc, "УЧАСТНИКИ")
            for participant in payload.participants:
                para = doc.add_paragraph(participant, style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

        # ----- Summary -----
        cls._add_heading(doc, "КРАТКОЕ СОДЕРЖАНИЕ")
        summary_text = payload.summary.strip() if payload.summary else "Нет"
        para = doc.add_paragraph(summary_text)
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_after = Pt(12)
        for run in para.runs:
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)

        # ----- Stenogram table -----
        cls._add_heading(doc, "СТЕНОГРАММА")

        if not payload.turns:
            para = doc.add_paragraph("Нет записи заседания.")
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)
                run.font.italic = True
        else:
            table = doc.add_table(rows=len(payload.turns) + 1, cols=2)
            # No built-in style — those introduce zebra-striping or
            # grey grid backgrounds. User requested 2026-05-04: every
            # cell white/transparent, only thin borders. We set the
            # bare-border style "Table Grid" (no shading) and then
            # explicitly clear shading on every cell.
            table.style = "Table Grid"
            table.autofit = False

            # Header row — keep the EPAM-red header (brand element),
            # no other coloured cells in the table.
            header_cells = table.rows[0].cells
            header_cells[0].text = "Участник"
            header_cells[1].text = "Речь"
            for cell in header_cells:
                cls._shade_cell(cell, "B2001F")
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.name = cls.HEADING_FONT
                        run.font.size = Pt(11)
                        run.font.color.rgb = RGBColor(255, 255, 255)

            # Column widths ~ 1.3 in / 5.2 in
            for row in table.rows:
                row.cells[0].width = Inches(1.5)
                row.cells[1].width = Inches(5.0)

            # Decide timestamp format once: HH:MM:SS for hearings >= 1h,
            # MM:SS for shorter ones. Lawyers cross-reference these
            # against the audio recording.
            max_start = max(
                (t.start_s for t in payload.turns if t.start_s is not None),
                default=0.0,
            )
            ts_format_hms = max_start >= 3600.0

            for i, turn in enumerate(payload.turns):
                row = table.rows[i + 1]
                # Speaker cell: bold speaker label, then [timestamp] in
                # smaller grey type below — keeps the speaker column
                # narrow but still surfaces the time code.
                speaker_cell = row.cells[0]
                speaker_cell.text = ""  # clear any default
                speaker_para = speaker_cell.paragraphs[0]
                speaker_run = speaker_para.add_run(turn.speaker or "")
                speaker_run.font.name = cls.BODY_FONT
                speaker_run.font.size = Pt(10)
                speaker_run.font.bold = True
                speaker_run.font.color.rgb = cls.EPAM_DARK_GRAY
                if turn.start_s is not None:
                    ts_text = cls._format_timestamp(turn.start_s, ts_format_hms)
                    ts_para = speaker_cell.add_paragraph()
                    ts_run = ts_para.add_run(ts_text)
                    ts_run.font.name = cls.BODY_FONT
                    ts_run.font.size = Pt(8)
                    ts_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

                row.cells[1].text = turn.text or ""
                row.cells[0].vertical_alignment = WD_ALIGN_VERTICAL.TOP
                row.cells[1].vertical_alignment = WD_ALIGN_VERTICAL.TOP
                for paragraph in row.cells[1].paragraphs:
                    for run in paragraph.runs:
                        run.font.name = cls.BODY_FONT
                        run.font.size = Pt(10)
                # Explicitly clear any shading on body cells. Some
                # Word table styles apply alternating row banding via
                # the conditional formatting block — we override it
                # by writing a `<w:shd w:val="clear" w:fill="auto"/>`
                # into each cell's tcPr. Belt-and-braces with the
                # "Table Grid" style above which doesn't ship banding.
                cls._clear_cell_shading(row.cells[0])
                cls._clear_cell_shading(row.cells[1])

        # Page numbers footer.
        cls._add_page_numbers_footer(doc)

        doc.save(str(output_path))
        logger.info(f"DOCX saved: {output_path}")
        return str(output_path)

    # =================================================================
    # Administrative meeting
    # =================================================================

    @classmethod
    def _format_administrative_docx(
        cls, payload: AdministrativeProtocol, output_path: str
    ) -> str:
        """Dispatch admin DOCX render across admin layouts.

        Three layouts can appear in stored payloads:

        * Practical v1.0 (2026-05-07): ``items`` is non-empty. Renders
          only participants plus one unified list of working takeaways.

        * Topic-segmented (Sprint 2026-04-30): ``topic_summaries``
          is non-empty. Each topic gets its own block with discussion,
          decisions, tasks, open_questions; risks remain at protocol
          level. Map-reduce extraction populated by the summarization
          engine when ``use_topic_segmented_admin`` is on (default).

        * Flat 9-section (T3.3, 2026-04-23): ``decisions``/``tasks``/
          ``open_questions``/``risks`` populated, ``topic_summaries``
          empty. Used when topic segmentation is disabled or fails.

        * Legacy department-grouped: ``departments`` populated and
          flat fields empty. Pre-T3.3 protocols still on disk.
        """
        if getattr(payload, "items", None):
            return cls._format_administrative_docx_practical(
                payload, output_path
            )
        if payload.topic_summaries:
            return cls._format_administrative_docx_topic_segmented(
                payload, output_path
            )
        # Legacy if departments has any non-empty block AND the new
        # flat fields are all empty. Otherwise use flat (default for
        # new runs and any payload that already has flat data).
        has_dept_content = any(
            (b.decisions or b.tasks or b.open_issues)
            for b in (payload.departments or [])
        )
        has_flat_content = bool(
            payload.topics
            or payload.decisions
            or payload.tasks
            or payload.open_questions
            or payload.risks
        )
        if has_dept_content and not has_flat_content:
            return cls._format_administrative_docx_legacy(payload, output_path)
        return cls._format_administrative_docx_flat(payload, output_path)

    @classmethod
    def _format_administrative_docx_practical(
        cls, payload: AdministrativeProtocol, output_path: str
    ) -> str:
        """Practical v1.0 admin formatter.

        User-facing principle: less report about the meeting, more what
        to remember and do. This DOCX intentionally omits goal, summary,
        topics, risks-as-a-separate-section, and stenogram. The full
        transcript remains available as a separate artifact in the app.
        """
        logger.info(f"Generating DOCX (administrative PRACTICAL): {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = Document()
        cls._set_default_margins(doc)

        epam_hdr = doc.add_paragraph()
        epam_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = epam_hdr.add_run("EPAM")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_RED

        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_before = Pt(18)
        title_para.paragraph_format.space_after = Pt(6)
        title_text = payload.title or "Протокол административного совещания"
        run = title_para.add_run(title_text)
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY

        if payload.meeting_date:
            date_para = doc.add_paragraph()
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_para.paragraph_format.space_after = Pt(18)
            run = date_para.add_run(f"Дата: {payload.meeting_date}")
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)
            run.font.color.rgb = cls.EPAM_DARK_GRAY

        if payload.participants:
            cls._add_heading(doc, "УЧАСТНИКИ")
            for participant in payload.participants:
                para = doc.add_paragraph(participant, style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

        cls._add_heading(doc, "ПРИНЯТЫЕ РЕШЕНИЯ И РАБОЧИЕ ТЕЗИСЫ")
        items = getattr(payload, "items", None) or []
        if not items:
            cls._render_empty_placeholder(doc)
        else:
            cls._render_practical_admin_items(doc, items)

        cls._add_page_numbers_footer(doc)
        doc.save(str(output_path))
        logger.info(f"DOCX saved: {output_path}")
        return str(output_path)

    @classmethod
    def format_admin_actions_docx(
        cls, payload: AdministrativeProtocol, output_path: str
    ) -> str:
        """Generate a compact task/action-list DOCX for admin meetings."""
        logger.info(f"Generating DOCX (administrative ACTIONS): {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = Document()
        cls._set_default_margins(doc)

        title = doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = title.add_run("СПИСОК ПОРУЧЕНИЙ")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY

        if payload.meeting_date:
            date_para = doc.add_paragraph()
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_run = date_para.add_run(f"Дата: {payload.meeting_date}")
            date_run.font.name = cls.BODY_FONT
            date_run.font.size = Pt(10)
            date_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        if payload.participants:
            cls._add_heading(doc, "УЧАСТНИКИ")
            para = doc.add_paragraph(", ".join(payload.participants))
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(10)

        items = []
        seen: set[tuple[str, str, str]] = set()
        for item in list(getattr(payload, "items", None) or []) + list(
            getattr(payload, "tasks", None) or []
        ):
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            is_task = getattr(item, "kind", "") == "task"
            has_action_fields = bool(
                (getattr(item, "owner", None) or "").strip()
                or (getattr(item, "deadline", None) or "").strip()
            )
            if not is_task and not has_action_fields:
                continue
            owner = (getattr(item, "owner", None) or "").strip()
            deadline = (getattr(item, "deadline", None) or "").strip()
            key = (text.lower(), owner.lower(), deadline.lower())
            if key in seen:
                continue
            seen.add(key)
            items.append(item)

        cls._add_heading(doc, "ПОРУЧЕНИЯ")
        if not items:
            para = doc.add_paragraph("Поручения не выявлены.")
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)
                run.font.italic = True
                run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            cls._add_page_numbers_footer(doc)
            doc.save(str(output_path))
            return str(output_path)

        table = doc.add_table(rows=1, cols=7)
        table.style = "Table Grid"
        headers = [
            "№",
            "Поручение",
            "Исполнитель",
            "Срок",
            "Основание / цитата",
            "Таймкод",
            "Надежность",
        ]
        for idx, header in enumerate(headers):
            cell = table.rows[0].cells[idx]
            cls._shade_cell(cell, "F2F2F2")
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.add_run(header)
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(9)
            run.font.bold = True

        confidence_labels = {
            "high": "высокая",
            "medium": "средняя",
            "low": "низкая",
        }
        for idx, item in enumerate(items, start=1):
            row = table.add_row().cells
            values = [
                str(idx),
                (getattr(item, "text", "") or "").strip(),
                (getattr(item, "owner", None) or "").strip() or "Не указано",
                (getattr(item, "deadline", None) or "").strip() or "Не указано",
                (getattr(item, "evidence", "") or "").strip(),
                (getattr(item, "timestamp", "") or "").strip(),
                confidence_labels.get(
                    getattr(item, "confidence", "medium"), "средняя"
                ),
            ]
            for cell, value in zip(row, values):
                cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_after = Pt(0)
                run = paragraph.add_run(value)
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(8 if len(value) > 80 else 9)

        cls._add_page_numbers_footer(doc)
        doc.save(str(output_path))
        logger.info(f"DOCX saved: {output_path}")
        return str(output_path)

    @classmethod
    def _render_practical_admin_items(cls, doc: Document, items) -> None:
        """Render practical admin items as numbered evidence-backed blocks."""
        kind_labels = {
            "decision": "Решение",
            "task": "Поручение",
            "open_question": "Открытый вопрос",
            "risk": "Риск",
            "thesis": "Тезис",
        }
        confidence_labels = {
            "high": "высокая",
            "medium": "средняя",
            "low": "низкая",
        }
        for idx, item in enumerate(items, start=1):
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            kind = kind_labels.get(getattr(item, "kind", "thesis"), "Тезис")
            confidence = confidence_labels.get(
                getattr(item, "confidence", "medium"), "средняя"
            )

            para = doc.add_paragraph()
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.space_after = Pt(2)
            num_run = para.add_run(f"{idx}. [{kind}] ")
            num_run.font.name = cls.BODY_FONT
            num_run.font.size = Pt(11)
            num_run.font.bold = True
            num_run.font.color.rgb = cls.EPAM_RED
            text_run = para.add_run(text)
            text_run.font.name = cls.BODY_FONT
            text_run.font.size = Pt(11)

            meta_parts: list[str] = []
            owner = (getattr(item, "owner", None) or "").strip()
            deadline = (getattr(item, "deadline", None) or "").strip()
            speaker = (getattr(item, "speaker", "") or "").strip()
            timestamp = (getattr(item, "timestamp", "") or "").strip()
            if owner:
                meta_parts.append(f"Ответственный: {owner}")
            if deadline:
                meta_parts.append(f"Срок: {deadline}")
            if speaker:
                meta_parts.append(f"Кто поднял: {speaker}")
            if timestamp:
                meta_parts.append(f"Таймкод: {timestamp}")
            meta_parts.append(f"Надежность: {confidence}")
            meta = doc.add_paragraph(" | ".join(meta_parts))
            meta.paragraph_format.left_indent = Inches(0.25)
            meta.paragraph_format.space_after = Pt(1)
            for run in meta.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

            evidence = (getattr(item, "evidence", "") or "").strip()
            if evidence:
                quote = doc.add_paragraph()
                quote.paragraph_format.left_indent = Inches(0.25)
                quote.paragraph_format.space_after = Pt(5)
                run = quote.add_run(f"Основание: «{evidence}»")
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(9)
                run.font.italic = True
                run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    @classmethod
    def _format_administrative_docx_legacy(
        cls, payload: AdministrativeProtocol, output_path: str
    ) -> str:
        """LEGACY admin formatter — department-grouped sections.

        Used only for protocols generated before T3.3 (2026-04-23) which
        populate the `departments` field. New protocols use the flat
        9-section formatter via `_format_administrative_docx_flat`.
        """
        logger.info(f"Generating DOCX (administrative LEGACY): {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = Document()
        cls._set_default_margins(doc)

        # ----- Title block -----
        epam_hdr = doc.add_paragraph()
        epam_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = epam_hdr.add_run("EPAM")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_RED

        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_before = Pt(18)
        title_para.paragraph_format.space_after = Pt(6)
        title_text = payload.title or "Протокол административного совещания"
        run = title_para.add_run(title_text)
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY

        if payload.meeting_date:
            date_para = doc.add_paragraph()
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_para.paragraph_format.space_after = Pt(18)
            run = date_para.add_run(f"Дата: {payload.meeting_date}")
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)
            run.font.color.rgb = cls.EPAM_DARK_GRAY

        # ----- Participants -----
        if payload.participants:
            cls._add_heading(doc, "УЧАСТНИКИ")
            for participant in payload.participants:
                para = doc.add_paragraph(participant, style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

        # ----- Summary -----
        cls._add_heading(doc, "КРАТКОЕ СОДЕРЖАНИЕ")
        summary_text = payload.summary.strip() if payload.summary else "Нет"
        para = doc.add_paragraph(summary_text)
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_after = Pt(12)
        for run in para.runs:
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)

        # ----- Department-indexed departments dict -----
        dept_map: dict[str, DepartmentBlock] = {}
        for block in payload.departments:
            if block.slug:
                dept_map[block.slug] = block

        # ----- The three sections, each subdivided by department -----
        cls._render_admin_section(
            doc, "ПРИНЯТЫЕ РЕШЕНИЯ", dept_map, kind="decisions"
        )
        cls._render_admin_section(
            doc, "ЗАДАЧИ И ПОРУЧЕНИЯ", dept_map, kind="tasks"
        )
        cls._render_admin_section(
            doc, "ОТКРЫТЫЕ ВОПРОСЫ", dept_map, kind="open_issues"
        )

        # ----- Stenogram (verbatim turns from aligned transcript) -----
        # Added 2026-04-23 matching EPAM IT 9-section layout. Populated
        # by the parser, not the LLM, so it's ground truth for the
        # reader to audit every extracted claim above.
        turns = getattr(payload, "turns", []) or []
        if turns:
            cls._add_heading(doc, "СТЕНОГРАММА")
            # Decide timestamp format once for the whole transcript.
            max_start = max(
                (
                    t.start_s
                    for t in turns
                    if getattr(t, "start_s", None) is not None
                ),
                default=0.0,
            )
            ts_format_hms = max_start >= 3600.0
            for turn in turns:
                speaker = (turn.speaker or "").strip() or "Говорящий"
                text = (turn.text or "").strip()
                if not text:
                    continue
                para = doc.add_paragraph()
                para.paragraph_format.space_after = Pt(4)
                # [HH:MM:SS] in light grey before the speaker label so the
                # reader can cross-reference the audio recording.
                start_s = getattr(turn, "start_s", None)
                if start_s is not None:
                    ts_text = cls._format_timestamp(start_s, ts_format_hms)
                    ts_run = para.add_run(f"{ts_text} ")
                    ts_run.font.name = cls.BODY_FONT
                    ts_run.font.size = Pt(9)
                    ts_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
                speaker_run = para.add_run(f"{speaker}: ")
                speaker_run.font.name = cls.BODY_FONT
                speaker_run.font.size = Pt(10)
                speaker_run.font.bold = True
                text_run = para.add_run(text)
                text_run.font.name = cls.BODY_FONT
                text_run.font.size = Pt(10)

        cls._add_page_numbers_footer(doc)

        doc.save(str(output_path))
        logger.info(f"DOCX saved: {output_path}")
        return str(output_path)

    @classmethod
    def _format_administrative_docx_topic_segmented(
        cls, payload: AdministrativeProtocol, output_path: str
    ) -> str:
        """Map-reduce topic-segmented admin formatter (Sprint 2026-04-30).

        Section order:
            1. EPAM header
            2. Title + meeting_date
            3. УЧАСТНИКИ
            4. ЦЕЛЬ ВСТРЕЧИ (if stated)
            5. КРАТКОЕ СОДЕРЖАНИЕ
            6. For each topic in payload.topic_summaries:
                 6a. Topic heading
                 6b. Обсуждение (1-3 sentences)
                 6c. Принятые решения (within this topic)
                 6d. Задачи и поручения (table within this topic)
                 6e. Открытые вопросы (within this topic)
            7. РИСКИ И БЛОКЕРЫ (cross-cutting)
            8. СТЕНОГРАММА (verbatim turns from aligned transcript)

        Reads top-down as a series of self-contained topic blocks
        rather than category-stacked flat lists. Each topic block
        becomes a forwardable mini-protocol for its owner.
        """
        logger.info(
            f"Generating DOCX (administrative TOPIC-SEGMENTED): {output_path}"
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = Document()
        cls._set_default_margins(doc)

        # ----- Title block -----
        epam_hdr = doc.add_paragraph()
        epam_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = epam_hdr.add_run("EPAM")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_RED

        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_before = Pt(18)
        title_para.paragraph_format.space_after = Pt(6)
        title_text = payload.title or "Протокол административного совещания"
        run = title_para.add_run(title_text)
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY

        if payload.meeting_date:
            date_para = doc.add_paragraph()
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_para.paragraph_format.space_after = Pt(18)
            run = date_para.add_run(f"Дата: {payload.meeting_date}")
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)
            run.font.color.rgb = cls.EPAM_DARK_GRAY

        # ----- Participants -----
        if payload.participants:
            cls._add_heading(doc, "УЧАСТНИКИ")
            for participant in payload.participants:
                para = doc.add_paragraph(participant, style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

        # ----- Meeting goal -----
        if payload.meeting_goal:
            cls._add_heading(doc, "ЦЕЛЬ ВСТРЕЧИ")
            para = doc.add_paragraph(payload.meeting_goal)
            para.paragraph_format.line_spacing = 1.15
            para.paragraph_format.space_after = Pt(12)
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)

        # ----- Summary -----
        cls._add_heading(doc, "КРАТКОЕ СОДЕРЖАНИЕ")
        summary_text = payload.summary.strip() if payload.summary else "Нет"
        para = doc.add_paragraph(summary_text)
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_after = Pt(12)
        for run in para.runs:
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)

        # ----- Per-topic blocks -----
        for i, topic in enumerate(payload.topic_summaries):
            # Topic heading: numbered for navigation in long protocols.
            heading_text = f"{i + 1}. {topic.name.strip() or 'Тема'}"
            heading_para = doc.add_paragraph()
            heading_para.paragraph_format.space_before = Pt(18)
            heading_para.paragraph_format.space_after = Pt(6)
            run = heading_para.add_run(heading_text.upper())
            run.font.name = cls.HEADING_FONT
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = cls.EPAM_RED

            # Discussion (prose paragraph; if empty, skip).
            if topic.discussion.strip():
                para = doc.add_paragraph(topic.discussion.strip())
                para.paragraph_format.line_spacing = 1.15
                para.paragraph_format.space_after = Pt(8)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

            # Per-topic decisions, tasks, open questions. Use the
            # same renderers as the flat path so styling is consistent.
            if topic.decisions:
                cls._render_flat_items_with_speaker(
                    doc, "Решения", topic.decisions
                )
            if topic.tasks:
                cls._render_flat_tasks_table(
                    doc, "Задачи и поручения", topic.tasks
                )
            if topic.open_questions:
                cls._render_flat_items_with_speaker(
                    doc, "Открытые вопросы", topic.open_questions
                )

            # Empty topic placeholder so readers know we considered
            # the topic but found nothing structured.
            if not (topic.discussion or topic.decisions or topic.tasks
                    or topic.open_questions):
                empty = doc.add_paragraph(
                    "По данной теме нет извлечённых решений, задач или вопросов."
                )
                empty.paragraph_format.left_indent = Inches(0.15)
                for run in empty.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(10)
                    run.font.italic = True
                    run.font.color.rgb = cls.EPAM_DARK_GRAY

        # ----- Risks (cross-cutting, after all topics) -----
        if payload.risks:
            cls._render_flat_items_with_speaker(
                doc, "РИСКИ И БЛОКЕРЫ", payload.risks
            )

        # ----- Stenogram (verbatim turns) — same as flat path -----
        turns = getattr(payload, "turns", []) or []
        if turns:
            cls._add_heading(doc, "СТЕНОГРАММА")
            max_start = max(
                (
                    t.start_s
                    for t in turns
                    if getattr(t, "start_s", None) is not None
                ),
                default=0,
            )
            use_hms = max_start >= 3600
            for turn in turns:
                start_s = getattr(turn, "start_s", None)
                speaker = getattr(turn, "speaker", "") or ""
                text = getattr(turn, "text", "") or ""
                if start_s is not None:
                    if use_hms:
                        hh = int(start_s // 3600)
                        mm = int((start_s % 3600) // 60)
                        ss = int(start_s % 60)
                        ts = f"[{hh:02d}:{mm:02d}:{ss:02d}]"
                    else:
                        mm = int(start_s // 60)
                        ss = int(start_s % 60)
                        ts = f"[{mm:02d}:{ss:02d}]"
                    line = f"{ts} {speaker}: {text}"
                else:
                    line = f"{speaker}: {text}"
                para = doc.add_paragraph(line)
                para.paragraph_format.line_spacing = 1.15
                para.paragraph_format.space_after = Pt(4)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(10)

        doc.save(str(output_path))
        return str(output_path)

    @classmethod
    def _format_administrative_docx_flat(
        cls, payload: AdministrativeProtocol, output_path: str
    ) -> str:
        """T3.3 flat 9-section admin formatter.

        Sections in render order:
            1. EPAM header
            2. Title + meeting_date
            3. УЧАСТНИКИ
            4. ЦЕЛЬ ВСТРЕЧИ (only if payload.meeting_goal is populated)
            5. КРАТКОЕ СОДЕРЖАНИЕ
            6. ОСНОВНЫЕ ТЕМЫ (bullet list of topics)
            7. ПРИНЯТЫЕ РЕШЕНИЯ (flat list with speaker)
            8. ЗАДАЧИ И ПОРУЧЕНИЯ (table: Задача / Исполнитель / Срок)
            9. ОТКРЫТЫЕ ВОПРОСЫ (flat list with speaker)
           10. РИСКИ И БЛОКЕРЫ (flat list with speaker; rendered only
               if payload.risks is non-empty)
           11. СТЕНОГРАММА (verbatim turns from aligned transcript)
        """
        logger.info(f"Generating DOCX (administrative FLAT): {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = Document()
        cls._set_default_margins(doc)

        # ----- Title block -----
        epam_hdr = doc.add_paragraph()
        epam_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = epam_hdr.add_run("EPAM")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_RED

        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_before = Pt(18)
        title_para.paragraph_format.space_after = Pt(6)
        title_text = payload.title or "Протокол административного совещания"
        run = title_para.add_run(title_text)
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY

        if payload.meeting_date:
            date_para = doc.add_paragraph()
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_para.paragraph_format.space_after = Pt(18)
            run = date_para.add_run(f"Дата: {payload.meeting_date}")
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)
            run.font.color.rgb = cls.EPAM_DARK_GRAY

        # ----- Participants -----
        if payload.participants:
            cls._add_heading(doc, "УЧАСТНИКИ")
            for participant in payload.participants:
                para = doc.add_paragraph(participant, style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

        # ----- Meeting goal (only if explicitly stated) -----
        if payload.meeting_goal:
            cls._add_heading(doc, "ЦЕЛЬ ВСТРЕЧИ")
            para = doc.add_paragraph(payload.meeting_goal)
            para.paragraph_format.line_spacing = 1.15
            para.paragraph_format.space_after = Pt(12)
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)

        # ----- Summary -----
        cls._add_heading(doc, "КРАТКОЕ СОДЕРЖАНИЕ")
        summary_text = payload.summary.strip() if payload.summary else "Нет"
        para = doc.add_paragraph(summary_text)
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_after = Pt(12)
        for run in para.runs:
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)

        # ----- Topics (bullet list) -----
        cls._add_heading(doc, "ОСНОВНЫЕ ТЕМЫ")
        if payload.topics:
            for topic in payload.topics:
                para = doc.add_paragraph(topic, style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)
        else:
            cls._render_empty_placeholder(doc)

        # ----- Decisions -----
        cls._render_flat_items_with_speaker(
            doc, "ПРИНЯТЫЕ РЕШЕНИЯ", payload.decisions
        )

        # ----- Tasks (table) -----
        cls._render_flat_tasks_table(
            doc, "ЗАДАЧИ И ПОРУЧЕНИЯ", payload.tasks
        )

        # ----- Open questions -----
        cls._render_flat_items_with_speaker(
            doc, "ОТКРЫТЫЕ ВОПРОСЫ", payload.open_questions
        )

        # ----- Risks (only if non-empty — many meetings have none) -----
        if payload.risks:
            cls._render_flat_items_with_speaker(
                doc, "РИСКИ И БЛОКЕРЫ", payload.risks
            )

        # ----- Stenogram (verbatim turns) -----
        turns = getattr(payload, "turns", []) or []
        if turns:
            cls._add_heading(doc, "СТЕНОГРАММА")
            max_start = max(
                (
                    t.start_s
                    for t in turns
                    if getattr(t, "start_s", None) is not None
                ),
                default=0.0,
            )
            ts_format_hms = max_start >= 3600.0
            for turn in turns:
                speaker = (turn.speaker or "").strip() or "Говорящий"
                text = (turn.text or "").strip()
                if not text:
                    continue
                para = doc.add_paragraph()
                para.paragraph_format.space_after = Pt(4)
                start_s = getattr(turn, "start_s", None)
                if start_s is not None:
                    ts_text = cls._format_timestamp(start_s, ts_format_hms)
                    ts_run = para.add_run(f"{ts_text} ")
                    ts_run.font.name = cls.BODY_FONT
                    ts_run.font.size = Pt(9)
                    ts_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
                speaker_run = para.add_run(f"{speaker}: ")
                speaker_run.font.name = cls.BODY_FONT
                speaker_run.font.size = Pt(10)
                speaker_run.font.bold = True
                text_run = para.add_run(text)
                text_run.font.name = cls.BODY_FONT
                text_run.font.size = Pt(10)

        cls._add_page_numbers_footer(doc)

        doc.save(str(output_path))
        logger.info(f"DOCX saved: {output_path}")
        return str(output_path)

    @classmethod
    def _render_empty_placeholder(cls, doc: Document) -> None:
        """Render 'Нет' italic placeholder for an empty section."""
        para = doc.add_paragraph("Нет")
        for run in para.runs:
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)
            run.font.italic = True
            run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    @classmethod
    def _render_flat_items_with_speaker(
        cls,
        doc: Document,
        section_title: str,
        items,
    ) -> None:
        """Render a flat list of decisions / open_questions / risks.

        Each item: bullet with text, then ' — ' speaker (italic, grey)
        when present. Empty list renders 'Нет' placeholder.

        2026-05-04 (task #45): low-confidence items are prefixed with
        a "[?]" marker in amber so the user knows to verify them.
        Confidence comes from the LLM's per-item assessment in the
        primary extraction call, replacing the deleted-on-no-evidence
        verification pass.
        """
        cls._add_heading(doc, section_title)
        if not items:
            cls._render_empty_placeholder(doc)
            return
        for item in items:
            text = (item.text or "").strip()
            if not text:
                continue
            para = doc.add_paragraph(style="List Bullet")
            para.paragraph_format.left_indent = Inches(0.25)
            # Confidence marker — only for "low", and only when the
            # field is present (legacy protocols missing the field
            # default to "medium" via Pydantic).
            confidence = getattr(item, "confidence", "medium")
            if confidence == "low":
                marker_run = para.add_run("[?] ")
                marker_run.font.name = cls.BODY_FONT
                marker_run.font.size = Pt(11)
                marker_run.font.bold = True
                # Amber — same as low-confidence speaker attribution
                # in the transcript editor, for visual consistency.
                marker_run.font.color.rgb = RGBColor(0xD9, 0x84, 0x10)
            text_run = para.add_run(text)
            text_run.font.name = cls.BODY_FONT
            text_run.font.size = Pt(11)
            speaker = (item.speaker or "").strip()
            if speaker:
                spk_run = para.add_run(f" — {speaker}")
                spk_run.font.name = cls.BODY_FONT
                spk_run.font.size = Pt(10)
                spk_run.font.italic = True
                spk_run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    @classmethod
    def _render_flat_tasks_table(
        cls,
        doc: Document,
        section_title: str,
        tasks,
    ) -> None:
        """Render tasks as a 3-column table: Задача | Исполнитель | Срок.

        Matches the IT-team baseline format. Empty list renders 'Нет'.
        """
        cls._add_heading(doc, section_title)
        if not tasks:
            cls._render_empty_placeholder(doc)
            return

        table = doc.add_table(rows=len(tasks) + 1, cols=3)
        table.style = "Light Grid"  # neutral grey grid (T1.1)
        table.autofit = False

        # Header
        header_cells = table.rows[0].cells
        header_cells[0].text = "Задача"
        header_cells[1].text = "Исполнитель"
        header_cells[2].text = "Срок"
        for cell in header_cells:
            cls._shade_cell(cell, "B2001F")
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
                    run.font.name = cls.HEADING_FONT
                    run.font.size = Pt(11)
                    run.font.color.rgb = RGBColor(255, 255, 255)

        for row in table.rows:
            row.cells[0].width = Inches(4.0)
            row.cells[1].width = Inches(1.7)
            row.cells[2].width = Inches(1.0)

        for i, task in enumerate(tasks):
            row = table.rows[i + 1]
            row.cells[0].text = (task.text or "").strip()
            row.cells[1].text = (task.owner or "").strip() or "Нет"
            row.cells[2].text = (task.deadline or "").strip() or "Нет"
            for cell in row.cells:
                cell.vertical_alignment = WD_ALIGN_VERTICAL.TOP
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.name = cls.BODY_FONT
                        run.font.size = Pt(10)

    @classmethod
    def _render_admin_section(
        cls,
        doc: Document,
        section_title: str,
        dept_map: dict[str, DepartmentBlock],
        *,
        kind: str,
    ) -> None:
        """Render one of the three admin top-level sections.

        Iterates DEPARTMENT_SLUGS_ORDERED so the display order is stable
        even if the LLM returned departments out of order or omitted
        empty ones. Empty departments render "Нет".
        """
        cls._add_heading(doc, section_title)

        for slug in DEPARTMENT_SLUGS_ORDERED:
            dept = DEPARTMENT_BY_SLUG.get(slug)
            if dept is None:
                continue
            block = dept_map.get(slug)
            items: list[DepartmentItem] = getattr(block, kind, []) if block else []

            # Department sub-heading
            sub = doc.add_paragraph()
            sub.paragraph_format.space_before = Pt(8)
            sub.paragraph_format.space_after = Pt(2)
            run = sub.add_run(dept.display_name)
            run.font.name = cls.HEADING_FONT
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.color.rgb = cls.EPAM_DARK_GRAY

            if not items:
                para = doc.add_paragraph("Нет")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)
                    run.font.italic = True
                    run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
                continue

            for item in items:
                text = item.text.strip() if item.text else ""
                if not text:
                    continue

                # Tasks: show department (implicit — we're already under
                # the department heading) + optional deadline. No individual
                # assignee per user requirement 2026-04-21.
                suffix_parts: list[str] = []
                if kind == "tasks" and item.deadline:
                    suffix_parts.append(f"срок: {item.deadline}")
                if item.speaker and kind != "tasks":
                    # Only surface speaker for decisions / open issues —
                    # tasks are department-level only.
                    suffix_parts.append(f"по поручению: {item.speaker}")

                if suffix_parts:
                    text = f"{text} ({'; '.join(suffix_parts)})"

                para = doc.add_paragraph(text, style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.35)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

    # =================================================================
    # Client meeting
    # =================================================================

    @classmethod
    def _format_client_meeting_docx(
        cls, payload: ClientMeetingProtocol, output_path: str
    ) -> str:
        """Client meeting: identification header + two rep tables + content.

        Per user requirement 2026-04-21: clear identification of the
        client, EPAM reps, and client reps at the top of the document.
        """
        logger.info(f"Generating DOCX (client meeting): {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = Document()
        cls._set_default_margins(doc)

        # ----- Title block -----
        epam_hdr = doc.add_paragraph()
        epam_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = epam_hdr.add_run("EPAM")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_RED

        title_bits = [payload.title or "Протокол встречи с клиентом"]
        if payload.client_name:
            title_bits.append(payload.client_name)
        title_text = " — ".join(title_bits)

        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_before = Pt(18)
        title_para.paragraph_format.space_after = Pt(6)
        run = title_para.add_run(title_text)
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY

        if payload.meeting_date:
            date_para = doc.add_paragraph()
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_para.paragraph_format.space_after = Pt(18)
            run = date_para.add_run(f"Дата: {payload.meeting_date}")
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)
            run.font.color.rgb = cls.EPAM_DARK_GRAY

        # ----- Identification block -----
        cls._add_heading(doc, "ИДЕНТИФИКАЦИЯ ВСТРЕЧИ")

        client_line = doc.add_paragraph()
        run = client_line.add_run("Клиент: ")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY
        run = client_line.add_run(payload.client_name or "Не указан")
        run.font.name = cls.BODY_FONT
        run.font.size = Pt(11)

        # EPAM representatives
        cls._add_rep_subtable(
            doc, "Представители EPAM", payload.epam_representatives
        )
        # Client representatives
        cls._add_rep_subtable(
            doc, "Представители клиента", payload.client_representatives
        )

        # ----- Summary -----
        cls._add_heading(doc, "КРАТКОЕ СОДЕРЖАНИЕ")
        summary_text = payload.summary.strip() if payload.summary else "Нет"
        para = doc.add_paragraph(summary_text)
        para.paragraph_format.line_spacing = 1.15
        para.paragraph_format.space_after = Pt(12)
        for run in para.runs:
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)

        # ----- Topics -----
        cls._add_simple_item_list(doc, "ОБСУЖДАВШИЕСЯ ВОПРОСЫ", payload.topics)

        # ----- Client requests -----
        cls._add_simple_item_list(
            doc, "ЗАПРОСЫ / ОБРАЩЕНИЯ КЛИЕНТА", payload.client_requests
        )

        # ----- Commitments -----
        cls._add_heading(doc, "ОБЯЗАТЕЛЬСТВА")
        if not payload.commitments:
            para = doc.add_paragraph("Нет")
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)
                run.font.italic = True
                run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        else:
            for commit in payload.commitments:
                side_label = "EPAM" if commit.party == "epam" else "Клиент"
                bits: list[str] = [f"[{side_label}]", commit.text.strip()]
                suffix_parts: list[str] = []
                if commit.speaker:
                    suffix_parts.append(commit.speaker)
                if commit.deadline:
                    suffix_parts.append(f"срок: {commit.deadline}")
                if suffix_parts:
                    bits.append(f"({'; '.join(suffix_parts)})")
                text = " ".join(bits)
                para = doc.add_paragraph(text, style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

        # ----- Follow-ups -----
        cls._add_simple_item_list(
            doc, "ДЕЙСТВИЯ EPAM", payload.follow_ups
        )

        # ----- Parked questions (plain strings) -----
        cls._add_heading(doc, "ОТКРЫТЫЕ ВОПРОСЫ")
        if not payload.parked_questions:
            para = doc.add_paragraph("Нет")
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)
                run.font.italic = True
                run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        else:
            for question in payload.parked_questions:
                if not question or not question.strip():
                    continue
                para = doc.add_paragraph(question.strip(), style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

        cls._add_page_numbers_footer(doc)

        doc.save(str(output_path))
        logger.info(f"DOCX saved: {output_path}")
        return str(output_path)

    @classmethod
    def _add_rep_subtable(
        cls, doc: Document, title: str, reps: list[str]
    ) -> None:
        """Render a small subsection listing representatives.

        One row per rep; shows "Нет" when empty so the section is visible
        regardless of content.
        """
        sub = doc.add_paragraph()
        sub.paragraph_format.space_before = Pt(8)
        sub.paragraph_format.space_after = Pt(2)
        run = sub.add_run(title)
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY

        if not reps:
            para = doc.add_paragraph("Нет")
            para.paragraph_format.left_indent = Inches(0.25)
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)
                run.font.italic = True
                run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            return

        for rep in reps:
            if not rep or not rep.strip():
                continue
            para = doc.add_paragraph(rep.strip(), style="List Bullet")
            para.paragraph_format.left_indent = Inches(0.25)
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)

    @classmethod
    def _add_simple_item_list(
        cls,
        doc: Document,
        heading: str,
        items: list[DepartmentItem],
    ) -> None:
        """Render a flat bullet list of DepartmentItem records.

        Reused for client meeting topics / requests / follow-ups. Each
        item can optionally carry a speaker and deadline.
        """
        cls._add_heading(doc, heading)
        if not items:
            para = doc.add_paragraph("Нет")
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)
                run.font.italic = True
                run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            return

        for item in items:
            text = item.text.strip() if item.text else ""
            if not text:
                continue
            suffix_parts: list[str] = []
            if item.speaker:
                suffix_parts.append(item.speaker)
            if item.deadline:
                suffix_parts.append(f"срок: {item.deadline}")
            if suffix_parts:
                text = f"{text} ({'; '.join(suffix_parts)})"
            para = doc.add_paragraph(text, style="List Bullet")
            para.paragraph_format.left_indent = Inches(0.25)
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)

    # =================================================================
    # Interview (dual output)
    # =================================================================

    @classmethod
    def _format_interview_docx(
        cls, payload: InterviewProtocol, output_path: str
    ) -> str:
        """Write TWO DOCX files for an interview.

        The `output_path` holds the INTERNAL report (confidential
        scorecard for the hiring team). Alongside it we write
        ``{stem}_external.docx`` with the EXTERNAL developmental feedback
        for the candidate (no scores, no pass/fail).

        Returns the internal report path.
        """
        logger.info(f"Generating DOCX (interview): {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        external_path = output_path.with_name(
            f"{output_path.stem}_external{output_path.suffix}"
        )

        cls._write_interview_internal(payload, output_path)
        cls._write_interview_external(payload, external_path)

        logger.info(
            f"Interview DOCX saved: internal={output_path}, external={external_path}"
        )
        return str(output_path)

    @classmethod
    def _write_interview_internal(
        cls, payload: InterviewProtocol, output_path: Path
    ) -> None:
        """Confidential scorecard for the hiring team.

        CAUTION: contains concerns, technical/communication signals and
        open questions. MUST NOT be shared with the candidate.
        """
        doc = Document()
        cls._set_default_margins(doc)

        internal = payload.internal

        # Header
        epam_hdr = doc.add_paragraph()
        epam_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = epam_hdr.add_run("EPAM")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_RED

        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_before = Pt(18)
        run = title_para.add_run("ВНУТРЕННИЙ ОТЧЁТ ПО СОБЕСЕДОВАНИЮ")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY

        confid_para = doc.add_paragraph()
        confid_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        confid_para.paragraph_format.space_after = Pt(12)
        run = confid_para.add_run(
            "КОНФИДЕНЦИАЛЬНО — только для команды по найму"
        )
        run.font.name = cls.BODY_FONT
        run.font.size = Pt(11)
        run.font.italic = True
        run.font.color.rgb = cls.EPAM_RED

        # Candidate / role info block
        if internal.candidate_name or internal.role or payload.meeting_date:
            info_para = doc.add_paragraph()
            info_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            info_para.paragraph_format.space_after = Pt(12)
            bits: list[str] = []
            if internal.candidate_name:
                bits.append(f"Кандидат: {internal.candidate_name}")
            if internal.role:
                bits.append(f"Роль: {internal.role}")
            if payload.meeting_date:
                bits.append(f"Дата: {payload.meeting_date}")
            run = info_para.add_run(" | ".join(bits))
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)
            run.font.color.rgb = cls.EPAM_DARK_GRAY

        # Summary
        cls._add_heading(doc, "ОБЩАЯ ОЦЕНКА")
        summary_text = internal.summary.strip() if internal.summary else "Нет"
        para = doc.add_paragraph(summary_text)
        para.paragraph_format.line_spacing = 1.15
        for run in para.runs:
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)

        # Sections with evidence
        cls._add_observation_list(doc, "СИЛЬНЫЕ СТОРОНЫ", internal.strengths)
        cls._add_observation_list(doc, "ОПАСЕНИЯ / ЗОНЫ РИСКА", internal.concerns)
        cls._add_observation_list(
            doc, "ТЕХНИЧЕСКИЕ СИГНАЛЫ", internal.technical_signals
        )
        cls._add_observation_list(
            doc, "КОММУНИКАЦИЯ / ПОДАЧА", internal.communication_signals
        )

        # Open questions for next round
        cls._add_heading(doc, "ВОПРОСЫ ДЛЯ СЛЕДУЮЩЕГО ИНТЕРВЬЮ")
        if not internal.open_questions_for_next_round:
            para = doc.add_paragraph("Нет")
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)
                run.font.italic = True
                run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
        else:
            for question in internal.open_questions_for_next_round:
                if not question or not question.strip():
                    continue
                para = doc.add_paragraph(question.strip(), style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

        cls._add_page_numbers_footer(doc)
        doc.save(str(output_path))

    @classmethod
    def _write_interview_external(
        cls, payload: InterviewProtocol, output_path: Path
    ) -> None:
        """Developmental feedback for the candidate (EPAM-branded).

        Per Block 12 vision: a gift to every candidate regardless of
        outcome. NO scores, NO pass/fail, NO internal assessments —
        purely developmental.
        """
        doc = Document()
        cls._set_default_margins(doc)

        external = payload.external

        # Header
        epam_hdr = doc.add_paragraph()
        epam_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = epam_hdr.add_run("EPAM")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_RED

        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_before = Pt(18)
        title_para.paragraph_format.space_after = Pt(12)
        run = title_para.add_run("ОБРАТНАЯ СВЯЗЬ ПО СОБЕСЕДОВАНИЮ")
        run.font.name = cls.HEADING_FONT
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = cls.EPAM_DARK_GRAY

        # Personal greeting
        if external.candidate_name:
            greet_para = doc.add_paragraph()
            greet_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            greet_para.paragraph_format.space_after = Pt(12)
            run = greet_para.add_run(f"Уважаемый(ая) {external.candidate_name},")
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)
            run.font.color.rgb = cls.EPAM_DARK_GRAY

        intro = doc.add_paragraph(
            "Благодарим Вас за интервью с EPAM. Ниже — наблюдения "
            "о Вашей манере общения и сильных моментах, которые мы "
            "заметили. Этот документ носит развивающий характер и не "
            "содержит оценок или решения по найму."
        )
        intro.paragraph_format.line_spacing = 1.15
        intro.paragraph_format.space_after = Pt(12)
        for run in intro.runs:
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)

        # Speaking-style notes
        cls._add_heading(doc, "МАНЕРА РЕЧИ И ПОДАЧА")
        if not external.speaking_style_notes:
            para = doc.add_paragraph("Нет заметок.")
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)
                run.font.italic = True
        else:
            for note in external.speaking_style_notes:
                if not note or not note.strip():
                    continue
                para = doc.add_paragraph(note.strip(), style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

        # Strongest moments (positive reinforcement)
        cls._add_observation_list(
            doc, "ВАШИ СИЛЬНЫЕ МОМЕНТЫ", external.strongest_moments
        )

        # Areas to develop
        cls._add_heading(doc, "НАПРАВЛЕНИЯ ДЛЯ РОСТА")
        if not external.areas_to_develop:
            para = doc.add_paragraph("Нет рекомендаций.")
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)
                run.font.italic = True
        else:
            for area in external.areas_to_develop:
                if not area or not area.strip():
                    continue
                para = doc.add_paragraph(area.strip(), style="List Bullet")
                para.paragraph_format.left_indent = Inches(0.25)
                for run in para.runs:
                    run.font.name = cls.BODY_FONT
                    run.font.size = Pt(11)

        # Closing note
        if external.closing_note and external.closing_note.strip():
            closing = doc.add_paragraph()
            closing.paragraph_format.space_before = Pt(18)
            closing.paragraph_format.line_spacing = 1.15
            run = closing.add_run(external.closing_note.strip())
            run.font.name = cls.BODY_FONT
            run.font.size = Pt(11)
            run.font.italic = True
            run.font.color.rgb = cls.EPAM_DARK_GRAY

        cls._add_page_numbers_footer(doc)
        doc.save(str(output_path))

    @classmethod
    def _add_observation_list(
        cls, doc: Document, heading: str, observations: list
    ) -> None:
        """Render a list of InterviewObservation (observation + evidence quote).

        Observations without evidence are still rendered (verification
        pass should have dropped unsupported ones, so what remains is
        supported but may have been parsed without a literal quote).
        """
        cls._add_heading(doc, heading)
        if not observations:
            para = doc.add_paragraph("Нет")
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)
                run.font.italic = True
                run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            return

        for obs in observations:
            text = getattr(obs, "observation", "")
            if not text or not text.strip():
                continue
            para = doc.add_paragraph(text.strip(), style="List Bullet")
            para.paragraph_format.left_indent = Inches(0.25)
            for run in para.runs:
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(11)

            evidence = getattr(obs, "evidence", "")
            if evidence and evidence.strip():
                quote_para = doc.add_paragraph()
                quote_para.paragraph_format.left_indent = Inches(0.6)
                quote_para.paragraph_format.space_after = Pt(4)
                run = quote_para.add_run(f"«{evidence.strip()}»")
                run.font.name = cls.BODY_FONT
                run.font.size = Pt(10)
                run.font.italic = True
                run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
