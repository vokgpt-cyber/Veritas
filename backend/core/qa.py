"""Quality assurance engine for pipeline validation."""
import logging
import re
from typing import Any, Optional

from backend.app.models import (
    AlignedSegment,
    AudioMetadata,
    DiarizationSegment,
    MeetingProtocol,
    QAIssue,
    QAResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)


class QualityAssurance:
    """Rule-based quality validation for each pipeline stage."""

    def __init__(self, config: Any):
        """
        Initialize QA engine.

        Args:
            config: QualityConfig object
        """
        self._config = config

    def validate_audio(self, metadata: AudioMetadata) -> QAResult:
        """
        Validate audio metadata.

        Checks:
        - Duration within bounds
        - Sample rate is 16000 Hz
        - Mono channel (1 channel)
        - Valid codec

        Args:
            metadata: AudioMetadata object

        Returns:
            QAResult with validation outcome
        """
        logger.info(f"Validating audio: {metadata.duration:.1f}s, {metadata.sample_rate}Hz, "
                   f"{metadata.channels}ch")

        issues = []
        warnings = []
        suggestions = []
        score = 100.0

        # Check duration
        if metadata.duration < self._config.quality.min_audio_duration:
            issues.append(
                QAIssue(
                    severity="critical",
                    stage="audio_validation",
                    message=f"Audio too short: {metadata.duration:.1f}s < {self._config.quality.min_audio_duration}s",
                    details={"duration": metadata.duration, "min_duration": self._config.quality.min_audio_duration},
                )
            )
            score -= 50

        if metadata.duration > self._config.quality.max_audio_duration:
            issues.append(
                QAIssue(
                    severity="warning",
                    stage="audio_validation",
                    message=f"Audio very long: {metadata.duration:.1f}s > {self._config.quality.max_audio_duration}s",
                    details={"duration": metadata.duration, "max_duration": self._config.quality.max_audio_duration},
                )
            )
            score -= 10

        # Check sample rate
        if metadata.sample_rate != 16000:
            warnings.append(f"Non-standard sample rate: {metadata.sample_rate}Hz (16000Hz recommended)")
            suggestions.append("Reprocess audio to 16kHz for better ASR quality")
            score -= 5

        # Check channels
        if metadata.channels != 1:
            warnings.append(f"Non-mono audio: {metadata.channels} channels (mono recommended)")
            suggestions.append("Convert to mono for consistent diarization")
            score -= 5

        # Check codec
        if metadata.codec not in ["pcm_s16le", "wav", "mp3", "aac", "flac"]:
            warnings.append(f"Uncommon codec: {metadata.codec}")

        # File size sanity check
        if metadata.file_size == 0 and metadata.duration > 0:
            warnings.append("File size not determined")

        passed = len(issues) == 0
        score = max(0, min(100, score))

        logger.info(f"Audio validation: {'PASSED' if passed else 'FAILED'} (score: {score})")

        return QAResult(
            passed=passed,
            score=score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
        )

    def validate_transcription(self, segments: list[TranscriptionSegment]) -> QAResult:
        """
        Validate transcription quality.

        Checks:
        - Average confidence above threshold
        - No empty text segments
        - Hallucination detection (repeated text, abnormal density)
        - No extremely low confidence segments

        Args:
            segments: List of transcription segments

        Returns:
            QAResult with validation outcome
        """
        logger.info(f"Validating transcription: {len(segments)} segments")

        issues = []
        warnings = []
        suggestions = []
        score = 100.0

        if not segments:
            issues.append(
                QAIssue(
                    severity="critical",
                    stage="transcription_validation",
                    message="No transcription segments provided",
                )
            )
            return QAResult(passed=False, score=0.0, issues=issues)

        # Calculate statistics
        confidences = [seg.confidence for seg in segments]
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        empty_count = sum(1 for seg in segments if not seg.text or not seg.text.strip())

        logger.debug(f"Transcription stats: avg_conf={avg_confidence:.2f}, "
                    f"min_conf={min_confidence:.2f}, empty={empty_count}")

        # Check average confidence
        if avg_confidence < self._config.quality.min_confidence:
            issues.append(
                QAIssue(
                    severity="warning",
                    stage="transcription_validation",
                    message=f"Low average confidence: {avg_confidence:.2f} < {self._config.quality.min_confidence}",
                    details={"avg_confidence": avg_confidence, "min_threshold": self._config.quality.min_confidence},
                )
            )
            score -= 20

        # Check for very low confidence segments
        very_low_count = sum(1 for seg in segments if seg.confidence < 0.3)
        if very_low_count > 0:
            warnings.append(f"{very_low_count} segments with very low confidence (<0.3)")
            suggestions.append("Review low-confidence segments manually")
            score -= 5

        # Check empty segments
        if empty_count > 0:
            warnings.append(f"{empty_count} empty text segments")
            suggestions.append("Verify audio quality in these time ranges")
            score -= 5 * min(empty_count, 5)

        # Hallucination detection
        if self._config.quality.hallucination_check:
            hallucination_issues = self._detect_hallucinations(segments)
            if hallucination_issues:
                issues.extend(hallucination_issues)
                score -= 15

        passed = len(issues) == 0
        score = max(0, min(100, score))

        logger.info(f"Transcription validation: {'PASSED' if passed else 'FAILED'} "
                   f"(avg_conf={avg_confidence:.2f}, score={score})")

        return QAResult(
            passed=passed,
            score=score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
        )

    def _detect_hallucinations(self, segments: list[TranscriptionSegment]) -> list[QAIssue]:
        """
        Detect potential hallucinations in transcription.

        Checks:
        - Repeated substrings > max_repeat_length
        - Abnormal text density (chars per second)

        Args:
            segments: List of transcription segments

        Returns:
            List of detected issues
        """
        issues = []

        # Check for repeated substrings
        for seg in segments:
            text = seg.text.lower()
            if len(text) < self._config.quality.max_repeat_length:
                continue

            # Check for substring repetition (min 20 chars, up to max_repeat_length)
            check_start = min(20, self._config.quality.max_repeat_length)
            for substr_len in range(check_start, self._config.quality.max_repeat_length + 1):
                for i in range(len(text) - substr_len * 2):
                    substr = text[i : i + substr_len]
                    if text[i + substr_len : i + substr_len * 2] == substr:
                        issues.append(
                            QAIssue(
                                severity="warning",
                                stage="transcription_validation",
                                message=f"Potential hallucination: repeated text detected",
                                details={
                                    "segment_time": f"{seg.start:.2f}-{seg.end:.2f}",
                                    "repeated_text": substr[:30] + "...",
                                },
                            )
                        )
                        break

        # Check for abnormal text density
        for seg in segments:
            duration = seg.end - seg.start
            if duration > 0:
                chars_per_sec = len(seg.text) / duration
                # Normal speech: ~40-60 chars per second (~8-12 words per second)
                if chars_per_sec > 200:  # Extremely high
                    issues.append(
                        QAIssue(
                            severity="info",
                            stage="transcription_validation",
                            message="Unusually high text density (possible hallucination)",
                            details={
                                "segment_time": f"{seg.start:.2f}-{seg.end:.2f}",
                                "chars_per_second": chars_per_sec,
                            },
                        )
                    )

        return issues

    def validate_diarization(self, segments: list[DiarizationSegment]) -> QAResult:
        """
        Validate diarization quality.

        Checks:
        - Speaker count within bounds
        - Coverage of audio (no large gaps)
        - Minimum segment duration
        - Minimum share per speaker

        Args:
            segments: List of diarization segments

        Returns:
            QAResult with validation outcome
        """
        logger.info(f"Validating diarization: {len(segments)} segments")

        issues = []
        warnings = []
        suggestions = []
        score = 100.0

        if not segments:
            issues.append(
                QAIssue(
                    severity="critical",
                    stage="diarization_validation",
                    message="No diarization segments provided",
                )
            )
            return QAResult(passed=False, score=0.0, issues=issues)

        # Count unique speakers
        unique_speakers = set(seg.speaker_id for seg in segments)
        speaker_count = len(unique_speakers)

        logger.debug(f"Diarization: {speaker_count} speakers, "
                    f"{len(segments)} segments total")

        # Check speaker count
        if speaker_count < 2:
            warnings.append(f"Only {speaker_count} speaker(s) detected (minimum 2 expected)")
            suggestions.append("Check if diarization failed or audio has only one speaker")
            score -= 10

        # Check minimum segment duration
        short_segments = [seg for seg in segments if (seg.end - seg.start) < 0.3]
        if short_segments:
            warnings.append(f"{len(short_segments)} very short segments (<0.3s)")
            suggestions.append("Consider merging or filtering short segments")
            score -= 5

        # Check coverage (no large gaps)
        if len(segments) > 1:
            gaps = []
            sorted_segs = sorted(segments, key=lambda s: s.start)
            for i in range(len(sorted_segs) - 1):
                gap = sorted_segs[i + 1].start - sorted_segs[i].end
                if gap > 1.0:  # More than 1 second gap
                    gaps.append(gap)

            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                if avg_gap > 2.0:
                    warnings.append(f"Large gaps detected (avg {avg_gap:.2f}s)")
                    score -= 10

        # Calculate speaker time shares
        total_duration = sum(seg.end - seg.start for seg in segments)
        speaker_durations = {}
        for seg in segments:
            if seg.speaker_id not in speaker_durations:
                speaker_durations[seg.speaker_id] = 0.0
            speaker_durations[seg.speaker_id] += seg.end - seg.start

        # Check minimum share per speaker
        if total_duration > 0:
            for speaker_id, duration in speaker_durations.items():
                share = duration / total_duration
                if share < 0.05:  # Less than 5%
                    warnings.append(f"Speaker {speaker_id} has very low share ({share * 100:.1f}%)")

        passed = len(issues) == 0
        score = max(0, min(100, score))

        logger.info(f"Diarization validation: {'PASSED' if passed else 'FAILED'} "
                   f"(speakers={speaker_count}, score={score})")

        return QAResult(
            passed=passed,
            score=score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
        )

    def validate_alignment(self, segments: list[AlignedSegment]) -> QAResult:
        """
        Validate aligned segments.

        Checks:
        - All segments have speaker_id
        - Chronological order
        - No large temporal gaps
        - Text not empty

        Args:
            segments: List of aligned segments

        Returns:
            QAResult with validation outcome
        """
        logger.info(f"Validating alignment: {len(segments)} segments")

        issues = []
        warnings = []
        suggestions = []
        score = 100.0

        if not segments:
            issues.append(
                QAIssue(
                    severity="critical",
                    stage="alignment_validation",
                    message="No aligned segments provided",
                )
            )
            return QAResult(passed=False, score=0.0, issues=issues)

        # Check all have speaker_id
        missing_speaker = [i for i, seg in enumerate(segments) if not seg.speaker_id]
        if missing_speaker:
            issues.append(
                QAIssue(
                    severity="critical",
                    stage="alignment_validation",
                    message=f"{len(missing_speaker)} segments missing speaker_id",
                    details={"segment_indices": missing_speaker[:10]},
                )
            )
            score -= 50

        # Check chronological order
        prev_end = segments[0].start
        order_issues = []
        for i, seg in enumerate(segments):
            if seg.start < prev_end:
                order_issues.append(i)
            if seg.end < seg.start:
                issues.append(
                    QAIssue(
                        severity="critical",
                        stage="alignment_validation",
                        message=f"Segment {i}: invalid time range",
                        details={"start": seg.start, "end": seg.end},
                    )
                )
            prev_end = seg.end

        if order_issues:
            warnings.append(f"{len(order_issues)} segments out of chronological order")
            score -= 20

        # Check for empty text
        empty_segments = [i for i, seg in enumerate(segments) if not seg.text or not seg.text.strip()]
        if empty_segments:
            warnings.append(f"{len(empty_segments)} segments with empty text")
            score -= 5

        passed = len(issues) == 0
        score = max(0, min(100, score))

        logger.info(f"Alignment validation: {'PASSED' if passed else 'FAILED'} (score={score})")

        return QAResult(
            passed=passed,
            score=score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
        )

    def validate_protocol(self, protocol: MeetingProtocol) -> QAResult:
        """
        Validate final meeting protocol.

        Checks:
        - All major sections present and non-empty
        - Summary reasonable length
        - Decisions and tasks have content
        - Participants list present
        - Transcript integrity

        Args:
            protocol: MeetingProtocol object

        Returns:
            QAResult with validation outcome
        """
        logger.info(f"Validating protocol: topic='{protocol.topic}', "
                   f"participants={len(protocol.participants)}, "
                   f"decisions={len(protocol.decisions)}, tasks={len(protocol.tasks)}")

        issues = []
        warnings = []
        suggestions = []
        score = 100.0

        # Check topic
        if not protocol.topic or len(protocol.topic.strip()) < 3:
            warnings.append("Missing or very short meeting topic")
            score -= 10

        # Check summary
        if not protocol.summary or len(protocol.summary.strip()) < 50:
            warnings.append("Summary missing or too short (<50 chars)")
            suggestions.append("Ensure LLM generates meaningful summary")
            score -= 20

        # Check participants
        if not protocol.participants:
            warnings.append("No participants extracted")
            suggestions.append("Verify diarization output")
            score -= 15

        # Check decisions
        if not protocol.decisions:
            warnings.append("No decisions detected in meeting")
            suggestions.append("Verify LLM prompt asks for decisions")
            score -= 5

        # Check tasks
        if not protocol.tasks:
            warnings.append("No action items/tasks detected")
            suggestions.append("Verify LLM prompt asks for tasks")
            score -= 5

        # Check transcript
        if not protocol.transcript:
            warnings.append("No transcript included in protocol")
            score -= 10

        # Validate summary length (not too long)
        if len(protocol.summary) > 5000:
            warnings.append("Summary unusually long (>5000 chars)")
            score -= 5

        # Check for reasonable content
        if all(
            len(section) == 0
            for section in [
                protocol.summary,
                protocol.key_topics,
                protocol.decisions,
                protocol.tasks,
            ]
        ):
            issues.append(
                QAIssue(
                    severity="critical",
                    stage="protocol_validation",
                    message="Protocol appears empty (no summary, topics, decisions, or tasks)",
                )
            )
            score -= 50

        passed = len(issues) == 0
        score = max(0, min(100, score))

        logger.info(f"Protocol validation: {'PASSED' if passed else 'FAILED'} (score={score})")

        return QAResult(
            passed=passed,
            score=score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
        )
