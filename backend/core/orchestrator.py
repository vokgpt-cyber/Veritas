"""Main pipeline orchestrator for meeting processing."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from backend.app.config import AppConfig
from backend.app.models import (
    AlignedSegment,
    DiarizationSegment,
    MeetingJob,
    MeetingProtocol,
    PipelineState,
    QAResult,
    TranscriptionSegment,
)
from backend.core.audit import AuditAction, audit_log
from backend.core.cleanup import CleanupConfig, FileCleanup
from backend.core.encryption import EncryptionConfig, FileEncryptor
from backend.core.postprocessor import postprocess_transcript
from backend.core.vram_manager import VRAMManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main pipeline coordinator. Manages the full meeting processing flow.

    Pipeline order (MVP):
      Preprocess -> ASR -> Diarization -> Align -> Post-process -> QA
      -> Summarize -> Format

    This is the proven pre-5d order that produces the best transcription
    quality. VAD and adaptive profiles are removed — they added complexity
    without improving output on 8GB VRAM hardware.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize orchestrator with configuration.

        Args:
            config: Application configuration
        """
        self._config = config
        self._vram = VRAMManager()
        self._jobs: dict[str, MeetingJob] = {}
        self._logger = logging.getLogger("Orchestrator")

        # Security: encryption at rest and file cleanup
        self._encryptor = FileEncryptor()
        self._cleanup = FileCleanup(output_dir=config.output.output_dir)

        # Lazy-load engines to avoid import cycles
        self._audio = None
        self._asr = None
        self._asr_engine_name: Optional[str] = None  # Track currently-loaded ASR class
        self._diarization = None
        self._summarization = None
        self._aligner = None
        self._qa = None
        self._formatter = None
        # Last-run language detection result (LanguageMix or None).
        # Populated by _resolve_asr_engine when asr.engine == "auto" and
        # an audio path is provided. Used for audit/logging.
        self._last_language_mix = None
        self._load_persisted_jobs()

    @staticmethod
    def _required_vram_gb(engine: Any, label: str) -> float:
        """Return an engine VRAM requirement or fail with a useful message."""
        if engine is None:
            raise RuntimeError(
                f"{label} engine was not initialized. "
                "This usually means pipeline settings were changed while a job "
                "was starting. Retry after the settings screen is closed."
            )
        required = getattr(engine, "required_vram_gb", None)
        if required is None:
            raise RuntimeError(f"{label} engine does not declare required_vram_gb")
        return float(required)

    def _cleanup_loaded_engines_after_error(self) -> None:
        """Best-effort cleanup so failed jobs do not leave VRAM occupied."""
        for attr, label in (
            ("_asr", "ASR"),
            ("_diarization", "diarization"),
            ("_summarization", "summarization"),
        ):
            engine = getattr(self, attr, None)
            if engine is None:
                continue
            try:
                if getattr(engine, "is_loaded", False):
                    self._logger.info("Unloading %s engine after error", label)
                    engine.unload()
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Failed to unload %s engine after error: %s",
                    label,
                    exc,
                )
        try:
            self._vram.unregister_model()
            self._vram.release_all()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to release VRAM after error: %s", exc)

    def _resolve_asr_engine(
        self,
        audio_path: Optional[str] = None,
        meeting_type: Optional[Any] = None,
    ) -> str:
        """Resolve which ASR engine key to use for this job.

        If config.asr.engine is explicit (gigaam|hf-whisper|whisper|qwen|nemo),
        honor it and skip detection. If "auto" (the default), route by
        measured language mix of the preprocessed audio:

          * EN probability >= config.asr.english_threshold (default 0.20)
            -> "whisper" (faster-whisper base large-v3, same as EPAM IT team)
          * else -> "gigaam" (SOTA Russian, WER 2.6-8.4%)

        meeting_type is NO LONGER a routing signal for ASR (2026-04-23
        rewrite). It stays on MeetingJob because the summarization
        builder and DOCX formatter still dispatch on it. Rationale:
        language is a property of the AUDIO, not of the meeting
        category. User-confirmed case: admin meetings used to route to
        hf-whisper on a weak heuristic (admin likely has English) and
        quality suffered; GigaAM outperforms HFWhisper on any primarily-
        Russian content regardless of the calendar label.

        Args:
            audio_path: Preprocessed 16kHz mono WAV. If None (e.g. called
                before preprocessing finishes, or from a hypothetical
                dry-run), we fall back to "gigaam" — the safer default
                for EPAM's mostly-Russian content.
            meeting_type: Kept for logging/audit; no longer drives ASR.

        Returns:
            Engine key string for _instantiate_asr().
        """
        configured = self._config.asr.engine.lower()
        if configured != "auto":
            self._logger.info(
                "ASR engine explicitly configured: %s (skipping language "
                "detection)",
                configured,
            )
            return configured

        if audio_path is None:
            # No audio yet → Russian fallback. This branch is only hit
            # if someone calls _resolve_asr_engine very early (before
            # preprocessing). The main pipeline always has audio_path.
            self._logger.info(
                "ASR auto-routing requested with no audio path — defaulting "
                "to gigaam (Russian)"
            )
            return "gigaam"

        # Language-based auto routing.
        try:
            from backend.engine.language_detector import LanguageDetector
        except ImportError as exc:
            self._logger.warning(
                "LanguageDetector unavailable (%s) — defaulting to gigaam", exc
            )
            return "gigaam"

        threshold = float(getattr(self._config.asr, "english_threshold", 0.20))
        detector = LanguageDetector(
            device=self._config.asr.device if self._config.asr.device != "auto"
            else "cuda",
            compute_type="float16",
        )
        try:
            mix = detector.detect(audio_path)
            self._logger.info(
                "Language mix: %s | threshold=%.2f english-heavy=%s",
                mix.summary(),
                threshold,
                mix.is_english_heavy(threshold),
            )
            # Keep detection result on the job for UI/audit.
            self._last_language_mix = mix
            if mix.is_english_heavy(threshold):
                return "whisper"
            return "gigaam"
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Language detection failed (%s) -- defaulting to GigaAM",
                exc,
            )
            return "gigaam"
        finally:
            # Always free the tiny model before the heavy ASR loads.
            detector.unload()

    def _instantiate_asr(self, asr_engine: str) -> None:
        """(Re)instantiate self._asr for a given engine key. No-op if
        the correct engine is already instantiated. Does NOT call load()."""
        if self._asr is not None and self._asr_engine_name == asr_engine:
            return

        # If we're swapping engines, drop the old reference first. The
        # previous engine's .unload() is already called in process_meeting
        # after ASR stage completes, so at this point it should be idle.
        if self._asr is not None and self._asr_engine_name != asr_engine:
            self._logger.info(
                f"Swapping ASR engine: {self._asr_engine_name} -> {asr_engine}"
            )
            self._asr = None

        if asr_engine == "gigaam":
            from backend.engine.gigaam_asr import GigaAMASREngine
            self._asr = GigaAMASREngine(self._config)
            self._asr_engine_name = "gigaam"
            self._logger.info("Using GigaAM v3_e2e_rnnt ASR engine")
        elif asr_engine == "hf-whisper":
            from backend.engine.hf_whisper_asr import HFWhisperASREngine
            self._asr = HFWhisperASREngine(self._config)
            self._asr_engine_name = "hf-whisper"
            self._logger.info("Using HuggingFace transformers Whisper ASR engine")
        elif asr_engine == "qwen":
            from backend.engine.qwen_asr import Qwen3ASREngine
            self._asr = Qwen3ASREngine(self._config)
            self._asr_engine_name = "qwen"
            self._logger.info("Using Qwen3-ASR engine")
        elif asr_engine == "whisper":
            from backend.engine.whisper_asr import FasterWhisperASREngine
            self._asr = FasterWhisperASREngine(self._config)
            self._asr_engine_name = "whisper"
            self._logger.info("Using faster-whisper ASR engine")
        elif asr_engine == "whisperx":
            # T3.6 (2026-04-23) pilot. Optional dep, NEVER auto-routed
            # — only used when config.asr.engine = "whisperx" explicitly.
            from backend.engine.whisperx_asr import WhisperXASREngine
            self._asr = WhisperXASREngine(self._config)
            self._asr_engine_name = "whisperx"
            self._logger.info(
                "Using WhisperX ASR engine (Whisper + wav2vec2 align)"
            )
        else:
            from backend.engine.asr import NeMoASREngine
            self._asr = NeMoASREngine(self._config)
            self._asr_engine_name = "nemo"
            self._logger.info("Using NeMo Conformer ASR engine")

    def _ensure_engines_loaded(self, meeting_type: Optional[Any] = None) -> None:
        """Lazy-load non-ASR engine dependencies and instantiate a
        placeholder ASR.

        ASR selection is finalised AFTER preprocessing via a second
        `_instantiate_asr(self._resolve_asr_engine(preprocessed_path, ...))`
        call inside `process_meeting` — by that point we have audio to
        run language detection on. Here we just pick a safe default
        (gigaam if "auto", or the explicit config value) so the
        instance attribute is set up.

        Instantiation is cheap (no model load); the real VRAM hit
        happens later in `.load()`. Swapping the instance after
        preprocessing is free."""
        from backend.core.audio import AudioPreprocessor
        from backend.core.aligner import TranscriptAligner
        from backend.core.qa import QualityAssurance
        from backend.core.formatter import ProtocolFormatter
        from backend.engine.summarization import SummarizationEngine

        if self._audio is None:
            self._audio = AudioPreprocessor()

        # Pre-preprocessing: pick placeholder ASR (gigaam if "auto",
        # else the explicit config value). Real language-based routing
        # happens after we have preprocessed audio — see process_meeting.
        if self._asr is None or self._asr_engine_name is not None:
            self._instantiate_asr(
                self._resolve_asr_engine(audio_path=None, meeting_type=meeting_type)
            )

        if self._diarization is None:
            # Select diarization engine based on config. Production quality
            # policy: pyannote Community-1 is the only supported diarization
            # engine. Do not silently fall back to weaker engines because
            # speaker attribution quality is part of the legal record.
            diarization_engine = self._config.diarization.engine.lower()
            if diarization_engine != "pyannote":
                raise RuntimeError(
                    "Unsupported diarization engine. VERITAS production pipeline "
                    "requires pyannote speaker-diarization-community-1."
                )

            from backend.engine.pyannote_diarization import PyannoteDiarizationEngine

            self._diarization = PyannoteDiarizationEngine(self._config)
            self._logger.info("Using pyannote-audio diarization engine")

        if self._summarization is None:
            self._summarization = SummarizationEngine(self._config)
        if self._aligner is None:
            self._aligner = TranscriptAligner()
        if self._qa is None:
            self._qa = QualityAssurance(self._config)
        if self._formatter is None:
            self._formatter = ProtocolFormatter()

    async def process_meeting(
        self,
        audio_path: str,
        job: MeetingJob,
        on_progress: Optional[Callable] = None,
        expected_speakers: Optional[int] = None,
        expected_speakers_min: Optional[int] = None,
        expected_speakers_max: Optional[int] = None,
        enable_llm_correction: Optional[bool] = None,
        enable_polish_pass: Optional[bool] = None,
        asr_engine_override: Optional[str] = None,
    ) -> Optional[MeetingProtocol]:
        """
        Process meeting through full pipeline.

        Pipeline stages with progress % (MVP order — proven pre-5d flow):
        1. PREPROCESSING (0-5%): Audio conversion + validation
        2. TRANSCRIBING (5-45%): ASR (Whisper large-v3)
        3. DIARIZING (45-60%): Speaker segmentation
        4. ALIGNING (60-65%): Merge transcription + diarization
        5. POST-PROCESSING (65-68%): Filler removal, sentence cleanup
        6. QA_VALIDATING (68-70%): Validate transcript quality
        7. SUMMARIZING (70-92%): LLM 1-paragraph summary
        8. FORMATTING (92-100%): Generate DOCX

        Args:
            audio_path: Path to audio file
            job: Meeting job to process
            on_progress: Callback function called with updated job
            expected_speakers: Optional expected speaker count from user

        Returns:
            Completed MeetingProtocol or None if failed
        """
        self._ensure_engines_loaded(meeting_type=job.meeting_type)
        self._logger.info(
            f"Meeting {job.id}: type={job.meeting_type.value if hasattr(job.meeting_type, 'value') else job.meeting_type}, "
            f"ASR engine={self._asr_engine_name}"
        )
        self._jobs[job.id] = job

        try:
            # Stage 1: PREPROCESSING
            self._update_job(
                job, PipelineState.PREPROCESSING, 0.0, "Preprocessing audio"
            )
            if on_progress:
                on_progress(job)

            preprocessed_path, audio_metadata = await self._run_with_retry(
                lambda: self._audio.process(audio_path, str(self.get_meeting_dir(job.id))),
                job,
                "PREPROCESSING",
            )

            if preprocessed_path is None:
                raise RuntimeError("Preprocessing failed")

            # Validate audio
            qa_audio = self._qa.validate_audio(audio_metadata)
            if not qa_audio.passed:
                raise RuntimeError(f"Audio validation failed: {qa_audio.issues[0].message if qa_audio.issues else 'Unknown'}")

            # Re-resolve ASR engine now that we have preprocessed audio.
            # Per-job override (UI checkbox) wins over auto-routing if
            # the user explicitly picked an engine for this meeting.
            if asr_engine_override:
                resolved_engine = asr_engine_override.lower()
                self._logger.info(
                    "ASR engine forced by per-job override: %s",
                    resolved_engine,
                )
            elif self._asr is not None and self._asr_engine_name is None:
                # Test harnesses and a few diagnostic scripts inject a mock
                # ASR engine directly. Preserve that engine instead of
                # replacing it with the config-derived class.
                resolved_engine = None
                self._logger.info(
                    "Using pre-initialized ASR engine instance: %s",
                    self._asr.__class__.__name__,
                )
            else:
                # When config.asr.engine == "auto", language detection
                # on the audio picks gigaam (Russian) or whisper
                # (English >= threshold). When config is explicit, this
                # is a no-op (same engine as _ensure_engines_loaded
                # already instantiated).
                resolved_engine = self._resolve_asr_engine(
                    audio_path=preprocessed_path,
                    meeting_type=job.meeting_type,
                )
            if resolved_engine is not None and resolved_engine != self._asr_engine_name:
                self._logger.info(
                    "Re-routing ASR post-detection: %s -> %s",
                    self._asr_engine_name,
                    resolved_engine,
                )
                self._instantiate_asr(resolved_engine)
            final_asr_name = self._asr_engine_name or self._asr.__class__.__name__
            self._logger.info(
                "Meeting %s: final ASR engine = %s (meeting_type=%s)",
                job.id,
                final_asr_name,
                job.meeting_type.value if hasattr(job.meeting_type, "value")
                else job.meeting_type,
            )
            asr_engine = self._asr
            asr_required_vram = self._required_vram_gb(asr_engine, "ASR")

            # Stage 2: TRANSCRIBING (ASR first — produces best quality)
            self._update_job(
                job, PipelineState.TRANSCRIBING, 5.0, "Transcribing audio"
            )
            if on_progress:
                on_progress(job)

            # Check VRAM before loading ASR
            if not await self._vram.wait_for_available(asr_required_vram):
                raise RuntimeError(
                    f"Insufficient VRAM for ASR ({asr_required_vram}GB)"
                )

            # Load the selected quality ASR engine.
            # If it cannot load, show a clear error instead of switching
            # to another engine behind the operator's back.
            try:
                asr_meeting_type = (
                    job.meeting_type.value
                    if hasattr(job.meeting_type, "value")
                    else str(job.meeting_type) if job.meeting_type else None
                )
            except Exception:
                asr_meeting_type = None
            asr_hot_words = (
                str(getattr(job, "court_dictionary", "") or "")[:1200]
            )

            try:
                setattr(asr_engine, "_current_meeting_type", asr_meeting_type)
                setattr(asr_engine, "_current_hot_words", asr_hot_words)
                asr_engine.load()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "ASR engine %r load failed: %s",
                    final_asr_name, exc,
                )
                raise RuntimeError(
                    f"ASR engine {final_asr_name!r} failed to load. "
                    "VERITAS did not switch to another ASR engine automatically."
                ) from exc

            self._vram.register_model("ASREngine")

            try:
                asr_meeting_type = (
                    job.meeting_type.value
                    if hasattr(job.meeting_type, "value")
                    else str(job.meeting_type) if job.meeting_type else None
                )
            except Exception:
                asr_meeting_type = None
            try:
                setattr(asr_engine, "_current_meeting_type", asr_meeting_type)
                if getattr(job, "court_dictionary", ""):
                    setattr(
                        asr_engine,
                        "_current_hot_words",
                        str(job.court_dictionary)[:1200],
                    )
                else:
                    setattr(asr_engine, "_current_hot_words", "")
            except Exception:
                pass

            def transcription_progress(progress: float, message: str = "") -> None:
                self._update_job(
                    job, PipelineState.TRANSCRIBING, 5.0 + progress * 40.0, message or None
                )
                if on_progress:
                    on_progress(job)

            transcription = await self._run_with_retry(
                lambda: asr_engine.process(
                    preprocessed_path, progress_callback=transcription_progress
                ),
                job,
                "TRANSCRIBING",
            )

            if not transcription:
                raise RuntimeError("Transcription failed")

            # Save intermediate results
            self._save_intermediate(job.id, "transcription", transcription)

            # Validate transcription
            qa_trans = self._qa.validate_transcription(transcription)
            if not qa_trans.passed:
                self._logger.warning(f"Transcription QA warning: {qa_trans.issues}")

            # Unload ASR and release VRAM
            self._logger.info("Unloading ASR engine...")
            asr_engine.unload()
            self._logger.info("ASR unloaded, releasing VRAM...")
            self._vram.unregister_model()
            self._vram.release_all()
            self._logger.info("VRAM released, starting diarization...")

            # Stage 3: DIARIZING (after ASR)
            self._update_job(job, PipelineState.DIARIZING, 45.0, "Diarizing speakers")
            if on_progress:
                on_progress(job)

            # Check VRAM before loading Diarization
            diarization_engine = self._diarization
            diarization_required_vram = self._required_vram_gb(
                diarization_engine,
                "Diarization",
            )
            if not await self._vram.wait_for_available(diarization_required_vram):
                raise RuntimeError(
                    f"Insufficient VRAM for diarization ({diarization_required_vram}GB)"
                )

            diarization_engine.load()
            self._vram.register_model("DiarizationEngine")

            def diarization_progress(progress: float, message: str = "") -> None:
                self._update_job(
                    job, PipelineState.DIARIZING, 45.0 + progress * 15.0, message or None
                )
                if on_progress:
                    on_progress(job)

            # Resolve speaker constraints in order of precedence:
            # 1. Explicit min+max (T3.1: bucket UI sends these directly)
            # 2. Legacy expected_speakers single value (widen ±1)
            # 3. Config defaults (broad pyannote range)
            min_spk = self._config.diarization.min_speakers
            max_spk = self._config.diarization.max_speakers
            if expected_speakers_min is not None or expected_speakers_max is not None:
                if expected_speakers_min is not None:
                    min_spk = max(1, expected_speakers_min)
                if expected_speakers_max is not None:
                    max_spk = expected_speakers_max
                if min_spk > max_spk:
                    # Defensive: shouldn't happen if upload validator did its job.
                    self._logger.warning(
                        "expected_speakers_min (%d) > _max (%d); swapping",
                        min_spk, max_spk,
                    )
                    min_spk, max_spk = max_spk, min_spk
            elif expected_speakers is not None and expected_speakers >= 2:
                # Constrain to narrow range around expected
                min_spk = max(2, expected_speakers - 1)
                max_spk = expected_speakers + 2

            # Pass meeting_type so the diarization engine can apply
            # type-specific tuning. Currently court_hearing uses lower
            # clustering threshold to keep same-gender lawyers separate
            # (see PyannoteDiarizationEngine._apply_meeting_type_overrides).
            # Other engines accept the kwarg via **kwargs and ignore it.
            meeting_type_str: Optional[str] = None
            try:
                meeting_type_str = (
                    job.meeting_type.value
                    if hasattr(job.meeting_type, "value")
                    else str(job.meeting_type) if job.meeting_type else None
                )
            except Exception:
                meeting_type_str = None

            diarization = await self._run_with_retry(
                lambda: diarization_engine.process(
                    preprocessed_path,
                    None,  # No VAD pre-filtering — diarization handles full audio
                    min_speakers=min_spk,
                    max_speakers=max_spk,
                    progress_callback=diarization_progress,
                    meeting_type=meeting_type_str,
                ),
                job,
                "DIARIZING",
            )

            if not diarization:
                raise RuntimeError("Diarization failed")

            # Temporal smoothing: merge short gaps and remove micro-segments
            diarization = self._smooth_diarization(diarization)

            # Save intermediate results
            self._save_intermediate(job.id, "diarization", diarization)

            # Validate diarization
            qa_diar = self._qa.validate_diarization(diarization)
            if not qa_diar.passed:
                self._logger.warning(f"Diarization QA warning: {qa_diar.issues}")

            # Count detected speakers
            unique_speakers = set()
            for seg in diarization:
                sid = seg.speaker_id if hasattr(seg, "speaker_id") else seg.get("speaker_id")
                unique_speakers.add(sid)
            n_speakers = len(unique_speakers)
            self._logger.info(f"Diarization detected {n_speakers} speakers")

            # Unload Diarization and release VRAM
            diarization_engine.unload()
            self._vram.unregister_model()
            self._vram.release_all()

            # Stage 4: ALIGNING
            self._update_job(job, PipelineState.ALIGNING, 60.0, "Aligning transcript")
            if on_progress:
                on_progress(job)

            aligned = await self._run_with_retry(
                lambda: self._aligner.align(
                    transcription,
                    diarization,
                    diarization_split=meeting_type_str == "court_hearing",
                ),
                job,
                "ALIGNING",
            )

            if not aligned:
                raise RuntimeError("Alignment failed")

            # Save intermediate results (raw alignment)
            self._save_intermediate(job.id, "aligned_raw", aligned)

            # Voice matching (Block 7, Sprint 2026-04-30, task #38).
            # For each speaker_id present in the alignment, compute a
            # representative embedding from the longest contiguous slice,
            # match against the encrypted voice DB, and replace the
            # SPEAKER_X label with the matched profile name. Runs on
            # CPU (a few seconds per speaker, lightweight). Failures
            # are non-fatal — keeps SPEAKER_X labels and continues.
            try:
                aligned = self._apply_voice_matching(
                    aligned,
                    preprocessed_path,
                    meeting_type=meeting_type_str,
                )
            except Exception as vm_err:  # noqa: BLE001
                self._logger.warning(
                    f"Voice matching failed: {vm_err}; keeping "
                    f"SPEAKER_X labels"
                )

            # Validate alignment
            qa_align = self._qa.validate_alignment(aligned)
            if not qa_align.passed:
                self._logger.warning(f"Alignment QA warning: {qa_align.issues}")

            # Stage 5: POST-PROCESSING (filler removal, sentence cleanup)
            self._update_job(
                job, PipelineState.ALIGNING, 65.0, "Post-processing transcript"
            )
            if on_progress:
                on_progress(job)

            pp_cfg = getattr(self._config, "postprocessing", None)
            aligned, detected_language = postprocess_transcript(
                aligned,
                restore_punctuation_enabled=bool(
                    pp_cfg and pp_cfg.restore_punctuation
                ),
                remove_fillers=bool(pp_cfg.remove_fillers) if pp_cfg else True,
                remove_repetitions=bool(pp_cfg.remove_repetitions) if pp_cfg else True,
                capitalize_sentences_enabled=bool(pp_cfg.capitalize_sentences) if pp_cfg else True,
                normalize_numbers_enabled=bool(
                    getattr(pp_cfg, "normalize_numbers", True)
                ) if pp_cfg else True,
                filter_hallucinations_enabled=bool(
                    getattr(pp_cfg, "filter_hallucinations", True)
                ) if pp_cfg else True,
            )
            self._logger.info(
                f"Post-processing complete: {len(aligned)} segments, "
                f"language: {detected_language}"
            )

            # T3.4 (2026-04-23): optional LLM post-correction pass.
            # Per-job UI override wins over the meeting-type-aware default.
            # When enabled, runs Gemma over batches of turns with a strict
            # "fix only obvious ASR errors, never paraphrase" prompt.
            # Reuses the summarization engine's Ollama client.
            #
            # 2026-04-28: court_hearing default flipped to ON. Bench
            # showed GigaAM at 25.7% WER on Russian legal speech — too
            # high to ship un-corrected. UI-side this is also auto-
            # checked when the user picks Судебное заседание; this
            # branch is the safety net for API-direct callers and
            # older frontends.
            if enable_llm_correction is not None:
                llm_correction_enabled = enable_llm_correction
            else:
                config_on = bool(
                    pp_cfg and getattr(pp_cfg, "llm_correction", False)
                )
                if meeting_type_str == "court_hearing":
                    config_on = False
                llm_correction_enabled = config_on
            if (
                llm_correction_enabled
                and self._config.summarization.enabled
            ):
                self._update_job(
                    job, PipelineState.ALIGNING, 67.0,
                    "LLM-correcting transcript",
                )
                if on_progress:
                    on_progress(job)
                try:
                    from backend.core.transcript_corrector import (
                        correct_transcript,
                    )
                    self._save_intermediate(
                        job.id,
                        "aligned_before_llm_correction",
                        aligned,
                    )
                    correction_engine = self._summarization
                    if correction_engine is None:
                        raise RuntimeError("Summarization engine was not initialized")
                    correction_engine.load()  # idempotent
                    aligned, corr_stats = correct_transcript(
                        aligned,
                        correction_engine,
                        chunk_chars=int(
                            getattr(pp_cfg, "llm_correction_chunk_chars", 4000)
                        ),
                        min_overlap=float(
                            getattr(pp_cfg, "llm_correction_min_overlap", 0.70)
                        ),
                    )
                    self._logger.info(
                        "LLM correction: %d/%d turns changed",
                        corr_stats.corrected_turns, corr_stats.total_turns,
                    )
                    self._save_intermediate(
                        job.id,
                        "llm_correction_stats",
                        corr_stats.__dict__,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "LLM correction failed (%s); keeping uncorrected "
                        "transcript", exc,
                    )

            # Save cleaned (and optionally LLM-corrected) alignment
            self._save_intermediate(job.id, "aligned", aligned)

            # Stage 6: QA_VALIDATING (transcript)
            self._update_job(
                job, PipelineState.QA_VALIDATING, 68.0, "Validating transcript"
            )
            if on_progress:
                on_progress(job)

            # Stage 7: SUMMARIZING (skip if disabled in config)
            if self._config.summarization.enabled:
                self._update_job(job, PipelineState.SUMMARIZING, 70.0, "Summarizing meeting")
                if on_progress:
                    on_progress(job)

                # Ollama manages its own VRAM — no VRAMManager needed
                summarization_engine = self._summarization
                if summarization_engine is None:
                    raise RuntimeError("Summarization engine was not initialized")
                summarization_engine.load()

                def summarization_progress(progress: float, message: str = "") -> None:
                    self._update_job(
                        job, PipelineState.SUMMARIZING, 70.0 + progress * 22.0, message or None
                    )
                    if on_progress:
                        on_progress(job)

                # Pass user-provided meeting context to LLM
                meeting_context = job.context or ""
                if meeting_context:
                    self._logger.info(f"Injecting meeting context ({len(meeting_context)} chars) into summarization")
                else:
                    self._logger.info("No meeting context provided")

                protocol_result = await self._run_with_retry(
                    lambda: summarization_engine.process(
                        aligned,
                        language=detected_language,
                        progress_callback=summarization_progress,
                        meeting_context=meeting_context,
                        meeting_type=job.meeting_type,
                        enable_polish_pass=enable_polish_pass,
                    ),
                    job,
                    "SUMMARIZING",
                )

                if protocol_result is None:
                    raise RuntimeError("Summarization failed")

                # ProtocolResult wraps a type-specific payload. Some tests
                # and legacy summarization paths still return MeetingProtocol
                # directly, so accept both shapes.
                protocol = getattr(protocol_result, "payload", protocol_result)

                # Lightweight completeness signal for admin meetings. This
                # does not block delivery; it writes an audit JSON so we can
                # spot sparse protocols after real runs and decide whether
                # to re-run with larger context/tokens or a different prompt.
                try:
                    meeting_type_value = (
                        job.meeting_type.value
                        if hasattr(job.meeting_type, "value")
                        else str(job.meeting_type)
                    )
                    if meeting_type_value == "administrative":
                        buckets = {
                            "items": len(getattr(protocol, "items", []) or []),
                            "decisions": len(getattr(protocol, "decisions", []) or []),
                            "tasks": len(getattr(protocol, "tasks", []) or []),
                            "open_questions": len(
                                getattr(protocol, "open_questions", []) or []
                            ),
                            "risks": len(getattr(protocol, "risks", []) or []),
                        }
                        total_items = buckets["items"] or (
                            buckets["decisions"]
                            + buckets["tasks"]
                            + buckets["open_questions"]
                            + buckets["risks"]
                        )
                        duration_s = 0.0
                        if aligned:
                            duration_s = max(
                                0.0,
                                float(getattr(aligned[-1], "end", 0.0))
                                - float(getattr(aligned[0], "start", 0.0)),
                            )
                        expected_min = max(8, int((duration_s / 60.0) / 5.0))
                        sparse = duration_s >= 1800 and total_items < expected_min
                        quality = {
                            "meeting_type": meeting_type_value,
                            "duration_s": duration_s,
                            "counts": buckets,
                            "total_items": total_items,
                            "expected_min_items": expected_min,
                            "sparse_protocol_warning": sparse,
                        }
                        self._save_intermediate(
                            job.id, "protocol_quality", quality
                        )
                        if sparse:
                            self._logger.warning(
                                "Admin protocol may be sparse: %d items for %.1f min "
                                "(expected at least %d)",
                                total_items, duration_s / 60.0, expected_min,
                            )
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "Protocol quality audit failed: %s", exc
                    )

                # Save intermediate results (JSON serialisation works for
                # any pydantic payload).
                self._save_intermediate(job.id, "protocol", protocol)

                # Ollama unload (sends keep_alive=0 to free GPU memory)
                summarization_engine.unload()

                # Protocol QA — only the legacy MeetingProtocol schema is
                # understood by QA. For type-specific payloads we skip QA
                # for now (the per-type verification pass in the builder
                # replaces it).
                if isinstance(protocol, MeetingProtocol):
                    qa_protocol = self._qa.validate_protocol(protocol)
                    if not qa_protocol.passed:
                        self._logger.warning(
                            f"Protocol QA warning: {qa_protocol.issues}"
                        )
            else:
                # Transcript-only mode: build protocol without LLM summary
                self._logger.info("Summarization disabled — transcript-only output")
                from backend.app.models import Participant
                speakers: dict[str, Participant] = {}
                total_duration = sum(seg.end - seg.start for seg in aligned)
                speaker_durations: dict[str, float] = {}
                for seg in aligned:
                    if seg.speaker_id not in speakers:
                        speakers[seg.speaker_id] = Participant(
                            speaker_id=seg.speaker_id,
                            speaker_name=seg.speaker_name,
                        )
                    speaker_durations[seg.speaker_id] = (
                        speaker_durations.get(seg.speaker_id, 0.0) + seg.end - seg.start
                    )
                participants = list(speakers.values())
                for p in participants:
                    dur = speaker_durations.get(p.speaker_id, 0.0)
                    p.speaking_time = dur
                    if total_duration > 0:
                        p.speaking_share = round(dur / total_duration * 100, 1)

                protocol = MeetingProtocol(
                    topic="",
                    participants=participants,
                    summary="",
                    key_topics=[],
                    decisions=[],
                    tasks=[],
                    open_questions=[],
                    transcript=aligned,
                )
                self._save_intermediate(job.id, "protocol", protocol)

            # Stage 8: FORMATTING
            self._update_job(job, PipelineState.FORMATTING, 92.0, "Formatting output")
            if on_progress:
                on_progress(job)

            await self._run_with_retry(
                lambda: self._formatter.format_docx(
                    protocol, str(self.get_meeting_dir(job.id) / f"{job.id}.docx")
                ),
                job,
                "FORMATTING",
            )

            # Also save JSON format
            await self._run_with_retry(
                lambda: self._formatter.format_json(
                    protocol, str(self.get_meeting_dir(job.id) / f"{job.id}.json")
                ),
                job,
                "FORMATTING",
            )

            # Archive results to data/archive/ for easy access
            try:
                self._archive_results(job)
            except Exception as archive_err:
                self._logger.warning(f"Results archiving failed: {archive_err}")

            # Completion
            self._update_job(
                job, PipelineState.COMPLETED, 100.0, "Meeting processing completed"
            )
            if on_progress:
                on_progress(job)

            # Cleanup intermediate files after successful processing
            try:
                self._cleanup.cleanup_meeting_intermediates(job.id)
            except Exception as cleanup_err:
                self._logger.warning(f"Post-processing cleanup failed: {cleanup_err}")

            # Audit log successful completion
            elapsed = (datetime.utcnow() - job.created_at).total_seconds()
            audit_log(
                action=AuditAction.MEETING_PROCESS_COMPLETE,
                resource_type="meeting",
                resource_id=job.id,
                details={
                    "elapsed_seconds": round(elapsed, 1),
                    "n_speakers": n_speakers,
                },
            )

            self._logger.info(f"Successfully processed meeting {job.id}")
            return protocol

        except Exception as e:
            self._logger.error(f"Pipeline error for job {job.id}: {e}", exc_info=True)
            self._cleanup_loaded_engines_after_error()
            job.error = str(e)
            job.retry_count += 1

            self._update_job(
                job, PipelineState.ERROR, job.progress, f"Failed: {str(e)}"
            )

            audit_log(
                action=AuditAction.MEETING_PROCESS_FAIL,
                resource_type="meeting",
                resource_id=job.id,
                success=False,
                error=str(e),
            )

            if on_progress:
                on_progress(job)

            return None

    def _update_job(
        self,
        job: MeetingJob,
        state: PipelineState,
        progress: float,
        stage: Optional[str],
    ) -> None:
        """
        Update job state and notify.

        Args:
            job: Job to update
            state: New pipeline state
            progress: Progress percentage (0-100)
            stage: Current stage name
        """
        job.state = state
        job.progress = min(100.0, max(0.0, progress))
        job.updated_at = datetime.utcnow()

        if stage:
            job.current_stage = stage

        # Estimate ETA based on progress rate
        if progress > 0 and progress < 100:
            elapsed = (job.updated_at - job.created_at).total_seconds()
            if elapsed > 0:
                rate = progress / elapsed
                remaining = (100 - progress) / rate if rate > 0 else None
                job.eta_seconds = remaining

        self._logger.debug(
            f"Job {job.id}: {state.value} - {progress:.1f}% ({stage})"
        )
        self._persist_job(job)

    async def _run_with_retry(
        self,
        func: Callable[[], Any],
        job: MeetingJob,
        stage_name: str,
        max_retries: int = 2,
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Async or sync function to execute
            job: Associated job
            stage_name: Stage name for logging
            max_retries: Maximum retry attempts

        Returns:
            Function result or None if all retries failed
        """
        import asyncio
        import inspect

        for attempt in range(max_retries + 1):
            try:
                result = func()
                if inspect.iscoroutine(result):
                    result = await result
                return result

            except Exception as e:
                self._logger.error(
                    f"{stage_name} attempt {attempt + 1}/{max_retries + 1} failed: {e}"
                )

                if attempt < max_retries:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    self._logger.info(f"Retrying {stage_name} in {wait_time}s...")

                    # Clean up CUDA memory between retries — OOM fragments
                    # memory, and retrying without cleanup always fails again.
                    try:
                        import gc
                        gc.collect()
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            self._logger.info("CUDA cache cleared between retries")
                    except Exception:
                        pass

                    await asyncio.sleep(wait_time)
                else:
                    self._logger.error(f"{stage_name} failed after {max_retries + 1} attempts")
                    return None

        return None

    def _smooth_diarization(
        self,
        segments: list[DiarizationSegment],
        min_duration_s: float = 0.15,
        merge_gap_s: float = 0.3,
    ) -> list[DiarizationSegment]:
        """Temporal smoothing for diarization output.

        Post-processes pyannote segments to:
          1. Remove micro-segments shorter than min_duration_s (noise).
          2. Merge consecutive same-speaker segments separated by gaps
             shorter than merge_gap_s (reduces fragmentation).

        This typically reduces DER by 5-10% on fragmented output by
        eliminating rapid false speaker switches.

        Args:
            segments: Raw diarization segments, sorted by start time.
            min_duration_s: Minimum segment duration to keep (default 150ms).
            merge_gap_s: Maximum gap between same-speaker segments to merge
                         (default 300ms).

        Returns:
            Smoothed list of DiarizationSegment.
        """
        if not segments:
            return segments

        # Sort by start time (should already be, but ensure)
        segments = sorted(segments, key=lambda s: s.start)

        # Step 1: Remove micro-segments
        filtered = [
            seg for seg in segments
            if (seg.end - seg.start) >= min_duration_s
        ]
        removed = len(segments) - len(filtered)
        if removed:
            self._logger.info(
                f"Diarization smoothing: removed {removed} micro-segments "
                f"(< {min_duration_s}s)"
            )

        # Step 2: Merge consecutive same-speaker segments with small gaps
        if not filtered:
            return filtered

        merged: list[DiarizationSegment] = [filtered[0]]
        for seg in filtered[1:]:
            prev = merged[-1]
            same_speaker = prev.speaker_id == seg.speaker_id
            gap = seg.start - prev.end

            if same_speaker and gap <= merge_gap_s:
                # Extend previous segment to cover this one
                merged[-1] = DiarizationSegment(
                    start=prev.start,
                    end=seg.end,
                    speaker_id=prev.speaker_id,
                )
            else:
                merged.append(seg)

        merge_count = len(filtered) - len(merged)
        if merge_count:
            self._logger.info(
                f"Diarization smoothing: merged {merge_count} fragments "
                f"(same speaker, gap < {merge_gap_s}s)"
            )

        n_before = len(segments)
        n_after = len(merged)
        n_speakers = len(set(s.speaker_id for s in merged))
        self._logger.info(
            f"Diarization smoothing complete: {n_before} -> {n_after} segments, "
            f"{n_speakers} speakers"
        )

        return merged

    def _save_intermediate(self, job_id: str, stage: str, data: Any) -> None:
        """
        Save intermediate results to disk for recovery.

        Args:
            job_id: Job identifier
            stage: Stage name
            data: Data to save (must be serializable)
        """
        try:
            meeting_dir = self.get_meeting_dir(job_id)
            intermediate_file = meeting_dir / f"{stage}.json"

            if isinstance(data, (list, dict)) or hasattr(data, "model_dump"):
                # For Pydantic models, convert to plain JSON-able data.
                if isinstance(data, list) and data and hasattr(data[0], "model_dump"):
                    data = [item.model_dump() for item in data]
                elif hasattr(data, "model_dump"):
                    data = data.model_dump()

                with open(intermediate_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)

                self._logger.debug(f"Saved intermediate results: {intermediate_file}")

        except Exception as e:
            self._logger.warning(f"Failed to save intermediate results for {stage}: {e}")

    def _load_intermediate(self, job_id: str, stage: str) -> Optional[Any]:
        """
        Load intermediate results if available for recovery.

        Args:
            job_id: Job identifier
            stage: Stage name

        Returns:
            Loaded data or None if not found
        """
        try:
            meeting_dir = self.get_meeting_dir(job_id)
            intermediate_file = meeting_dir / f"{stage}.json"

            if not intermediate_file.exists():
                return None

            with open(intermediate_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._logger.debug(f"Loaded intermediate results: {intermediate_file}")
            return data

        except Exception as e:
            self._logger.warning(f"Failed to load intermediate results for {stage}: {e}")
            return None

    def _apply_voice_matching(
        self,
        aligned: list,
        preprocessed_path: str,
        meeting_type: Optional[str] = None,
    ) -> list:
        """Replace SPEAKER_X labels with enrolled profile names.

        Block 7 voice enrollment integration. For every distinct
        speaker_id in the aligned transcript:

        1. Pick the longest contiguous segment from that speaker (most
           likely to be solo speech, gives the cleanest embedding).
        2. Slice that span out of the preprocessed audio.
        3. Embed via the same ECAPA model used at enrollment time.
        4. Match against the voice DB at the configured threshold.
        5. If matched, set ``speaker_name`` on every aligned segment
           with that ``speaker_id``. For court hearings, store the match
           as a suggestion only so a human confirms legal speaker names.

        Failures (no DB, no enrolment, no torchaudio, threshold not
        met for a particular speaker) are non-fatal — the affected
        segments simply keep their SPEAKER_X labels and the user can
        rename manually. Returns the (possibly updated) aligned list.
        """
        # Bail early if voice enrollment isn't installed.
        try:
            from backend.engine import voice_enrollment as ve
        except ImportError:
            self._logger.info(
                "Voice enrollment module unavailable; skipping match step"
            )
            return aligned

        # Bail early if there's nothing enrolled — saves the cost of
        # loading the embedding model.
        try:
            profiles = ve.list_profiles()
        except Exception as e:  # noqa: BLE001
            self._logger.warning(f"Voice DB read failed: {e}")
            return aligned
        if not profiles:
            return aligned

        # Build {speaker_id: longest_segment} map. We anchor on the
        # longest contiguous turn rather than concatenating because
        # concatenation introduces spectral discontinuities that
        # hurt embedding quality.
        longest: dict[str, tuple[float, float]] = {}
        for seg in aligned:
            sid = getattr(seg, "speaker_id", None)
            if not sid:
                continue
            start = float(getattr(seg, "start", 0) or 0)
            end = float(getattr(seg, "end", 0) or 0)
            duration = max(0.0, end - start)
            if duration < ve.MIN_SAMPLE_DURATION_S:
                continue
            cur = longest.get(sid)
            if cur is None or duration > (cur[1] - cur[0]):
                longest[sid] = (start, end)

        if not longest:
            self._logger.info(
                "No speaker has a single segment >= %.1fs; "
                "skipping voice match",
                ve.MIN_SAMPLE_DURATION_S,
            )
            return aligned

        # Slice + embed + match per speaker. We open the audio once
        # and reuse the tensor to avoid re-reading from disk.
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(preprocessed_path)
        except Exception as e:  # noqa: BLE001
            self._logger.warning(
                f"Could not load preprocessed audio for voice match: {e}"
            )
            return aligned

        import tempfile
        import os
        sid_to_match: dict[str, dict] = {}
        for sid, (start_s, end_s) in longest.items():
            start_idx = int(start_s * sample_rate)
            end_idx = int(end_s * sample_rate)
            slice_wav = waveform[:, start_idx:end_idx]
            if slice_wav.shape[-1] < int(
                ve.MIN_SAMPLE_DURATION_S * sample_rate
            ):
                continue

            # Write the slice to a temp WAV so the pyannote Inference
            # path stays unchanged. Cleaned up immediately after the
            # embedding call.
            fd, tmp_path = tempfile.mkstemp(
                prefix=f"voicematch_{sid}_", suffix=".wav"
            )
            os.close(fd)
            try:
                torchaudio.save(tmp_path, slice_wav, sample_rate)
                vec = ve.extract_embedding(tmp_path)
                match = ve.match_speaker(vec)
            except Exception as e:  # noqa: BLE001
                self._logger.warning(
                    f"Voice matching failed for {sid}: {e}"
                )
                match = None
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            if match is not None:
                sid_to_match[sid] = match
                self._logger.info(
                    "Voice match: %s → %s (similarity=%.3f)",
                    sid, match["name"], match["similarity"],
                )

        if not sid_to_match:
            return aligned

        court_mode = meeting_type == "court_hearing"
        # Apply matches. We mutate in place rather than rebuilding the
        # list because the segments are large (have word-level info)
        # and rebuilding loses anything the aligner stored that we
        # don't explicitly know about.
        for seg in aligned:
            sid = getattr(seg, "speaker_id", None)
            if sid in sid_to_match:
                match = sid_to_match[sid]
                if court_mode:
                    seg.suggested_speaker_name = match["name"]
                    seg.suggested_speaker_confidence = round(
                        float(match["similarity"]), 3
                    )
                else:
                    seg.speaker_name = match["name"]

        return aligned

    def _archive_results(self, job: MeetingJob) -> None:
        """Backup final outputs for easy user access.

        Copies DOCX, JSON protocol, and aligned transcript to a
        month-grouped folder under ``backup.backup_dir``. Folder name is
        ``<filename-stem>_<short-job-id>`` so two meetings with the same
        original filename don't collide. Honours
        ``backup.retention_days`` to optionally prune older folders.

        Method retains the legacy name ``_archive_results`` for caller
        stability — the underlying behaviour was upgraded 2026-04-30 from
        the daily-grouped ``data/archive/`` to the new month-grouped
        ``data/backups/``. The legacy archive layout is no longer
        produced (existing ``data/archive/`` folders are left in place
        but no new content is written there).

        Args:
            job: Completed meeting job.
        """
        import shutil

        backup_cfg = self._config.backup
        if not backup_cfg.enabled:
            self._logger.info("Backup disabled (backup.enabled=false); skipping")
            return

        meeting_dir = self.get_meeting_dir(job.id)

        # Folder name parts. Month grouping (YYYY-MM) avoids the hundreds
        # of daily folders the old archive/ layout produced. Short
        # job-id suffix prevents same-filename collisions on the same
        # day. Sanitise filename to strip path separators and trim.
        month_str = job.created_at.strftime("%Y-%m")
        safe_name = Path(job.filename).stem.replace(" ", "_")
        # Strip any path separators the filename may have leaked (defensive).
        safe_name = safe_name.replace("/", "_").replace("\\", "_")
        # Cap at 80 chars so we don't blow Windows MAX_PATH on deep trees.
        if len(safe_name) > 80:
            safe_name = safe_name[:80]
        short_job_id = job.id[-8:] if len(job.id) >= 8 else job.id
        folder_name = f"{safe_name}_{short_job_id}"

        backup_root = Path(backup_cfg.backup_dir)
        backup_dir = backup_root / month_str / folder_name
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copy final outputs (DOCX + JSON protocol).
        copied = 0
        for pattern in ["*.docx", "*.json"]:
            for f in meeting_dir.glob(pattern):
                shutil.copy2(f, backup_dir / f.name)
                copied += 1

        # Copy aligned transcript with a stable name regardless of the
        # internal filename (it's "aligned.json" today but the renamer
        # may produce "aligned_edited.json" later).
        aligned_file = meeting_dir / "aligned.json"
        if aligned_file.exists():
            shutil.copy2(aligned_file, backup_dir / "transcript.json")
            copied += 1

        self._logger.info(
            f"Backed up {copied} files to {backup_dir}"
        )

        # Retention: prune older backup folders if a retention window is set.
        # No-op when retention_days is None (default — keep forever).
        if backup_cfg.retention_days is not None and backup_cfg.retention_days > 0:
            try:
                self._prune_old_backups(backup_root, backup_cfg.retention_days)
            except Exception as prune_err:
                # Pruning is best-effort — never fail the pipeline over it.
                self._logger.warning(f"Backup pruning failed: {prune_err}")

    def _prune_old_backups(self, backup_root: Path, retention_days: int) -> None:
        """Delete backup folders older than ``retention_days``.

        Walks the month-grouped layout (``<root>/YYYY-MM/<folder>/``) and
        removes folders whose mtime is older than the retention window.
        Empty month directories are removed afterwards. Logs counts at
        INFO. Best-effort — exceptions during deletion of an individual
        folder are logged at WARNING and don't stop the sweep.
        """
        import shutil
        import time

        if not backup_root.exists():
            return

        cutoff = time.time() - retention_days * 24 * 3600
        deleted = 0
        for month_dir in backup_root.iterdir():
            if not month_dir.is_dir():
                continue
            for folder in month_dir.iterdir():
                if not folder.is_dir():
                    continue
                try:
                    if folder.stat().st_mtime < cutoff:
                        shutil.rmtree(folder)
                        deleted += 1
                except Exception as e:
                    self._logger.warning(
                        f"Failed to prune backup folder {folder}: {e}"
                    )
            # Drop empty month directories so the layout stays tidy.
            try:
                if month_dir.is_dir() and not any(month_dir.iterdir()):
                    month_dir.rmdir()
            except Exception:
                pass

        if deleted > 0:
            self._logger.info(
                f"Backup retention: pruned {deleted} folder(s) older than "
                f"{retention_days} days"
            )

    def get_meeting_dir(self, job_id: str) -> Path:
        """
        Get or create directory for meeting data.

        Args:
            job_id: Job identifier

        Returns:
            Path to meeting directory
        """
        meeting_dir = Path(self._config.output.output_dir) / job_id
        meeting_dir.mkdir(parents=True, exist_ok=True)
        return meeting_dir

    def _job_metadata_path(self, job_id: str) -> Path:
        return Path(self._config.output.output_dir) / job_id / "job.json"

    def _persist_job(self, job: MeetingJob) -> None:
        """Persist job state so the UI survives backend restarts."""
        try:
            path = self._job_metadata_path(job.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(job.model_dump(mode="json"), f, indent=2)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to persist job %s: %s", job.id, exc)

    def _load_job_metadata(self, job_id: str) -> Optional[MeetingJob]:
        path = self._job_metadata_path(job_id)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            job = MeetingJob.model_validate(data)
            active_states = {
                PipelineState.UPLOADED,
                PipelineState.PREPROCESSING,
                PipelineState.TRANSCRIBING,
                PipelineState.DIARIZING,
                PipelineState.ALIGNING,
                PipelineState.SUMMARIZING,
                PipelineState.QA_VALIDATING,
                PipelineState.FORMATTING,
                PipelineState.RETRYING,
            }
            if job.state in active_states:
                job.state = PipelineState.ERROR
                job.progress = min(job.progress, 99.0)
                job.error = (
                    "Processing was interrupted by backend restart. "
                    "Use retry to run the job again."
                )
                job.current_stage = "Interrupted by backend restart"
                job.updated_at = datetime.utcnow()
                self._persist_job(job)
            self._jobs[job.id] = job
            return job
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to load job metadata %s: %s", path, exc)
            return None

    def _load_persisted_jobs(self) -> None:
        output_dir = Path(self._config.output.output_dir)
        if not output_dir.exists():
            return
        for metadata_path in output_dir.glob("*/job.json"):
            if metadata_path.parent.name in self._jobs:
                continue
            self._load_job_metadata(metadata_path.parent.name)

    def has_active_job(self) -> bool:
        """Check if any job is currently processing.

        Single-GPU hardware can only process one meeting at a time.
        Returns True if a job is in an active (non-terminal) state.
        """
        active_states = {
            PipelineState.UPLOADED,
            PipelineState.PREPROCESSING,
            PipelineState.TRANSCRIBING,
            PipelineState.DIARIZING,
            PipelineState.ALIGNING,
            PipelineState.SUMMARIZING,
            PipelineState.QA_VALIDATING,
            PipelineState.FORMATTING,
            PipelineState.RETRYING,
        }
        return any(job.state in active_states for job in self._jobs.values())

    def active_job_count(self) -> int:
        """Return the number of jobs currently occupying the pipeline."""
        active_states = {
            PipelineState.UPLOADED,
            PipelineState.PREPROCESSING,
            PipelineState.TRANSCRIBING,
            PipelineState.DIARIZING,
            PipelineState.ALIGNING,
            PipelineState.SUMMARIZING,
            PipelineState.QA_VALIDATING,
            PipelineState.FORMATTING,
            PipelineState.RETRYING,
        }
        return sum(job.state in active_states for job in self._jobs.values())

    def get_job(self, job_id: str) -> Optional[MeetingJob]:
        """
        Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            MeetingJob or None if not found
        """
        return self._jobs.get(job_id) or self._load_job_metadata(job_id)

    def list_jobs(self) -> list[MeetingJob]:
        """
        List all tracked jobs.

        Returns:
            List of MeetingJob objects
        """
        self._load_persisted_jobs()
        return list(self._jobs.values())

    def delete_job(self, job_id: str) -> bool:
        """
        Delete job and its data.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted, False if not found
        """
        meeting_dir = Path(self._config.output.output_dir) / job_id
        if job_id not in self._jobs and not meeting_dir.exists():
            return False

        # Delete meeting directory
        try:
            import shutil

            shutil.rmtree(meeting_dir, ignore_errors=True)
        except Exception as e:
            self._logger.warning(f"Failed to delete meeting directory: {e}")

        self._jobs.pop(job_id, None)
        self._logger.info(f"Deleted job {job_id}")
        return True
