"""Application configuration using Pydantic Settings with YAML support."""
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class PipelineProfile(str, Enum):
    """Pipeline profile determines ASR/diarization/summarization parameters.

    SMALL: optimized for 2-4 speaker meetings (current defaults).
    LARGE: tuned for 5+ speaker meetings (finer granularity, stricter params).
    AUTO: run diarization first, detect speaker count, then select profile.
    """

    SMALL = "small"
    LARGE = "large"
    AUTO = "auto"


class ServerConfig(BaseSettings):
    """Server configuration."""
    host: str = "127.0.0.1"
    # 8765 (was 8000): picked to avoid collisions with FastAPI/Django
    # default 8000, alternative HTTP servers on 8080, and common dev
    # tools on 5000. Override via EPAM_BACKEND_PORT env var (the
    # launcher and frontend Vite proxy honour the same var).
    port: int = 8765
    workers: int = 1
    cors_origins: list[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(env_prefix="EPAM_SERVER_")


class ASRConfig(BaseSettings):
    """Automatic Speech Recognition configuration."""
    # engine: gigaam|hf-whisper|whisper|qwen|nemo|auto
    # "auto" (default, 2026-04-23 rewrite) routes by detected LANGUAGE
    # MIX of the preprocessed audio, not by MeetingType:
    #   english_probability >= english_threshold -> "whisper"
    #     (faster-whisper base large-v3, multilingual, same as EPAM IT)
    #   else -> "gigaam" (SOTA Russian, WER 2.6-8.4%)
    # Language detection runs Whisper-tiny on 3x 30s windows between
    # preprocessing and ASR. MeetingType no longer affects ASR; it
    # drives summarization prompts and DOCX templates only.
    # Explicit values (gigaam/hf-whisper/whisper/qwen/nemo) bypass
    # detection entirely.
    engine: str = "auto"
    # English-probability threshold for routing "auto" to the
    # multilingual Whisper fallback.
    #
    # 2026-04-23 retune: first shipped at 0.20, which routed a
    # Russian-primary admin meeting (with a few English brand names
    # and tech terms) to base Whisper large-v3. Base large-v3 is
    # materially worse on Russian than the antony66 Russian fine-tune
    # AND less accurate than GigaAM. Result was a blank-looking
    # protocol because Gemma couldn't extract anything from the noisy
    # transcript. Raised to 0.50 — we want a meeting to be genuinely
    # English-majority (half or more) before paying the Russian-
    # accuracy penalty for switching to multilingual Whisper.
    # Tune via EPAM_ASR_ENGLISH_THRESHOLD env var if needed.
    english_threshold: float = 0.50
    model: str = "nvidia/stt_ru_conformer_transducer_large"
    whisper_model: str = "large-v3"  # Whisper model size or HF repo path
    qwen_model: str = "Qwen/Qwen3-ASR-1.7B"  # Qwen3-ASR HuggingFace model
    qwen_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"  # ForcedAligner for timestamps
    # T3.6 (2026-04-23) WhisperX pilot. wav2vec2 alignment model. The
    # default Russian XLSR-53 fine-tune from jonatasgrosman is the
    # WhisperX-recommended choice for Russian. Override only if a
    # better-fit model is available.
    whisperx_align_model: str = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
    # GigaAM longform batch size. This affects speed/VRAM only, not
    # recognition quality. RTX 3090 target value is 16; if there is not
    # enough free VRAM, the pipeline must fail clearly instead of
    # silently using a lower-throughput mode.
    gigaam_batch_size: int = 16
    gigaam_min_batch_size: int = 1
    gigaam_retry_smaller_batch: bool = False
    # Qwen3-ASR parallel-chunk batch size. batch=1 leaves the GPU mostly idle;
    # batch=8 is comfortable on a 24GB card. Lower on 8-12GB cards.
    qwen_batch_size: int = 8
    device: str = "auto"  # auto|cuda|cpu
    compute_type: str = "float16"
    language: str = "ru"
    beam_size: int = 5
    chunk_duration: float = 30.0  # seconds per chunk for streaming
    # See settings.yaml notes: thresholds tuned 2026-04-23 to recover
    # mid-clause content the previous (0.5 / 100) values were clipping.
    vad_threshold: float = 0.4
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 350
    # Domain vocabulary hint for Whisper transcribe(). Biases the
    # decoder toward terms that appear in EPAM legal speech. Capped
    # at ~224 tokens by Whisper. Expanded 2026-04-23 after stakeholder
    # review of Court hearing 129 found 76 phrase/clause edits and 6
    # date/number edits — many tied to legal-domain vocabulary.
    # NOTE: GigaAM does NOT consume this (no equivalent API in the
    # gigaam pip package); only Whisper-family engines use it.
    initial_prompt: str = (
        "Протокол совещания юридической фирмы EPAM. "
        "Участники обсуждают правовые вопросы, контракты, судебные дела. "
        "Истец, ответчик, председательствующий, секретарь, представитель. "
        "Исковое заявление, ходатайство, апелляция, кассация, постановление, "
        "решение суда, протокол судебного заседания, доверенность, свидетель. "
        "Требование, обязательство, расчёт, штраф, неустойка, договор, "
        "приложение, пояснения, дополнения. "
        "Рублей, копеек, миллионов, тысяч, долларов США. "
        "Участники могут перебивать друг друга."
    )
    condition_on_previous_text: bool = False  # Use previous chunk as context
    no_speech_threshold: float = 0.6  # Whisper no_speech_threshold
    compression_ratio_threshold: float = 2.4  # Whisper compression_ratio_threshold

    model_config = SettingsConfigDict(env_prefix="EPAM_ASR_")


class DiarizationConfig(BaseSettings):
    """Speaker diarization configuration."""
    engine: str = "pyannote"  # pyannote is the only production diarization engine
    embedding_model: str = ""  # Legacy field; ignored by pyannote
    pyannote_model: str = "pyannote/speaker-diarization-community-1"  # pyannote 4.0 community model
    clustering: Literal["spectral", "agglomerative"] = "spectral"
    min_speakers: int = 2
    max_speakers: int = 20
    identification_threshold: float = 0.75
    min_segment_duration: float = 0.5
    window_duration: float = 1.0  # sliding window duration in seconds
    step_duration: float = 0.5  # sliding window step in seconds
    hf_token: Optional[str] = Field(default=None, json_schema_extra={"env": "HF_TOKEN"})
    # ----- Court-hearing-specific overrides (Sprint 2026-04-30) -------
    # Court hearings have multiple same-gender lawyers (plaintiff's
    # rep + defendant's rep + judge) whose voices VBx clustering tends
    # to merge into one. court_hearing_129 produced a 59/33/8 split
    # where the two lawyers should have been roughly 35/35. These
    # knobs apply ONLY when meeting_type == "court_hearing"; other
    # meeting types use the standard fields above. Set individually
    # via env (EPAM_DIARIZATION_COURT_HEARING_*).
    #
    # Lower than the global min_segment_duration above. Court
    # interjections ("Возражение!", "Не слышу") are often <0.5s and
    # currently get filtered out, hiding evidence of distinct speakers.
    court_hearing_min_segment_duration: float = 0.2
    # pyannote 4.0 community-1 clustering.threshold override.
    # None = use pipeline's pretrained default (~0.7).
    # Lower values make clusters split more aggressively (less merging
    # of similar voices). 0.55 is the calibration target on
    # court_hearing_129 — separates the two lawyers without spuriously
    # multiplying the judge's cluster.
    court_hearing_clustering_threshold: Optional[float] = 0.55
    # pyannote 4.0 community-1 segmentation.min_duration_off override.
    # None = use pipeline default (0.0). Setting >0 forces silence-based
    # turn breaks, useful for fast-paced cross-examination.
    court_hearing_segmentation_min_duration_off: Optional[float] = None

    model_config = SettingsConfigDict(env_prefix="EPAM_DIARIZATION_")


class SummarizationConfig(BaseSettings):
    """Summarization engine configuration.

    Uses Ollama HTTP API for LLM inference. Default model: Gemma 4 26B MoE
    via Ollama (~16GB VRAM, 128K context). Ollama manages GPU memory lifecycle.
    """
    enabled: bool = True  # Set False for transcript-only output (no LLM needed)
    ollama_base_url: str = "http://localhost:11434"  # Ollama API base URL
    ollama_model: str = "gemma4:26b"  # Ollama model tag (configurable for fallbacks)
    ollama_timeout: int = 300  # Seconds — LLM can be slow on long transcripts
    # Ollama does not automatically use a model's full context window.
    # "auto" sizes the context from the prompt length:
    #   short/normal meetings -> 64K
    #   long meetings         -> 128K
    #   very long materials   -> up to 256K, capped by model metadata
    # Set an explicit integer string (e.g. "131072") or "max" to force it.
    ollama_num_ctx: str = "auto"
    ollama_num_ctx_auto_min: int = 65536
    ollama_num_ctx_auto_max: int = 262144
    max_new_tokens: int = 8192  # Max output tokens per generation
    # Administrative protocols are extraction-heavy and can legitimately
    # contain many decisions/tasks. Keep a larger budget than generic
    # summaries so the answer is not silently cut short.
    admin_max_new_tokens: int = 12288
    # Use Ollama's structured-output `format` schema for admin protocol
    # JSON. If a fallback model rejects schemas, disable via env.
    structured_outputs: bool = True
    # Administrative recall pass: when the first single-pass extraction
    # returns too few practical items for a long meeting, make additional
    # Gemma calls over chronological windows and merge non-duplicate items.
    admin_recall_pass: bool = True
    admin_recall_min_items_per_hour: int = 15
    admin_recall_window_minutes: int = 12
    admin_recall_max_windows: int = 12
    # Supporting docs/agenda text added to the prompt. 2K was too small
    # for legal materials; keep enough context while avoiding bloat.
    context_file_max_chars: int = 20000
    # Generation sampling: tightened for anti-hallucination (2026-04-17).
    # 0.1 is near-greedy; 0.80 top-p narrows token distribution further.
    temperature: float = 0.1
    top_p: float = 0.80
    # T3.5 (2026-04-23): polish pass over admin protocol JSON. Format-
    # only edits (whitespace, capitalization, punctuation, deduplication);
    # validation rejects any factual change. Adds one Ollama call per
    # admin run (~30s on 60-min meeting). Default OFF.
    polish_pass: bool = False
    # Sprint 2026-04-30 (task #35): map-reduce topic-segmented admin
    # protocol. When True, replaces the single-call flat 9-section
    # extraction with a two-step process: (1) detect topic boundaries
    # in the transcript, (2) extract structured discussion/decisions/
    # tasks/questions per topic.
    #
    # 2026-05-04 (task #43): DISABLED BY DEFAULT after evidence
    # showed map-reduce hurts more than it helps for our typical
    # 30-90 minute meetings. Industry consensus (Otter, Fireflies,
    # Fathom, Granola, Microsoft Teams Recap, Notion AI, Anthropic
    # Meeting Scribe) is single-pass with full transcript when the
    # context window fits (we have 128K, meetings are <50K tokens).
    # Map-reduce fragments cross-topic context — discussions that
    # span topic boundaries get truncated mid-thought. Empty topic
    # blocks ("По данной теме нет извлечённых...") were the visible
    # symptom on protocol_20.
    #
    # Code path remains for explicit opt-in via this flag (e.g. for
    # 3+ hour meetings that exceed Gemma 4's effective attention).
    use_topic_segmented_admin: bool = False

    model_config = SettingsConfigDict(env_prefix="EPAM_SUMMARIZATION_")


class QualityConfig(BaseSettings):
    """Quality assurance and validation configuration."""
    min_confidence: float = 0.6
    hallucination_check: bool = True
    max_repeat_length: int = 100
    retry_count: int = 2
    min_audio_duration: float = 10.0  # seconds
    max_audio_duration: float = 7200.0  # 2 hours

    model_config = SettingsConfigDict(env_prefix="EPAM_QUALITY_")


class OutputConfig(BaseSettings):
    """Output formatting configuration."""
    format: Literal["docx", "pdf", "md", "json"] = "docx"
    branding: str = "epam"
    output_dir: str = "data/meetings"

    model_config = SettingsConfigDict(env_prefix="EPAM_OUTPUT_")


class BackupConfig(BaseSettings):
    """Backup configuration for completed meeting protocols.

    After a meeting reaches the COMPLETED state, the orchestrator copies
    the final outputs (DOCX, JSON protocol, aligned transcript) to the
    backup folder so they're recoverable independently of the
    job-id-named working directory under output.output_dir.

    Folder layout: ``<backup_dir>/YYYY-MM/<filename-stem>_<short-job-id>/``
    Month-grouped (vs the legacy daily archive/ which created hundreds
    of folders) and tagged with the last 8 chars of the job id to
    prevent collisions when two meetings share a filename.

    Retention policy is intentionally permissive — law-firm protocols
    are valuable, and disk is cheap. Set ``retention_days`` to a
    positive integer to opt into automatic cleanup of older backup
    folders. Default ``None`` keeps everything forever.
    """

    # Master toggle. Disable for tests or environments where backup is
    # handled by an external system (e.g. Veeam, Acronis on the host).
    enabled: bool = True
    # Backup root. Sibling of output.output_dir by default. Override
    # via EPAM_BACKUP_BACKUP_DIR for shared network storage.
    backup_dir: str = "data/backups"
    # Retention in days. None = keep forever (default — recommended for
    # law firm). Set to e.g. 365 to keep one year of backups and let
    # the orchestrator prune older folders on each new completion.
    retention_days: Optional[int] = None

    model_config = SettingsConfigDict(env_prefix="EPAM_BACKUP_")


class BrandingConfig(BaseSettings):
    """EPAM branding configuration."""
    primary_color: str = "#B2001F"
    secondary_color: str = "#333333"
    heading_font: str = "Georgia"
    body_font: str = "Arial Narrow"
    logo_path: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="EPAM_BRANDING_")


class SecurityConfig(BaseSettings):
    """Security configuration for encryption, audit, and cleanup."""
    # Encryption at rest
    encryption_enabled: bool = True
    # Audit logging
    audit_enabled: bool = True
    audit_log_dir: str = "data/audit"
    # Temp file cleanup
    auto_cleanup: bool = True
    retention_hours: int = 720  # 30 days
    secure_delete: bool = True
    # CORS: production origins (empty = use server.cors_origins)
    cors_production_origins: list[str] = []
    # Online comparison endpoint. Disabled by default because it sends
    # transcript content outside the on-premise boundary.
    cloud_compare_enabled: bool = False

    model_config = SettingsConfigDict(env_prefix="EPAM_SECURITY_")


class PipelineConfig(BaseSettings):
    """Pipeline-level configuration for adaptive processing."""
    profile: PipelineProfile = PipelineProfile.AUTO
    # LARGE profile speaker count threshold (>= this triggers LARGE)
    large_profile_threshold: int = 5

    model_config = SettingsConfigDict(env_prefix="EPAM_PIPELINE_")


class PostProcessingConfig(BaseSettings):
    """Transcript post-processing configuration.

    GigaAM v3 and antony66/whisper-large-v3-russian both emit punctuation
    natively with much higher fidelity than any post-hoc restorer (benchmark
    2026-04-20 on court_hearing_129: GigaAM native punct F1 73.2%,
    deepmultilingualpunctuation 53.7%, Silero ru_punct 39.9%). Keep
    restore_punctuation disabled for those engines — it is strictly a
    regression. Enable only if using a no-punct ASR (rare).
    """
    # Run deepmultilingualpunctuation over each segment's text. Default False
    # because both our default engines emit punct natively and DMP degrades
    # their output. Set True for engines that drop punct.
    restore_punctuation: bool = False
    # Remove filler words and discourse markers ("ну", "вот", "эээ" etc.).
    # Conservative by default: keep turned on because the filler list is
    # unambiguous fillers only.
    remove_fillers: bool = True
    # Collapse Whisper-style repeated phrase hallucinations.
    remove_repetitions: bool = True
    # Capitalize the first word after sentence-ending punctuation.
    # Benchmark 2026-04-20 on court_hearing_129 (see
    # benchmarks/reports/cap_scoreboard.json): GigaAM native casing
    # accuracy 94.6% (F1 0.734) against gold. Every rule-based overlay
    # we tested (sentence-start rule, oracle proper-noun dict) made it
    # slightly worse — residual cap errors are downstream of
    # sentence-boundary disagreements, not missing capitalization logic.
    # The per-segment rule below is near-idempotent on GigaAM output
    # (segments already start with a capital) so this is safe on.
    # Would matter for a no-caps ASR engine.
    capitalize_sentences: bool = True
    # Convert spelled-out Russian numerals to digits via text2num.
    # DEFAULT FALSE as of 2026-04-23 second pass:
    # text2num.alpha2digit("ru", relaxed=True) is too aggressive for
    # Russian legal speech. Confirmed regressions on protocol (14):
    #   * Ordinals fragment: "двадцать пятого" -> "20 пятого" (text2num
    #     converts the cardinal portion but leaves the ordinal genitive
    #     suffix "пятого" alone, breaking dates like "июля двадцать
    #     пятого года")
    #   * Standalone narrative "миллион" -> "1 000 000" even when used
    #     as a regular noun ("где-то миллион проблем", "5 с половиной
    #     миллион")
    # Reference baseline (Протокол_129_01) had zero such artifacts;
    # text2num introduced 23+ new ones in 60 minutes of audio. Net
    # negative ROI — disabling. Code path retained so a future,
    # narrowly-scoped normalizer (e.g. only converts X тысяч/миллионов
    # рублей patterns with explicit currency anchors) can drop in.
    normalize_numbers: bool = False
    # Drop short segments that match known Whisper hallucination
    # patterns (timecode-like artifacts at chunk boundaries, YouTube-
    # training-data leakage like "Спасибо за просмотр" / "Subscribe").
    # Conservative — only fires on segments <=30 chars dominated by
    # the pattern. Real speech that happens to mention these words is
    # never affected.
    filter_hallucinations: bool = True
    # T3.4 (2026-04-23): opt-in LLM post-correction pass over aligned
    # turns. Runs Gemma 4 over batches of ~4000 chars with a strict
    # "fix only obvious ASR errors, never paraphrase" prompt. Adds
    # ~1 minute of latency on a 60-min meeting and one Ollama call
    # per batch. Default OFF — enable per run via env var
    # `EPAM_POSTPROCESSING_LLM_CORRECTION=true` while we gather data.
    # Validation: rejects batches that change turn count, speakers,
    # or paraphrase below 70% Jaccard overlap.
    llm_correction: bool = False
    # Minimum Jaccard token overlap between original and corrected
    # text per turn. Below this, the individual turn reverts to
    # original (anti-paraphrase guard).
    llm_correction_min_overlap: float = 0.70
    # Approximate batch size for the corrector. Smaller = more LLM
    # calls but tighter context. 4000 chars ≈ 30-50 short turns.
    llm_correction_chunk_chars: int = 4000

    model_config = SettingsConfigDict(env_prefix="EPAM_POSTPROCESSING_")


class AppConfig(BaseSettings):
    """Main application configuration aggregating all sub-configs."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    backup: BackupConfig = Field(default_factory=BackupConfig)
    branding: BrandingConfig = Field(default_factory=BrandingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    postprocessing: PostProcessingConfig = Field(default_factory=PostProcessingConfig)

    model_config = SettingsConfigDict(env_prefix="EPAM_")

    def apply_large_profile(self) -> None:
        """Apply LARGE profile settings for 5+ speaker meetings.

        Adjusts ASR, diarization, and summarization parameters for complex
        multi-speaker meetings. Called by orchestrator after diarization
        determines speaker count >= large_profile_threshold.
        """
        logger.info("Applying LARGE pipeline profile for multi-speaker meeting")

        # ASR: higher beam, lower VAD threshold to catch quiet speakers
        self.asr.beam_size = 7
        self.asr.vad_threshold = 0.35
        self.asr.vad_min_speech_ms = 150
        self.asr.vad_min_silence_ms = 80
        # NOT enabling condition_on_previous_text: cross-contamination risk
        self.asr.condition_on_previous_text = False

        # Diarization: finer windows (already aggressive, but even finer)
        self.diarization.window_duration = 0.75
        self.diarization.step_duration = 0.25

        # Summarization: Ollama handles its own context — no changes needed

    def apply_small_profile(self) -> None:
        """Apply SMALL profile settings for 2-4 speaker meetings.

        Uses conservative defaults that work well for simple meetings.
        """
        logger.info("Applying SMALL pipeline profile for simple meeting")
        # Keep defaults — no changes needed

    @classmethod
    def load_from_yaml(cls, yaml_path: str | Path) -> "AppConfig":
        """
        Load configuration from YAML file and overlay environment variables.

        Args:
            yaml_path: Path to settings.yaml file

        Returns:
            AppConfig instance with merged YAML and environment settings
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            logger.warning(f"Config file not found: {yaml_path}. Using defaults.")
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_dict = yaml.safe_load(f) or {}

        logger.info(f"Loaded configuration from {yaml_path}")

        # Extract sub-config sections from YAML
        server_dict = yaml_dict.get("server", {})
        asr_dict = yaml_dict.get("asr", {})
        diarization_dict = yaml_dict.get("diarization", {})
        summarization_dict = yaml_dict.get("summarization", {})
        quality_dict = yaml_dict.get("quality", {})
        output_dict = yaml_dict.get("output", {})
        backup_dict = yaml_dict.get("backup", {})
        branding_dict = yaml_dict.get("branding", {})
        security_dict = yaml_dict.get("security", {})
        pipeline_dict = yaml_dict.get("pipeline", {})
        postprocessing_dict = yaml_dict.get("postprocessing", {})

        # Overlay env vars on top of YAML. In Pydantic v2, explicit kwargs to
        # __init__ outrank env vars — so a YAML value silently beats an
        # EPAM_* env var. That was masking EPAM_DIARIZATION_ENGINE=diarizen
        # from the bat file. Here we pre-merge env vars into the YAML dict so
        # the expected precedence (env > YAML > defaults) holds. We key on
        # the field name declared on each BaseSettings subclass.
        def _apply_env_overrides(prefix: str, model_cls: type, yaml_values: dict) -> dict:
            merged = dict(yaml_values)
            for field_name in model_cls.model_fields:
                env_key = f"{prefix}{field_name.upper()}"
                if env_key in os.environ:
                    merged[field_name] = os.environ[env_key]
            return merged

        server_dict = _apply_env_overrides("EPAM_SERVER_", ServerConfig, server_dict)
        asr_dict = _apply_env_overrides("EPAM_ASR_", ASRConfig, asr_dict)
        diarization_dict = _apply_env_overrides("EPAM_DIARIZATION_", DiarizationConfig, diarization_dict)
        summarization_dict = _apply_env_overrides("EPAM_SUMMARIZATION_", SummarizationConfig, summarization_dict)
        quality_dict = _apply_env_overrides("EPAM_QUALITY_", QualityConfig, quality_dict)
        output_dict = _apply_env_overrides("EPAM_OUTPUT_", OutputConfig, output_dict)
        backup_dict = _apply_env_overrides("EPAM_BACKUP_", BackupConfig, backup_dict)
        branding_dict = _apply_env_overrides("EPAM_BRANDING_", BrandingConfig, branding_dict)
        security_dict = _apply_env_overrides("EPAM_SECURITY_", SecurityConfig, security_dict)
        pipeline_dict = _apply_env_overrides("EPAM_PIPELINE_", PipelineConfig, pipeline_dict)
        postprocessing_dict = _apply_env_overrides("EPAM_POSTPROCESSING_", PostProcessingConfig, postprocessing_dict)

        # Create config instances with merged (env-over-YAML) values
        config = cls(
            server=ServerConfig(**server_dict),
            asr=ASRConfig(**asr_dict),
            diarization=DiarizationConfig(**diarization_dict),
            summarization=SummarizationConfig(**summarization_dict),
            quality=QualityConfig(**quality_dict),
            output=OutputConfig(**output_dict),
            backup=BackupConfig(**backup_dict),
            branding=BrandingConfig(**branding_dict),
            security=SecurityConfig(**security_dict),
            pipeline=PipelineConfig(**pipeline_dict),
            postprocessing=PostProcessingConfig(**postprocessing_dict),
        )

        # Per-user data isolation (Sprint 2026-04-30, task #34): all
        # path-typed config fields are resolved through paths.py so
        # data lives under %APPDATA%\VERITAS\<user>\ instead of the
        # project-relative data/. Operators can override globally via
        # EPAM_DATA_ROOT=data to revert to legacy behaviour, or per
        # field by setting an absolute path in YAML/env.
        config._apply_user_data_root()

        return config

    def _apply_user_data_root(self) -> None:
        """Rewrite relative path config fields to land under user data root.

        Called once after ``load_from_yaml`` (or any other code path
        that constructs an AppConfig from defaults). Idempotent —
        absolute paths already produced by a previous resolve are left
        alone, and explicit absolute paths in YAML/env always win.

        Touches three fields:

        * ``output.output_dir``  ("data/meetings"  → user-scoped)
        * ``backup.backup_dir``  ("data/backups"   → user-scoped)
        * ``security.audit_log_dir`` ("data/audit" → user-scoped)
        """
        from backend.app.paths import resolve_user_path
        self.output.output_dir = str(resolve_user_path(self.output.output_dir))
        self.backup.backup_dir = str(resolve_user_path(self.backup.backup_dir))
        self.security.audit_log_dir = str(
            resolve_user_path(self.security.audit_log_dir)
        )

    @field_validator("output", mode="before")
    @classmethod
    def ensure_output_dir_exists(cls, v):
        """Ensure output directory exists."""
        if isinstance(v, dict):
            output_dir = v.get("output_dir", "data/meetings")
        else:
            output_dir = v.output_dir

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return v
