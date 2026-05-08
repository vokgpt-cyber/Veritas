"""ASR engine using faster-whisper (CTranslate2) for Windows and cross-platform use.

Uses OpenAI Whisper large-v3 with CTranslate2 optimization.
Supports BatchedInferencePipeline for full GPU utilization on RTX 3090.
Excellent Russian support, CUDA-accelerated, word-level timestamps.
Drop-in replacement for NeMoASREngine via BaseEngine interface.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# BatchedInferencePipeline available in faster-whisper >= 1.0
try:
    from faster_whisper import BatchedInferencePipeline
    BATCHED_AVAILABLE = True
except ImportError:
    BATCHED_AVAILABLE = False

from backend.app.models import TranscriptionSegment, WordInfo
from backend.engine.base import BaseEngine

logger = logging.getLogger(__name__)


class FasterWhisperASREngine(BaseEngine):
    """
    ASR engine using faster-whisper (CTranslate2-optimized Whisper).

    Supports Russian and English with automatic language detection.
    Provides word-level timestamps and confidence scores.
    Uses BatchedInferencePipeline for parallel chunk processing on GPU.
    Works on both Windows and Linux with CUDA support.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize faster-whisper ASR engine.

        Args:
            config: Application configuration with ASR settings.
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "faster-whisper is required for ASR on Windows. "
                "Install with: pip install faster-whisper"
            )

        super().__init__(config)
        self._device: Optional[str] = None
        self._compute_type: Optional[str] = None
        self._batched_pipeline = None  # BatchedInferencePipeline wrapper

    @property
    def required_vram_gb(self) -> float:
        """VRAM required for Whisper large-v3 in float16."""
        return 3.0

    def load(self) -> None:
        """
        Load Whisper model with optional BatchedInferencePipeline.

        Automatically selects CUDA if available, falls back to CPU.
        Uses float16 on GPU for optimal VRAM usage.
        Wraps model in BatchedInferencePipeline for parallel processing.
        """
        if self.is_loaded:
            logger.warning("Whisper ASR engine already loaded")
            return

        try:
            # Determine device and compute type
            import torch
            if self._config.asr.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self._config.asr.device
                if self._device.startswith("cuda"):
                    self._device = "cuda"

            # Use config compute_type for GPU, valid CPU type for CPU
            if self._device == "cuda":
                self._compute_type = self._config.asr.compute_type
            else:
                # "int8" is not a valid standalone CTranslate2 CPU type;
                # use "int8_float32" for CPU quantization or "float32" for full precision
                cpu_type = self._config.asr.compute_type
                if cpu_type in ("float16", "int8"):
                    cpu_type = "int8_float32"
                self._compute_type = cpu_type

            # Map NeMo model names to Whisper model sizes
            model_size = self._get_whisper_model_size()

            # Resolve local directory paths (relative to project root)
            import os
            project_root = Path(__file__).parent.parent.parent
            local_model_dir = project_root / model_size
            if local_model_dir.is_dir():
                # Local model directory found — use absolute path
                model_size = str(local_model_dir)
                logger.info(
                    f"Loading Whisper from local directory: {model_size} on "
                    f"{self._device} ({self._compute_type})..."
                )
            elif Path(model_size).is_dir():
                # Already an absolute path to local directory
                logger.info(
                    f"Loading Whisper from local directory: {model_size} on "
                    f"{self._device} ({self._compute_type})..."
                )
            else:
                # HuggingFace model name or repo path — check cache
                model_cached = self._is_whisper_cached(model_size)

                if model_cached:
                    logger.info(
                        f"Loading Whisper {model_size} from local cache on "
                        f"{self._device} ({self._compute_type}, offline mode)..."
                    )
                    os.environ["HF_HUB_OFFLINE"] = "1"
                else:
                    logger.info(
                        f"First run — downloading Whisper {model_size} on "
                        f"{self._device} ({self._compute_type}). "
                        f"Will be cached for offline use."
                    )

            # Keep HF_HUB_OFFLINE set permanently for on-premise deployment
            # (don't pop it after loading — faster-whisper also checks during transcription)
            self._model = WhisperModel(
                model_size,
                device=self._device,
                compute_type=self._compute_type,
            )

            # Wrap in BatchedInferencePipeline for parallel chunk processing.
            # This is the KEY optimization for GPU utilization — processes
            # multiple 30s chunks simultaneously instead of one at a time.
            # On RTX 3090 (24GB), batch_size=16 is optimal for large-v3 float16.
            if BATCHED_AVAILABLE and self._device == "cuda":
                self._batched_pipeline = BatchedInferencePipeline(
                    model=self._model,
                )
                logger.info(
                    "BatchedInferencePipeline enabled — GPU will process "
                    "multiple chunks in parallel"
                )
            else:
                self._batched_pipeline = None
                if not BATCHED_AVAILABLE:
                    logger.warning(
                        "BatchedInferencePipeline not available — "
                        "upgrade faster-whisper for parallel processing"
                    )
                logger.info("Using sequential transcription")

            logger.info("Whisper ASR engine loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Whisper model loading failed: {e}") from e

    @staticmethod
    def _is_whisper_cached(model_size: str) -> bool:
        """Check if faster-whisper model is already cached locally."""
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            # Check for exact repo match (HF path) or Systran/faster-whisper-*
            if "/" in model_size:
                target = model_size
            else:
                target = f"Systran/faster-whisper-{model_size}"
            for repo in cache_info.repos:
                if repo.repo_id == target:
                    return True
        except Exception:
            pass

        # Fallback: check common cache locations
        import os
        cache_dir = os.environ.get(
            "HF_HOME",
            os.environ.get(
                "HUGGINGFACE_HUB_CACHE",
                str(Path.home() / ".cache" / "huggingface" / "hub"),
            ),
        )
        if "/" in model_size:
            model_dir_name = "models--" + model_size.replace("/", "--")
        else:
            model_dir_name = f"models--Systran--faster-whisper-{model_size}"
        model_dir = Path(cache_dir) / model_dir_name
        return model_dir.exists()

    def _get_whisper_model_size(self) -> str:
        """
        Determine Whisper model size from config.

        If whisper_model is a HuggingFace repo path (contains '/'), return it
        directly for faster-whisper to download/load. Otherwise, map standard
        size names.

        Returns:
            Whisper model size string or HF repo path.
        """
        # First check whisper_model config (preferred for explicit model selection)
        whisper_model = self._config.asr.whisper_model
        if "/" in whisper_model:
            # Full HuggingFace repo path (e.g., "bzikst/faster-whisper-large-v3-russian")
            return whisper_model

        # Direct Whisper model sizes from whisper_model config
        whisper_sizes = [
            "tiny", "base", "small", "medium",
            "large-v1", "large-v2", "large-v3", "large",
        ]
        if whisper_model.lower() in whisper_sizes:
            return whisper_model.lower()

        # Fallback: try to extract from generic model name
        model_name = self._config.asr.model.lower()
        for size in whisper_sizes:
            if size in model_name:
                return size

        # Default: large-v3 for best Russian quality
        return "large-v3"

    # Class-level list that prevents GC from calling CTranslate2 __del__
    # by keeping at least one Python reference alive until process exit.
    _parked_models: list = []

    def unload(self) -> None:
        """Unload model and free VRAM.

        CRITICAL: CTranslate2 WhisperModel.__del__ can deadlock or segfault
        when triggered from the main/asyncio thread due to CUDA cleanup.

        Strategy:
        1. Call unload_model() to release GPU memory (official CTranslate2 API)
        2. Park the Python object in a class-level list so refcount never hits 0
        3. Set self._model = None so is_loaded returns False
        4. __del__ never fires because _parked_models keeps a reference
        5. Parked objects are cleaned up at process exit (safe: no async context)
        """
        try:
            # Clear batched pipeline reference first
            self._batched_pipeline = None

            if self._model is not None:
                # Step 1: Free GPU memory via CTranslate2 API
                try:
                    self._model.model.unload_model()
                    logger.info("CTranslate2 model GPU memory released via unload_model()")
                except AttributeError:
                    logger.warning("CTranslate2 unload_model() not available")
                except Exception as e:
                    logger.warning(f"unload_model() error: {e}")

                # Step 2: Park reference so __del__ never fires during pipeline
                FasterWhisperASREngine._parked_models.append(self._model)

                # Step 3: Clear self._model (refcount stays >0 via parked list)
                self._model = None

            logger.info("Whisper ASR engine unloaded")

        except Exception as e:
            logger.error(f"Error during Whisper unload: {e}")

    async def process(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float], Any]] = None,
    ) -> list[TranscriptionSegment]:
        """
        Transcribe audio file with word-level timestamps.

        Uses BatchedInferencePipeline when available for parallel GPU
        processing (5-10x faster than sequential on RTX 3090).

        Args:
            audio_path: Path to audio file (any format ffmpeg supports).
            progress_callback: Optional callback(progress: float [0-1]).

        Returns:
            List of TranscriptionSegment with timestamps and word info.
        """
        if not self.is_loaded:
            raise RuntimeError("Whisper ASR engine not loaded. Call load() first.")

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Starting Whisper transcription of {audio_path}")

            # Build transcribe parameters
            # CRITICAL: task="transcribe" must be explicit — some fine-tuned models
            # (e.g., antony66/whisper-large-v3-russian) default to "translate" mode
            # which converts Russian speech to English text instead of keeping Russian.
            transcribe_kwargs = dict(
                task="transcribe",
                language=self._get_language(),
                beam_size=self._config.asr.beam_size,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_speech_duration_ms=self._config.asr.vad_min_speech_ms,
                    min_silence_duration_ms=self._config.asr.vad_min_silence_ms,
                ),
            )
            # Domain vocabulary hint — biases the decoder toward legal
            # terms ("истец", "ходатайство", "тысяч рублей" etc.).
            # Whisper ignores empty strings, so guard against blanks.
            #
            # Sprint 2026-04-30 (task #39): hot_words.yaml provides
            # additional names + addresses + counterparty terms,
            # appended after the static initial_prompt. Voice DB
            # employees are merged in automatically by the loader.
            initial_prompt = (self._config.asr.initial_prompt or "").strip()
            try:
                from backend.core.hot_words import load_hot_words_prompt
                meeting_type_str = getattr(
                    self, "_current_meeting_type", None
                )
                hw_prompt = load_hot_words_prompt(
                    meeting_type=meeting_type_str
                )
                if hw_prompt:
                    initial_prompt = (
                        f"{initial_prompt} {hw_prompt}".strip()
                        if initial_prompt
                        else hw_prompt
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"hot_words load failed (non-fatal): {e}")
            custom_hot_words = str(
                getattr(self, "_current_hot_words", "") or ""
            ).strip()
            if custom_hot_words:
                initial_prompt = (
                    f"{initial_prompt} {custom_hot_words}".strip()
                    if initial_prompt
                    else custom_hot_words
                )
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            # Choose transcription method: batched (fast) or sequential (fallback)
            if self._batched_pipeline is not None:
                # Batched inference: processes multiple 30s chunks in parallel
                # batch_size=16 is optimal for RTX 3090 with large-v3 float16
                # (~3GB model + ~8GB for 16 concurrent chunks = ~11GB, well within 24GB)
                logger.info("Using BatchedInferencePipeline (parallel GPU processing)")
                segments_gen, info = self._batched_pipeline.transcribe(
                    audio_path,
                    batch_size=16,
                    **transcribe_kwargs,
                )
            else:
                # Sequential fallback
                logger.info("Using sequential transcription")
                segments_gen, info = self._model.transcribe(
                    audio_path,
                    **transcribe_kwargs,
                )

            logger.info(
                f"Detected language: {info.language} "
                f"(probability: {info.language_probability:.2f})"
            )

            # Convert faster-whisper segments to our TranscriptionSegment format
            result_segments: list[TranscriptionSegment] = []
            total_duration = info.duration or 1.0
            segment_list = list(segments_gen)

            for idx, segment in enumerate(segment_list):
                # Extract word-level info
                words: list[WordInfo] = []
                if segment.words:
                    for w in segment.words:
                        words.append(WordInfo(
                            start=w.start,
                            end=w.end,
                            word=w.word.strip(),
                            confidence=w.probability,
                        ))

                # Average word confidence or use segment-level fallback
                if words:
                    avg_confidence = sum(w.confidence for w in words) / len(words)
                else:
                    # Some model formats don't have avg_log_prob — use safe fallback
                    avg_confidence = getattr(segment, "avg_log_prob", 0.0) or 0.0
                    # avg_log_prob is negative log probability; convert to 0-1 range
                    if avg_confidence < 0:
                        import math
                        avg_confidence = math.exp(avg_confidence)

                result_segments.append(TranscriptionSegment(
                    start=round(segment.start, 3),
                    end=round(segment.end, 3),
                    text=segment.text.strip(),
                    confidence=max(0.0, min(1.0, avg_confidence)),
                    words=words,
                ))

                if progress_callback:
                    progress = min(1.0, segment.end / total_duration)
                    result = progress_callback(progress)
                    if hasattr(result, "__await__"):
                        await result

            # Merge very short adjacent segments from same speaker context
            merged = self._merge_short_segments(result_segments)

            logger.info(
                f"Whisper transcription complete: {len(merged)} segments, "
                f"language={info.language}"
            )

            if progress_callback:
                result = progress_callback(1.0)
                if hasattr(result, "__await__"):
                    await result

            return merged

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}", exc_info=True)
            raise RuntimeError(f"Whisper ASR error: {e}") from e

    def _get_language(self) -> Optional[str]:
        """Get language code from config, or None for auto-detect."""
        lang = self._config.asr.language
        if lang in ("auto", ""):
            return None
        return lang

    def _merge_short_segments(
        self,
        segments: list[TranscriptionSegment],
        min_duration: float = 0.5,
        max_gap: float = 0.3,
    ) -> list[TranscriptionSegment]:
        """
        Merge very short adjacent segments for cleaner output.

        Args:
            segments: Input segments.
            min_duration: Minimum segment duration in seconds.
            max_gap: Maximum gap between segments to merge.

        Returns:
            Merged segments.
        """
        if len(segments) <= 1:
            return segments

        merged: list[TranscriptionSegment] = []
        current = segments[0]

        for next_seg in segments[1:]:
            current_duration = current.end - current.start
            gap = next_seg.start - current.end

            if current_duration < min_duration and gap < max_gap:
                # Merge into current
                current = TranscriptionSegment(
                    start=current.start,
                    end=next_seg.end,
                    text=current.text + " " + next_seg.text,
                    confidence=max(current.confidence, next_seg.confidence),
                    words=current.words + next_seg.words,
                )
            else:
                merged.append(current)
                current = next_seg

        merged.append(current)
        return merged
