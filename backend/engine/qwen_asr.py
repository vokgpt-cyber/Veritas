"""ASR engine using Qwen3-ASR-1.7B for state-of-the-art speech recognition.

Qwen3-ASR-1.7B (January 2026) outperforms Whisper large-v3 on Russian
benchmarks (MLS, Common Voice, MLC-SLM). Supports 52 languages/dialects
including Russian. Uses ForcedAligner for word-level timestamps.

Requires: pip install qwen-asr
VRAM: ~3.5-4GB (ASR 1.7B bf16 + ForcedAligner 0.6B bf16).
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from qwen_asr import Qwen3ASRModel
    QWEN_ASR_AVAILABLE = True
except ImportError:
    QWEN_ASR_AVAILABLE = False

from backend.app.models import TranscriptionSegment, WordInfo
from backend.engine.base import BaseEngine

logger = logging.getLogger(__name__)

# Language name mapping for Qwen3-ASR (uses full language names, not ISO codes)
_LANG_CODE_TO_NAME = {
    "ru": "Russian",
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "tr": "Turkish",
    "hi": "Hindi",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "el": "Greek",
    "hu": "Hungarian",
    "ro": "Romanian",
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
    "fa": "Persian",
    "fil": "Filipino",
    "mk": "Macedonian",
    "yue": "Cantonese",
}


class Qwen3ASREngine(BaseEngine):
    """ASR engine using Qwen3-ASR-1.7B with ForcedAligner for timestamps.

    Qwen3-ASR uses a transformer encoder-decoder architecture with a
    separate ForcedAligner model for word-level timestamp prediction.
    The pipeline:
      1. Audio -> Qwen3-ASR-1.7B -> text + language detection
      2. Audio + text -> Qwen3-ForcedAligner-0.6B -> word timestamps

    Output matches the TranscriptionSegment format used by other ASR engines.
    """

    def __init__(self, config: Any) -> None:
        """Initialize Qwen3-ASR engine.

        Args:
            config: AppConfig instance with ASR settings.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for Qwen3-ASR. "
                "Install with: pip install torch"
            )
        if not QWEN_ASR_AVAILABLE:
            raise ImportError(
                "qwen-asr is required for Qwen3-ASR engine. "
                "Install with: pip install qwen-asr"
            )

        super().__init__(config)

    @property
    def required_vram_gb(self) -> float:
        """Qwen3-ASR-1.7B bf16 (~3.5GB) + ForcedAligner-0.6B bf16 (~1.2GB)."""
        return 4.0

    def load(self) -> None:
        """Load Qwen3-ASR model with ForcedAligner, offline-first.

        Loads both the ASR model and the ForcedAligner for timestamp
        support. Both are loaded onto the GPU in bfloat16 precision.
        """
        if self.is_loaded:
            logger.warning("Qwen3-ASR engine already loaded")
            return

        import os

        asr_model_name = self._config.asr.qwen_model
        aligner_model_name = self._config.asr.qwen_aligner_model

        try:
            # Check for cached model
            is_cached = self._is_model_cached(asr_model_name)

            if is_cached:
                logger.info(
                    f"Loading Qwen3-ASR from local cache: {asr_model_name}"
                )
                os.environ["HF_HUB_OFFLINE"] = "1"
            else:
                logger.info(
                    f"First run -- downloading Qwen3-ASR: {asr_model_name}. "
                    f"Will be cached for offline use."
                )

            # Determine device
            if self._config.asr.device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                device = self._config.asr.device
                if device == "cuda":
                    device = "cuda:0"

            # Use bfloat16 on GPU, float32 on CPU
            dtype = torch.bfloat16 if "cuda" in device else torch.float32

            # Batch size controls how many 30s chunks are processed in parallel.
            # Qwen3-ASR-1.7B @ bf16 is ~3.5GB; each chunk needs ~1-2GB of KV
            # cache; ForcedAligner-0.6B is ~1.2GB. On a 24GB card, batch=8 is
            # comfortable. On smaller cards, lower via EPAM_ASR_QWEN_BATCH_SIZE
            # (falls through to 8 if the config field is absent).
            batch_size = getattr(self._config.asr, "qwen_batch_size", 8)

            try:
                # Load ASR model with ForcedAligner for timestamps
                load_kwargs = dict(
                    dtype=dtype,
                    device_map=device,
                    max_inference_batch_size=batch_size,
                    max_new_tokens=512,  # Sufficient for 30s chunks
                    forced_aligner=aligner_model_name,
                    forced_aligner_kwargs=dict(
                        dtype=dtype,
                        device_map=device,
                    ),
                )
                logger.info(
                    f"Qwen3-ASR batch size: {batch_size} "
                    f"(increase via config.asr.qwen_batch_size if VRAM allows)"
                )

                self._model = Qwen3ASRModel.from_pretrained(
                    asr_model_name, **load_kwargs
                )
            finally:
                os.environ.pop("HF_HUB_OFFLINE", None)

            logger.info(
                f"Qwen3-ASR loaded on {device} ({dtype}): "
                f"{asr_model_name} + {aligner_model_name}"
            )

        except Exception as e:
            logger.error(f"Failed to load Qwen3-ASR: {e}")
            raise RuntimeError(f"Qwen3-ASR loading failed: {e}") from e

    @staticmethod
    def _is_model_cached(model_name: str) -> bool:
        """Check if Qwen3-ASR model is cached locally."""
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if repo.repo_id == model_name:
                    return True
        except Exception:
            pass
        return False

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if not self.is_loaded:
            return

        try:
            if self._model is not None:
                del self._model
                self._model = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Qwen3-ASR engine unloaded")

        except Exception as e:
            logger.error(f"Error during Qwen3-ASR unload: {e}")

    async def process(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float], Any]] = None,
    ) -> list[TranscriptionSegment]:
        """Transcribe audio file with word-level timestamps.

        Qwen3-ASR transcribes the full audio and returns text with
        language detection. ForcedAligner provides word-level timestamps.

        Args:
            audio_path: Path to audio file (WAV, 16kHz mono recommended).
            progress_callback: Optional callback(progress: float [0-1]).

        Returns:
            List of TranscriptionSegment with timestamps and word info.
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            self._logger.info(f"Starting Qwen3-ASR transcription of {audio_path}")

            if progress_callback:
                result = progress_callback(0.1)
                if hasattr(result, "__await__"):
                    await result

            # Get language name for Qwen3-ASR (uses full names, not ISO codes)
            language = self._get_language_name()

            # Transcribe with timestamps
            results = self._model.transcribe(
                audio=str(audio_path),
                language=language,
                return_time_stamps=True,
            )

            if progress_callback:
                result = progress_callback(0.6)
                if hasattr(result, "__await__"):
                    await result

            if not results:
                self._logger.warning("Qwen3-ASR returned empty results")
                return []

            asr_result = results[0]
            self._logger.info(
                f"Qwen3-ASR detected language: {asr_result.language}, "
                f"text length: {len(asr_result.text)} chars"
            )

            # Convert to TranscriptionSegment format
            segments = self._convert_to_segments(asr_result)

            if progress_callback:
                result = progress_callback(0.9)
                if hasattr(result, "__await__"):
                    await result

            # Merge very short adjacent segments
            merged = self._merge_short_segments(segments)

            self._logger.info(
                f"Qwen3-ASR transcription complete: {len(merged)} segments"
            )

            if progress_callback:
                result = progress_callback(1.0)
                if hasattr(result, "__await__"):
                    await result

            return merged

        except FileNotFoundError:
            raise
        except Exception as e:
            self._logger.error(f"Qwen3-ASR transcription failed: {e}", exc_info=True)
            raise RuntimeError(f"Qwen3-ASR error: {e}") from e

    def _get_language_name(self) -> Optional[str]:
        """Convert ISO language code from config to full language name.

        Qwen3-ASR uses full language names (e.g., 'Russian') instead of
        ISO codes (e.g., 'ru'). Returns None for auto-detection.
        """
        lang = self._config.asr.language
        if lang in ("auto", ""):
            return None
        return _LANG_CODE_TO_NAME.get(lang, None)

    def _convert_to_segments(self, asr_result: Any) -> list[TranscriptionSegment]:
        """Convert Qwen3-ASR result to TranscriptionSegment list.

        If timestamps are available (from ForcedAligner), creates segments
        by grouping words into sentence-like chunks based on punctuation
        and pauses. If no timestamps, creates a single segment for the
        entire transcription.

        Args:
            asr_result: Qwen3-ASR transcription result with .text and
                        optional .time_stamps attributes.

        Returns:
            List of TranscriptionSegment objects.
        """
        full_text = asr_result.text.strip()
        if not full_text:
            return []

        timestamps = getattr(asr_result, "time_stamps", None)

        # If no timestamps, return single segment covering full audio
        if not timestamps or len(timestamps) == 0:
            return [TranscriptionSegment(
                start=0.0,
                end=0.0,  # Unknown duration without timestamps
                text=full_text,
                confidence=0.95,  # Qwen3-ASR doesn't provide confidence scores
                words=[],
            )]

        # Convert timestamp objects to WordInfo and group into segments
        words: list[WordInfo] = []
        for ts in timestamps:
            words.append(WordInfo(
                start=round(float(ts.start_time), 3),
                end=round(float(ts.end_time), 3),
                word=ts.text.strip(),
                confidence=0.95,  # Qwen3-ASR doesn't expose per-word confidence
            ))

        # Group words into segments by sentence boundaries
        segments = self._group_words_into_segments(words)
        return segments

    def _group_words_into_segments(
        self,
        words: list[WordInfo],
        max_segment_duration: float = 15.0,
        pause_threshold: float = 0.8,
    ) -> list[TranscriptionSegment]:
        """Group word-level timestamps into sentence-like segments.

        Splits on:
          1. Sentence-ending punctuation (. ! ? and Russian equivalents)
          2. Long pauses between words (> pause_threshold seconds)
          3. Maximum segment duration exceeded

        Args:
            words: Word-level timestamps from ForcedAligner.
            max_segment_duration: Max duration per segment in seconds.
            pause_threshold: Pause duration that triggers a segment break.

        Returns:
            List of TranscriptionSegment grouped from words.
        """
        if not words:
            return []

        segments: list[TranscriptionSegment] = []
        current_words: list[WordInfo] = [words[0]]

        for i in range(1, len(words)):
            prev_word = words[i - 1]
            curr_word = words[i]

            # Check segment break conditions
            should_break = False

            # 1. Sentence-ending punctuation on previous word
            if prev_word.word and prev_word.word[-1] in ".!?\u2026":
                should_break = True

            # 2. Long pause between words
            gap = curr_word.start - prev_word.end
            if gap > pause_threshold:
                should_break = True

            # 3. Maximum segment duration exceeded
            seg_duration = curr_word.end - current_words[0].start
            if seg_duration > max_segment_duration:
                should_break = True

            if should_break and current_words:
                segments.append(self._words_to_segment(current_words))
                current_words = [curr_word]
            else:
                current_words.append(curr_word)

        # Final segment
        if current_words:
            segments.append(self._words_to_segment(current_words))

        return segments

    @staticmethod
    def _words_to_segment(words: list[WordInfo]) -> TranscriptionSegment:
        """Create a TranscriptionSegment from a group of WordInfo objects."""
        text = " ".join(w.word for w in words)
        avg_conf = sum(w.confidence for w in words) / len(words) if words else 0.95
        return TranscriptionSegment(
            start=round(words[0].start, 3),
            end=round(words[-1].end, 3),
            text=text,
            confidence=round(avg_conf, 4),
            words=words,
        )

    def _merge_short_segments(
        self,
        segments: list[TranscriptionSegment],
        min_duration: float = 0.5,
        max_gap: float = 0.3,
    ) -> list[TranscriptionSegment]:
        """Merge very short adjacent segments for cleaner output.

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
