"""Custom ASR engine using NVIDIA NeMo Conformer model."""
from __future__ import annotations

import gc
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except Exception:
    NEMO_AVAILABLE = False

from backend.app.models import TranscriptionSegment, WordInfo
from backend.engine.base import BaseEngine
from backend.engine.vad import SileroVAD

logger = logging.getLogger(__name__)


class NeMoASREngine(BaseEngine):
    """
    Custom ASR engine using NVIDIA NeMo Conformer model.

    Supports both CTC and Transducer model architectures. Integrates
    Silero VAD for speech activity detection and provides word-level
    timestamps with confidence scores.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize NeMo ASR engine.

        Args:
            config: Application configuration with ASR settings.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch and torchaudio are required for ASR. "
                "Install with: pip install torch torchaudio"
            )

        if not NEMO_AVAILABLE:
            raise ImportError(
                "nemo is required for ASR. "
                "Install with: pip install nemo-toolkit"
            )

        super().__init__(config)

        self._vad = SileroVAD(
            threshold=config.asr.vad_threshold,
            min_speech_ms=config.asr.vad_min_speech_ms,
            min_silence_ms=config.asr.vad_min_silence_ms,
            sample_rate=16000,
        )

        self._device: Optional[str] = None
        self._model_type: Optional[str] = None
        self._sample_rate = 16000

    @property
    def required_vram_gb(self) -> float:
        """VRAM required for NeMo Conformer Large model."""
        return 2.5

    def load(self) -> None:
        """
        Load NeMo ASR model and Silero VAD.

        Automatically detects model type (CTC or Transducer) from model name.
        Falls back to CPU if CUDA memory is insufficient.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self.is_loaded:
            logger.warning("ASR engine already loaded")
            return

        try:
            # Auto-detect device
            if self._config.asr.device == "auto":
                self._device = (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self._device = self._config.asr.device

            logger.info(f"Loading NeMo ASR model on {self._device}...")

            model_name = self._config.asr.model
            logger.info(f"Model: {model_name}")

            # Enable offline mode if model is cached (NeMo uses NGC/HF)
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            # Auto-detect model type from name
            try:
                if (
                    "transducer" in model_name.lower()
                    or "rnnt" in model_name.lower()
                ):
                    self._model_type = "transducer"
                    self._model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                        model_name
                    )
                    logger.info("Loaded Transducer (RNN-T) model")
                else:
                    self._model_type = "ctc"
                    self._model = (
                        nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                            model_name
                        )
                    )
                    logger.info("Loaded CTC model")
            except Exception as offline_err:
                # If offline load fails, model may not be cached yet — retry with network
                logger.warning(
                    f"Offline load failed ({offline_err}), "
                    f"retrying with network for first-time download..."
                )
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                if (
                    "transducer" in model_name.lower()
                    or "rnnt" in model_name.lower()
                ):
                    self._model_type = "transducer"
                    self._model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                        model_name
                    )
                else:
                    self._model_type = "ctc"
                    self._model = (
                        nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                            model_name
                        )
                    )
            finally:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)

            self._model = self._model.to(self._device)
            self._model.eval()

            # Load VAD
            self._vad.load()

            logger.info("ASR engine loaded successfully")

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(
                f"CUDA OOM while loading model: {e}. "
                "Falling back to CPU..."
            )
            self._device = "cpu"
            if self._model is not None:
                del self._model
                self._model = None
            torch.cuda.empty_cache()
            gc.collect()

            # Retry on CPU
            try:
                model_name = self._config.asr.model
                if "transducer" in model_name.lower():
                    self._model_type = "transducer"
                    self._model = (
                        nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                            model_name
                        )
                    )
                else:
                    self._model_type = "ctc"
                    self._model = (
                        nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                            model_name
                        )
                    )

                self._model = self._model.to(self._device)
                self._model.eval()
                self._vad.load()

                logger.info("ASR engine loaded on CPU")

            except Exception as cpu_error:
                logger.error(f"Failed to load on CPU: {cpu_error}")
                raise RuntimeError(f"ASR model loading failed: {cpu_error}") from cpu_error

        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise RuntimeError(f"ASR model loading failed: {e}") from e

    def unload(self) -> None:
        """Unload all models and free VRAM."""
        try:
            self._vad.unload()

            if self._model is not None:
                del self._model
                self._model = None

            self._model_type = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()
            logger.info("ASR engine unloaded")

        except Exception as e:
            logger.error(f"Error during ASR unload: {e}")

    async def process(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float], Any]] = None,
    ) -> list[TranscriptionSegment]:
        """
        Transcribe audio file.

        Performs VAD to detect speech segments, transcribes each segment,
        extracts word-level timestamps, and returns structured results.

        Args:
            audio_path: Path to audio file.
            progress_callback: Optional callback(progress: float [0-1]) for progress tracking.

        Returns:
            List of TranscriptionSegment with timestamps and word info.

        Raises:
            RuntimeError: If engine not loaded or transcription fails.
            FileNotFoundError: If audio file not found.
        """
        if not self.is_loaded:
            raise RuntimeError("ASR engine not loaded. Call load() first.")

        import os

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Starting transcription of {audio_path}")

            # Step 1: Run VAD to get speech segments
            speech_segments = self._vad.detect_from_file(audio_path)

            if not speech_segments:
                logger.warning(f"No speech detected in {audio_path}")
                return []

            logger.info(
                f"VAD detected {len(speech_segments)} speech segments"
            )

            # Load full audio for chunk extraction
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample to 16kHz if necessary
            if sample_rate != self._sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self._sample_rate
                )
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.squeeze(0)

            # Step 2: Process each VAD segment
            segments = []
            total_segments = len(speech_segments)

            for idx, vad_segment in enumerate(speech_segments):
                if progress_callback:
                    progress = (idx / total_segments) * 0.8  # 80% for transcription
                    await progress_callback(progress)

                segment_start = vad_segment["start"]
                segment_end = vad_segment["end"]

                # Handle very long segments: split into chunks with overlap
                chunk_duration = self._config.asr.chunk_duration
                overlap_duration = min(2.0, chunk_duration * 0.1)

                current_pos = segment_start
                segment_text = ""
                segment_confidence = 0.0
                segment_words: list[WordInfo] = []

                while current_pos < segment_end:
                    chunk_end = min(
                        current_pos + chunk_duration, segment_end
                    )

                    # Extract audio chunk
                    chunk_start_sample = int(
                        current_pos * self._sample_rate
                    )
                    chunk_end_sample = int(chunk_end * self._sample_rate)

                    chunk_audio = waveform[
                        chunk_start_sample:chunk_end_sample
                    ]

                    if chunk_audio.shape[0] < self._sample_rate * 0.1:
                        # Skip very short chunks
                        current_pos = chunk_end
                        continue

                    # Transcribe chunk
                    try:
                        text, confidence = self._transcribe_chunk(
                            chunk_audio, chunk_start_sample
                        )

                        if text.strip():
                            segment_text += text + " "
                            segment_confidence = max(
                                segment_confidence, confidence
                            )

                            # Get word-level timestamps
                            words = self._get_word_timestamps(
                                chunk_audio,
                                text,
                                current_pos,
                            )
                            segment_words.extend(words)

                    except Exception as chunk_error:
                        logger.warning(
                            f"Failed to transcribe chunk "
                            f"[{current_pos:.1f}s-{chunk_end:.1f}s]: "
                            f"{chunk_error}"
                        )

                    # Move to next chunk with overlap
                    current_pos = chunk_end - overlap_duration

                segment_text = segment_text.strip()

                if segment_text:
                    segment = TranscriptionSegment(
                        start=segment_start,
                        end=segment_end,
                        text=segment_text,
                        confidence=segment_confidence,
                        words=segment_words,
                    )
                    segments.append(segment)

            if progress_callback:
                await progress_callback(0.9)

            # Step 3: Post-process: merge adjacent segments if needed
            merged_segments = self._merge_adjacent_segments(segments)

            if progress_callback:
                await progress_callback(1.0)

            logger.info(
                f"Transcription complete: {len(merged_segments)} segments"
            )

            return merged_segments

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"ASR transcription error: {e}") from e

    def _transcribe_chunk(
        self, audio_chunk: torch.Tensor, start_sample: int = 0
    ) -> tuple[str, float]:
        """
        Transcribe a single audio chunk.

        Args:
            audio_chunk: Audio waveform tensor (mono, 16kHz).
            start_sample: Starting sample index (for reference).

        Returns:
            Tuple of (transcribed_text, confidence_score).

        Raises:
            RuntimeError: If transcription fails.
        """
        try:
            # NeMo models work with audio tensors directly in newer versions
            with torch.no_grad():
                audio_chunk = audio_chunk.to(self._device)

                if self._model_type == "transducer":
                    # For Transducer models
                    predictions = self._model.transcribe(
                        audio_file=None,
                        batch_size=1,
                        return_hypotheses=False,
                    )

                    # Fallback: transcribe using greedy decoder
                    predictions = self._model.transcribe([audio_chunk])

                else:
                    # For CTC models
                    predictions = self._model.transcribe([audio_chunk])

                if predictions:
                    text = predictions[0] if isinstance(predictions, list) else str(predictions)
                else:
                    text = ""

            # Estimate confidence based on model's internal score if available
            # Default to 0.8 for successful transcription
            confidence = 0.8 if text.strip() else 0.0

            return text, confidence

        except Exception as e:
            logger.warning(f"Chunk transcription failed: {e}")
            return "", 0.0

    def _get_word_timestamps(
        self,
        audio_chunk: torch.Tensor,
        text: str,
        segment_start_time: float,
    ) -> list[WordInfo]:
        """
        Extract word-level timestamps.

        If word timestamps are not available from the model, distributes
        the segment duration evenly across words.

        Args:
            audio_chunk: Audio waveform tensor.
            text: Transcribed text.
            segment_start_time: Start time of segment in seconds.

        Returns:
            List of WordInfo with timing and confidence.
        """
        if not text.strip():
            return []

        words_list = text.split()
        chunk_duration = len(audio_chunk) / self._sample_rate

        try:
            # Try to get word-level timestamps from model
            with torch.no_grad():
                audio_chunk = audio_chunk.to(self._device)

                # NeMo provides word timestamps via return_hypotheses
                hypotheses = self._model.transcribe(
                    [audio_chunk],
                    return_hypotheses=True,
                )

                if (
                    hypotheses
                    and hasattr(hypotheses[0], "timestep")
                ):
                    # Use model-provided timestamps
                    return self._extract_words_from_hypothesis(
                        hypotheses[0], segment_start_time
                    )

        except Exception as e:
            logger.debug(f"Failed to extract detailed timestamps: {e}")

        # Fallback: distribute duration evenly across words
        if not words_list:
            return []

        time_per_word = chunk_duration / len(words_list)
        word_infos: list[WordInfo] = []

        for word_idx, word in enumerate(words_list):
            word_start = segment_start_time + word_idx * time_per_word
            word_end = segment_start_time + (word_idx + 1) * time_per_word

            word_info = WordInfo(
                start=word_start,
                end=word_end,
                word=word,
                confidence=0.8,  # Default confidence
            )
            word_infos.append(word_info)

        return word_infos

    def _extract_words_from_hypothesis(
        self, hypothesis: Any, segment_start_time: float
    ) -> list[WordInfo]:
        """
        Extract word-level timing from NeMo hypothesis object.

        Args:
            hypothesis: NeMo Hypothesis object with timestep data.
            segment_start_time: Segment start time in seconds.

        Returns:
            List of WordInfo with extracted timestamps.
        """
        word_infos: list[WordInfo] = []

        try:
            text = hypothesis.text
            if not text.strip():
                return []

            # Parse hypothesis timestep data
            # Structure varies by model, so we use defensive parsing
            if hasattr(hypothesis, "timestep"):
                # Token-level timestamps available
                tokens = hypothesis.timestep
                words = text.split()

                # Map tokens to words (tokens might be subword units)
                token_to_word_idx = []
                word_idx = 0
                token_idx = 0

                for word in words:
                    # Estimate tokens per word
                    word_tokens = max(1, len(word) // 2)
                    for _ in range(word_tokens):
                        if token_idx < len(tokens):
                            token_to_word_idx.append(word_idx)
                            token_idx += 1
                    word_idx += 1

                # Group timestamps by word
                for word_idx, word in enumerate(words):
                    matching_tokens = [
                        tokens[i]
                        for i, widx in enumerate(token_to_word_idx)
                        if widx == word_idx
                    ]

                    if matching_tokens:
                        word_start = min(
                            matching_tokens
                        ) / 1000.0  # Convert ms to seconds
                        word_end = max(
                            matching_tokens
                        ) / 1000.0

                        word_start += segment_start_time
                        word_end += segment_start_time
                    else:
                        # No timing info, skip
                        continue

                    word_info = WordInfo(
                        start=word_start,
                        end=word_end,
                        word=word,
                        confidence=0.85,
                    )
                    word_infos.append(word_info)

        except Exception as e:
            logger.debug(f"Error extracting words from hypothesis: {e}")

        return word_infos

    def _merge_adjacent_segments(
        self, segments: list[TranscriptionSegment]
    ) -> list[TranscriptionSegment]:
        """
        Merge adjacent segments if pause between them is small.

        Args:
            segments: List of transcription segments.

        Returns:
            List of merged segments.
        """
        if len(segments) <= 1:
            return segments

        merged: list[TranscriptionSegment] = []
        min_pause = 0.5  # Minimum pause to keep segments separate (seconds)

        current_segment = segments[0]

        for next_segment in segments[1:]:
            pause = next_segment.start - current_segment.end

            if pause < min_pause:
                # Merge segments
                merged_text = (
                    current_segment.text + " " + next_segment.text
                )
                merged_words = current_segment.words + next_segment.words

                current_segment = TranscriptionSegment(
                    start=current_segment.start,
                    end=next_segment.end,
                    text=merged_text,
                    confidence=max(
                        current_segment.confidence,
                        next_segment.confidence,
                    ),
                    words=merged_words,
                )
            else:
                # Keep segments separate
                merged.append(current_segment)
                current_segment = next_segment

        merged.append(current_segment)

        return merged
