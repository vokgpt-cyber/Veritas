"""ASR engine using HuggingFace transformers with direct model.generate().

Loads Whisper models in their original HuggingFace format — no CTranslate2
conversion needed. This preserves all model settings (task, language, etc.)
exactly as the model author intended.

Used for fine-tuned models like antony66/whisper-large-v3-russian that break
when manually converted to CTranslate2 format (Bug 56).

Performance strategy: manual 30s chunking + batched encoder + batched decoder
via model.generate(). This avoids the slow, experimental pipeline() chunking
and achieves speed comparable to faster-whisper's BatchedInferencePipeline.

Expected performance on RTX 3090 (24GB):
  - 30-min audio: ~1-3 minutes (vs 20+ min with pipeline())
  - VRAM: ~7-8GB peak (vs 23.7GB with pipeline() batch_size=16)
"""
from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from backend.app.models import TranscriptionSegment, WordInfo
from backend.engine.base import BaseEngine

logger = logging.getLogger(__name__)

# Whisper processes audio in 30-second windows
CHUNK_LENGTH_S = 30
SAMPLE_RATE = 16000
CHUNK_SAMPLES = CHUNK_LENGTH_S * SAMPLE_RATE


class HFWhisperASREngine(BaseEngine):
    """
    ASR engine using HuggingFace transformers with direct model.generate().

    Loads Whisper models in original format from local directory or HF Hub.
    Uses float16 on GPU. Manually chunks long audio into 30s segments and
    batches them through the encoder for GPU parallelism.
    """

    def __init__(self, config: Any) -> None:
        """Initialize HF Whisper ASR engine."""
        super().__init__(config)
        self._processor = None
        self._device: Optional[str] = None
        self._torch_dtype = None

    @property
    def required_vram_gb(self) -> float:
        """VRAM required: ~3.5GB model + ~4GB working memory for batched inference."""
        return 3.5

    def load(self) -> None:
        """Load Whisper model and processor from local directory or HF Hub.

        Uses float16 on CUDA for optimal VRAM. Loads from local directory
        models/whisper-large-v3-russian/ if available, otherwise from HF Hub.
        """
        if self.is_loaded:
            logger.warning("HF Whisper ASR engine already loaded")
            return

        try:
            import torch
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )

            # Determine device
            if self._config.asr.device == "auto":
                self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self._config.asr.device
                if self._device == "cuda":
                    self._device = "cuda:0"

            # Resolve model path — check local directory first
            model_id = self._resolve_model_path()

            # Use float16 on GPU for speed + VRAM savings
            self._torch_dtype = (
                torch.float16 if "cuda" in self._device else torch.float32
            )

            logger.info(
                f"Loading HF Whisper from {model_id} on {self._device} "
                f"(dtype={self._torch_dtype})"
            )

            # Load processor (tokenizer + feature extractor)
            self._processor = WhisperProcessor.from_pretrained(model_id)

            # Load model with low_cpu_mem_usage to avoid double memory during load
            self._model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=self._torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self._device)

            # Verify the model's generation config has correct task
            gen_config = self._model.generation_config
            logger.info(
                f"Model generation_config: task={getattr(gen_config, 'task', 'N/A')}, "
                f"language={getattr(gen_config, 'language', 'N/A')}"
            )

            logger.info(
                f"HF Whisper ASR engine loaded on {self._device} "
                f"(dtype={self._torch_dtype})"
            )

        except Exception as e:
            logger.error(f"Failed to load HF Whisper model: {e}")
            raise RuntimeError(f"HF Whisper model loading failed: {e}") from e

    def _resolve_model_path(self) -> str:
        """Resolve model path: configured local dir > legacy local dir > HF Hub.

        Priority:
        1. If config.asr.whisper_model points to an existing local directory
           (absolute path), use it directly. This lets benchmark overrides or
           explicit configs pick an arbitrary fine-tune on disk.
        2. Legacy default: models/whisper-large-v3-russian/ (antony66 fine-tune).
        3. Otherwise, treat config.asr.whisper_model as an HF repo path or name.
        """
        configured = self._config.asr.whisper_model
        if configured and Path(configured).is_dir():
            logger.info(f"Using configured local model directory: {configured}")
            return configured

        project_root = Path(__file__).parent.parent.parent
        local_model_dir = project_root / "models" / "whisper-large-v3-russian"
        if local_model_dir.is_dir():
            logger.info(f"Found local model directory: {local_model_dir}")
            return str(local_model_dir)

        logger.info(f"Using HuggingFace model: {configured}")
        return configured

    def unload(self) -> None:
        """Unload model and free VRAM."""
        try:
            if self._model is not None:
                try:
                    self._model.cpu()
                except Exception:
                    pass
                del self._model
                self._model = None

            if self._processor is not None:
                del self._processor
                self._processor = None

            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info("HF Whisper ASR engine unloaded")

        except Exception as e:
            logger.error(f"Error during HF Whisper unload: {e}")

    async def process(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float], Any]] = None,
    ) -> list[TranscriptionSegment]:
        """Transcribe audio file with segment-level timestamps.

        Strategy for speed:
        1. Load audio as numpy array at 16kHz mono
        2. Split into 30-second chunks (Whisper's native window)
        3. Batch chunks through encoder + decoder via model.generate()
        4. Parse timestamp tokens from output to get segment boundaries
        5. Offset timestamps by chunk position for absolute times

        This avoids the slow experimental pipeline() chunking and achieves
        speed comparable to faster-whisper's BatchedInferencePipeline.

        Args:
            audio_path: Path to audio file (WAV, 16kHz mono preferred).
            progress_callback: Optional progress callback.

        Returns:
            List of TranscriptionSegment with timestamps.
        """
        if not self.is_loaded or self._model is None or self._processor is None:
            raise RuntimeError("HF Whisper ASR engine not loaded. Call load() first.")

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            import torch

            logger.info(f"Starting HF Whisper transcription of {audio_path}")

            if progress_callback:
                result = progress_callback(0.02)
                if hasattr(result, "__await__"):
                    await result

            # Step 1: Load audio
            audio = self._load_audio(audio_path)
            duration_s = len(audio) / SAMPLE_RATE
            logger.info(
                f"Audio loaded: {duration_s:.1f}s ({len(audio)} samples at {SAMPLE_RATE}Hz)"
            )

            if progress_callback:
                result = progress_callback(0.05)
                if hasattr(result, "__await__"):
                    await result

            # Step 2: Split into 30s chunks
            chunks = self._split_into_chunks(audio)
            total_chunks = len(chunks)
            logger.info(f"Split audio into {total_chunks} chunks of {CHUNK_LENGTH_S}s")

            # Step 3: Process chunks in batches
            beam_size = getattr(self._config.asr, "beam_size", 1) or 1
            # batch_size=8 balances GPU utilization vs VRAM on RTX 3090
            # Each chunk: ~3MB features + ~73MB KV cache = ~76MB per chunk
            # 8 chunks: ~608MB working memory + 3.5GB model = ~4.1GB total
            batch_size = 8

            logger.info(
                f"Processing {total_chunks} chunks with batch_size={batch_size}, "
                f"num_beams={beam_size}"
            )

            all_segments: list[TranscriptionSegment] = []

            for batch_start in range(0, total_chunks, batch_size):
                batch_end = min(batch_start + batch_size, total_chunks)
                batch_chunks = chunks[batch_start:batch_end]
                actual_batch_size = len(batch_chunks)

                # Create mel features for batch
                features = self._processor.feature_extractor(
                    [c for c in batch_chunks],
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                )
                input_features = features.input_features.to(
                    self._device, dtype=self._torch_dtype
                )

                # Build the domain-vocabulary prompt as token IDs once
                # per call (not once per chunk — same prompt every chunk).
                # `prompt_ids` biases decoding toward legal terms;
                # see config.asr.initial_prompt.
                generate_kwargs = dict(
                    return_timestamps=True,
                    task="transcribe",
                    language=self._config.asr.language or "ru",
                    num_beams=beam_size,
                    max_new_tokens=440,  # Whisper's max sequence length
                )
                initial_prompt = (
                    self._config.asr.initial_prompt or ""
                ).strip()
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
                    try:
                        prompt_ids = self._processor.get_prompt_ids(
                            initial_prompt, return_tensors="pt"
                        ).to(self._device)
                        generate_kwargs["prompt_ids"] = prompt_ids
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Could not encode initial_prompt for HFWhisper: %s",
                            exc,
                        )

                # Generate transcriptions with timestamps
                with torch.no_grad():
                    output_ids = self._model.generate(
                        input_features,
                        **generate_kwargs,
                    )

                # Decode each output in the batch
                for j in range(actual_batch_size):
                    chunk_idx = batch_start + j
                    chunk_offset = chunk_idx * float(CHUNK_LENGTH_S)

                    # Check if this is the last chunk (may be shorter)
                    chunk_duration = len(batch_chunks[j]) / SAMPLE_RATE

                    segments = self._decode_output(
                        output_ids[j], chunk_offset, chunk_duration
                    )
                    all_segments.extend(segments)

                # Free batch tensors
                del input_features, output_ids

                # Progress callback
                if progress_callback:
                    progress = 0.05 + 0.85 * (batch_end / total_chunks)
                    result = progress_callback(progress)
                    if hasattr(result, "__await__"):
                        await result

                logger.info(
                    f"Processed chunks {batch_start+1}-{batch_end}/{total_chunks}"
                )

            # Step 4: Merge short segments and clean up
            merged = self._merge_short_segments(all_segments)

            logger.info(
                f"HF Whisper transcription complete: {len(merged)} segments "
                f"from {duration_s:.1f}s audio"
            )

            if progress_callback:
                result = progress_callback(1.0)
                if hasattr(result, "__await__"):
                    await result

            return merged

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"HF Whisper transcription failed: {e}", exc_info=True)
            raise RuntimeError(f"HF Whisper ASR error: {e}") from e

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file as numpy array at 16kHz mono.

        Tries librosa first (robust format support), falls back to
        torchaudio if librosa fails.
        """
        try:
            import librosa
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            return audio
        except Exception as e:
            logger.warning(f"librosa failed ({e}), trying torchaudio")

        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample if needed
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)
            return waveform.squeeze().numpy()
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load audio with both librosa and torchaudio: {e2}"
            ) from e2

    @staticmethod
    def _split_into_chunks(audio: np.ndarray) -> list[np.ndarray]:
        """Split audio into 30-second chunks.

        The last chunk may be shorter than 30s — it will be padded by
        the feature extractor during batching.
        """
        chunks = []
        for i in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[i : i + CHUNK_SAMPLES]
            chunks.append(chunk)
        return chunks

    def _decode_output(
        self,
        output_ids,
        chunk_offset: float,
        chunk_duration: float,
    ) -> list[TranscriptionSegment]:
        """Decode model output tokens into TranscriptionSegments.

        Whisper's model.generate() with return_timestamps=True produces
        output that includes special timestamp tokens like <|0.00|>, <|2.56|>.
        We parse these to extract segment boundaries.

        Uses processor.decode(output_offsets=True) for structured output
        when available, falls back to regex parsing of timestamp tokens.

        Args:
            output_ids: Token IDs from model.generate() for one chunk.
            chunk_offset: Time offset of this chunk in the full audio (seconds).
            chunk_duration: Actual duration of this chunk (seconds).

        Returns:
            List of TranscriptionSegment with absolute timestamps.
        """
        # Try structured decode with offsets first (cleaner)
        try:
            result = self._processor.tokenizer.decode(
                output_ids,
                skip_special_tokens=False,
                output_offsets=True,
            )
            if isinstance(result, dict) and "offsets" in result:
                return self._parse_offsets(
                    result["offsets"], chunk_offset, chunk_duration
                )
        except (TypeError, AttributeError):
            # output_offsets not supported in this transformers version
            pass

        # Fallback: decode with timestamp tokens and parse manually
        decoded = self._processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=False,
            decode_with_timestamps=True,
        )
        return self._parse_timestamp_text(decoded, chunk_offset, chunk_duration)

    @staticmethod
    def _parse_offsets(
        offsets: list[dict],
        chunk_offset: float,
        chunk_duration: float,
    ) -> list[TranscriptionSegment]:
        """Parse structured offset output from tokenizer.decode(output_offsets=True).

        Each offset has: {"text": "...", "timestamp": (start, end)}
        """
        segments = []
        for offset in offsets:
            text = offset.get("text", "").strip()
            if not text:
                continue

            ts = offset.get("timestamp")
            if ts is None:
                continue

            start, end = ts
            if start is None:
                start = 0.0
            if end is None:
                end = min(start + 5.0, chunk_duration)

            # Clamp to chunk duration
            start = min(start, chunk_duration)
            end = min(end, chunk_duration)

            if end <= start:
                end = min(start + 0.5, chunk_duration)

            segments.append(
                TranscriptionSegment(
                    start=round(start + chunk_offset, 3),
                    end=round(end + chunk_offset, 3),
                    text=text,
                    confidence=0.9,  # HF model doesn't expose per-segment confidence
                    words=[],
                )
            )
        return segments

    @staticmethod
    def _parse_timestamp_text(
        decoded: str,
        chunk_offset: float,
        chunk_duration: float,
    ) -> list[TranscriptionSegment]:
        """Parse timestamp tokens from decoded text.

        Whisper output format with timestamps:
          <|startoftranscript|><|ru|><|transcribe|><|0.00|> text here <|4.52|>
          <|4.52|> more text <|8.10|> ...

        We extract pairs of (start_time, text, end_time) using regex.
        """
        # Remove non-timestamp special tokens
        cleaned = re.sub(
            r"<\|(startoftranscript|endoftext|startoflm|startofprev|"
            r"nospeech|notimestamps|transcribe|translate|"
            r"[a-z]{2,3})\|>",
            "",
            decoded,
        )

        # Pattern: <|start_time|> text <|end_time|>
        # Timestamp tokens: <|0.00|> through <|30.00|> in 0.02s increments
        timestamp_pattern = r"<\|(\d+\.\d+)\|>"

        # Split by timestamp tokens
        parts = re.split(timestamp_pattern, cleaned)

        # Parts alternate: [text_before, timestamp, text, timestamp, text, ...]
        # First element may be empty text before first timestamp
        segments = []
        i = 0
        while i < len(parts):
            # Skip empty text parts
            if i < len(parts) and not parts[i].strip():
                i += 1
                continue

            # Look for: timestamp, text, timestamp pattern
            if (
                i + 2 < len(parts)
                and _is_timestamp(parts[i])
                and parts[i + 1].strip()
            ):
                start = float(parts[i])
                text = parts[i + 1].strip()
                end = float(parts[i + 2]) if _is_timestamp(parts[i + 2]) else None

                if end is None:
                    end = min(start + 5.0, chunk_duration)

                # Clamp to chunk duration
                start = min(start, chunk_duration)
                end = min(end, chunk_duration)
                if end <= start:
                    end = min(start + 0.5, chunk_duration)

                segments.append(
                    TranscriptionSegment(
                        start=round(start + chunk_offset, 3),
                        end=round(end + chunk_offset, 3),
                        text=text,
                        confidence=0.9,
                        words=[],
                    )
                )
                i += 3
            else:
                # Text without proper timestamp pair — try to use it
                if parts[i].strip() and not _is_timestamp(parts[i]):
                    # Orphaned text — assign estimated timestamps
                    text = parts[i].strip()
                    est_start = (
                        segments[-1].end - chunk_offset if segments else 0.0
                    )
                    est_end = min(est_start + 5.0, chunk_duration)
                    segments.append(
                        TranscriptionSegment(
                            start=round(est_start + chunk_offset, 3),
                            end=round(est_end + chunk_offset, 3),
                            text=text,
                            confidence=0.7,  # lower confidence for estimated timestamps
                            words=[],
                        )
                    )
                i += 1

        return segments

    def _merge_short_segments(
        self,
        segments: list[TranscriptionSegment],
        min_duration: float = 0.5,
        max_gap: float = 0.3,
    ) -> list[TranscriptionSegment]:
        """Merge very short adjacent segments for cleaner output."""
        if len(segments) <= 1:
            return segments

        merged: list[TranscriptionSegment] = []
        current = segments[0]

        for next_seg in segments[1:]:
            current_duration = current.end - current.start
            gap = next_seg.start - current.end

            if current_duration < min_duration and gap < max_gap:
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


def _is_timestamp(s: str) -> bool:
    """Check if a string looks like a timestamp value (e.g. '0.00', '15.24')."""
    try:
        val = float(s)
        return 0.0 <= val <= 30.0
    except (ValueError, TypeError):
        return False
