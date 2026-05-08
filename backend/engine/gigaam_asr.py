"""ASR engine using GigaAM v3_e2e_rnnt from Sber.

GigaAM v3 is a 240M-parameter Conformer-based model pretrained on 700K+ hours
of Russian speech. The e2e_rnnt variant outputs punctuated, capitalized,
text-normalized Russian directly — no separate punctuation model needed.

Performance on RTX 3090:
  - ~3 GB VRAM
  - ~250x realtime (1-hour audio in ~15 seconds)
  - WER 2.6-8.4% on Russian benchmarks (vs Whisper's 12-25%)

Uses transcribe_longform() with pyannote VAD for files >25 seconds.
Requires HF_TOKEN for pyannote/segmentation-3.0 model (first download only).

Critical: base .transcribe() fails on audio >25 seconds. Always use
transcribe_longform() for meeting recordings.
"""
from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from backend.app.models import TranscriptionSegment, WordInfo
from backend.engine.base import BaseEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# GigaAM base transcribe() limit is ~25 seconds
LONGFORM_THRESHOLD_S = 25.0


class GigaAMASREngine(BaseEngine):
    """ASR engine using GigaAM v3_e2e_rnnt.

    Produces punctuated, capitalized Russian text directly.
    Uses transcribe_longform() with pyannote VAD segmentation for long audio.
    """

    def __init__(self, config: Any) -> None:
        """Initialize GigaAM ASR engine.

        Args:
            config: AppConfig with asr settings.
        """
        super().__init__(config)
        self._device: Optional[str] = None

    @property
    def required_vram_gb(self) -> float:
        """Estimated VRAM budget for GigaAM longform inference.

        The model weights are ~3GB, but batched longform inference needs
        extra activation/allocator headroom. We reserve enough VRAM for the
        configured batch so other GPU apps are caught before transcription
        starts instead of causing a mid-run CUDA OOM.
        """
        try:
            batch_size = int(getattr(self._config.asr, "gigaam_batch_size", 16) or 16)
        except Exception:
            batch_size = 16
        if batch_size >= 16:
            return 12.0
        if batch_size >= 8:
            return 8.0
        if batch_size >= 4:
            return 6.0
        return 4.0

    def load(self) -> None:
        """Load GigaAM v3_e2e_rnnt model onto GPU.

        Uses fp16 encoder for VRAM savings. Auto-detects CUDA device.
        Downloads model from Sber CDN on first run (~500MB), then uses
        local cache at ~/.cache/gigaam/.
        """
        if self.is_loaded:
            logger.warning("GigaAM ASR engine already loaded")
            return

        try:
            import torch
            import gigaam

            # Determine device
            device_cfg = self._config.asr.device
            if device_cfg == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = device_cfg

            # Ensure HF_TOKEN is available for pyannote/segmentation-3.0
            # (needed by transcribe_longform VAD)
            hf_token = (
                os.environ.get("HF_TOKEN")
                or getattr(self._config.diarization, "hf_token", None)
            )
            if hf_token and "HF_TOKEN" not in os.environ:
                os.environ["HF_TOKEN"] = hf_token

            logger.info(
                f"Loading GigaAM v3_e2e_rnnt on {self._device} (fp16_encoder=True)"
            )

            # Load model — downloads from CDN on first run
            # fp16_encoder=True halves encoder VRAM (default)
            # use_flash=False for broad compatibility (flash_attn optional)
            self._model = gigaam.load_model(
                "v3_e2e_rnnt",
                fp16_encoder=True,
                use_flash=False,
                device=self._device,
            )

            logger.info(
                f"GigaAM v3_e2e_rnnt loaded on {self._device} "
                f"({self.required_vram_gb:.1f} GB VRAM)"
            )

        except Exception as e:
            logger.error(f"Failed to load GigaAM model: {e}")
            self._model = None
            raise RuntimeError(f"GigaAM model loading failed: {e}") from e

    def unload(self) -> None:
        """Unload GigaAM model and free VRAM.

        Follows correct unload order: model.cpu() -> del model -> gc -> empty_cache.
        """
        try:
            if self._model is not None:
                try:
                    self._model.cpu()
                except Exception:
                    pass
                del self._model
                self._model = None

            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            logger.info("GigaAM ASR engine unloaded")

        except Exception as e:
            logger.error(f"Error during GigaAM unload: {e}")

    async def process(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float], Any]] = None,
    ) -> list[TranscriptionSegment]:
        """Transcribe audio file using GigaAM v3_e2e_rnnt.

        Uses transcribe_longform() for all meeting-length audio (>25s).
        Falls back to transcribe() for very short clips.

        The e2e_rnnt model outputs punctuated, capitalized, text-normalized
        Russian — no post-processing punctuation model needed.

        Args:
            audio_path: Path to audio file (any format supported by librosa).
            progress_callback: Optional callback for progress updates (0.0-1.0).

        Returns:
            List of TranscriptionSegment with timestamps and text.
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError("GigaAM ASR engine not loaded. Call load() first.")

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Starting GigaAM transcription of {audio_path}")

            if progress_callback:
                result = progress_callback(0.02)
                if hasattr(result, "__await__"):
                    await result

            # Determine audio duration to decide transcribe vs transcribe_longform
            duration_s = self._get_audio_duration(audio_path)
            logger.info(f"Audio duration: {duration_s:.1f}s")

            if progress_callback:
                result = progress_callback(0.05)
                if hasattr(result, "__await__"):
                    await result

            if duration_s > LONGFORM_THRESHOLD_S:
                segments = self._transcribe_longform(audio_path, progress_callback)
            else:
                segments = self._transcribe_short(audio_path)

            logger.info(
                f"GigaAM transcription complete: {len(segments)} segments "
                f"from {duration_s:.1f}s audio"
            )

            if progress_callback:
                result = progress_callback(1.0)
                if hasattr(result, "__await__"):
                    await result

            return segments

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"GigaAM transcription failed: {e}", exc_info=True)
            raise RuntimeError(f"GigaAM ASR error: {e}") from e

    def _transcribe_short(self, audio_path: str) -> list[TranscriptionSegment]:
        """Transcribe short audio (<25s) using base transcribe().

        Args:
            audio_path: Path to audio file.

        Returns:
            Single-segment transcription result.
        """
        result = self._model.transcribe(audio_path, word_timestamps=True)

        # Get audio duration for segment end time
        duration_s = self._get_audio_duration(audio_path)

        words = []
        if result.words:
            words = [
                WordInfo(
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                    word=w.text,
                    confidence=0.95,
                )
                for w in result.words
            ]

        text = result.text.strip()
        if not text:
            return []

        return [
            TranscriptionSegment(
                start=0.0,
                end=round(duration_s, 3),
                text=text,
                confidence=0.95,
                words=words,
            )
        ]

    def _transcribe_longform(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float], Any]] = None,
    ) -> list[TranscriptionSegment]:
        """Transcribe long audio using Silero VAD + batched GigaAM inference.

        pyannote's built-in transcribe_longform() fails on Windows due to
        torchcodec incompatibility with torch 2.8.0. This method replaces it
        with Silero VAD for speech segmentation, then batches chunks through
        the GigaAM encoder/decoder directly.

        Pipeline:
          1. Load audio as tensor
          2. Run Silero VAD to detect speech segments
          3. Merge/split VAD segments into chunks (15-25s target, 30s max)
          4. Batch chunks through model.forward() + model._decode()
          5. Collect results with absolute timestamps

        Args:
            audio_path: Path to audio file.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of TranscriptionSegment with timestamps.
        """
        import torch

        logger.info("Using Silero VAD + batched GigaAM inference (longform)")

        # Step 1: Load audio
        audio_tensor = self._load_audio_tensor(audio_path)
        duration_s = audio_tensor.shape[0] / SAMPLE_RATE
        logger.info(f"Audio loaded: {duration_s:.1f}s")

        # Step 2: Run Silero VAD to get speech timestamps. Pass config
        # values so the same retune that helps Whisper longform helps
        # GigaAM longform (otherwise we'd be cutting clauses on one
        # path while preserving them on the other).
        vad_segments = self._run_silero_vad(
            audio_tensor,
            threshold=float(self._config.asr.vad_threshold),
            min_speech_ms=int(self._config.asr.vad_min_speech_ms),
            min_silence_ms=int(self._config.asr.vad_min_silence_ms),
        )
        logger.info(f"Silero VAD found {len(vad_segments)} speech segments")

        if not vad_segments:
            logger.warning("No speech detected by VAD")
            return []

        # Step 3: Merge VAD segments into transcription chunks
        chunks, boundaries = self._merge_vad_segments(
            audio_tensor, vad_segments, sr=SAMPLE_RATE
        )
        logger.info(f"Merged into {len(chunks)} chunks for transcription")
        total_chunks = len(chunks)

        # Step 4: Batch through GigaAM encoder/decoder. Batch size changes
        # speed/VRAM only; it does not change recognition quality. By default
        # VERITAS fails clearly on CUDA memory pressure rather than silently
        # changing runtime mode. Smaller-batch retry is available only as an
        # explicit ops override for lower-VRAM machines.
        configured_batch_size = max(
            1, int(getattr(self._config.asr, "gigaam_batch_size", 16) or 16)
        )
        min_batch_size = max(
            1, int(getattr(self._config.asr, "gigaam_min_batch_size", 1) or 1)
        )
        retry_smaller_batch = bool(
            getattr(self._config.asr, "gigaam_retry_smaller_batch", False)
        )
        min_batch_size = min(min_batch_size, configured_batch_size)
        batch_size = configured_batch_size

        while True:
            try:
                segments = self._transcribe_chunks_with_batch(
                    chunks=chunks,
                    boundaries=boundaries,
                    batch_size=batch_size,
                    progress_callback=progress_callback,
                )
                break
            except RuntimeError as exc:
                if self._is_cuda_memory_error(exc) and not retry_smaller_batch:
                    self._clear_cuda_cache()
                    raise RuntimeError(
                        "GigaAM CUDA memory error. Close other GPU-heavy "
                        "applications, wait for RTX 3090 VRAM to be free, "
                        f"and retry. Configured batch_size={batch_size}; "
                        "VERITAS did not switch to a lower batch automatically."
                    ) from exc
                if (
                    not self._is_cuda_memory_error(exc)
                    or batch_size <= min_batch_size
                ):
                    raise

                next_batch_size = max(min_batch_size, batch_size // 2)
                if next_batch_size == batch_size:
                    raise

                logger.warning(
                    "GigaAM longform batch_size=%d failed with CUDA memory "
                    "pressure; retrying with batch_size=%d",
                    batch_size,
                    next_batch_size,
                )
                self._clear_cuda_cache()
                batch_size = next_batch_size

        # Dedupe overlap region between adjacent chunks. Each chunk was
        # transcribed with +1.5s tail audio for better right-context
        # (T2.1, 2026-04-23); the next chunk's first ~1.5s is then a
        # duplicate transcription of the same audio. Strip that prefix.
        before_dedupe = sum(len((s.text or "").split()) for s in segments)
        segments = self._dedupe_chunk_overlap(segments)
        after_dedupe = sum(len((s.text or "").split()) for s in segments)
        if before_dedupe != after_dedupe:
            logger.info(
                "Chunk-overlap dedupe: %d -> %d words across %d segments",
                before_dedupe, after_dedupe, len(segments),
            )

        logger.info(f"Transcribed {len(segments)} segments from {total_chunks} chunks")
        return segments

    def _transcribe_chunks_with_batch(
        self,
        chunks: list[Any],
        boundaries: list[tuple[float, float]],
        batch_size: int,
        progress_callback: Optional[Callable[[float], Any]] = None,
    ) -> list[TranscriptionSegment]:
        """Transcribe already-prepared audio chunks with one batch size."""
        import torch
        from torch.utils.data import DataLoader
        from gigaam.utils import AudioDataset

        logger.info(
            "Running GigaAM longform inference with batch_size=%d over %d chunks",
            batch_size,
            len(chunks),
        )

        ds = AudioDataset(chunks, tokenizer=None)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=AudioDataset.collate,
            num_workers=0,
        )

        segments: list[TranscriptionSegment] = []
        idx = 0
        total_chunks = len(chunks)

        for wav_pad, wav_lens in dl:
            encoded = None
            encoded_len = None
            decoded = None
            try:
                wav_pad = wav_pad.to(self._model._device).to(self._model._dtype)
                wav_lens = wav_lens.to(self._model._device)

                with torch.inference_mode():
                    encoded, encoded_len = self._model.forward(wav_pad, wav_lens)
                    decoded = list(
                        self._model._decode(encoded, encoded_len, wav_lens, True)
                    )

                for text, words in decoded:
                    if idx >= total_chunks:
                        logger.warning(
                            "GigaAM decoder returned more chunks than expected"
                        )
                        break

                    seg_start, seg_end = boundaries[idx]
                    idx += 1

                    text = text.strip()
                    if not text:
                        continue

                    word_infos = []
                    if words:
                        word_infos = [
                            WordInfo(
                                start=round(w.start + seg_start, 3),
                                end=round(w.end + seg_start, 3),
                                word=w.text,
                                confidence=0.95,
                            )
                            for w in words
                        ]

                    segments.append(
                        TranscriptionSegment(
                            start=round(seg_start, 3),
                            end=round(seg_end, 3),
                            text=text,
                            confidence=0.95,
                            words=word_infos,
                        )
                    )

                # Progress update per batch
                if progress_callback and total_chunks > 0:
                    progress = 0.05 + 0.85 * (
                        min(idx, total_chunks) / total_chunks
                    )
                    cb_result = progress_callback(progress)
                    if hasattr(cb_result, "__await__"):
                        pass
            finally:
                del wav_pad
                del wav_lens
                if encoded is not None:
                    del encoded
                if encoded_len is not None:
                    del encoded_len
                if decoded is not None:
                    del decoded
                self._clear_cuda_cache()

        logger.info("GigaAM batch inference completed %d/%d chunks", idx, total_chunks)
        return segments

    @staticmethod
    def _is_cuda_memory_error(exc: BaseException) -> bool:
        """Return true for CUDA memory/allocator failures worth retrying smaller."""
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "out of memory",
                "cuda driver error",
                "cudacachingallocator",
                "internal assert failed",
            )
        )

    @staticmethod
    def _clear_cuda_cache() -> None:
        """Best-effort CUDA allocator cleanup between GigaAM batches."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def _dedupe_chunk_overlap(
        segments: list[TranscriptionSegment],
        max_overlap_words: int = 8,
        min_match_words: int = 2,
    ) -> list[TranscriptionSegment]:
        """Remove duplicate transcription of the tail-overlap region.

        For each adjacent pair of chunks (N, N+1), find the longest
        sequence of words that appears at the END of N AND the START
        of N+1 (case-insensitive, punct-stripped comparison). If a
        match of length >= `min_match_words` is found, strip those
        words from N+1's text and word list.

        Conservative: requires at least 2 matching words to avoid false
        positives on common short words ("я", "это"). Caps the search
        window at 8 words on each side — overlap region is ~1.5s of
        speech which is rarely more than 8 words.
        """
        import string as _string

        if len(segments) < 2:
            return segments

        # Punctuation we strip when comparing word texts.
        punct_strip = _string.punctuation + "—–«»… "

        def _norm(word: str) -> str:
            return word.lower().strip(punct_strip).strip()

        out: list[TranscriptionSegment] = [segments[0]]
        for cur in segments[1:]:
            prev = out[-1]
            prev_words = list(prev.words or [])
            cur_words = list(cur.words or [])
            if not prev_words or not cur_words:
                out.append(cur)
                continue

            k = min(max_overlap_words, len(prev_words), len(cur_words))
            prev_tail_norm = [_norm(w.word) for w in prev_words[-k:]]
            cur_head_norm = [_norm(w.word) for w in cur_words[:k]]

            best = 0
            for length in range(k, min_match_words - 1, -1):
                if prev_tail_norm[-length:] == cur_head_norm[:length]:
                    best = length
                    break

            if best == 0:
                out.append(cur)
                continue

            # Strip first `best` words from cur. For text: walk
            # cur.text left-to-right matching word.word strings, find
            # the position after the `best`-th word, slice text from
            # there. Skip leading whitespace/punct after slice.
            new_words = cur_words[best:]
            text = cur.text or ""
            pos = 0
            ok = True
            for w in cur_words[:best]:
                wt = w.word or ""
                if not wt:
                    continue
                idx = text.find(wt, pos)
                if idx < 0:
                    ok = False
                    break
                pos = idx + len(wt)
            if not ok:
                # Couldn't locate words in text — punt and keep cur as-is.
                out.append(cur)
                continue
            new_text = text[pos:].lstrip(" \t,.;:!?-—–")

            out.append(
                TranscriptionSegment(
                    start=cur.start,
                    end=cur.end,
                    text=new_text,
                    confidence=cur.confidence,
                    words=new_words,
                )
            )

        return out

    @staticmethod
    def _load_audio_tensor(audio_path: str):
        """Load audio as a 1D torch tensor at 16kHz mono.

        Uses torchaudio (no torchcodec dependency).
        """
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        return waveform.squeeze(0)

    @staticmethod
    def _run_silero_vad(
        audio_tensor,
        threshold: float = 0.4,
        min_speech_ms: int = 250,
        min_silence_ms: int = 350,
    ):
        """Run Silero VAD to detect speech segments.

        Returns list of dicts with 'start' and 'end' in samples.
        Silero VAD is ~1MB, runs on CPU, no HF token needed.

        Defaults track config.asr defaults — see settings.yaml. Lower
        threshold = treat softer speech as speech (recovers quiet
        clause endings). Higher min_silence_ms = only break at real
        speaker-change-sized gaps, not at brief breaths.
        """
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        get_speech_timestamps = utils[0]

        # Silero VAD expects 16kHz mono float tensor
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            threshold=threshold,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            return_seconds=False,
        )

        return speech_timestamps

    @staticmethod
    def _merge_vad_segments(
        audio_tensor,
        vad_segments: list,
        sr: int = 16000,
        max_duration: float = 22.0,
        min_duration: float = 15.0,
        strict_limit: float = 30.0,
        new_chunk_threshold: float = 0.2,
        tail_overlap_s: float = 1.5,
    ):
        """Merge VAD speech segments into chunks suitable for GigaAM.

        GigaAM's base transcribe() handles up to ~25s. We target 15-22s
        chunks and hard-split anything over 30s.

        Tail-overlap (added Tier 2, 2026-04-23): each chunk's AUDIO is
        extended by `tail_overlap_s` seconds past its logical end (clamped
        to file length). This gives GigaAM's encoder right-context for
        words near the chunk boundary that would otherwise be truncated.
        The boundary tuple records the LOGICAL end (unchanged); the
        downstream dedupe pass uses suffix-prefix word matching to remove
        the duplicate transcription of the overlap region from the next
        chunk's start.

        Mirrors gigaam's own segment_audio_file logic but with Silero VAD
        output format (sample-based start/end).

        Returns:
            chunks: list of audio tensors (each potentially +tail_overlap_s
                longer than its logical duration)
            boundaries: list of (logical_start_sec, logical_end_sec) tuples
        """
        import torch

        chunks = []
        boundaries = []
        curr_start = 0.0
        curr_end = 0.0
        curr_duration = 0.0

        # Total audio length (samples / sr) — clamp tail extension here.
        audio_len_s = float(audio_tensor.shape[0]) / float(sr)

        def _slice_with_tail(start_s: float, end_s: float):
            """Slice audio[start_s : end_s + tail_overlap_s] clamped to
            file end. Returns the audio tensor."""
            extended_end = min(end_s + tail_overlap_s, audio_len_s)
            return audio_tensor[int(start_s * sr) : int(extended_end * sr)]

        def _flush(start_s: float, end_s: float, duration_s: float):
            if duration_s > strict_limit:
                # Split long segments into equal parts. Each part still
                # gets the tail overlap from _slice_with_tail.
                n_parts = int(duration_s / strict_limit) + 1
                part_dur = duration_s / n_parts
                for k in range(n_parts):
                    ps = start_s + k * part_dur
                    pe = start_s + (k + 1) * part_dur
                    chunks.append(_slice_with_tail(ps, pe))
                    boundaries.append((ps, pe))
            else:
                chunks.append(_slice_with_tail(start_s, end_s))
                boundaries.append((start_s, end_s))

        for seg in vad_segments:
            start = seg["start"] / sr
            end = seg["end"] / sr

            if curr_duration == 0.0:
                curr_start = start
            elif curr_duration > new_chunk_threshold and (
                curr_duration + (end - curr_end) > max_duration
                or curr_duration > min_duration
            ):
                _flush(curr_start, curr_end, curr_duration)
                curr_start = start

            curr_end = end
            curr_duration = curr_end - curr_start

        if curr_duration > new_chunk_threshold:
            _flush(curr_start, curr_end, curr_duration)

        return chunks, boundaries

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float:
        """Get audio duration in seconds.

        Tries librosa (robust), falls back to torchaudio.
        """
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            return float(duration)
        except Exception:
            pass

        try:
            import torchaudio
            info = torchaudio.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception:
            pass

        # Last resort: load the file and count samples
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            return len(audio) / sr
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            # Assume long-form to be safe
            return LONGFORM_THRESHOLD_S + 1.0
