"""Pre-ASR language detection.

Runs faster-whisper's tiny model (~39M params, ~150MB VRAM) on a few
30-second windows of the preprocessed audio to measure the RU/EN mix
before the heavy ASR engine loads. The orchestrator uses the result
to pick between GigaAM (Russian) and faster-whisper base large-v3
(multilingual, for English-heavy content).

Design rationale (2026-04-23):
- GigaAM v3 is SOTA on Russian (WER 2.6-8.4%) but Russian-only.
- faster-whisper base large-v3 is what IT uses for mixed-language content.
- MeetingType is no longer the routing signal — it drives prompts and
  DOCX templates only. Language drives ASR.
- Detection is cheap: tiny model + 3 × 30s windows ≈ 2-4s after first
  load, and it's unloaded before GigaAM/Whisper loads to free VRAM.

Sampling strategy:
- 3 windows at 0%, 50%, 100% of duration (clamped) so we catch
  language shifts that often happen in meetings (Russian opener →
  English technical discussion → Russian close).
- If duration < 30s, use the whole audio as one window.
- Failures on individual windows are logged and skipped; at least one
  window must succeed or we fall back to Russian (GigaAM) which is
  the safer default for EPAM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LanguageMix:
    """Aggregated language-detection result for one audio file."""

    #: Dominant language code ("ru", "en", etc.) by majority vote.
    primary: str
    #: Average probability of Russian across all sampled windows.
    ru_probability: float
    #: Average probability of English across all sampled windows.
    en_probability: float
    #: Count of windows that successfully produced a detection.
    windows_sampled: int
    #: Per-window (lang_code, probability) for logging/debug.
    per_window: list[tuple[str, float]] = field(default_factory=list)

    def is_english_heavy(self, threshold: float = 0.20) -> bool:
        """Route to multilingual Whisper if EN share >= threshold."""
        return self.en_probability >= threshold

    def summary(self) -> str:
        """Short human-readable summary for logs and UI."""
        return (
            f"primary={self.primary} "
            f"ru={self.ru_probability:.2f} "
            f"en={self.en_probability:.2f} "
            f"windows={self.windows_sampled}"
        )


class LanguageDetector:
    """faster-whisper tiny wrapper for pre-ASR language detection.

    Not a BaseEngine subclass because it's lightweight and stateless
    from the pipeline's perspective — orchestrator calls detect() once,
    gets a result, unloads the model.

    Args:
        device: "cuda" or "cpu". Tiny is fast enough for CPU but we
            prefer GPU when available to avoid dragging model weights
            across the PCIe bus on every job.
        compute_type: "float16" for GPU, "int8_float32" for CPU.
    """

    def __init__(
        self,
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self._device = device
        self._compute_type = compute_type
        self._model: Optional[Any] = None  # faster_whisper.WhisperModel

    def load(self) -> None:
        """Lazy load. Safe to call multiple times — no-op if loaded."""
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info(
            "Loading Whisper tiny for language detection on %s (%s)",
            self._device,
            self._compute_type,
        )
        self._model = WhisperModel(
            "tiny",
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Whisper tiny loaded")

    def unload(self) -> None:
        """Release the model. Called before the heavy ASR engine loads."""
        if self._model is None:
            return
        logger.info("Unloading Whisper tiny language detector")
        # Must null the reference before gc.collect() + empty_cache for
        # CUDA memory to actually free — see Bug 31 (orchestrator notes).
        self._model = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def detect(
        self,
        audio_path: str,
        num_windows: int = 3,
        window_s: float = 30.0,
    ) -> LanguageMix:
        """Sample windows, detect language in each, aggregate.

        Args:
            audio_path: Preprocessed 16kHz mono WAV path.
            num_windows: How many 30s windows to sample across duration.
                3 is a sensible default — start, middle, end. Use 1 for
                very short audio, 5+ for long multi-hour recordings.
            window_s: Length of each sample in seconds. 30s matches
                Whisper's internal window and avoids edge effects.

        Returns:
            LanguageMix. ru_probability / en_probability are averages
            across all successfully-detected windows. primary is the
            majority-vote language code. If all detections fail,
            primary="ru" and ru_probability=1.0 (Russian fallback).
        """
        self.load()

        import soundfile as sf
        import numpy as np

        # Get duration without loading full file into memory.
        info = sf.info(audio_path)
        total_samples = info.frames
        sr = info.samplerate
        duration_s = total_samples / sr

        # Compute window start offsets (in seconds, clamped to file).
        window_samples = int(window_s * sr)
        if duration_s <= window_s or num_windows <= 1:
            offsets_s = [0.0]
        else:
            # Evenly spaced starts: first at 0, last at (duration - window).
            last_start = max(0.0, duration_s - window_s)
            offsets_s = [
                last_start * i / (num_windows - 1) for i in range(num_windows)
            ]

        per_window: list[tuple[str, float]] = []
        ru_probs: list[float] = []
        en_probs: list[float] = []
        primary_votes: list[str] = []

        for offset_s in offsets_s:
            start_frame = int(offset_s * sr)
            frames_to_read = min(window_samples, total_samples - start_frame)
            if frames_to_read <= 0:
                continue

            try:
                audio, _ = sf.read(
                    audio_path,
                    start=start_frame,
                    frames=frames_to_read,
                    dtype="float32",
                    always_2d=False,
                )
                # Stereo to mono (preprocess should already have mono'd,
                # but be defensive).
                if getattr(audio, "ndim", 1) > 1:
                    audio = audio.mean(axis=1)
                audio = np.ascontiguousarray(audio, dtype=np.float32)

                # faster-whisper's detect_language returns a tuple whose
                # shape varies by version: (code, prob) or (code, prob,
                # all_probs). Handle both.
                result = self._model.detect_language(audio)
                if isinstance(result, tuple) and len(result) >= 2:
                    lang = str(result[0])
                    prob = float(result[1])
                    all_probs = result[2] if len(result) >= 3 else None
                else:
                    logger.warning(
                        "Unexpected detect_language return shape at %.1fs: %r",
                        offset_s,
                        result,
                    )
                    continue

                per_window.append((lang, prob))
                primary_votes.append(lang)

                # all_probs can be dict {"ru": 0.5, ...} or list of
                # tuples [("ru", 0.5), ...] depending on faster-whisper
                # version. Extract RU/EN defensively.
                ru_p = 0.0
                en_p = 0.0
                if isinstance(all_probs, dict):
                    ru_p = float(all_probs.get("ru", 0.0))
                    en_p = float(all_probs.get("en", 0.0))
                elif isinstance(all_probs, list):
                    for entry in all_probs:
                        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                            continue
                        code = str(entry[0])
                        p = float(entry[1])
                        if code == "ru":
                            ru_p = p
                        elif code == "en":
                            en_p = p
                else:
                    # No distribution available — fall back to primary.
                    if lang == "ru":
                        ru_p = prob
                    elif lang == "en":
                        en_p = prob

                ru_probs.append(ru_p)
                en_probs.append(en_p)

            except Exception as exc:  # noqa: BLE001 — non-fatal per window
                logger.warning(
                    "Language detection failed on window at %.1fs: %s",
                    offset_s,
                    exc,
                )
                continue

        if not primary_votes:
            # Every window failed. Default to Russian so we land on
            # GigaAM — safer for EPAM than accidentally routing to
            # Whisper on an unreadable audio.
            logger.warning(
                "Language detection produced no results; defaulting to Russian"
            )
            return LanguageMix(
                primary="ru",
                ru_probability=1.0,
                en_probability=0.0,
                windows_sampled=0,
                per_window=[],
            )

        ru_avg = sum(ru_probs) / len(ru_probs) if ru_probs else 0.0
        en_avg = sum(en_probs) / len(en_probs) if en_probs else 0.0

        # Majority vote for primary (ties broken by first seen).
        primary = max(set(primary_votes), key=primary_votes.count)

        result = LanguageMix(
            primary=primary,
            ru_probability=ru_avg,
            en_probability=en_avg,
            windows_sampled=len(primary_votes),
            per_window=per_window,
        )

        logger.info(
            "Language detection: %s | per-window: %s",
            result.summary(),
            [f"{code}={p:.2f}" for code, p in per_window],
        )
        return result
