"""Voice Activity Detection using Silero VAD model."""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Voice Activity Detection using Silero VAD model.

    Detects speech segments in audio using Silero's pre-trained VAD model.
    Handles mono/stereo conversion and provides segment detection in both
    sample indices and seconds.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        sample_rate: int = 16000,
    ) -> None:
        """
        Initialize Silero VAD detector.

        Args:
            threshold: Speech probability threshold (0.0-1.0). Default 0.5.
            min_speech_ms: Minimum duration for speech segment in milliseconds.
            min_silence_ms: Minimum duration for silence segment in milliseconds.
            sample_rate: Audio sample rate in Hz. Default 16000.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch and torchaudio are required for VAD. "
                "Install with: pip install torch torchaudio"
            )

        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._sample_rate = sample_rate

        self._model: Optional[torch.nn.Module] = None
        self._get_speech_timestamps = None
        self._is_loaded = False

    def load(self) -> None:
        """
        Load Silero VAD model via torch.hub.

        Retrieves the pretrained Silero VAD model and utility functions
        for speech timestamp extraction.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._is_loaded:
            logger.warning("Silero VAD model already loaded")
            return

        try:
            # Check if Silero VAD is already cached in torch hub
            import os
            hub_dir = torch.hub.get_dir()
            cached_repo = Path(hub_dir) / "snakers4_silero-vad_master"
            if not cached_repo.exists():
                # Also check without _master suffix
                cached_repo = Path(hub_dir) / "snakers4_silero-vad"

            if cached_repo.exists():
                logger.info(
                    f"Loading Silero VAD from local cache: {cached_repo}"
                )
                # Load from local cache without network access
                self._model, utils = torch.hub.load(
                    repo_or_dir=str(cached_repo),
                    model="silero_vad",
                    source="local",
                    trust_repo=True,
                )
            else:
                logger.info(
                    "First run — downloading Silero VAD from torch.hub "
                    "(will be cached for offline use)..."
                )
                self._model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )

            (
                self._get_speech_timestamps,
                _,
                _,
                _,
                _,
            ) = utils

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)
            self._model.eval()

            self._is_loaded = True
            logger.info(f"Silero VAD loaded successfully on {device}")

        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise RuntimeError(f"Silero VAD loading failed: {e}") from e

    def unload(self) -> None:
        """Free model memory and release resources."""
        if not self._is_loaded:
            return

        try:
            if self._model is not None:
                # Move to CPU first to release CUDA tensors
                try:
                    self._model.cpu()
                except Exception:
                    pass
                del self._model
                self._model = None

            self._get_speech_timestamps = None

            # gc FIRST, then empty_cache
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._is_loaded = False
            logger.info("Silero VAD model unloaded")

        except Exception as e:
            logger.error(f"Error during VAD unload: {e}")

    @property
    def is_loaded(self) -> bool:
        """Check if VAD model is currently loaded."""
        return self._is_loaded

    def detect(
        self, audio: torch.Tensor | np.ndarray
    ) -> list[dict[str, int]]:
        """
        Detect speech segments in audio tensor.

        Args:
            audio: Audio waveform as torch.Tensor or np.ndarray.
                   Expected shape: (samples,) for mono or (channels, samples).

        Returns:
            List of dicts with 'start' and 'end' keys (sample indices).

        Raises:
            RuntimeError: If VAD model is not loaded.
            ValueError: If audio format is invalid.
        """
        if not self._is_loaded:
            raise RuntimeError("VAD model not loaded. Call load() first.")

        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        elif not isinstance(audio, torch.Tensor):
            raise ValueError(
                f"Expected torch.Tensor or np.ndarray, got {type(audio)}"
            )

        # Handle stereo: convert to mono by averaging channels
        if audio.dim() == 2:
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0)
            else:
                audio = audio.squeeze(0)

        if audio.dim() != 1:
            raise ValueError(
                f"Expected 1D audio tensor, got shape {audio.shape}"
            )

        # Ensure audio is on correct device
        device = next(self._model.parameters()).device
        audio = audio.to(device)

        try:
            with torch.no_grad():
                speech_timestamps = self._get_speech_timestamps(
                    audio,
                    self._model,
                    threshold=self._threshold,
                    min_speech_duration_ms=self._min_speech_ms,
                    min_silence_duration_ms=self._min_silence_ms,
                    sampling_rate=self._sample_rate,
                )

            return speech_timestamps

        except Exception as e:
            logger.error(f"VAD detection failed: {e}")
            raise RuntimeError(f"VAD detection error: {e}") from e

    def detect_from_file(self, audio_path: str) -> list[dict[str, float]]:
        """
        Load audio file and detect speech segments.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)

        Returns:
            List of dicts with 'start' and 'end' keys (in seconds).

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If audio loading or VAD fails.
        """
        import os

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample to 16kHz if necessary
            if sample_rate != self._sample_rate:
                logger.debug(
                    f"Resampling audio from {sample_rate}Hz to "
                    f"{self._sample_rate}Hz"
                )
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self._sample_rate
                )
                waveform = resampler(waveform)

            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.squeeze(0)

            # Run VAD detection
            speech_timestamps_samples = self.detect(waveform)

            # Convert sample indices to seconds
            speech_timestamps_seconds = [
                {
                    "start": ts["start"] / self._sample_rate,
                    "end": ts["end"] / self._sample_rate,
                }
                for ts in speech_timestamps_samples
            ]

            logger.info(
                f"Detected {len(speech_timestamps_seconds)} speech segments "
                f"in {audio_path}"
            )

            return speech_timestamps_seconds

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to process audio file {audio_path}: {e}")
            raise RuntimeError(f"Audio processing failed: {e}") from e
