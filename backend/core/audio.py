"""Audio preprocessing with ffmpeg (primary) and torchaudio (fallback).

On Linux/Docker: uses ffmpeg for conversion and loudness normalization.
On Windows dev: falls back to torchaudio if ffmpeg is not installed,
which handles resampling and mono conversion (no loudnorm filter).
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from backend.app.models import AudioMetadata

logger = logging.getLogger(__name__)

# Check for torchaudio availability (fallback backend)
try:
    import torch
    import torchaudio
    import soundfile as sf
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


def _ffmpeg_available() -> bool:
    """Check if ffmpeg/ffprobe are on PATH and runnable.

    Note: earlier versions probed for a non-existent flag ``-print_json`` and
    therefore always returned False. The correct flag is ``-print_format json``
    (used in _get_metadata_ffprobe). We now just verify both binaries launch.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


class AudioPreprocessor:
    """Audio preprocessing with ffmpeg or torchaudio fallback."""

    # FFmpeg binary name
    FFMPEG_BIN = "ffmpeg"
    FFPROBE_BIN = "ffprobe"

    # Target audio specifications
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1

    @classmethod
    async def process(
        cls, input_path: str, output_dir: str
    ) -> tuple[str, AudioMetadata]:
        """
        Convert audio to WAV 16kHz mono.

        Uses ffmpeg if available (includes loudness normalization).
        Falls back to torchaudio on Windows when ffmpeg is missing.

        Args:
            input_path: Path to input audio file
            output_dir: Directory to save output

        Returns:
            Tuple of (output_path, metadata)
        """
        logger.info(f"Processing audio: {input_path}")

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Input audio file not found: {input_path}"
            )

        output_filename = input_path.stem + "_processed.wav"
        output_path = output_dir / output_filename

        use_ffmpeg = _ffmpeg_available()

        if use_ffmpeg:
            try:
                logger.info("Using ffmpeg for audio preprocessing")
                metadata = await cls._get_metadata_ffprobe(str(input_path))
                logger.info(
                    f"Input: {metadata.sample_rate}Hz, "
                    f"{metadata.channels}ch, {metadata.duration:.2f}s"
                )
                await cls._convert_ffmpeg(str(input_path), str(output_path))
            except Exception as ffmpeg_err:
                if TORCHAUDIO_AVAILABLE:
                    logger.warning(
                        f"ffmpeg failed ({ffmpeg_err}), "
                        "falling back to torchaudio"
                    )
                    use_ffmpeg = False
                else:
                    raise

        # Torchaudio fallback path — only runs if ffmpeg is unavailable or failed.
        # When ffmpeg succeeded above, `use_ffmpeg` stays True and we skip this
        # block entirely (metadata + output_path are already set).
        if not use_ffmpeg:
            # Video containers (mp4/mov/avi/mkv/webm) require ffmpeg to
            # demux. torchaudio's libsndfile/sox backends generally
            # don't decode them. Surface a clear error rather than
            # letting torchaudio crash with a cryptic message.
            video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
            if input_path.suffix.lower() in video_exts:
                raise RuntimeError(
                    f"Cannot preprocess {input_path.suffix} without ffmpeg. "
                    "Install ffmpeg and ensure it is on PATH, or convert "
                    "the file to WAV/MP3 first."
                )
            if TORCHAUDIO_AVAILABLE:
                logger.info(
                    "ffmpeg not found, using torchaudio fallback "
                    "(no loudness normalization)"
                )
                metadata = await cls._get_metadata_torchaudio(str(input_path))
                logger.info(
                    f"Input: {metadata.sample_rate}Hz, "
                    f"{metadata.channels}ch, {metadata.duration:.2f}s"
                )
                await cls._convert_torchaudio(str(input_path), str(output_path))
            else:
                raise RuntimeError(
                    "No audio backend available. Install ffmpeg or "
                    "torch+torchaudio+soundfile."
                )

        logger.info(f"Audio processing complete: {output_path}")
        return str(output_path), metadata

    @classmethod
    async def get_metadata(cls, audio_path: str) -> AudioMetadata:
        """
        Get audio metadata using best available backend.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioMetadata object
        """
        if _ffmpeg_available():
            try:
                return await cls._get_metadata_ffprobe(audio_path)
            except Exception as e:
                if TORCHAUDIO_AVAILABLE:
                    logger.warning(
                        f"ffprobe failed ({e}), using torchaudio fallback"
                    )
                else:
                    raise
        if TORCHAUDIO_AVAILABLE:
            return await cls._get_metadata_torchaudio(audio_path)
        else:
            raise RuntimeError(
                "No audio backend available for metadata extraction."
            )

    # ------------------------------------------------------------------
    # ffmpeg-based methods (primary)
    # ------------------------------------------------------------------

    @classmethod
    async def _get_metadata_ffprobe(cls, audio_path: str) -> AudioMetadata:
        """Extract metadata via ffprobe."""
        logger.debug(f"Extracting metadata (ffprobe): {audio_path}")

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [
                    cls.FFPROBE_BIN,
                    "-v", "error",
                    "-show_format",
                    "-show_streams",
                    "-print_format", "json",
                    audio_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    stderr=result.stderr,
                )

            data = json.loads(result.stdout)
            format_info = data.get("format", {})
            stream_info = data.get("streams", [{}])[0]

            duration = float(format_info.get("duration", 0))
            sample_rate = int(stream_info.get("sample_rate", 16000))
            channels = int(stream_info.get("channels", 1))
            codec = stream_info.get("codec_name", "unknown")
            bitrate = int(format_info.get("bit_rate", 0))
            file_size = int(format_info.get("size", 0))

            return AudioMetadata(
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                codec=codec,
                bitrate=bitrate,
                file_size=file_size,
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed: {e.stderr}")
            raise RuntimeError(
                f"Failed to extract audio metadata: {e.stderr}"
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse ffprobe output: {e}")
            raise RuntimeError(f"Invalid audio metadata format: {e}")
        except asyncio.TimeoutError:
            logger.error("ffprobe timed out")
            raise RuntimeError("Audio metadata extraction timed out")

    @classmethod
    async def _convert_ffmpeg(
        cls, input_path: str, output_path: str
    ) -> None:
        """Convert audio via ffmpeg with loudness normalization."""
        logger.debug(f"Converting (ffmpeg): {input_path} -> {output_path}")

        cmd = [
            cls.FFMPEG_BIN,
            "-i", input_path,
            "-ar", str(cls.TARGET_SAMPLE_RATE),
            "-ac", str(cls.TARGET_CHANNELS),
            "-af", "loudnorm=I=-20:TP=-1.5:LRA=11",
            "-y",
            output_path,
        ]

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    stderr=result.stderr,
                )

            if not Path(output_path).exists():
                raise RuntimeError("Output file was not created")

            output_size = Path(output_path).stat().st_size
            logger.info(
                f"Audio converted (ffmpeg): {output_size / 1e6:.1f} MB"
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg conversion failed: {e.stderr}")
            raise RuntimeError(f"Audio conversion failed: {e.stderr}")
        except asyncio.TimeoutError:
            logger.error("Audio conversion timed out")
            raise RuntimeError("Audio conversion timed out (>5 minutes)")

    # ------------------------------------------------------------------
    # torchaudio-based methods (fallback for Windows without ffmpeg)
    # ------------------------------------------------------------------

    @classmethod
    async def _get_metadata_torchaudio(
        cls, audio_path: str
    ) -> AudioMetadata:
        """Extract metadata via torchaudio/soundfile."""
        logger.debug(f"Extracting metadata (torchaudio): {audio_path}")

        def _extract():
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            file_size = Path(audio_path).stat().st_size

            # Determine codec from file extension
            ext = Path(audio_path).suffix.lower()
            codec_map = {
                ".wav": "pcm_s16le",
                ".mp3": "mp3",
                ".m4a": "aac",
                ".ogg": "vorbis",
                ".flac": "flac",
            }
            codec = codec_map.get(ext, "unknown")

            return AudioMetadata(
                duration=duration,
                sample_rate=info.sample_rate,
                channels=info.num_channels,
                codec=codec,
                bitrate=int(file_size * 8 / max(duration, 0.01)),
                file_size=file_size,
            )

        try:
            return await asyncio.to_thread(_extract)
        except Exception as e:
            logger.error(f"torchaudio metadata extraction failed: {e}")
            raise RuntimeError(
                f"Failed to extract audio metadata: {e}"
            ) from e

    @classmethod
    async def _convert_torchaudio(
        cls, input_path: str, output_path: str
    ) -> None:
        """Convert audio via torchaudio (resample + mono, no loudnorm)."""
        logger.debug(
            f"Converting (torchaudio): {input_path} -> {output_path}"
        )

        def _convert():
            waveform, sample_rate = torchaudio.load(input_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to target rate
            if sample_rate != cls.TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=cls.TARGET_SAMPLE_RATE,
                )
                waveform = resampler(waveform)

            # Simple peak normalization (no loudnorm filter available)
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak * 0.95

            # Save as 16-bit PCM WAV
            torchaudio.save(
                output_path,
                waveform,
                cls.TARGET_SAMPLE_RATE,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            output_size = Path(output_path).stat().st_size
            logger.info(
                f"Audio converted (torchaudio): {output_size / 1e6:.1f} MB"
            )

        try:
            await asyncio.to_thread(_convert)

            if not Path(output_path).exists():
                raise RuntimeError("Output file was not created")

        except Exception as e:
            if "Output file" in str(e):
                raise
            logger.error(f"torchaudio conversion failed: {e}")
            raise RuntimeError(
                f"Audio conversion failed (torchaudio): {e}"
            ) from e

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @classmethod
    async def validate(cls, audio_path: str) -> bool:
        """
        Validate if audio file is readable and valid.

        Args:
            audio_path: Path to audio file

        Returns:
            True if valid, False otherwise
        """
        try:
            metadata = await cls.get_metadata(audio_path)
            if metadata.duration <= 0:
                logger.warning(
                    f"Invalid audio duration: {metadata.duration}"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"Audio validation failed: {e}")
            return False

    @classmethod
    async def get_duration(cls, audio_path: str) -> float:
        """
        Get audio duration in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        metadata = await cls.get_metadata(audio_path)
        return metadata.duration
