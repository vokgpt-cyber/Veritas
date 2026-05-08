"""Speaker diarization engine using pyannote-audio 4.0 pipeline.

pyannote-audio 4.0 with the community-1 model provides state-of-the-art
diarization using VBx clustering (replacing agglomerative from 3.x).
Natively supports min_speakers/max_speakers constraints.

Requires: pip install pyannote-audio>=4.0
VRAM: ~9-10GB for inference (community-1 model on RTX 3090).
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
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

from backend.app.models import DiarizationSegment
from backend.engine.base import BaseEngine

logger = logging.getLogger(__name__)


class PyannoteDiarizationEngine(BaseEngine):
    """Speaker diarization using pyannote-audio 4.0 pipeline.

    Uses the community-1 model which employs VBx (Variational Bayes)
    clustering for significantly better speaker counting and assignment
    compared to the agglomerative clustering in pyannote 3.x.

    The pipeline performs:
      1. Neural speaker segmentation (local speaker detection)
      2. ECAPA-TDNN embedding extraction
      3. VBx clustering with speaker count hints

    Output is a DiarizeOutput dataclass with:
      - speaker_diarization: standard Annotation (with overlapping speech)
      - exclusive_speaker_diarization: non-overlapping version
      - speaker_embeddings: per-speaker embeddings (num_speakers, dim)
    """

    def __init__(self, config: Any) -> None:
        """Initialize pyannote diarization engine.

        Args:
            config: AppConfig instance with diarization settings.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for pyannote diarization. "
                "Install with: pip install torch"
            )
        if not PYANNOTE_AVAILABLE:
            raise ImportError(
                "pyannote-audio is required for pyannote diarization engine. "
                "Install with: pip install pyannote-audio>=4.0"
            )

        super().__init__(config)
        self._pipeline: Optional[PyannotePipeline] = None

    @property
    def required_vram_gb(self) -> float:
        """pyannote 4.0 community-1 uses ~9-10GB VRAM."""
        return 9.5

    def load(self) -> None:
        """Load pyannote diarization pipeline, offline-first.

        pyannote 4.0 breaking changes from 3.x:
          - use_auth_token -> token
          - PYANNOTE_CACHE env removed, uses huggingface_hub cache
        """
        if self.is_loaded:
            logger.warning("Pyannote diarization engine already loaded")
            return

        import os

        model_name = self._config.diarization.pyannote_model
        hf_token = os.environ.get("HF_TOKEN") or self._config.diarization.hf_token

        try:
            is_cached = self._is_model_cached(model_name)

            if is_cached:
                logger.info(
                    f"Loading pyannote pipeline from local cache: {model_name}"
                )
                previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "1"
            else:
                previous_hf_offline = None
                if not hf_token:
                    raise RuntimeError(
                        "HF_TOKEN is required to download "
                        "pyannote/speaker-diarization-community-1. Accept "
                        "the model terms on Hugging Face, create a read "
                        "token, set HF_TOKEN in .env.docker, and restart "
                        "Docker."
                    )
                logger.info(
                    f"First run -- downloading pyannote pipeline: {model_name}. "
                    f"Will be cached for offline use."
                )

            try:
                load_kwargs = {}
                if hf_token:
                    load_kwargs["token"] = hf_token

                self._pipeline = PyannotePipeline.from_pretrained(
                    model_name, **load_kwargs
                )
            finally:
                if is_cached:
                    if previous_hf_offline is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = previous_hf_offline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline = self._pipeline.to(torch.device(device))

            self._model = True  # Mark as loaded
            logger.info(f"Pyannote diarization pipeline loaded on {device}")

        except Exception as e:
            logger.error(f"Failed to load pyannote pipeline: {e}")
            raise RuntimeError(f"Pyannote pipeline loading failed: {e}") from e

    @staticmethod
    def _mark_overlaps(
        segments: list[DiarizationSegment],
        min_overlap_s: float = 0.05,
    ) -> None:
        """Mark segments that overlap in time with a different speaker.

        Edits `segments` in place. A segment is flagged when any other
        segment from a DIFFERENT speaker shares more than min_overlap_s
        seconds of time with it. Tiny overlaps (<50 ms, common at
        clustering boundaries) are ignored to avoid false positives.

        Used by the aligner to downgrade attribution_confidence so the
        editor UI can flag interruption regions as "verify here".
        """
        n = len(segments)
        if n < 2:
            return
        # segments are sorted by start; sweep forward checking only
        # segments whose start < cur.end + min_overlap_s.
        for i, cur in enumerate(segments):
            for j in range(i + 1, n):
                other = segments[j]
                if other.start >= cur.end:
                    break  # sorted; no further overlap possible
                if other.speaker_id == cur.speaker_id:
                    continue
                ov_start = max(cur.start, other.start)
                ov_end = min(cur.end, other.end)
                if ov_end - ov_start >= min_overlap_s:
                    cur.is_overlap = True
                    other.is_overlap = True

    @staticmethod
    def _is_model_cached(model_name: str) -> bool:
        """Check if pyannote model is cached locally."""
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
        """Unload pipeline and free VRAM."""
        if not self.is_loaded:
            return

        try:
            if self._pipeline is not None:
                del self._pipeline
                self._pipeline = None

            self._model = None

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Pyannote diarization engine unloaded")

        except Exception as e:
            logger.error(f"Error during pyannote unload: {e}")

    def _apply_meeting_type_overrides(self, meeting_type: Optional[str]) -> None:
        """Apply meeting-type-specific pyannote hyperparameter overrides.

        Currently only ``court_hearing`` has overrides — other types use
        the pipeline's pretrained defaults. Wrapped in try/except so a
        param-name mismatch (e.g. pyannote internal rename) logs a
        warning and falls through to defaults instead of crashing the
        whole pipeline. The pipeline is reloaded between meetings, so
        overrides applied here don't leak into the next run.

        Calibration evidence (court_hearing_129, 2026-04-30):
        - Default clustering.threshold (~0.7) merged two same-gender
          lawyers into one cluster, producing a 59/33/8 speaker share.
        - Lowering to 0.55 separates them while keeping the judge's
          cluster intact.
        """
        if meeting_type != "court_hearing":
            return
        cfg = self._config.diarization
        overrides: dict[str, Any] = {}
        if cfg.court_hearing_clustering_threshold is not None:
            overrides.setdefault("clustering", {})["threshold"] = (
                cfg.court_hearing_clustering_threshold
            )
        if cfg.court_hearing_segmentation_min_duration_off is not None:
            overrides.setdefault("segmentation", {})["min_duration_off"] = (
                cfg.court_hearing_segmentation_min_duration_off
            )
        if not overrides:
            return
        try:
            self._pipeline.instantiate(overrides)
            logger.info(
                "Applied court_hearing pyannote overrides: %s", overrides
            )
        except Exception as e:
            logger.warning(
                "Failed to apply court_hearing pyannote overrides %s: %s. "
                "Falling back to pipeline defaults.",
                overrides, e,
            )

    async def process(
        self,
        audio_path: str,
        speech_segments: Optional[list[dict]] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        meeting_type: Optional[str] = None,
    ) -> list[DiarizationSegment]:
        """Run pyannote 4.0 diarization pipeline.

        pyannote 4.0 returns DiarizeOutput with .speaker_diarization
        (Annotation) instead of returning Annotation directly.

        Args:
            audio_path: Path to 16kHz mono WAV file.
            speech_segments: Optional VAD output (not used by pyannote).
            min_speakers: Minimum expected number of speakers.
            max_speakers: Maximum expected number of speakers.
            progress_callback: Optional callback(progress, message).
            meeting_type: One of MeetingType enum values. When
                ``"court_hearing"``, applies aggressive segmentation
                overrides (lower clustering threshold, shorter min
                segment duration) tuned for multiple same-gender
                lawyers. Other values use defaults.

        Returns:
            List of DiarizationSegment with speaker assignments.
        """
        if not self.is_loaded or self._pipeline is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            self._logger.info(
                f"Starting pyannote diarization of {audio_path} "
                f"(meeting_type={meeting_type or 'default'})"
            )

            # Apply court_hearing-specific clustering/segmentation overrides
            # if applicable. No-op for other meeting types. Must come
            # before the pipeline is invoked.
            self._apply_meeting_type_overrides(meeting_type)

            if progress_callback:
                progress_callback(0.1, "Loading audio for diarization")

            pipeline_params = {}
            exact_speakers = (
                min_speakers
                if min_speakers is not None
                and max_speakers is not None
                and min_speakers == max_speakers
                else None
            )
            if exact_speakers is not None:
                # Exact speaker count is stronger than min=max for
                # pyannote. Court hearings rely on this: when the user
                # says "4 speakers", treat it as a hard clustering target.
                pipeline_params["num_speakers"] = exact_speakers
            elif min_speakers is not None:
                pipeline_params["min_speakers"] = min_speakers
            elif self._config.diarization.min_speakers:
                pipeline_params["min_speakers"] = self._config.diarization.min_speakers
            if exact_speakers is None and max_speakers is not None:
                pipeline_params["max_speakers"] = max_speakers
            elif exact_speakers is None and self._config.diarization.max_speakers:
                pipeline_params["max_speakers"] = self._config.diarization.max_speakers

            if progress_callback:
                progress_callback(0.2, "Running diarization pipeline")

            # Pre-load audio via torchaudio to bypass pyannote's torchcodec
            # dependency (torchcodec is incompatible with torch 2.8.0 on Windows).
            # pyannote accepts {"waveform": Tensor, "sample_rate": int} dicts.
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}

            try:
                raw_output = self._pipeline(audio_input, **pipeline_params)
            except TypeError:
                if exact_speakers is None:
                    raise
                self._logger.warning(
                    "pyannote rejected num_speakers; falling back to "
                    "min_speakers=max_speakers=%d",
                    exact_speakers,
                )
                raw_output = self._pipeline(
                    audio_input,
                    min_speakers=exact_speakers,
                    max_speakers=exact_speakers,
                )

            if progress_callback:
                progress_callback(0.8, "Extracting speaker segments")

            # pyannote 4.0 returns DiarizeOutput dataclass.
            # Access .speaker_diarization for the Annotation object.
            # Fall back to raw_output if it's already an Annotation (3.x compat).
            if hasattr(raw_output, "speaker_diarization"):
                diarization_annotation = raw_output.speaker_diarization
            else:
                diarization_annotation = raw_output

            # Convert pyannote Annotation to our DiarizationSegment format
            segments = []
            for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
                segments.append(DiarizationSegment(
                    start=round(turn.start, 3),
                    end=round(turn.end, 3),
                    speaker_id=speaker,
                ))

            segments.sort(key=lambda s: s.start)

            # Court hearings keep shorter interjections (0.2s default vs
            # the 0.5s global) to surface quick speaker changes that
            # clustering can use as evidence of distinct voices.
            if meeting_type == "court_hearing":
                min_duration = self._config.diarization.court_hearing_min_segment_duration
            else:
                min_duration = self._config.diarization.min_segment_duration
            filtered = [
                seg for seg in segments
                if seg.end - seg.start >= min_duration
            ]

            # Tier 2 (2026-04-23): mark segments that overlap in time
            # with another speaker's segment as is_overlap=True.
            # pyannote 4.0's `speaker_diarization` already includes
            # overlap regions (vs `exclusive_speaker_diarization` which
            # snaps to one speaker per moment); we just need to flag
            # the segments that participate. Sweep-line algorithm: for
            # each segment, check whether any OTHER-speaker segment
            # intersects. O(N log N) sort + O(N*window) scan.
            self._mark_overlaps(filtered)
            overlap_count = sum(1 for s in filtered if s.is_overlap)

            n_speakers = len(set(s.speaker_id for s in filtered))
            if exact_speakers is not None and n_speakers != exact_speakers:
                self._logger.warning(
                    "Expected exactly %d speaker(s), but pyannote returned %d "
                    "after filtering. Human review is required.",
                    exact_speakers,
                    n_speakers,
                )
            self._logger.info(
                f"Pyannote diarization complete: {len(filtered)} segments, "
                f"{n_speakers} speakers, {overlap_count} in overlap regions"
            )

            if progress_callback:
                progress_callback(1.0, "Completed")

            return filtered

        except Exception as e:
            self._logger.error(f"Pyannote diarization failed: {e}")
            raise RuntimeError(f"Pyannote diarization error: {e}") from e
