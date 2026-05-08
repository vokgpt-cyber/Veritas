"""WhisperX ASR engine — Whisper transcription + wav2vec2 forced alignment.

T3.6 (2026-04-23) pilot. WhisperX combines:
  - faster-whisper as the ASR backend (Whisper large-v3)
  - VAD-based chunking
  - wav2vec2 forced alignment for word-level timestamps significantly
    tighter than vanilla Whisper's ~chunk-level output

Why pilot this when GigaAM already wins on Russian word accuracy:
The stakeholder's biggest pain point is speaker attribution at
boundaries (interruptions, brief turns). Tight word-level timestamps
let pyannote attribute words individually rather than whole segments,
which would help the 15+ "speaker label fixes" she made by hand.

Trade-off: WhisperX uses base Whisper large-v3 (multilingual). On
pure Russian, GigaAM v3 has materially better WER (2.6-8.4% vs
Whisper base ~12-15%). So this is NOT a wholesale GigaAM replacement.
It's an alternative for cases where speaker boundaries matter most
(court hearings) at the cost of slightly worse Russian word recognition.

INSTALL (optional dependency, NOT in requirements.txt):
    venv\\Scripts\\python.exe -m pip install whisperx
    # wav2vec2 model auto-downloads on first run.

The engine is hidden behind a try-import. If `whisperx` isn't
installed, this module raises ImportError on `import` and the
orchestrator's `_instantiate_asr` will fall back to faster-whisper.
NEVER auto-routed by language detection — only used when
`config.asr.engine = "whisperx"` is set explicitly.
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

# WhisperX is intentionally optional. Don't fail at module import; let
# the engine class raise a clean error in __init__ if it's needed.
try:
    import whisperx  # type: ignore[import-not-found]
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    whisperx = None  # type: ignore[assignment]

from backend.app.models import TranscriptionSegment, WordInfo
from backend.engine.base import BaseEngine

logger = logging.getLogger(__name__)


# Default Russian wav2vec2 alignment model — community fine-tune of
# XLSR-53 on Russian. ~1.5GB VRAM. Override via
# `config.asr.whisperx_align_model` if needed.
DEFAULT_RUSSIAN_ALIGN_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"


class WhisperXASREngine(BaseEngine):
    """WhisperX wrapper: faster-whisper transcribe + wav2vec2 align.

    Engine key: "whisperx" (explicit only, never auto-routed).

    Pipeline:
      1. Load faster-whisper Whisper model (`large-v3` by default).
      2. Load wav2vec2 alignment model for the detected language.
      3. Transcribe → align in one pass.
      4. Return TranscriptionSegment with word-level timestamps from
         the wav2vec2 forced alignment, NOT from Whisper's coarse
         segment output.

    VRAM budget: ~3 GB Whisper large-v3 (float16) + ~1.5 GB wav2vec2
    = ~4.5 GB peak. Comfortable on RTX 3090.
    """

    def __init__(self, config: Any) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for WhisperXASREngine. "
                "Install: pip install torch"
            )
        if not WHISPERX_AVAILABLE:
            raise ImportError(
                "whisperx is not installed. This engine is optional — "
                "install with: pip install whisperx"
            )
        super().__init__(config)
        self._model = None
        self._align_model = None
        self._align_metadata: Optional[dict[str, Any]] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._compute_type = "float16" if self._device == "cuda" else "int8"

    @property
    def required_vram_gb(self) -> float:
        # Whisper large-v3 ~3GB + wav2vec2 ~1.5GB peak.
        return 4.5

    def load(self) -> None:
        """Load Whisper + wav2vec2 alignment model."""
        if self.is_loaded:
            return

        whisper_model = self._config.asr.whisper_model or "large-v3"
        align_model_name = getattr(
            self._config.asr,
            "whisperx_align_model",
            DEFAULT_RUSSIAN_ALIGN_MODEL,
        )
        language = self._config.asr.language or "ru"

        try:
            logger.info(
                "Loading WhisperX: %s on %s (%s)",
                whisper_model, self._device, self._compute_type,
            )
            # whisperx.load_model() takes the same compute_type strings
            # as faster-whisper.
            initial_prompt = (self._config.asr.initial_prompt or "").strip()
            custom_hot_words = str(
                getattr(self, "_current_hot_words", "") or ""
            ).strip()
            if custom_hot_words:
                initial_prompt = (
                    f"{initial_prompt} {custom_hot_words}".strip()
                    if initial_prompt
                    else custom_hot_words
                )

            self._model = whisperx.load_model(
                whisper_model,
                device=self._device,
                compute_type=self._compute_type,
                language=language,
                # asr_options: forwarded into faster-whisper. Use the
                # same initial_prompt as the standalone whisper engine
                # so legal vocab biases the decoder consistently.
                asr_options={
                    "initial_prompt": initial_prompt or None,
                    "beam_size": int(self._config.asr.beam_size or 1),
                },
            )
            logger.info("Whisper model loaded; loading wav2vec2 align: %s", align_model_name)
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language,
                device=self._device,
                model_name=align_model_name,
            )
            logger.info("WhisperX engine loaded (whisper + wav2vec2 alignment)")
        except Exception as exc:
            logger.error("WhisperX load failed: %s", exc)
            raise RuntimeError(f"WhisperX load failed: {exc}") from exc

    def unload(self) -> None:
        """Free both models from VRAM."""
        if not self.is_loaded:
            return
        try:
            self._model = None
            self._align_model = None
            self._align_metadata = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("WhisperX engine unloaded")
        except Exception as exc:
            logger.warning("WhisperX unload error (non-fatal): %s", exc)

    async def process(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> list[TranscriptionSegment]:
        """Run WhisperX transcribe + align."""
        if not self.is_loaded:
            raise RuntimeError("WhisperXASREngine not loaded. Call load() first.")

        audio_path = str(audio_path)
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            if progress_callback:
                progress_callback(0.05, "Loading audio for WhisperX")

            # whisperx.load_audio reads WAV at 16kHz mono. We've already
            # preprocessed to that format upstream so this is fast.
            audio = whisperx.load_audio(audio_path)

            if progress_callback:
                progress_callback(0.10, "Transcribing")

            # Transcribe → returns dict with "segments" and "language".
            # batch_size=16 is the documented WhisperX default for
            # large-v3 on a 24 GB GPU.
            transcribe_result = self._model.transcribe(
                audio,
                batch_size=16,
                language=self._config.asr.language or "ru",
            )
            language = transcribe_result.get(
                "language", self._config.asr.language or "ru"
            )
            logger.info(
                "WhisperX transcribe: %d segments, language=%s",
                len(transcribe_result.get("segments", [])), language,
            )

            if progress_callback:
                progress_callback(0.55, "Aligning words to audio")

            # Align word timestamps. The aligner returns the SAME
            # segments but with `words` populated by phoneme-level
            # forced alignment.
            aligned_result = whisperx.align(
                transcribe_result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                device=self._device,
                # return_char_alignments=False keeps the result compact.
                return_char_alignments=False,
            )

            # Convert WhisperX segments → our TranscriptionSegment
            # format. Word-level timestamps come from `words` list
            # populated by the aligner.
            segments: list[TranscriptionSegment] = []
            for seg in aligned_result.get("segments", []):
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                start = float(seg.get("start") or 0.0)
                end = float(seg.get("end") or start)
                # WhisperX confidence is per-word. We don't have a
                # segment-level score so use a flat 0.95 the way we do
                # for GigaAM (which also doesn't surface a per-segment
                # confidence). UI uses attribution_confidence instead.
                seg_conf = 0.95

                words: list[WordInfo] = []
                for w in seg.get("words", []) or []:
                    if "start" not in w or "end" not in w:
                        continue
                    word_text = (w.get("word") or "").strip()
                    if not word_text:
                        continue
                    words.append(
                        WordInfo(
                            start=round(float(w["start"]), 3),
                            end=round(float(w["end"]), 3),
                            word=word_text,
                            confidence=float(w.get("score", 0.95)),
                        )
                    )

                segments.append(
                    TranscriptionSegment(
                        start=round(start, 3),
                        end=round(end, 3),
                        text=text,
                        confidence=seg_conf,
                        words=words,
                    )
                )

            if progress_callback:
                progress_callback(1.0, "Completed")

            logger.info(
                "WhisperX aligned: %d segments, total words: %d",
                len(segments),
                sum(len(s.words or []) for s in segments),
            )
            return segments

        except Exception as exc:
            logger.error("WhisperX process failed: %s", exc)
            raise RuntimeError(f"WhisperX transcription error: {exc}") from exc
