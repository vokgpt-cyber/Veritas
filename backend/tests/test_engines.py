"""Tests for ML engine modules (import, init, and interface validation)."""
import numpy as np
import pytest
import torch

from backend.app.config import AppConfig
from backend.engine.base import BaseEngine
from backend.engine.vad import SileroVAD


class TestBaseEngine:
    """Abstract engine base class tests."""

    def test_cannot_instantiate_directly(self):
        """BaseEngine is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseEngine(config=None)

    def test_concrete_implementation(self):
        """A concrete subclass must implement all abstract methods."""

        class DummyEngine(BaseEngine):
            @property
            def required_vram_gb(self) -> float:
                return 0.0

            def load(self) -> None:
                self._model = "loaded"

            def unload(self) -> None:
                self._model = None

            async def process(self, *args, **kwargs):
                return "result"

        engine = DummyEngine(config=None)
        assert engine.name == "DummyEngine"
        assert not engine.is_loaded

        engine.load()
        assert engine.is_loaded

        engine.unload()
        assert not engine.is_loaded


class TestSileroVAD:
    """Silero VAD engine tests."""

    def test_init_params(self):
        """Test VAD initialization with custom parameters."""
        vad = SileroVAD(
            threshold=0.7,
            min_speech_ms=300,
            min_silence_ms=150,
            sample_rate=16000,
        )
        assert vad._threshold == 0.7
        assert vad._min_speech_ms == 300
        assert not vad.is_loaded

    def test_detect_requires_loaded_model(self):
        """Calling detect() before load() should raise RuntimeError."""
        vad = SileroVAD()
        with pytest.raises(RuntimeError, match="not loaded"):
            vad.detect(torch.zeros(16000))

    def test_detect_invalid_input(self):
        """Detect should reject invalid audio types."""
        vad = SileroVAD()
        vad._is_loaded = True  # Force loaded state
        vad._model = torch.nn.Linear(1, 1)  # Dummy model
        with pytest.raises((ValueError, RuntimeError)):
            vad.detect("not_a_tensor")

    def test_detect_from_file_missing(self):
        """Detect from nonexistent file should raise FileNotFoundError."""
        vad = SileroVAD()
        vad._is_loaded = True
        with pytest.raises(FileNotFoundError):
            vad.detect_from_file("/nonexistent/audio.wav")


class TestNeMoASREngine:
    """NeMo ASR engine tests (import and interface only, no model loading)."""

    def test_import(self):
        """ASR engine module should import successfully."""
        from backend.engine.asr import NeMoASREngine
        assert NeMoASREngine is not None

    def test_inherits_base(self):
        """ASR engine should inherit from BaseEngine."""
        from backend.engine.asr import NeMoASREngine
        assert issubclass(NeMoASREngine, BaseEngine)

    def test_vram_requirement(self):
        """ASR engine should declare VRAM requirement."""
        from backend.engine.asr import NeMoASREngine
        config = AppConfig()
        try:
            engine = NeMoASREngine(config)
            assert engine.required_vram_gb > 0
        except ImportError:
            pytest.skip("NeMo not available")


class TestDiarizationEngine:
    """pyannote diarization engine tests."""

    def test_import(self):
        """pyannote diarization engine should import successfully."""
        from backend.engine.pyannote_diarization import PyannoteDiarizationEngine

        assert PyannoteDiarizationEngine is not None

    def test_inherits_base(self):
        """pyannote diarization engine should inherit from BaseEngine."""
        from backend.engine.pyannote_diarization import PyannoteDiarizationEngine

        assert issubclass(PyannoteDiarizationEngine, BaseEngine)

    def test_vram_requirement(self):
        """pyannote diarization engine should declare VRAM requirement."""
        from backend.engine.pyannote_diarization import PyannoteDiarizationEngine

        engine = PyannoteDiarizationEngine(AppConfig())
        assert engine.required_vram_gb >= 9.0


class TestSummarizationEngine:
    """Summarization engine tests (import, config, and parsing only)."""

    def test_import(self):
        """Summarization engine should import successfully."""
        from backend.engine.summarization import SummarizationEngine
        assert SummarizationEngine is not None

    def test_inherits_base(self):
        """Summarization engine should inherit from BaseEngine."""
        from backend.engine.summarization import SummarizationEngine
        assert issubclass(SummarizationEngine, BaseEngine)

    def test_vram_zero_ollama_managed(self):
        """Ollama-based engine reports 0 VRAM (Ollama manages its own GPU)."""
        from backend.engine.summarization import SummarizationEngine
        config = AppConfig()
        engine = SummarizationEngine(config)
        assert engine.required_vram_gb == 0.0

    def test_config_defaults(self):
        """Verify default Ollama config values."""
        from backend.app.config import SummarizationConfig
        cfg = SummarizationConfig()
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.ollama_model == "gemma4:26b"
        assert cfg.ollama_timeout == 300
        assert cfg.temperature == 0.1
        assert cfg.max_new_tokens == 8192
        assert cfg.admin_max_new_tokens == 12288
        assert cfg.ollama_num_ctx == "auto"

    def test_parse_protocol_sections_russian(self):
        """Parser should extract all 7 sections from structured output."""
        from backend.engine.summarization import SummarizationEngine

        raw = (
            "KRATKOYE SODERZHANIYE:\n"
            "Na soveshchanii obsuzhdalis' voprosy po delu Ivanova.\n"
            "Resheno podgotovit' apellyatsiyu.\n\n"
            "KLYUCHEVYE TEMY OBSUZHDENIYA:\n"
            "- Analiz sudebnogo resheniya\n"
            "- Strategiya apellyatsii\n\n"
            "RESHENIYA:\n"
            "- Podgotovit' apellyatsionnuyu zhalobu\n\n"
            "PORUCHENIYA:\n"
            "- Sostavit' proekt zhaloby | Petrov A.V. | 15.04.2026\n"
            "- Podgotovit' dokazatel'stva | Sidorov B.C. | 20.04.2026\n\n"
            "OTKRYTYE VOPROSY:\n"
            "- Srok podachi zhaloby\n"
        )

        parsed = SummarizationEngine._parse_protocol_sections(raw)

        assert "soveshchanii" in parsed["summary"]
        assert len(parsed["topics"]) == 2
        assert len(parsed["decisions"]) == 1
        assert len(parsed["tasks"]) == 2
        assert parsed["tasks"][0]["assignee"] == "Petrov A.V."
        assert parsed["tasks"][0]["deadline"] == "15.04.2026"
        assert len(parsed["questions"]) == 1

    def test_parse_protocol_sections_english(self):
        """Parser should handle English section headers."""
        from backend.engine.summarization import SummarizationEngine

        raw = (
            "BRIEF SUMMARY:\n"
            "The meeting discussed the appeal strategy.\n\n"
            "KEY DISCUSSION POINTS:\n"
            "- Court ruling analysis\n\n"
            "DECISIONS:\n"
            "- None\n\n"
            "ACTION ITEMS:\n"
            "- Draft appeal | Smith | 2026-04-15\n\n"
            "OPEN QUESTIONS:\n"
            "- Filing deadline\n"
        )

        parsed = SummarizationEngine._parse_protocol_sections(raw)

        assert "appeal" in parsed["summary"]
        assert len(parsed["topics"]) == 1
        assert len(parsed["decisions"]) == 0  # "None" should be filtered
        assert len(parsed["tasks"]) == 1
        assert parsed["tasks"][0]["assignee"] == "Smith"
        assert len(parsed["questions"]) == 1

    def test_parse_protocol_sections_empty(self):
        """Parser should handle empty/garbled input gracefully."""
        from backend.engine.summarization import SummarizationEngine

        parsed = SummarizationEngine._parse_protocol_sections("")
        assert parsed["summary"] == ""
        assert parsed["topics"] == []
        assert parsed["decisions"] == []
        assert parsed["tasks"] == []
        assert parsed["questions"] == []

    def test_format_transcript(self):
        """Transcript formatting should produce readable text."""
        from backend.engine.summarization import SummarizationEngine
        from backend.app.models import AlignedSegment

        segments = [
            AlignedSegment(
                text="Privet", start=0.0, end=1.0,
                speaker_id="spk_0", speaker_name="Ivanov",
                confidence=0.95,
            ),
            AlignedSegment(
                text="Dobryj den'", start=1.5, end=3.0,
                speaker_id="spk_1", speaker_name="Petrov",
                confidence=0.90,
            ),
        ]
        result = SummarizationEngine._format_transcript(segments)
        assert "Ivanov [00:00:00]: Privet" in result
        assert "Petrov [00:00:01]:" in result


class TestQwen3ASREngine:
    """Qwen3-ASR engine tests (import, config, and word grouping logic)."""

    def test_import(self):
        """Qwen3-ASR engine module should import successfully."""
        from backend.engine.qwen_asr import Qwen3ASREngine
        assert Qwen3ASREngine is not None

    def test_inherits_base(self):
        """Qwen3-ASR engine should inherit from BaseEngine."""
        from backend.engine.qwen_asr import Qwen3ASREngine
        assert issubclass(Qwen3ASREngine, BaseEngine)

    def test_vram_requirement(self):
        """Qwen3-ASR should report ~4GB VRAM (1.7B + 0.6B aligner)."""
        from backend.engine.qwen_asr import Qwen3ASREngine
        config = AppConfig()
        try:
            engine = Qwen3ASREngine(config)
            assert engine.required_vram_gb == 4.0
        except ImportError:
            pytest.skip("qwen-asr not installed")

    def test_config_defaults(self):
        """Verify default Qwen3-ASR config values."""
        from backend.app.config import ASRConfig
        cfg = ASRConfig()
        assert cfg.engine == "qwen"
        assert cfg.qwen_model == "Qwen/Qwen3-ASR-1.7B"
        assert cfg.qwen_aligner_model == "Qwen/Qwen3-ForcedAligner-0.6B"

    def test_language_mapping(self):
        """Language code to name mapping should work for Russian."""
        from backend.engine.qwen_asr import _LANG_CODE_TO_NAME
        assert _LANG_CODE_TO_NAME["ru"] == "Russian"
        assert _LANG_CODE_TO_NAME["en"] == "English"
        assert "zh" in _LANG_CODE_TO_NAME

    def test_group_words_into_segments(self):
        """Word grouping should split on punctuation and pauses."""
        from backend.engine.qwen_asr import Qwen3ASREngine
        from backend.app.models import WordInfo

        try:
            engine = Qwen3ASREngine(AppConfig())
        except ImportError:
            pytest.skip("qwen-asr not installed")

        words = [
            WordInfo(start=0.0, end=0.3, word="Dobryj", confidence=0.95),
            WordInfo(start=0.3, end=0.6, word="den'.", confidence=0.95),
            WordInfo(start=1.5, end=1.8, word="Kak", confidence=0.95),
            WordInfo(start=1.8, end=2.1, word="dela?", confidence=0.95),
        ]

        segments = engine._group_words_into_segments(words, pause_threshold=0.5)
        # Should split after "den'." (punctuation) and before "Kak" (pause > 0.5s)
        assert len(segments) == 2
        assert "den'" in segments[0].text
        assert "Kak" in segments[1].text

    def test_words_to_segment(self):
        """Static method should create proper TranscriptionSegment from words."""
        from backend.engine.qwen_asr import Qwen3ASREngine
        from backend.app.models import WordInfo

        words = [
            WordInfo(start=1.0, end=1.5, word="Test", confidence=0.9),
            WordInfo(start=1.5, end=2.0, word="slovo.", confidence=0.8),
        ]

        seg = Qwen3ASREngine._words_to_segment(words)
        assert seg.start == 1.0
        assert seg.end == 2.0
        assert seg.text == "Test slovo."
        assert len(seg.words) == 2
        assert seg.confidence == pytest.approx(0.85, abs=0.01)
