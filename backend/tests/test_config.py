"""Tests for configuration loading and validation."""
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from backend.app.config import (
    AppConfig,
    ASRConfig,
    BrandingConfig,
    DiarizationConfig,
    OutputConfig,
    QualityConfig,
    ServerConfig,
    SummarizationConfig,
)


class TestServerConfig:
    """Server configuration tests."""

    def test_defaults(self):
        config = ServerConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.workers == 1
        assert "http://localhost:3000" in config.cors_origins

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EPAM_SERVER_HOST", "0.0.0.0")
        monkeypatch.setenv("EPAM_SERVER_PORT", "9000")
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 9000


class TestASRConfig:
    """ASR configuration tests."""

    def test_defaults(self):
        config = ASRConfig()
        assert config.engine == "qwen"
        assert config.qwen_model == "Qwen/Qwen3-ASR-1.7B"
        assert config.qwen_aligner_model == "Qwen/Qwen3-ForcedAligner-0.6B"
        assert "conformer" in config.model.lower()
        assert config.device == "auto"
        assert config.language == "ru"
        assert config.chunk_duration == 30.0
        assert 0 < config.vad_threshold < 1

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EPAM_ASR_DEVICE", "cpu")
        config = ASRConfig()
        assert config.device == "cpu"


class TestDiarizationConfig:
    """Diarization configuration tests."""

    def test_defaults(self):
        config = DiarizationConfig()
        assert config.engine == "pyannote"
        assert "community-1" in config.pyannote_model
        assert "ecapa" in config.embedding_model.lower()
        assert config.clustering in ("spectral", "agglomerative")
        assert config.min_speakers >= 1
        assert config.max_speakers <= 50

    def test_speaker_bounds(self):
        config = DiarizationConfig()
        assert config.min_speakers < config.max_speakers


class TestSummarizationConfig:
    """Summarization configuration tests."""

    def test_defaults(self):
        config = SummarizationConfig()
        assert config.ollama_base_url == "http://localhost:11434"
        assert "gemma" in config.ollama_model.lower()
        assert config.ollama_timeout > 0
        assert config.max_new_tokens > 0
        assert config.admin_max_new_tokens >= config.max_new_tokens
        assert config.ollama_num_ctx in {"auto", "max"} or str(
            config.ollama_num_ctx
        ).isdigit()
        assert 0.0 <= config.temperature <= 2.0


class TestQualityConfig:
    """Quality assurance configuration tests."""

    def test_defaults(self):
        config = QualityConfig()
        assert 0 < config.min_confidence <= 1
        assert config.min_audio_duration > 0
        assert config.max_audio_duration > config.min_audio_duration
        assert config.retry_count >= 0


class TestAppConfig:
    """Main application configuration tests."""

    def test_defaults(self):
        config = AppConfig()
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.asr, ASRConfig)
        assert isinstance(config.diarization, DiarizationConfig)
        assert isinstance(config.summarization, SummarizationConfig)
        assert isinstance(config.quality, QualityConfig)
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.branding, BrandingConfig)

    def test_yaml_loading(self, tmp_path):
        yaml_content = {
            "server": {"host": "0.0.0.0", "port": 9000},
            "asr": {"device": "cpu", "language": "ru"},
            "diarization": {"min_speakers": 1, "max_speakers": 10},
            "summarization": {"ollama_model": "qwen3:32b"},
            "quality": {"min_confidence": 0.7},
            "output": {"format": "json", "output_dir": str(tmp_path / "output")},
            "branding": {"primary_color": "#FF0000"},
        }

        yaml_path = tmp_path / "settings.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = AppConfig.load_from_yaml(yaml_path)
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 9000
        assert config.asr.device == "cpu"
        assert config.diarization.max_speakers == 10
        assert config.summarization.ollama_model == "qwen3:32b"
        assert config.quality.min_confidence == 0.7
        assert config.output.format == "json"

    def test_yaml_missing_file(self):
        config = AppConfig.load_from_yaml("/nonexistent/path.yaml")
        assert config.server.host == "127.0.0.1"  # Falls back to defaults

    def test_branding_defaults(self):
        config = BrandingConfig()
        assert config.primary_color == "#B2001F"  # EPAM red
        assert config.heading_font == "Georgia"
        assert config.body_font == "Arial Narrow"
