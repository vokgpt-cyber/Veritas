"""Block 5: Docker compose and deployment verification tests.

Tests verify:
- Dockerfile syntax and security hardening
- docker-compose.yml structure and configuration
- Environment variable handling
- Security settings (non-root, read-only rootfs, dropped caps)
- Network isolation settings
- Volume mounts and resource limits
- Configuration file validity
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.yaml"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


# ---------------------------------------------------------------------------
# Dockerfile Tests
# ---------------------------------------------------------------------------


class TestDockerfile:
    """Verify Dockerfile security hardening and structure."""

    @pytest.fixture(autouse=True)
    def _load_dockerfile(self):
        if not DOCKERFILE.exists():
            pytest.skip("Dockerfile not found")
        self.content = DOCKERFILE.read_text()

    def test_base_image_is_cuda(self):
        """Base image should be NVIDIA CUDA."""
        assert "nvidia/cuda" in self.content or "cuda" in self.content.lower(), \
            "Dockerfile should use NVIDIA CUDA base image"

    def test_non_root_user(self):
        """Container should run as non-root user."""
        assert "USER" in self.content, "Dockerfile must specify a non-root USER"
        # Should not be USER root at the end
        user_lines = [line.strip() for line in self.content.split("\n")
                      if line.strip().startswith("USER")]
        if user_lines:
            last_user = user_lines[-1]
            assert "root" not in last_user.lower(), "Final USER should not be root"

    def test_no_curl_wget_in_final(self):
        """curl/wget/git should be removed from final image."""
        # Look for removal commands
        has_removal = ("rm" in self.content and ("curl" in self.content or "wget" in self.content)) or \
                      "apt-get remove" in self.content or "apt-get purge" in self.content
        # Or the tools were never installed
        no_install = "curl" not in self.content and "wget" not in self.content
        assert has_removal or no_install, \
            "curl/wget should be removed from final image for security"

    def test_workdir_set(self):
        """WORKDIR should be explicitly set."""
        assert "WORKDIR" in self.content

    def test_healthcheck_present(self):
        """Dockerfile should include a HEALTHCHECK."""
        # HEALTHCHECK can be in Dockerfile or docker-compose
        compose_content = COMPOSE_FILE.read_text() if COMPOSE_FILE.exists() else ""
        has_healthcheck = "HEALTHCHECK" in self.content or "healthcheck" in compose_content
        assert has_healthcheck, "Should have a healthcheck defined"

    def test_expose_port(self):
        """Should EXPOSE the API port."""
        assert "EXPOSE" in self.content or "8000" in self.content


# ---------------------------------------------------------------------------
# Docker Compose Tests
# ---------------------------------------------------------------------------


class TestDockerCompose:
    """Verify docker-compose.yml structure and security settings."""

    @pytest.fixture(autouse=True)
    def _load_compose(self):
        if not COMPOSE_FILE.exists():
            pytest.skip("docker-compose.yml not found")
        with open(COMPOSE_FILE) as f:
            self.compose = yaml.safe_load(f)
        self.content = COMPOSE_FILE.read_text()

    def test_compose_has_services(self):
        """Compose file should define services."""
        assert "services" in self.compose
        assert len(self.compose["services"]) >= 1

    def test_service_has_security_options(self):
        """Services should have security_opt configured."""
        for name, service in self.compose["services"].items():
            if "build" in service or "image" in service:
                security_opt = service.get("security_opt", [])
                has_no_new_privs = any("no-new-privileges" in str(opt) for opt in security_opt)
                assert has_no_new_privs, \
                    f"Service '{name}' should have no-new-privileges security option"

    def test_service_has_resource_limits(self):
        """Services should have resource limits."""
        for name, service in self.compose["services"].items():
            if "build" in service or "image" in service:
                deploy = service.get("deploy", {})
                resources = deploy.get("resources", {})
                # Check either deploy.resources or direct mem_limit
                has_limits = bool(resources) or "mem_limit" in service
                assert has_limits, f"Service '{name}' should have resource limits"

    def test_service_has_read_only_rootfs(self):
        """Services should have read_only rootfs."""
        for name, service in self.compose["services"].items():
            if "build" in service or "image" in service:
                read_only = service.get("read_only", False)
                assert read_only, f"Service '{name}' should have read_only: true"

    def test_service_drops_capabilities(self):
        """Services should drop all capabilities."""
        for name, service in self.compose["services"].items():
            if "build" in service or "image" in service:
                cap_drop = service.get("cap_drop", [])
                assert "ALL" in cap_drop, \
                    f"Service '{name}' should drop ALL capabilities"

    def test_service_has_tmpfs(self):
        """Services should use tmpfs for /tmp."""
        for name, service in self.compose["services"].items():
            if "build" in service or "image" in service:
                tmpfs = service.get("tmpfs", [])
                has_tmp = any("/tmp" in str(t) for t in (tmpfs if isinstance(tmpfs, list) else [tmpfs]))
                assert has_tmp, f"Service '{name}' should have tmpfs for /tmp"

    def test_network_isolation(self):
        """Compose should define isolated networks."""
        networks = self.compose.get("networks", {})
        # Either explicit network definition or default bridge
        assert networks or "network_mode" in str(self.compose), \
            "Should define isolated networks"

    def test_no_host_network_mode(self):
        """Services should not use host network mode."""
        for name, service in self.compose["services"].items():
            network_mode = service.get("network_mode", "")
            assert network_mode != "host", \
                f"Service '{name}' should not use host network mode"


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestConfigFiles:
    """Verify configuration files are valid and complete."""

    def test_settings_yaml_valid(self):
        """settings.yaml should be valid YAML."""
        if not SETTINGS_FILE.exists():
            pytest.skip("settings.yaml not found")

        with open(SETTINGS_FILE) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert isinstance(config, dict)

    def test_settings_yaml_has_all_sections(self):
        """settings.yaml should have all configuration sections."""
        if not SETTINGS_FILE.exists():
            pytest.skip("settings.yaml not found")

        with open(SETTINGS_FILE) as f:
            config = yaml.safe_load(f)

        expected_sections = ["server", "asr", "diarization", "summarization",
                             "quality", "output", "branding"]
        for section in expected_sections:
            assert section in config, f"settings.yaml missing '{section}' section"

    def test_env_example_exists(self):
        """An .env.example file should exist."""
        assert ENV_EXAMPLE.exists(), ".env.example should exist for documentation"

    def test_env_example_has_critical_vars(self):
        """The .env.example should document critical variables."""
        if not ENV_EXAMPLE.exists():
            pytest.skip(".env.example not found")

        content = ENV_EXAMPLE.read_text()
        critical_vars = ["EPAM_AUTH_SECRET_KEY", "EPAM_ENCRYPTION_SECRET_KEY"]

        for var in critical_vars:
            assert var in content, f".env.example should document {var}"

    def test_requirements_pinned(self):
        """All requirements should have pinned versions."""
        if not REQUIREMENTS.exists():
            pytest.skip("requirements.txt not found")

        content = REQUIREMENTS.read_text()
        lines = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]

        for line in lines:
            # Skip comments and empty lines
            if line.startswith("-") or line.startswith("#"):
                continue
            # Should have == for pinned version
            assert "==" in line, f"Dependency '{line}' should be pinned with =="

    def test_settings_yaml_loads_as_config(self):
        """settings.yaml should load into AppConfig without errors."""
        if not SETTINGS_FILE.exists():
            pytest.skip("settings.yaml not found")

        from backend.app.config import AppConfig
        config = AppConfig.load_from_yaml(str(SETTINGS_FILE))

        assert config.server.port == 8000
        assert config.asr.model is not None
        assert config.quality.min_confidence > 0

    def test_gitignore_covers_sensitive_files(self):
        """Gitignore should cover sensitive files."""
        gitignore = PROJECT_ROOT / ".gitignore"
        if not gitignore.exists():
            pytest.skip(".gitignore not found")

        content = gitignore.read_text()

        patterns = [".env", "*.wav", "*.mp3", "models/"]
        for pattern in patterns:
            assert pattern in content, f".gitignore should cover '{pattern}'"
