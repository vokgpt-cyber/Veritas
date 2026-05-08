"""Tests for FastAPI endpoints with JWT authentication."""
import struct

import pytest
from fastapi.testclient import TestClient

from backend.app.auth import create_access_token
from backend.app.main import app


@pytest.fixture
def client():
    """Create test client with lifespan events to initialize orchestrator."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    """Create valid auth headers for testing."""
    token, _ = create_access_token("admin", role="admin")
    return {"Authorization": f"Bearer {token}"}


class TestPublicEndpoints:
    """Endpoints that should NOT require authentication."""

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "VERITAS" in data["message"]

    def test_docs_accessible(self, client):
        response = client.get("/api/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        response = client.get("/api/openapi.json")
        assert response.status_code == 200


class TestAuthEndpoints:
    """Authentication endpoint tests."""

    def test_login_success(self, client):
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_wrong_password(self, client):
        response = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        response = client.post(
            "/api/auth/login",
            json={"username": "ghost", "password": "password"},
        )
        assert response.status_code == 401

    def test_verify_valid_token(self, client):
        # Get a token first
        login_resp = client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admin"},
        )
        token = login_resp.json()["access_token"]

        # Verify it
        response = client.post(f"/api/auth/verify?token={token}")
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["username"] == "admin"

    def test_verify_invalid_token(self, client):
        response = client.post("/api/auth/verify?token=bad.token.here")
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False


class TestProtectedEndpoints:
    """Protected endpoints should return 403 without auth and 200 with auth."""

    def test_meetings_list_requires_auth(self, client):
        response = client.get("/api/meetings/")
        assert response.status_code in (401, 403)

    def test_meetings_list_with_auth(self, client, auth_headers):
        response = client.get("/api/meetings/", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == []

    def test_meeting_status_requires_auth(self, client):
        response = client.get("/api/meetings/fake-id/status")
        assert response.status_code in (401, 403)

    def test_meeting_status_with_auth(self, client, auth_headers):
        response = client.get("/api/meetings/fake-id/status", headers=auth_headers)
        assert response.status_code == 404  # Not found, but auth passed

    def test_meeting_delete_requires_auth(self, client):
        response = client.delete("/api/meetings/fake-id")
        assert response.status_code in (401, 403)

    def test_meeting_delete_with_auth(self, client, auth_headers):
        response = client.delete("/api/meetings/fake-id", headers=auth_headers)
        assert response.status_code == 404  # Not found, but auth passed

    def test_upload_requires_auth(self, client):
        response = client.post(
            "/api/meetings/upload",
            files={"file": ("test.wav", b"fake", "audio/wav")},
        )
        assert response.status_code in (401, 403)

    def test_upload_with_auth(self, client, auth_headers):
        sample_rate = 16000
        num_samples = 1600
        data_size = num_samples * 2

        wav_data = b"RIFF"
        wav_data += struct.pack("<I", 36 + data_size)
        wav_data += b"WAVE"
        wav_data += b"fmt "
        wav_data += struct.pack("<I", 16)
        wav_data += struct.pack("<H", 1)
        wav_data += struct.pack("<H", 1)
        wav_data += struct.pack("<I", sample_rate)
        wav_data += struct.pack("<I", sample_rate * 2)
        wav_data += struct.pack("<H", 2)
        wav_data += struct.pack("<H", 16)
        wav_data += b"data"
        wav_data += struct.pack("<I", data_size)
        wav_data += b"\x00" * data_size

        response = client.post(
            "/api/meetings/upload",
            files={"file": ("test.wav", wav_data, "audio/wav")},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.wav"

    def test_system_health_requires_auth(self, client):
        response = client.get("/api/system/health")
        assert response.status_code in (401, 403)

    def test_system_health_with_auth(self, client, auth_headers):
        response = client.get("/api/system/health", headers=auth_headers)
        assert response.status_code == 200

    def test_system_info_with_auth(self, client, auth_headers):
        response = client.get("/api/system/info", headers=auth_headers)
        assert response.status_code == 200

    def test_speakers_list_requires_auth(self, client):
        response = client.get("/api/speakers/")
        assert response.status_code in (401, 403)

    def test_speakers_list_with_auth(self, client, auth_headers):
        response = client.get("/api/speakers/", headers=auth_headers)
        assert response.status_code == 200

    def test_speaker_get_requires_auth(self, client):
        response = client.get("/api/speakers/fake-id")
        assert response.status_code in (401, 403)

    def test_speaker_get_with_auth(self, client, auth_headers):
        response = client.get("/api/speakers/fake-id", headers=auth_headers)
        assert response.status_code == 404  # Auth passed, speaker not found


class TestTokenExpiredOrInvalid:
    """Test rejected auth scenarios."""

    def test_expired_token_format(self, client):
        """Malformed bearer token should be rejected."""
        response = client.get(
            "/api/meetings/",
            headers={"Authorization": "Bearer invalid.jwt.token"},
        )
        assert response.status_code == 401

    def test_no_bearer_prefix(self, client):
        """Token without Bearer prefix should be rejected."""
        token, _ = create_access_token("admin")
        response = client.get(
            "/api/meetings/",
            headers={"Authorization": token},  # Missing "Bearer "
        )
        assert response.status_code in (401, 403)
