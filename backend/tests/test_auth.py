"""Tests for JWT authentication module."""
import time

import pytest

from backend.app.auth import (
    authenticate_user,
    auth_config,
    create_access_token,
    decode_token,
    verify_password,
    pwd_context,
)


class TestPasswordHashing:
    """Password hashing and verification tests."""

    def test_hash_and_verify(self):
        password = "test_password_123"
        hashed = pwd_context.hash(password)
        assert verify_password(password, hashed)

    def test_wrong_password(self):
        hashed = pwd_context.hash("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_hash_is_not_plaintext(self):
        password = "my_secret"
        hashed = pwd_context.hash(password)
        assert hashed != password
        assert len(hashed) > len(password)


class TestTokenCreation:
    """JWT token creation tests."""

    def test_create_token(self):
        token, expires_at = create_access_token("admin", role="admin")
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are long
        assert expires_at is not None

    def test_token_has_three_parts(self):
        """JWT tokens have header.payload.signature format."""
        token, _ = create_access_token("user1")
        parts = token.split(".")
        assert len(parts) == 3


class TestTokenDecoding:
    """JWT token decoding and validation tests."""

    def test_decode_valid_token(self):
        token, _ = create_access_token("admin", role="admin")
        data = decode_token(token)
        assert data.username == "admin"
        assert data.role == "admin"

    def test_decode_invalid_token(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            decode_token("invalid.token.here")
        assert exc_info.value.status_code == 401

    def test_decode_tampered_token(self):
        from fastapi import HTTPException

        token, _ = create_access_token("admin")
        # Tamper with the token
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(HTTPException) as exc_info:
            decode_token(tampered)
        assert exc_info.value.status_code == 401


class TestUserAuthentication:
    """User authentication tests."""

    def test_authenticate_default_admin(self):
        user = authenticate_user("admin", "admin")
        assert user is not None
        assert user["username"] == "admin"
        assert user["role"] == "admin"

    def test_authenticate_wrong_password(self):
        user = authenticate_user("admin", "wrong_password")
        assert user is None

    def test_authenticate_nonexistent_user(self):
        user = authenticate_user("nonexistent", "password")
        assert user is None


class TestAuthConfig:
    """Authentication configuration tests."""

    def test_config_defaults(self):
        assert auth_config.algorithm == "HS256"
        assert auth_config.access_token_expire_minutes > 0
        assert len(auth_config.secret_key) > 20  # Should be a long random key
