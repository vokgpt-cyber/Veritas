"""Tests for encryption at rest module."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from backend.core.encryption import (
    ENCRYPTED_EXTENSION,
    HEADER_SIZE,
    NONCE_SIZE,
    SALT_SIZE,
    TAG_SIZE,
    EncryptionConfig,
    FileEncryptor,
)


@pytest.fixture
def encryptor():
    """Create an encryptor with a test secret key."""
    config = EncryptionConfig(secret_key="test-secret-key-for-unit-tests-only")
    return FileEncryptor(config)


@pytest.fixture
def disabled_encryptor():
    """Create a disabled encryptor."""
    config = EncryptionConfig(enabled=False)
    return FileEncryptor(config)


@pytest.fixture
def no_key_encryptor():
    """Create an encryptor with no key set."""
    config = EncryptionConfig(secret_key=None, enabled=True)
    return FileEncryptor(config)


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample file for testing."""
    file_path = tmp_path / "test_audio.wav"
    content = os.urandom(1024)  # 1KB of random data
    file_path.write_bytes(content)
    return file_path, content


class TestFileEncryptor:
    """Tests for FileEncryptor class."""

    def test_encryptor_enabled_with_key(self, encryptor):
        """Encryptor is enabled when secret key is provided."""
        assert encryptor.is_enabled is True

    def test_encryptor_disabled_explicitly(self, disabled_encryptor):
        """Encryptor is disabled when enabled=False."""
        assert disabled_encryptor.is_enabled is False

    def test_encryptor_disabled_without_key(self, no_key_encryptor):
        """Encryptor is disabled when no secret key is provided."""
        assert no_key_encryptor.is_enabled is False

    def test_encrypt_file_roundtrip(self, encryptor, sample_file):
        """Encrypting then decrypting produces the original file content."""
        file_path, original_content = sample_file

        # Encrypt
        encrypted_path = encryptor.encrypt_file(file_path)
        assert encrypted_path.exists()
        assert encrypted_path.suffix == ENCRYPTED_EXTENSION
        assert not file_path.exists()  # Original should be securely deleted

        # Encrypted content should differ from original
        encrypted_content = encrypted_path.read_bytes()
        assert encrypted_content != original_content
        assert len(encrypted_content) > len(original_content)

        # Decrypt
        decrypted_path = encryptor.decrypt_file(encrypted_path)
        assert decrypted_path.exists()
        decrypted_content = decrypted_path.read_bytes()
        assert decrypted_content == original_content

    def test_encrypt_file_custom_output(self, encryptor, sample_file, tmp_path):
        """Encrypting to a custom output path works correctly."""
        file_path, original_content = sample_file
        custom_output = tmp_path / "custom_encrypted.bin"

        encrypted_path = encryptor.encrypt_file(file_path, custom_output)
        assert encrypted_path == custom_output
        assert encrypted_path.exists()

    def test_encrypt_file_header_format(self, encryptor, sample_file):
        """Encrypted file has correct header structure (salt + nonce)."""
        file_path, _ = sample_file

        encrypted_path = encryptor.encrypt_file(file_path)
        encrypted_data = encrypted_path.read_bytes()

        # Must have at least salt + nonce + tag
        min_size = SALT_SIZE + NONCE_SIZE + TAG_SIZE
        assert len(encrypted_data) >= min_size

    def test_encrypt_different_files_produce_different_output(self, encryptor, tmp_path):
        """Same content encrypted twice produces different ciphertext (unique salt+nonce)."""
        content = b"identical content for both files"

        file1 = tmp_path / "file1.txt"
        file1.write_bytes(content)
        enc1 = encryptor.encrypt_file(file1, tmp_path / "enc1.bin")

        file2 = tmp_path / "file2.txt"
        file2.write_bytes(content)
        enc2 = encryptor.encrypt_file(file2, tmp_path / "enc2.bin")

        assert enc1.read_bytes() != enc2.read_bytes()

    def test_decrypt_with_wrong_key_fails(self, encryptor, sample_file):
        """Decryption with wrong key raises ValueError."""
        file_path, _ = sample_file
        encrypted_path = encryptor.encrypt_file(file_path)

        # Create encryptor with different key
        wrong_config = EncryptionConfig(secret_key="wrong-key-should-fail")
        wrong_encryptor = FileEncryptor(wrong_config)

        with pytest.raises(ValueError, match="Decryption failed"):
            wrong_encryptor.decrypt_file(encrypted_path)

    def test_decrypt_corrupted_file_fails(self, encryptor, sample_file, tmp_path):
        """Decryption of corrupted file raises ValueError."""
        file_path, _ = sample_file
        encrypted_path = encryptor.encrypt_file(file_path)

        # Corrupt the encrypted file
        data = bytearray(encrypted_path.read_bytes())
        if len(data) > SALT_SIZE + NONCE_SIZE + 10:
            data[SALT_SIZE + NONCE_SIZE + 5] ^= 0xFF  # Flip a byte in ciphertext
        corrupted = tmp_path / "corrupted.enc"
        corrupted.write_bytes(bytes(data))

        with pytest.raises(ValueError):
            encryptor.decrypt_file(corrupted)

    def test_decrypt_too_small_file_fails(self, encryptor, tmp_path):
        """Decryption of file smaller than minimum header fails."""
        small_file = tmp_path / "too_small.enc"
        small_file.write_bytes(b"tiny")

        with pytest.raises(ValueError, match="too small"):
            encryptor.decrypt_file(small_file)

    def test_encrypt_nonexistent_file_raises(self, encryptor, tmp_path):
        """Encrypting a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            encryptor.encrypt_file(tmp_path / "does_not_exist.wav")

    def test_decrypt_nonexistent_file_raises(self, encryptor, tmp_path):
        """Decrypting a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            encryptor.decrypt_file(tmp_path / "does_not_exist.enc")

    def test_encrypt_disabled_raises(self, disabled_encryptor, sample_file):
        """Encrypting with disabled encryptor raises RuntimeError."""
        file_path, _ = sample_file
        with pytest.raises(RuntimeError, match="disabled"):
            disabled_encryptor.encrypt_file(file_path)

    def test_decrypt_disabled_raises(self, disabled_encryptor, tmp_path):
        """Decrypting with disabled encryptor raises RuntimeError."""
        dummy = tmp_path / "dummy.enc"
        dummy.write_bytes(os.urandom(100))
        with pytest.raises(RuntimeError, match="disabled"):
            disabled_encryptor.decrypt_file(dummy)

    def test_encrypt_bytes_roundtrip(self, encryptor):
        """Encrypting then decrypting bytes produces original data."""
        original = b"sensitive transcript data that must be protected"
        encrypted = encryptor.encrypt_bytes(original)

        assert encrypted != original
        assert len(encrypted) > len(original)

        decrypted = encryptor.decrypt_bytes(encrypted)
        assert decrypted == original

    def test_encrypt_bytes_disabled_raises(self, disabled_encryptor):
        """Encrypting bytes with disabled encryptor raises RuntimeError."""
        with pytest.raises(RuntimeError):
            disabled_encryptor.encrypt_bytes(b"data")

    def test_decrypt_bytes_too_small_raises(self, encryptor):
        """Decrypting bytes smaller than minimum size raises ValueError."""
        with pytest.raises(ValueError, match="too small"):
            encryptor.decrypt_bytes(b"tiny")

    def test_encrypt_empty_file(self, encryptor, tmp_path):
        """Encrypting an empty file works correctly."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_bytes(b"")

        encrypted_path = encryptor.encrypt_file(empty_file)
        assert encrypted_path.exists()

        decrypted_path = encryptor.decrypt_file(encrypted_path)
        assert decrypted_path.read_bytes() == b""

    def test_encrypt_large_file(self, encryptor, tmp_path):
        """Encrypting a larger file (1MB) works correctly."""
        large_file = tmp_path / "large.bin"
        large_content = os.urandom(1024 * 1024)  # 1MB
        large_file.write_bytes(large_content)

        encrypted_path = encryptor.encrypt_file(large_file)
        decrypted_path = encryptor.decrypt_file(encrypted_path)
        assert decrypted_path.read_bytes() == large_content
