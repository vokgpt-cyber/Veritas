"""Encryption at rest for audio files and transcripts.

Uses AES-256-GCM via the cryptography library (Fernet wraps AES-CBC,
but for this security-critical use case we use AES-GCM directly for
authenticated encryption with associated data).

Key management:
- Encryption key is derived from a master secret via PBKDF2-HMAC-SHA256.
- Master secret is set via EPAM_ENCRYPTION_SECRET_KEY env var.
- A unique random nonce (96-bit) is generated for each encryption operation.
- Encrypted file format: [16-byte salt][12-byte nonce][16-byte tag][ciphertext]
"""
from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Constants
SALT_SIZE = 16  # 128-bit salt for key derivation
NONCE_SIZE = 12  # 96-bit nonce for AES-GCM (standard)
TAG_SIZE = 16  # 128-bit authentication tag
KEY_SIZE = 32  # 256-bit key for AES-256
KDF_ITERATIONS = 600_000  # OWASP 2023 recommendation for PBKDF2-SHA256
HEADER_SIZE = SALT_SIZE + NONCE_SIZE  # Salt + Nonce stored before ciphertext

# File extension for encrypted files
ENCRYPTED_EXTENSION = ".enc"


class EncryptionConfig(BaseSettings):
    """Encryption configuration.

    EPAM_ENCRYPTION_SECRET_KEY MUST be set in production.
    If not set, encryption is disabled and a warning is logged.
    """

    secret_key: Optional[str] = None
    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="EPAM_ENCRYPTION_")


class FileEncryptor:
    """Handles encryption and decryption of files at rest.

    Uses AES-256-GCM for authenticated encryption. Each file gets a
    unique salt (for key derivation) and nonce (for encryption), ensuring
    that identical files produce different ciphertext.

    Attributes:
        _master_secret: The master secret from which per-file keys are derived.
        _enabled: Whether encryption is active.
    """

    def __init__(self, config: Optional[EncryptionConfig] = None) -> None:
        """Initialize encryptor with configuration.

        Args:
            config: Encryption configuration. If None, loads from env vars.
        """
        if config is None:
            config = EncryptionConfig()

        self._enabled = config.enabled and config.secret_key is not None

        if self._enabled:
            self._master_secret = config.secret_key.encode("utf-8")
            logger.info("File encryption enabled (AES-256-GCM)")
        else:
            self._master_secret = None
            if config.enabled:
                logger.warning(
                    "Encryption enabled but EPAM_ENCRYPTION_SECRET_KEY not set. "
                    "Files will NOT be encrypted. Set the key in production!"
                )
            else:
                logger.info("File encryption disabled by configuration")

    @property
    def is_enabled(self) -> bool:
        """Whether encryption is currently active."""
        return self._enabled

    def _derive_key(self, salt: bytes) -> bytes:
        """Derive a per-file encryption key from master secret and salt.

        Uses PBKDF2-HMAC-SHA256 with high iteration count per OWASP guidelines.

        Args:
            salt: Random salt for key derivation (16 bytes).

        Returns:
            32-byte derived key for AES-256.
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=KDF_ITERATIONS,
            backend=default_backend(),
        )
        return kdf.derive(self._master_secret)

    def encrypt_file(self, input_path: str | Path, output_path: Optional[str | Path] = None) -> Path:
        """Encrypt a file at rest.

        Reads the input file, encrypts its contents with AES-256-GCM,
        and writes the encrypted output. The original file is securely
        overwritten and deleted if output_path differs or is None.

        File format: [16-byte salt][12-byte nonce][ciphertext+tag]
        The GCM tag is appended to ciphertext by the AESGCM implementation.

        Args:
            input_path: Path to the plaintext file.
            output_path: Path for the encrypted file. If None, appends .enc
                to input_path and removes the original.

        Returns:
            Path to the encrypted file.

        Raises:
            RuntimeError: If encryption is disabled.
            FileNotFoundError: If input file does not exist.
        """
        if not self._enabled:
            raise RuntimeError("Encryption is disabled. Set EPAM_ENCRYPTION_SECRET_KEY.")

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path is None:
            output_path = input_path.with_suffix(input_path.suffix + ENCRYPTED_EXTENSION)
            remove_original = True
        else:
            output_path = Path(output_path)
            remove_original = input_path != output_path

        # Generate unique salt and nonce
        salt = secrets.token_bytes(SALT_SIZE)
        nonce = secrets.token_bytes(NONCE_SIZE)

        # Derive per-file key
        key = self._derive_key(salt)
        aesgcm = AESGCM(key)

        # Read plaintext
        plaintext = input_path.read_bytes()

        # Encrypt (GCM appends the 16-byte tag to ciphertext)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Write encrypted file: salt + nonce + ciphertext_with_tag
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(salt)
            f.write(nonce)
            f.write(ciphertext)

        # Securely remove original file
        if remove_original and input_path.exists() and input_path != output_path:
            self._secure_delete(input_path)

        logger.debug(f"Encrypted: {input_path} -> {output_path}")
        return output_path

    def decrypt_file(self, input_path: str | Path, output_path: Optional[str | Path] = None) -> Path:
        """Decrypt an encrypted file.

        Reads the encrypted file, extracts salt and nonce from header,
        derives the key, and decrypts the ciphertext.

        Args:
            input_path: Path to the encrypted file.
            output_path: Path for the decrypted file. If None, removes
                .enc extension from input_path.

        Returns:
            Path to the decrypted file.

        Raises:
            RuntimeError: If encryption is disabled.
            FileNotFoundError: If input file does not exist.
            ValueError: If file is too small or authentication fails.
        """
        if not self._enabled:
            raise RuntimeError("Encryption is disabled. Set EPAM_ENCRYPTION_SECRET_KEY.")

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {input_path}")

        if output_path is None:
            # Strip .enc extension
            name = str(input_path)
            if name.endswith(ENCRYPTED_EXTENSION):
                output_path = Path(name[: -len(ENCRYPTED_EXTENSION)])
            else:
                output_path = input_path.with_suffix(".dec")
        else:
            output_path = Path(output_path)

        # Read encrypted file
        encrypted_data = input_path.read_bytes()

        min_size = SALT_SIZE + NONCE_SIZE + TAG_SIZE
        if len(encrypted_data) < min_size:
            raise ValueError(
                f"Encrypted file too small ({len(encrypted_data)} bytes, "
                f"minimum {min_size})"
            )

        # Extract header components
        salt = encrypted_data[:SALT_SIZE]
        nonce = encrypted_data[SALT_SIZE : SALT_SIZE + NONCE_SIZE]
        ciphertext = encrypted_data[SALT_SIZE + NONCE_SIZE :]

        # Derive key and decrypt
        key = self._derive_key(salt)
        aesgcm = AESGCM(key)

        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as e:
            raise ValueError(
                f"Decryption failed (wrong key or corrupted file): {e}"
            ) from e

        # Write decrypted file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(plaintext)

        logger.debug(f"Decrypted: {input_path} -> {output_path}")
        return output_path

    def encrypt_bytes(self, data: bytes) -> bytes:
        """Encrypt raw bytes in memory.

        Args:
            data: Plaintext bytes.

        Returns:
            Encrypted bytes with header (salt + nonce + ciphertext_with_tag).

        Raises:
            RuntimeError: If encryption is disabled.
        """
        if not self._enabled:
            raise RuntimeError("Encryption is disabled.")

        salt = secrets.token_bytes(SALT_SIZE)
        nonce = secrets.token_bytes(NONCE_SIZE)
        key = self._derive_key(salt)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, data, None)
        return salt + nonce + ciphertext

    def decrypt_bytes(self, encrypted_data: bytes) -> bytes:
        """Decrypt raw bytes in memory.

        Args:
            encrypted_data: Encrypted bytes with header.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            RuntimeError: If encryption is disabled.
            ValueError: If data is too small or authentication fails.
        """
        if not self._enabled:
            raise RuntimeError("Encryption is disabled.")

        min_size = SALT_SIZE + NONCE_SIZE + TAG_SIZE
        if len(encrypted_data) < min_size:
            raise ValueError(f"Encrypted data too small ({len(encrypted_data)} bytes)")

        salt = encrypted_data[:SALT_SIZE]
        nonce = encrypted_data[SALT_SIZE : SALT_SIZE + NONCE_SIZE]
        ciphertext = encrypted_data[SALT_SIZE + NONCE_SIZE :]

        key = self._derive_key(salt)
        aesgcm = AESGCM(key)

        try:
            return aesgcm.decrypt(nonce, ciphertext, None)
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}") from e

    @staticmethod
    def _secure_delete(file_path: Path) -> None:
        """Securely delete a file by overwriting with random data before unlinking.

        This provides defense-in-depth against file recovery on spinning disks.
        On SSDs with TRIM, this is less effective but still better than plain delete.

        Args:
            file_path: Path to the file to securely delete.
        """
        try:
            file_size = file_path.stat().st_size
            if file_size > 0:
                # Overwrite with random data
                with open(file_path, "wb") as f:
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            file_path.unlink()
            logger.debug(f"Securely deleted: {file_path}")
        except OSError as e:
            # Fall back to normal deletion
            logger.warning(f"Secure delete failed for {file_path}, using normal delete: {e}")
            try:
                file_path.unlink()
            except OSError:
                pass
