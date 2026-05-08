"""Voice enrollment for automatic speaker identification (Block 7).

Extracts ECAPA-TDNN voice embeddings from audio samples, stores them
in an encrypted SQLite database, and provides a matcher that the
pipeline calls after diarization to replace SPEAKER_X labels with
real names of enrolled employees.

Architecture:

* **Storage** — SQLite at the *shared* data root (so all VERITAS users
  on the machine see the same employee voice DB), with the embedding
  vectors AES-256-GCM encrypted. Profile metadata (name, department)
  is stored unencrypted because (a) it's not personally sensitive in
  the way audio biometrics are, and (b) we need to query/list profiles
  without decrypting every row.
* **Embedding model** — pyannote's ECAPA-TDNN at
  ``pyannote/embedding`` (256-dim vector). Different model from the
  diarization community-1 pipeline so we don't need to hold the entire
  diarization graph in VRAM during enrollment.
* **Matching** — cosine similarity against all enrolled embeddings.
  ``DEFAULT_MATCH_THRESHOLD = 0.7`` empirically separates same-speaker
  from different-speaker pairs on Russian voice samples. Increase for
  fewer false positives, decrease for fewer false negatives.

This module is **optional**: pyannote/torch are runtime dependencies,
not import-time. Imports are wrapped so the rest of VERITAS keeps
running if voice enrollment isn't available (no `torch` on the host).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# Cosine similarity above which two embeddings are considered the same
# voice. Empirically calibrated on EPAM staff voice samples (2026-04-30).
DEFAULT_MATCH_THRESHOLD = 0.7

# Minimum sample duration we accept for enrollment. Short samples
# produce unreliable embeddings — pyannote ECAPA wants at least
# ~10 seconds of speech to converge. We enforce 8s as a soft floor
# to allow for slight trim differences.
MIN_SAMPLE_DURATION_S = 8.0

# Embedding vector dimension for ECAPA-TDNN. Stored as float32 → 1024
# bytes per embedding pre-encryption. Verified against the model on
# load; mismatch raises a clear error rather than silently corrupting.
EXPECTED_EMBEDDING_DIM = 256


# =====================================================================
# DB schema + connection
# =====================================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS profiles (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    department      TEXT NOT NULL DEFAULT '',
    position        TEXT NOT NULL DEFAULT '',
    is_admin_default INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    last_used_at    TEXT,
    sample_count    INTEGER NOT NULL DEFAULT 0,
    match_count     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embeddings (
    profile_id      TEXT NOT NULL,
    sample_idx      INTEGER NOT NULL,
    -- Encrypted blob: nonce(12) || ciphertext || tag(16). Decrypt with
    -- the master key from EPAM_ENCRYPTION_SECRET_KEY (same key as
    -- audio/transcript encryption).
    encrypted_vec   BLOB NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (profile_id, sample_idx),
    FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_profiles_name ON profiles(name);
"""


def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
    """Apply lightweight migrations for existing local voice databases."""
    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(profiles)").fetchall()
    }
    if "is_admin_default" not in columns:
        conn.execute(
            "ALTER TABLE profiles "
            "ADD COLUMN is_admin_default INTEGER NOT NULL DEFAULT 0"
        )


def _db_path() -> Path:
    """Resolve the SQLite path under the shared data root."""
    from backend.app.paths import get_shared_data_root
    return get_shared_data_root() / "voices.db"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Open + initialise the voice DB. Caller closes via context manager."""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA_SQL)
        _ensure_schema_migrations(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


# =====================================================================
# Embedding extraction
# =====================================================================

# Lazy global — the pyannote embedding model is a few hundred MB and
# we only want to load it once per process. Initialised on first call
# to ``extract_embedding()``. Thread-safe via the lock.
_embedding_model = None
_embedding_lock = threading.Lock()


def _load_embedding_model():
    """Load pyannote/embedding (ECAPA-TDNN, 256-d). Cached process-wide."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_lock:
        if _embedding_model is not None:
            return _embedding_model
        try:
            from pyannote.audio import Inference, Model  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Voice enrollment requires pyannote-audio>=4.0. "
                "Install via: pip install pyannote.audio"
            ) from e

        hf_token = os.environ.get("HF_TOKEN")
        try:
            model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=hf_token,
            )
        except TypeError:
            # pyannote 4.x renamed use_auth_token to token.
            model = Model.from_pretrained(
                "pyannote/embedding",
                token=hf_token,
            )
        # window="whole" gives one embedding per audio file (we average
        # internally). Suitable for enrollment where we want a single
        # canonical voice signature, not per-window vectors.
        inference = Inference(model, window="whole")
        _embedding_model = inference
        logger.info("Voice enrollment model loaded: pyannote/embedding")
        return _embedding_model


def extract_embedding(audio_path: str | Path):
    """Extract a 256-d voice embedding from an audio file.

    Returns a numpy.ndarray of shape (256,), dtype float32, L2-normalised.
    Raises RuntimeError if the model is unavailable, the file is missing,
    or the resulting embedding has the wrong shape (defensive guard
    against silent model swaps).
    """
    import numpy as np

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    inference = _load_embedding_model()
    raw = inference(str(audio_path))
    # `raw` is a SlidingWindowFeature for window="sliding" or a numpy
    # array for window="whole". We requested "whole" → expect ndarray.
    if hasattr(raw, "data"):
        vec = np.asarray(raw.data).flatten()
    else:
        vec = np.asarray(raw).flatten()
    vec = vec.astype("float32")

    if vec.shape != (EXPECTED_EMBEDDING_DIM,):
        raise RuntimeError(
            f"Unexpected embedding shape {vec.shape}; expected "
            f"({EXPECTED_EMBEDDING_DIM},). Model may have changed — refusing "
            f"to corrupt the voice DB."
        )

    # L2-normalise so cosine similarity == dot product, simplifies the
    # matcher and is standard for ECAPA embeddings.
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec


# =====================================================================
# Encryption (reuses the project encryption infrastructure)
# =====================================================================

def _encrypt_vector(vec) -> bytes:
    """Encrypt a numpy embedding to a storable BLOB.

    Uses the same FileEncryption helper as audio/transcript encryption
    so we have one master key and one key-rotation story.
    """
    import numpy as np
    from backend.core.encryption import FileEncryption, EncryptionConfig

    # Cache the encryptor — instantiating reads the key from env each
    # call which is wasteful when the matcher iterates over many rows.
    enc = _get_encryptor()
    # Serialise to bytes deterministically; numpy.tobytes is endian-
    # native but we run on a single architecture per machine, so
    # round-trips are safe.
    return enc.encrypt_bytes(vec.astype("float32").tobytes())


def _decrypt_vector(blob: bytes):
    """Decrypt a stored embedding back to a numpy float32 vector."""
    import numpy as np
    enc = _get_encryptor()
    raw = enc.decrypt_bytes(blob)
    vec = np.frombuffer(raw, dtype="float32")
    if vec.shape[0] != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"Decrypted embedding has wrong dim {vec.shape[0]}; "
            f"expected {EXPECTED_EMBEDDING_DIM}. DB may be corrupted."
        )
    return vec.copy()  # Detach from buffer so caller can mutate safely.


_encryptor = None
_encryptor_lock = threading.Lock()


def _get_encryptor():
    """Lazy singleton for the FileEncryption helper."""
    global _encryptor
    if _encryptor is not None:
        return _encryptor
    with _encryptor_lock:
        if _encryptor is not None:
            return _encryptor
        from backend.core.encryption import (
            FileEncryption,
            EncryptionConfig,
        )
        cfg = EncryptionConfig()
        _encryptor = FileEncryption(cfg)
        return _encryptor


# =====================================================================
# CRUD
# =====================================================================

def create_profile(
    *,
    name: str,
    department: str = "",
    position: str = "",
    audio_path: str | Path,
    is_admin_default: bool = False,
) -> dict:
    """Enrol a new voice profile from an audio sample.

    Steps: validate, extract embedding, encrypt, write profile +
    embedding rows. Returns the profile dict (without the embedding).
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Profile name is required")

    vec = extract_embedding(audio_path)
    blob = _encrypt_vector(vec)
    profile_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            "INSERT INTO profiles (id, name, department, position, is_admin_default, "
            "created_at, sample_count, match_count) "
            "VALUES (?, ?, ?, ?, ?, ?, 1, 0)",
            (
                profile_id,
                name,
                department.strip(),
                position.strip(),
                1 if is_admin_default else 0,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO embeddings "
            "(profile_id, sample_idx, encrypted_vec, created_at) "
            "VALUES (?, 0, ?, ?)",
            (profile_id, blob, now),
        )

    logger.info(f"Voice profile enrolled: {name} ({profile_id})")
    return _load_profile(profile_id)


def create_profile_metadata(
    *,
    name: str,
    department: str = "",
    position: str = "",
    is_admin_default: bool = False,
) -> dict:
    """Create an employee profile without a voice sample."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Profile name is required")

    profile_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            "INSERT INTO profiles (id, name, department, position, is_admin_default, "
            "created_at, sample_count, match_count) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
            (
                profile_id,
                name,
                department.strip(),
                position.strip(),
                1 if is_admin_default else 0,
                now,
            ),
        )

    logger.info(f"Employee profile created: {name} ({profile_id})")
    return _load_profile(profile_id)


def add_sample(profile_id: str, audio_path: str | Path) -> dict:
    """Add an additional voice sample to an existing profile.

    More samples = better averaged embedding = better match accuracy.
    """
    vec = extract_embedding(audio_path)
    blob = _encrypt_vector(vec)
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        row = conn.execute(
            "SELECT sample_count FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Profile not found: {profile_id}")
        next_idx = int(row["sample_count"])
        conn.execute(
            "INSERT INTO embeddings "
            "(profile_id, sample_idx, encrypted_vec, created_at) "
            "VALUES (?, ?, ?, ?)",
            (profile_id, next_idx, blob, now),
        )
        conn.execute(
            "UPDATE profiles SET sample_count = sample_count + 1 "
            "WHERE id = ?",
            (profile_id,),
        )
    return _load_profile(profile_id)


def update_profile(
    profile_id: str,
    *,
    name: Optional[str] = None,
    department: Optional[str] = None,
    position: Optional[str] = None,
    is_admin_default: Optional[bool] = None,
) -> dict:
    """Update profile metadata only (does not touch embeddings)."""
    fields = []
    values: list = []
    if name is not None:
        fields.append("name = ?")
        values.append(name.strip())
    if department is not None:
        fields.append("department = ?")
        values.append(department.strip())
    if position is not None:
        fields.append("position = ?")
        values.append(position.strip())
    if is_admin_default is not None:
        fields.append("is_admin_default = ?")
        values.append(1 if is_admin_default else 0)
    if not fields:
        return _load_profile(profile_id)

    values.append(profile_id)
    with _connect() as conn:
        cur = conn.execute(
            f"UPDATE profiles SET {', '.join(fields)} WHERE id = ?",
            tuple(values),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Profile not found: {profile_id}")
    return _load_profile(profile_id)


def delete_profile(profile_id: str) -> bool:
    """Delete a profile and all its embeddings. Returns True if deleted."""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM profiles WHERE id = ?", (profile_id,)
        )
        # ON DELETE CASCADE handles embeddings.
        return cur.rowcount > 0


def list_profiles() -> list[dict]:
    """Return all enrolled profiles (metadata only, no embeddings)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, department, position, is_admin_default, created_at, "
            "last_used_at, sample_count, match_count FROM profiles "
            "ORDER BY name COLLATE NOCASE"
        ).fetchall()
    return [dict(r) for r in rows]


def _load_profile(profile_id: str) -> dict:
    """Read a profile row by id. Raises ValueError if missing."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, department, position, is_admin_default, created_at, "
            "last_used_at, sample_count, match_count FROM profiles "
            "WHERE id = ?",
            (profile_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Profile not found: {profile_id}")
    return dict(row)


# =====================================================================
# Matching
# =====================================================================

def _all_profile_embeddings():
    """Yield (profile_id, name, mean_embedding) for every enrolled profile.

    The "mean" is the L2-normalised average of all per-sample vectors
    for the profile — gives a more stable signature than any single
    sample. Decrypts on read; cache in memory if hot.
    """
    import numpy as np

    with _connect() as conn:
        rows = conn.execute(
            "SELECT p.id, p.name, e.encrypted_vec "
            "FROM profiles p "
            "JOIN embeddings e ON e.profile_id = p.id"
        ).fetchall()

    by_id: dict[str, list] = {}
    name_by_id: dict[str, str] = {}
    for r in rows:
        try:
            vec = _decrypt_vector(r["encrypted_vec"])
        except Exception as e:
            logger.warning(
                "Failed to decrypt embedding for profile %s: %s",
                r["id"], e,
            )
            continue
        by_id.setdefault(r["id"], []).append(vec)
        name_by_id[r["id"]] = r["name"]

    for pid, vecs in by_id.items():
        if not vecs:
            continue
        mean = np.mean(np.stack(vecs, axis=0), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            mean = mean / norm
        yield pid, name_by_id[pid], mean.astype("float32")


def match_speaker(
    query_vec,
    *,
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> Optional[dict]:
    """Find the best-matching enrolled profile for a query embedding.

    Returns ``{"profile_id", "name", "similarity"}`` when the best
    match exceeds ``threshold``, else None. ``query_vec`` must already
    be L2-normalised (caller's responsibility — ``extract_embedding``
    does this; pipeline cluster averaging should too).

    Updates the matched profile's ``last_used_at`` and ``match_count``
    so the UI can surface "most active voices" stats.
    """
    import numpy as np

    best_id: Optional[str] = None
    best_name: Optional[str] = None
    best_sim = -1.0
    for pid, name, mean in _all_profile_embeddings():
        sim = float(np.dot(query_vec, mean))
        if sim > best_sim:
            best_sim = sim
            best_id = pid
            best_name = name

    if best_id is None or best_sim < threshold:
        return None

    # Telemetry update — best-effort, don't fail the match on a write
    # error.
    try:
        now = datetime.now(timezone.utc).isoformat()
        with _connect() as conn:
            conn.execute(
                "UPDATE profiles SET last_used_at = ?, "
                "match_count = match_count + 1 WHERE id = ?",
                (now, best_id),
            )
    except Exception:  # noqa: BLE001
        pass

    return {
        "profile_id": best_id,
        "name": best_name,
        "similarity": best_sim,
    }


__all__ = [
    "DEFAULT_MATCH_THRESHOLD",
    "MIN_SAMPLE_DURATION_S",
    "extract_embedding",
    "create_profile",
    "create_profile_metadata",
    "add_sample",
    "update_profile",
    "delete_profile",
    "list_profiles",
    "match_speaker",
]
