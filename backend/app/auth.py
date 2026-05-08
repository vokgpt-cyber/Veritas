"""JWT authentication module for EPAM VERITAS."""
from __future__ import annotations

import logging
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Security scheme for Swagger UI
security = HTTPBearer()


class AuthConfig(BaseSettings):
    """Authentication configuration.

    All values can be overridden via environment variables with EPAM_AUTH_ prefix.
    SECRET_KEY MUST be set in production via EPAM_AUTH_SECRET_KEY env var.
    """

    secret_key: str = secrets.token_urlsafe(64)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 480  # 8 hours
    # Default admin credentials (MUST be changed in production)
    default_username: str = "admin"
    default_password_hash: str = ""
    default_password: str = "admin"

    model_config = SettingsConfigDict(env_prefix="EPAM_AUTH_")


# Module-level config (loaded once)
auth_config = AuthConfig()

# Password hashing
# Note: passlib 1.7.4 has a compatibility issue with bcrypt>=4.1 where
# internal bug detection passes strings >72 bytes. We patch around this
# by catching the error during context initialization.
import bcrypt as _bcrypt_mod
if not hasattr(_bcrypt_mod, '_original_hashpw'):
    _bcrypt_mod._original_hashpw = _bcrypt_mod.hashpw
    def _safe_hashpw(password: bytes, salt: bytes) -> bytes:
        if len(password) > 72:
            password = password[:72]
        return _bcrypt_mod._original_hashpw(password, salt)
    _bcrypt_mod.hashpw = _safe_hashpw

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# SQLite-backed user/session store. The in-memory dict stays only as
# emergency fallback if the local DB cannot be opened.
_users_db: dict[str, dict] = {}

_AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username             TEXT PRIMARY KEY,
    password_hash        TEXT NOT NULL,
    role                 TEXT NOT NULL DEFAULT 'operator',
    is_active            INTEGER NOT NULL DEFAULT 1,
    must_change_password INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    jti         TEXT PRIMARY KEY,
    username    TEXT NOT NULL,
    role        TEXT NOT NULL,
    issued_at   TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
);
"""


def _auth_db_path():
    from backend.app.paths import resolve_user_path
    path = resolve_user_path("data/auth/veritas_auth.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_auth_db_path()))
    conn.row_factory = sqlite3.Row
    conn.executescript(_AUTH_SCHEMA)
    return conn


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _row_to_user(row: sqlite3.Row | None) -> Optional[dict]:
    if row is None:
        return None
    return {
        "username": row["username"],
        "password_hash": row["password_hash"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "must_change_password": bool(row["must_change_password"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def hash_password(password: str) -> str:
    """Hash a plain-text password for storage."""
    return pwd_context.hash(password)


def _ensure_default_user() -> None:
    """Create default admin user if no users exist.

    Uses auth_config.default_password_hash if set, otherwise generates
    a hash of 'admin' (MUST be changed in production).
    """
    try:
        with _connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
            if row and int(row["n"]) > 0:
                return

            if auth_config.default_password_hash:
                password_hash = auth_config.default_password_hash
                must_change = 0
            else:
                password_hash = pwd_context.hash(auth_config.default_password or "admin")
                must_change = 1
                if auth_config.default_password == "admin":
                    logger.warning(
                        "Using default admin password. Set "
                        "EPAM_AUTH_DEFAULT_PASSWORD_HASH in production!"
                    )

            now = _now_iso()
            conn.execute(
                "INSERT INTO users "
                "(username, password_hash, role, is_active, "
                "must_change_password, created_at, updated_at) "
                "VALUES (?, ?, 'admin', 1, ?, ?, ?)",
                (
                    auth_config.default_username,
                    password_hash,
                    must_change,
                    now,
                    now,
                ),
            )
            return
    except Exception as exc:  # noqa: BLE001
        logger.warning("SQLite auth store unavailable, using memory: %s", exc)

    if _users_db:
        return
    password_hash = (
        auth_config.default_password_hash
        or pwd_context.hash(auth_config.default_password or "admin")
    )
    _users_db[auth_config.default_username] = {
        "username": auth_config.default_username,
        "password_hash": password_hash,
        "is_active": True,
        "role": "admin",
        "must_change_password": not bool(auth_config.default_password_hash),
    }


# Initialize default user
_ensure_default_user()


class TokenData(BaseModel):
    """JWT token payload data."""

    username: str
    role: str = "user"
    jti: Optional[str] = None
    exp: Optional[datetime] = None


class TokenResponse(BaseModel):
    """Response model for token endpoint."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    must_change_password: bool = False
    username: str = ""
    role: str = "user"


class LoginRequest(BaseModel):
    """Login request body."""

    username: str
    password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(username: str, role: str = "user") -> tuple[str, datetime]:
    """
    Create a new JWT access token.

    Args:
        username: Username to encode in the token
        role: User role

    Returns:
        Tuple of (token_string, expiration_datetime)
    """
    expires_at = datetime.utcnow() + timedelta(
        minutes=auth_config.access_token_expire_minutes
    )

    jti = str(uuid.uuid4())
    payload = {
        "sub": username,
        "role": role,
        "jti": jti,
        "exp": expires_at,
        "iat": datetime.utcnow(),
    }

    token = jwt.encode(
        payload,
        auth_config.secret_key,
        algorithm=auth_config.algorithm,
    )

    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO sessions "
                "(jti, username, role, issued_at, expires_at, is_active) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (
                    jti,
                    username,
                    role,
                    datetime.utcnow().isoformat(),
                    expires_at.isoformat(),
                ),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist session %s: %s", jti, exc)

    return token, expires_at


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData with decoded claims

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            auth_config.secret_key,
            algorithms=[auth_config.algorithm],
        )

        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return TokenData(
            username=username,
            role=payload.get("role", "user"),
            jti=payload.get("jti"),
        )

    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Authenticate user by username and password.

    Args:
        username: Username
        password: Plain text password

    Returns:
        User dict if authentication succeeds, None otherwise
    """
    user = get_user(username)
    if not user:
        return None

    if not user.get("is_active", False):
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    return user


def get_user(username: str) -> Optional[dict]:
    """Return a user from SQLite, falling back to in-memory store."""
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            user = _row_to_user(row)
            if user:
                return user
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auth DB read failed, falling back to memory: %s", exc)
    return _users_db.get(username)


def list_users() -> list[dict]:
    """List users for admin UI."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT username, role, is_active, must_change_password, "
                "created_at, updated_at FROM users ORDER BY username"
            ).fetchall()
            return [
                {
                    "username": r["username"],
                    "role": r["role"],
                    "is_active": bool(r["is_active"]),
                    "must_change_password": bool(r["must_change_password"]),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]
    except Exception:
        return [
            {
                "username": u["username"],
                "role": u.get("role", "user"),
                "is_active": bool(u.get("is_active", False)),
                "must_change_password": bool(u.get("must_change_password", False)),
                "created_at": "",
                "updated_at": "",
            }
            for u in _users_db.values()
        ]


def create_user(
    *,
    username: str,
    password: str,
    role: str = "operator",
    must_change_password: bool = True,
) -> dict:
    """Create a persistent user."""
    username = username.strip()
    if not username:
        raise ValueError("username is required")
    if role not in {"admin", "developer", "operator", "viewer", "lawyer"}:
        raise ValueError("invalid role")
    now = _now_iso()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users "
            "(username, password_hash, role, is_active, "
            "must_change_password, created_at, updated_at) "
            "VALUES (?, ?, ?, 1, ?, ?, ?)",
            (
                username,
                hash_password(password),
                role,
                1 if must_change_password else 0,
                now,
                now,
            ),
        )
    return get_user(username)


def update_user(
    username: str,
    *,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    must_change_password: Optional[bool] = None,
) -> dict:
    """Update user metadata."""
    fields = []
    values: list = []
    if role is not None:
        if role not in {"admin", "developer", "operator", "viewer", "lawyer"}:
            raise ValueError("invalid role")
        fields.append("role = ?")
        values.append(role)
    if is_active is not None:
        fields.append("is_active = ?")
        values.append(1 if is_active else 0)
    if must_change_password is not None:
        fields.append("must_change_password = ?")
        values.append(1 if must_change_password else 0)
    if not fields:
        user = get_user(username)
        if not user:
            raise ValueError("user not found")
        return user
    fields.append("updated_at = ?")
    values.append(_now_iso())
    values.append(username)
    with _connect() as conn:
        cur = conn.execute(
            f"UPDATE users SET {', '.join(fields)} WHERE username = ?",
            tuple(values),
        )
        if cur.rowcount == 0:
            raise ValueError("user not found")
        if is_active is False:
            conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE username = ?",
                (username,),
            )
    return get_user(username)


def reset_user_password(username: str, password: str) -> dict:
    """Set a temporary password and require change on next login."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = ?, must_change_password = 1, "
            "updated_at = ? WHERE username = ?",
            (hash_password(password), _now_iso(), username),
        )
        if cur.rowcount == 0:
            raise ValueError("user not found")
        conn.execute(
            "UPDATE sessions SET is_active = 0 WHERE username = ?",
            (username,),
        )
    return get_user(username)


def delete_user(username: str) -> None:
    """Delete a user and revoke their sessions."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT role FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            raise ValueError("user not found")
        if row["role"] == "admin":
            admins = conn.execute(
                "SELECT COUNT(*) AS n FROM users "
                "WHERE role = 'admin' AND is_active = 1"
            ).fetchone()
            if admins and int(admins["n"]) <= 1:
                raise ValueError("cannot delete the last active admin")
        conn.execute("DELETE FROM sessions WHERE username = ?", (username,))
        conn.execute("DELETE FROM users WHERE username = ?", (username,))


def change_user_password(
    username: str,
    current_password: str,
    new_password: str,
) -> None:
    """Change own password and clear must_change_password."""
    user = authenticate_user(username, current_password)
    if not user:
        raise ValueError("invalid current password")
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, must_change_password = 0, "
            "updated_at = ? WHERE username = ?",
            (hash_password(new_password), _now_iso(), username),
        )


def is_session_active(jti: Optional[str]) -> bool:
    if not jti:
        return True  # legacy tokens from before session persistence
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT is_active, expires_at FROM sessions WHERE jti = ?",
                (jti,),
            ).fetchone()
            if row is None or not bool(row["is_active"]):
                return False
            try:
                expires_at = datetime.fromisoformat(row["expires_at"])
                return expires_at > datetime.utcnow()
            except ValueError:
                return False
    except Exception:
        return True


def deactivate_session(jti: Optional[str]) -> None:
    if not jti:
        return
    try:
        with _connect() as conn:
            conn.execute(
                "UPDATE sessions SET is_active = 0 WHERE jti = ?",
                (jti,),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to deactivate session %s: %s", jti, exc)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """
    FastAPI dependency to validate JWT token and return current user.

    Usage in route:
        @router.get("/protected")
        async def protected_route(user: TokenData = Depends(get_current_user)):
            return {"username": user.username}
    """
    token_data = decode_token(credentials.credentials)

    # Verify user and session still exist and are active
    user = get_user(token_data.username)
    if not user or not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account disabled or not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Role changes made by an admin should take effect immediately,
    # even if the user still holds an older token.
    token_data.role = user.get("role", token_data.role)
    if not is_session_active(token_data.jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token_data


def require_admin(user: TokenData = Depends(get_current_user)) -> TokenData:
    """
    FastAPI dependency requiring admin role.

    Usage in route:
        @router.post("/admin-only")
        async def admin_route(user: TokenData = Depends(require_admin)):
            return {"admin": user.username}
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user
