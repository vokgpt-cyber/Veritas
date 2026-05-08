"""User-scoped data root resolution for VERITAS.

Returns the active user's VERITAS data directory. On Windows this is
``%APPDATA%\\VERITAS\\<windows-username>\\``, isolating each Windows
account from others on the same machine. On other OSes it falls back
to ``~/.veritas/``. For tests or environments without a real user
(e.g. headless CI) it falls back to the project-relative ``data/`` so
existing dev workflows keep working.

**Why per-user matters:** in a law firm, each lawyer must see only
their own meetings, audio, and protocols. A shared ``data/`` folder
under the project root violates that isolation. Voice profiles are
explicitly *shared* (see ``get_shared_data_root()``) because the
employee voice DB is a cross-user resource — different lawyers all
benefit from the same enrolled voices.

**All callers MUST go through ``resolve_user_path()``** rather than
building paths from ``Path(self._config.X.Y)`` directly. This keeps
the policy centralized and makes future moves (e.g. from %APPDATA%
to %LOCALAPPDATA%) one-edit changes.

**Override for ops:** set ``EPAM_DATA_ROOT`` to force a specific path
(useful for shared network drives, smoke tests, or single-tenant
deployments where the operator wants explicit control). Setting
``EPAM_DATA_ROOT=data`` reverts to legacy project-relative behaviour.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Sanitise the Windows username for filesystem use. Most usernames are
# already filesystem-safe but we strip path separators / control chars
# defensively so a poisoned USERNAME env var can't escape the data root.
_UNSAFE_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitise_username(name: str) -> str:
    """Strip filesystem-unsafe characters from a username.

    Returns ``"default"`` when the input is empty or sanitisation
    leaves nothing — never returns an empty string, since that would
    collapse the per-user path to the parent directory.
    """
    if not name:
        return "default"
    cleaned = _UNSAFE_CHARS_RE.sub("_", name).strip(" .")
    return cleaned or "default"


def get_user_data_root() -> Path:
    """Resolve the active user's VERITAS data directory.

    Resolution order:

    1. ``EPAM_DATA_ROOT`` env var override (verbatim, used by tests
       and ops). Pass ``data`` to revert to project-relative.
    2. Windows: ``%APPDATA%\\VERITAS\\<username>\\``
    3. POSIX: ``~/.veritas/`` (note: NOT per-user-of-same-home — Linux
       users typically have unique home directories already, so we
       don't double-namespace)
    4. Fallback: project-relative ``data/`` (CI, broken env)
    """
    override = os.environ.get("EPAM_DATA_ROOT")
    if override:
        # Honour exactly what the operator asked for — no /<username>
        # auto-append, since they may have already pointed us at a
        # specific user folder.
        return Path(override).expanduser()

    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            username = _sanitise_username(os.environ.get("USERNAME", ""))
            return Path(appdata) / "VERITAS" / username
        # APPDATA missing on Windows is unusual but possible in CI.
        # Fall through to the home-dir branch below.
        logger.warning(
            "APPDATA env var not set on Windows; falling back to user home"
        )

    home = Path.home()
    # Path.home() on misconfigured environments can return "/" or
    # similar unhelpful defaults. Guard against that to avoid creating
    # a top-level /.veritas directory.
    if home and str(home) not in ("/", ".", ""):
        return home / ".veritas"

    # Final fallback: project-relative. Preserves the dev workflow
    # for anyone who runs the server without a configured home.
    return Path.cwd() / "data"


def get_shared_data_root() -> Path:
    """Cross-user shared VERITAS data — currently only the voice DB.

    Voice profiles for EPAM employees are shared across all VERITAS
    users on the machine: when lawyer A enrolls lawyer B's voice once,
    lawyer C also benefits when B speaks in C's meeting. So we keep
    this on a separate root from the per-user data.

    Resolution order:

    1. ``EPAM_SHARED_DATA_ROOT`` env override
    2. Windows: ``%PROGRAMDATA%\\VERITAS\\``
    3. POSIX: ``/var/lib/veritas/`` if writable, else
       ``<user-data-root>/shared/`` as fallback
    """
    override = os.environ.get("EPAM_SHARED_DATA_ROOT")
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        programdata = os.environ.get("PROGRAMDATA")
        if programdata:
            return Path(programdata) / "VERITAS"

    # Try the system-wide POSIX location, fall back to user-scoped if
    # we can't write there (developer machine, no sudo).
    posix_shared = Path("/var/lib/veritas")
    try:
        posix_shared.mkdir(parents=True, exist_ok=True)
        # Probe writability without leaving a file behind.
        probe = posix_shared / ".write_probe"
        probe.touch(exist_ok=True)
        probe.unlink()
        return posix_shared
    except (PermissionError, OSError):
        return get_user_data_root() / "shared"


def resolve_user_path(relative_path: str | Path) -> Path:
    """Resolve a config-declared relative path against the user data root.

    Absolute paths pass through unchanged so an operator can override
    individual subdirs in YAML without losing the per-user policy
    everywhere else. Relative paths get prefixed with the user data
    root.

    Examples:
        resolve_user_path("data/meetings")
        # On Windows: WindowsPath('C:/Users/BAZA/AppData/Roaming/VERITAS/BAZA/data/meetings')
        # With EPAM_DATA_ROOT=data: PosixPath('data/meetings') (legacy behaviour)

        resolve_user_path("/srv/shared/meetings")
        # PosixPath('/srv/shared/meetings') — absolute paths pass through
    """
    p = Path(relative_path).expanduser()
    if p.is_absolute():
        return p
    return get_user_data_root() / p


def ensure_user_data_dirs(*subdirs: str) -> Path:
    """Create the user data root + subdirectories on first run.

    Called once at server startup (see ``main.py:lifespan``). Listed
    subdirectories are created underneath the resolved user data
    root. Returns the resolved root for logging.
    """
    root = get_user_data_root()
    root.mkdir(parents=True, exist_ok=True)
    for sub in subdirs:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


__all__ = [
    "get_user_data_root",
    "get_shared_data_root",
    "resolve_user_path",
    "ensure_user_data_dirs",
]
