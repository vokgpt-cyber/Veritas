"""Hot-words / vocabulary biasing loader for ASR (Sprint 2026-04-30, #39).

Loads ``config/hot_words.yaml`` and produces a single string suitable
for passing to Whisper-family ASR engines as ``initial_prompt``. The
prompt biases token probabilities toward the listed names and domain
terms — particularly useful for Russian proper names that the model
has never seen in training, and for legal/financial vocabulary that
Whisper splits inconsistently.

**Usage from ASR engines:**

    from backend.core.hot_words import load_hot_words_prompt
    prompt = load_hot_words_prompt(meeting_type="court_hearing")
    # Pass `prompt` as initial_prompt to faster-whisper / HF whisper.

**meeting_type filtering** (optional): only some sections apply to
each meeting type. court_hearing always wants ``addresses`` and
``counterparties``; administrative meetings want ``employees`` +
``finance`` + ``legal_terms``. Passing ``None`` (default) loads the
union of all sections — safe but spends more of the prompt window.

**Voice DB enrichment:** the ``employees`` section can be augmented
at runtime with names from the voice enrollment DB so newly enrolled
staff start being biased immediately without a YAML edit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


# Per-meeting-type section whitelists. Empty = load everything.
_MEETING_TYPE_SECTIONS = {
    "court_hearing": [
        "addresses",
        "counterparties",
        "legal_terms",
        "clients",
        "finance",
    ],
    "administrative": [
        "employees",
        "clients",
        "finance",
        "legal_terms",
    ],
    "client_meeting": [
        "employees",
        "counterparties",
        "clients",
        "finance",
    ],
    "interview": [
        "employees",
    ],
    "generic": [],  # load all
}


def _yaml_path() -> Path:
    """Locate the hot_words.yaml config file."""
    # Look relative to the project root first (where settings.yaml
    # lives), then current working dir as fallback.
    candidates = [
        Path("config/hot_words.yaml"),
        Path("hot_words.yaml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # default; load_yaml will return {} if absent


def _load_yaml() -> dict:
    """Read the YAML file; return empty dict on missing/invalid file."""
    path = _yaml_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(
                f"Hot-words YAML at {path} is not a dict; ignoring"
            )
            return {}
        return data
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to read hot-words YAML {path}: {e}")
        return {}


def _enrich_with_voice_db(employees: list[str]) -> list[str]:
    """Append names from the voice enrollment DB to the employees list.

    Best-effort: if voice_enrollment isn't available or the DB is
    empty, return the input unchanged. De-duplicated case-insensitively.
    """
    try:
        from backend.engine import voice_enrollment as ve
        profiles = ve.list_profiles()
    except Exception:  # noqa: BLE001
        return employees

    seen_lower = {n.strip().lower() for n in employees}
    out = list(employees)
    for p in profiles:
        name = (p.get("name") or "").strip()
        if name and name.lower() not in seen_lower:
            out.append(name)
            seen_lower.add(name.lower())
    return out


def load_hot_words_prompt(
    *,
    meeting_type: Optional[str] = None,
    max_chars: int = 800,
) -> str:
    """Build the ASR ``initial_prompt`` string for a meeting type.

    ``max_chars`` caps the total length so we never overflow Whisper's
    finite prompt window. We greedily pack sections in declaration
    order; later sections may be truncated. Set higher if you've
    confirmed your model accepts longer prompts.

    Returns an empty string when no entries apply — caller should
    treat empty as "no biasing", not as an error.
    """
    data = _load_yaml()
    if not data:
        return ""

    if meeting_type and meeting_type in _MEETING_TYPE_SECTIONS:
        sections = _MEETING_TYPE_SECTIONS[meeting_type]
        if not sections:
            sections = list(data.keys())
    else:
        sections = list(data.keys())

    parts: list[str] = []
    for section in sections:
        items = data.get(section, [])
        if section == "employees":
            items = _enrich_with_voice_db(items)
        if not isinstance(items, list):
            continue
        for entry in items:
            if isinstance(entry, str) and entry.strip():
                parts.append(entry.strip())

    # Deduplicate case-insensitively, preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    # Pack into a single comma-separated string until we hit the
    # length cap. Whisper accepts free-form text here; we're not
    # constrained to a particular format.
    out: list[str] = []
    total = 0
    for p in deduped:
        addition = (", " if out else "") + p
        if total + len(addition) > max_chars:
            break
        out.append(p)
        total += len(addition)

    if not out:
        return ""
    prefix = "Имена и термины: "
    return prefix + ", ".join(out) + "."


__all__ = ["load_hot_words_prompt"]
