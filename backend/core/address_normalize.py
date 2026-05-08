"""Address + legal abbreviation normalization (Sprint 2026-04-30, #40).

Postprocessor pass that applies regex substitutions from
``config/address_corrections.yaml`` to the aligned transcript text.
Fixes consistent ASR mis-spellings of Moscow street names, legal
references, and monetary expressions that we've observed across
protocols.

**Why this exists:** WER on Russian legal speech (~22%) is dominated
by proper-name errors. Hot-words biasing (task #39) helps prevent
new mis-spellings; this module repairs the residue post-ASR. Two
defenses are better than one — biasing is best-effort, postprocessor
substitution is deterministic.

**Why YAML (not code):** the corrections list grows as new meetings
expose new mis-spellings. Editing YAML doesn't require a redeploy;
restarting the backend picks up the new entries.

**Word boundaries:** patterns are applied with ``re.sub`` honouring
the YAML's regex syntax. Most entries use ``\b...\b`` to avoid
matching substrings inside longer words. Bare-string keys (no
regex metachars) are escaped automatically and word-boundary-anchored.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


_RULES_CACHE: Optional[list[tuple[re.Pattern, str]]] = None


def _yaml_path() -> Path:
    candidates = [
        Path("config/address_corrections.yaml"),
        Path("address_corrections.yaml"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _looks_like_regex(s: str) -> bool:
    """Heuristic: does this string contain regex metacharacters?"""
    return any(ch in s for ch in r".\^$*+?()[]{}|")


def _compile_rules() -> list[tuple[re.Pattern, str]]:
    """Read the YAML and compile (pattern, replacement) pairs.

    Bare-string keys without regex metacharacters are auto-escaped
    and wrapped in ``\b...\b`` so they only match whole words. Keys
    that look like regex are compiled as-is so authors can express
    custom anchoring.
    """
    path = _yaml_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to read address_corrections.yaml: {e}")
        return []
    if not isinstance(data, dict):
        return []

    compiled: list[tuple[re.Pattern, str]] = []
    for section_name, section in data.items():
        if not isinstance(section, dict):
            continue
        for raw_pattern, replacement in section.items():
            if not isinstance(raw_pattern, str) or not isinstance(
                replacement, str
            ):
                continue
            try:
                if _looks_like_regex(raw_pattern):
                    pat = re.compile(raw_pattern, re.UNICODE)
                else:
                    pat = re.compile(
                        rf"\b{re.escape(raw_pattern)}\b", re.UNICODE
                    )
                compiled.append((pat, replacement))
            except re.error as e:
                logger.warning(
                    f"Skipping invalid regex in {section_name}: "
                    f"{raw_pattern!r} ({e})"
                )
    logger.info(
        f"Loaded {len(compiled)} address/legal correction rules"
    )
    return compiled


def get_rules() -> list[tuple[re.Pattern, str]]:
    """Return compiled rules, cached after first call."""
    global _RULES_CACHE
    if _RULES_CACHE is None:
        _RULES_CACHE = _compile_rules()
    return _RULES_CACHE


def normalize_text(text: str) -> str:
    """Apply all corrections to a single text string."""
    if not text:
        return text
    out = text
    for pat, replacement in get_rules():
        out = pat.sub(replacement, out)
    return out


def normalize_segments(segments: list) -> list:
    """Apply normalization to every segment's ``text`` field in place.

    Returns the same list — the caller can chain with other
    postprocessor passes. Segments without a ``text`` attribute are
    skipped silently.
    """
    rules = get_rules()
    if not rules:
        return segments
    for seg in segments:
        text = getattr(seg, "text", None)
        if not text or not isinstance(text, str):
            continue
        new_text = text
        for pat, replacement in rules:
            new_text = pat.sub(replacement, new_text)
        if new_text != text:
            seg.text = new_text
    return segments


def reset_cache() -> None:
    """Clear the compiled-rules cache. Useful for tests."""
    global _RULES_CACHE
    _RULES_CACHE = None


__all__ = [
    "normalize_text",
    "normalize_segments",
    "reset_cache",
    "get_rules",
]
