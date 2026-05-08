"""Meeting-type-specific protocol builders.

Each submodule owns the prompt, the verification prompt, and the parser
for one MeetingType. The SummarizationEngine dispatches to the right
builder based on MeetingJob.meeting_type.

This package replaces the single generic prompt with a family of
type-specific prompts, each tuned to the structure that type requires:
    - court_hearing: verbatim Q&A transcript + summary header
    - administrative: department-grouped Decisions / Tasks / Open Issues
    - client_meeting: client identification + topics + commitments
    - interview: dual output (internal scorecard + external feedback)
    - generic: legacy 7-section protocol (backward compatibility)

Design rules:
- Every builder returns a dict of Ollama chat messages for _generate().
- Every builder's parser returns a typed result usable by the formatter.
- Anti-hallucination rules (speaker citation, quote-or-drop, verification
  pass) are baked into every prompt — not just the generic one.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from backend.app.models import MeetingType

if TYPE_CHECKING:
    from backend.engine.protocols.schemas import ProtocolResult

# Registry: MeetingType -> module with build_prompt / parse_response /
# build_verification_prompt functions. Populated lazily on first use so we
# don't pay import cost if summarization is disabled.
_REGISTRY: dict[MeetingType, Any] = {}


def get_builder(meeting_type: MeetingType) -> Any:
    """Return the builder module for a meeting type.

    Lazy-loads the module on first request. Falls back to the generic
    builder for unknown types so new enum values never crash the pipeline.
    """
    if meeting_type in _REGISTRY:
        return _REGISTRY[meeting_type]

    if meeting_type == MeetingType.COURT_HEARING:
        from backend.engine.protocols import court_hearing as mod
    elif meeting_type == MeetingType.ADMINISTRATIVE:
        from backend.engine.protocols import administrative as mod
    elif meeting_type == MeetingType.CLIENT_MEETING:
        from backend.engine.protocols import client_meeting as mod
    elif meeting_type == MeetingType.INTERVIEW:
        from backend.engine.protocols import interview as mod
    else:
        from backend.engine.protocols import generic as mod

    _REGISTRY[meeting_type] = mod
    return mod


__all__ = ["get_builder"]
