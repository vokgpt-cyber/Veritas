"""Summarization engine using Ollama HTTP API.

Calls a locally-running Ollama service for LLM inference. Default model:
Gemma 4 26B MoE (~16GB VRAM, 128K context). Ollama manages GPU memory
lifecycle — no torch/CUDA code needed in this module.

Produces a full 7-section Russian legal meeting protocol:
  1. Topic
  2. Participants
  3. Brief summary
  4. Key discussion points
  5. Decisions
  6. Action items (who / what / when)
  7. Open questions
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

import httpx

from backend.app.models import (
    AlignedSegment,
    Decision,
    MeetingProtocol,
    MeetingType,
    Participant,
    TaskItem,
    TopicItem,
)
from backend.engine.base import BaseEngine
from backend.engine.protocols import get_builder
from backend.engine.protocols.schemas import (
    GenericProtocolPayload,
    ProtocolResult,
)

logger = logging.getLogger(__name__)


class SummarizationEngine(BaseEngine):
    """LLM inference for meeting summarization via Ollama HTTP API.

    Replaces the previous llama-cpp-python engine. Ollama runs as a separate
    service and manages its own GPU memory, so this engine is a thin HTTP
    client. VRAMManager is NOT used for this engine.
    """

    def __init__(self, config: Any) -> None:
        """Initialize summarization engine.

        Args:
            config: AppConfig instance with summarization settings.
        """
        super().__init__(config)
        self._client: Optional[httpx.Client] = None
        self._provider: str = getattr(config.summarization, "provider", "ollama")
        self._base_url: str = (
            config.summarization.openai_base_url.rstrip("/")
            if self._provider == "openai_compatible"
            else config.summarization.ollama_base_url.rstrip("/")
        )
        self._model_name: str = config.summarization.ollama_model
        self._timeout: int = config.summarization.ollama_timeout
        self._model_context_limit: Optional[int] = None

    @property
    def required_vram_gb(self) -> float:
        """VRAM managed by Ollama — report 0 so VRAMManager skips checks."""
        return 0.0

    def load(self) -> None:
        """Verify Ollama connectivity and model availability.

        Does NOT load GPU memory — Ollama does that on first inference.
        We verify the service is reachable and the configured model exists.
        """
        if self.is_loaded:
            logger.warning("Summarization engine already loaded")
            return

        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout=float(self._timeout)),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )

        if self._provider == "openai_compatible":
            self._load_openai_compatible()
            return

        # Verify Ollama is running
        try:
            resp = self._client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Ollama not reachable at {self._base_url}. "
                "Start Ollama with: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama health check failed: {e}") from e

        # Check model is available
        tags_data = resp.json()
        available_models = [
            m.get("name", "") for m in tags_data.get("models", [])
        ]
        # Exact tag match: do not let Gemma 3 or Qwen masquerade as Gemma 4.
        model_found = self._model_name in available_models
        if not model_found:
            raise RuntimeError(
                f"Model '{self._model_name}' not found in Ollama. "
                f"Available: {available_models}. "
                f"Pull it with: ollama pull {self._model_name}"
            )
            # Don't raise — Ollama will pull on first use if configured

        self._model_context_limit = self._detect_model_context_limit()
        if self._model_context_limit:
            logger.info(
                "Ollama model context limit detected: %d tokens",
                self._model_context_limit,
            )

        self._model = True  # Mark as loaded (sentinel for BaseEngine.is_loaded)
        logger.info(
            f"Summarization engine ready (Ollama @ {self._base_url}, "
            f"model={self._model_name})"
        )

    def _openai_headers(self) -> dict[str, str]:
        """Headers for OpenAI-compatible providers such as vLLM."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = getattr(self._config.summarization, "openai_api_key", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _load_openai_compatible(self) -> None:
        """Verify a vLLM/OpenAI-compatible chat service."""
        if self._client is None:
            raise RuntimeError("HTTP client not initialized")
        try:
            resp = self._client.get(
                f"{self._base_url}/models",
                headers=self._openai_headers(),
            )
            resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"OpenAI-compatible LLM service not reachable at {self._base_url}. "
                "Start vLLM with its OpenAI-compatible server."
            )
        except Exception as e:
            raise RuntimeError(
                f"OpenAI-compatible LLM health check failed: {e}"
            ) from e

        available_models: list[str] = []
        try:
            data = resp.json()
            available_models = [
                str(m.get("id", ""))
                for m in data.get("data", [])
                if isinstance(m, dict)
            ]
        except Exception:
            available_models = []
        if available_models and self._model_name not in available_models:
            logger.warning(
                "Configured LLM model %r not listed by provider. Available: %s",
                self._model_name,
                available_models,
            )
        self._model_context_limit = None
        self._model = True
        logger.info(
            "Summarization engine ready (OpenAI-compatible @ %s, model=%s)",
            self._base_url,
            self._model_name,
        )

    def unload(self) -> None:
        """Release the HTTP client and ask Ollama to unload when applicable."""
        if not self.is_loaded:
            return

        try:
            if self._client is not None and self._provider == "ollama":
                # Send keep_alive=0 to free GPU memory
                self._client.post(
                    f"{self._base_url}/api/chat",
                    json={
                        "model": self._model_name,
                        "messages": [],
                        "keep_alive": 0,
                    },
                    timeout=30.0,
                )
                logger.info(
                    f"Sent keep_alive=0 to Ollama for model {self._model_name}"
                )
            if self._client is not None:
                self._client.close()
                self._client = None
        except Exception as e:
            logger.warning(f"LLM unload request failed (non-fatal): {e}")

        self._model = None
        logger.info("Summarization engine unloaded")

    async def process(
        self,
        transcript: list[AlignedSegment],
        participants: Optional[list[Participant]] = None,
        language: str = "ru",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        meeting_context: str = "",
        meeting_type: MeetingType = MeetingType.GENERIC,
        enable_polish_pass: Optional[bool] = None,
    ) -> ProtocolResult:
        """Generate a meeting protocol from transcript using the
        meeting-type-specific prompt and parser.

        With Gemma 4's 128K context, most meetings fit in a single pass.

        Args:
            transcript: List of aligned transcript segments.
            participants: Optional pre-computed participants.
            language: Detected language ("ru" or "en").
            progress_callback: Optional progress updates.
            meeting_context: User-provided meeting context.
            meeting_type: Selects the prompt template and output schema.

        Returns:
            ProtocolResult with the type-specific payload. For
            MeetingType.GENERIC the payload is a MeetingProtocol; for
            other types it is the appropriate schema (see
            backend.engine.protocols.schemas).
        """
        if not self.is_loaded or self._client is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        if not transcript:
            logger.warning("Empty transcript provided")
            no_content = (
                "Stenogramma otsutstvuet."
                if language == "ru"
                else "No transcript content available."
            )
            empty_protocol = MeetingProtocol(
                topic="",
                summary=no_content,
                participants=participants or [],
                key_topics=[],
                decisions=[],
                tasks=[],
                open_questions=[],
                transcript=transcript,
            )
            return ProtocolResult(
                meeting_type=meeting_type.value,
                payload=empty_protocol,
            )

        logger.info(
            "Processing transcript with %d segments (language: %s, type: %s)",
            len(transcript), language, meeting_type.value,
        )

        if progress_callback:
            progress_callback(0.05, "Formatting transcript")

        transcript_text = self._format_transcript(transcript)
        logger.info(f"Formatted transcript: {len(transcript_text)} chars")

        if progress_callback:
            progress_callback(0.1, "Generating protocol")

        # Dispatch to the meeting-type-specific builder for prompt
        # construction. The builder owns the prompt, the verification
        # prompt, and the response parser; this engine only runs the HTTP
        # transport, the verification pass, and (for GENERIC) the final
        # MeetingProtocol assembly with participants/topic/transcript.
        builder = get_builder(meeting_type)
        messages = builder.build_prompt(
            transcript_text, language, meeting_context
        )
        try:
            from backend.core.developer_settings import (
                get_prompt_override,
                render_prompt_override,
            )

            override = get_prompt_override(meeting_type.value, "main")
            if override is not None:
                override_messages = render_prompt_override(
                    override,
                    transcript=transcript_text,
                    meeting_context=meeting_context,
                    language=language,
                )
                if override_messages:
                    logger.info(
                        "Using developer prompt override: %s",
                        override.get("id", "unknown"),
                    )
                    messages = override_messages
        except Exception as exc:  # noqa: BLE001
            logger.warning("Prompt override lookup failed: %s", exc)
        response_format = None
        max_tokens = None
        if meeting_type == MeetingType.ADMINISTRATIVE:
            max_tokens = int(
                getattr(
                    self._config.summarization,
                    "admin_max_new_tokens",
                    self._config.summarization.max_new_tokens,
                )
            )
            if bool(
                getattr(self._config.summarization, "structured_outputs", True)
            ):
                from backend.engine.protocols import administrative as admin_proto
                response_format = admin_proto.response_format_schema()

        try:
            raw_protocol = await asyncio.to_thread(
                self._generate,
                messages,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except RuntimeError as exc:
            if response_format is None or "HTTP error: 400" not in str(exc):
                raise
            logger.warning(
                "Structured-output admin call failed (%s); retrying without "
                "Ollama format schema",
                exc,
            )
            raw_protocol = await asyncio.to_thread(
                self._generate,
                messages,
                max_tokens=max_tokens,
            )

        logger.info(
            "Raw LLM output (first 500 chars): %s",
            raw_protocol[:500] if raw_protocol else "<empty>"
        )

        if progress_callback:
            progress_callback(0.5, "Verifying protocol against transcript")

        # Verification pass — historically a second LLM call that
        # re-checked every claim and DROPPED items without explicit
        # transcript evidence. 2026-05-04 (task #45): SKIPPED for
        # administrative meetings because the deletion behaviour
        # combined with the (now-replaced) defensive prompt was
        # multiplicatively reducing extracted content. The new admin
        # prompt asks the LLM to attach `confidence: high|medium|low`
        # per item in the FIRST call; UI surfaces low-confidence with
        # an amber tag and the user curates manually. This matches
        # the Otter / Fireflies / Notion AI production pattern of
        # "extract permissively, let the user filter".
        #
        # Other meeting types (court_hearing, client_meeting,
        # interview) keep the verification pass — they have stricter
        # accuracy requirements (verbatim stenogram, candidate-facing
        # reports) where over-extraction is more dangerous than
        # missing items.
        skip_verification = (meeting_type == MeetingType.ADMINISTRATIVE)
        if skip_verification:
            logger.info(
                "Skipping verification pass for administrative meeting "
                "(task #45 — confidence comes from primary call)"
            )
        else:
            verification_messages = builder.build_verification_prompt(
                raw_protocol, transcript_text, language
            )
            try:
                verified_protocol = await asyncio.to_thread(
                    self._generate, verification_messages
                )
            except Exception as exc:
                logger.warning("Verification pass failed: %s", exc)
                verified_protocol = ""

            if verified_protocol and len(verified_protocol.strip()) > 50:
                logger.info(
                    "Verification pass produced %d chars (original: %d chars)",
                    len(verified_protocol), len(raw_protocol),
                )
                raw_protocol = verified_protocol
            else:
                logger.info("Verification pass returned empty — keeping original")

        if progress_callback:
            progress_callback(0.7, "Parsing structured output")

        # Each builder returns a ProtocolResult whose payload is the
        # type-specific schema. Court hearings and admin meetings
        # consume the aligned transcript for verbatim turns; admin also
        # consumes meeting_context to recover operator-provided
        # attendees if the LLM's participants list is empty.
        try:
            result = builder.parse_response(
                raw_protocol,
                aligned=transcript,
                meeting_context=meeting_context,
            )
        except TypeError:
            # Builders that haven't been updated to accept
            # meeting_context (court, client, interview, generic) still
            # work with the legacy two-arg signature.
            result = builder.parse_response(raw_protocol, aligned=transcript)

        if (
            meeting_type == MeetingType.ADMINISTRATIVE
            and bool(getattr(self._config.summarization, "admin_recall_pass", True))
        ):
            try:
                if progress_callback:
                    progress_callback(0.72, "Expanding admin protocol coverage")
                result = await self._expand_admin_practical_recall(
                    result=result,
                    transcript=transcript,
                    meeting_context=meeting_context,
                    progress_callback=progress_callback,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Admin recall pass failed (%s); keeping primary extraction",
                    exc,
                )

        # Speaker resolution post-pass (Sprint 2026-05-04, task #46).
        # After the protocol is built, make ONE focused LLM call asking
        # "given attendees + sample turns, map SPEAKER_X to names".
        # Apply the mapping across decisions/tasks/open_questions/risks
        # /turns. SPEAKER_X labels for which the model is unsure (or
        # which represent external participants not in the attendee
        # list) are relabelled "Спикер #N" — more readable than the
        # technical SPEAKER_06.
        #
        # Why a separate pass and not part of extraction: focused
        # single-task calls produce consistent mappings; asking the
        # extraction LLM to also do attribution led to inconsistent
        # results across topics (some resolved, others left raw).
        try:
            await self._apply_speaker_resolution(
                result=result,
                aligned=transcript,
                meeting_context=meeting_context,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Speaker resolution post-pass failed (%s); "
                "leaving SPEAKER_X labels in place", exc,
            )

        # Sprint 2026-04-30 (task #35): map-reduce topic-segmented
        # admin protocol. The flat result from `builder.parse_response`
        # gives us meeting_date, meeting_goal, summary, participants,
        # and risks; we now re-extract decisions/tasks/open_questions
        # per detected topic so the protocol reads as topic-grouped
        # mini-protocols rather than collapsed flat lists.
        #
        # Failure modes are non-fatal: if topic detection returns no
        # topics, or any per-topic call fails, we fall back to the
        # flat result we already have. The user toggle/feature flag
        # gates the whole step.
        if (
            meeting_type == MeetingType.ADMINISTRATIVE
            and bool(
                getattr(
                    self._config.summarization,
                    "use_topic_segmented_admin",
                    False,
                )
            )
        ):
            try:
                if progress_callback:
                    progress_callback(0.72, "Detecting topic boundaries")
                ts_result = await self._build_topic_segmented_admin(
                    flat_result=result,
                    aligned=transcript,
                    language=language,
                    meeting_context=meeting_context,
                    progress_callback=progress_callback,
                )
                if ts_result is not None:
                    result = ts_result
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Topic-segmented admin pass failed (%s); "
                    "keeping flat 9-section result", exc,
                )

        # T3.5 (2026-04-23): polish pass over admin protocol JSON.
        # Format-only edits: whitespace, capitalization, punctuation,
        # deduplication. Validates that counts and speakers are
        # unchanged before accepting. Admin meetings only.
        # Per-job UI toggle wins over config default.
        polish_enabled = (
            enable_polish_pass
            if enable_polish_pass is not None
            else bool(
                getattr(self._config.summarization, "polish_pass", False)
            )
        )
        if (
            meeting_type == MeetingType.ADMINISTRATIVE
            and polish_enabled
        ):
            try:
                if progress_callback:
                    progress_callback(0.8, "Polishing protocol")
                result = await asyncio.to_thread(
                    self._polish_admin_protocol, result
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Polish pass failed (%s); keeping unpolished output", exc,
                )

        # GENERIC: upgrade the lightweight GenericProtocolPayload into a
        # full MeetingProtocol so the existing formatter, frontend, and
        # archive paths see the same shape they have always seen.
        if meeting_type == MeetingType.GENERIC:
            if progress_callback:
                progress_callback(0.85, "Generating title")

            generic_payload = result.payload
            summary_text = ""
            topics_list: list[str] = []
            decisions_list: list[str] = []
            tasks_list: list[dict] = []
            questions_list: list[str] = []
            if isinstance(generic_payload, GenericProtocolPayload):
                summary_text = generic_payload.summary
                topics_list = list(generic_payload.topics)
                decisions_list = list(generic_payload.decisions)
                tasks_list = list(generic_payload.tasks)
                questions_list = list(generic_payload.questions)
            elif isinstance(generic_payload, MeetingProtocol):
                # Already a full protocol — return as-is.
                return result

            topic = await asyncio.to_thread(
                self._generate_topic, summary_text, language,
            )
            participants = self._build_participants(transcript, participants)

            full_protocol = MeetingProtocol(
                topic=topic,
                participants=participants,
                summary=summary_text.strip(),
                key_topics=[
                    TopicItem(title=t, content=t, speakers=[])
                    for t in topics_list
                ],
                decisions=[Decision(text=d) for d in decisions_list],
                tasks=[
                    TaskItem(
                        text=t.get("text", ""),
                        assignee=t.get("assignee"),
                        deadline=t.get("deadline"),
                    )
                    for t in tasks_list
                ],
                open_questions=questions_list,
                transcript=transcript,
            )

            result = ProtocolResult(
                meeting_type=meeting_type.value,
                payload=full_protocol,
            )

        if progress_callback:
            progress_callback(1.0, "Complete")

        logger.info(
            "Protocol generated (type=%s, payload=%s)",
            meeting_type.value, result.payload.__class__.__name__,
        )
        return result

    # ------------------------------------------------------------------
    # Ollama HTTP call
    # ------------------------------------------------------------------

    def _detect_model_context_limit(self) -> Optional[int]:
        """Best-effort read of the model's max context from Ollama."""
        if self._client is None or self._provider != "ollama":
            return None
        try:
            response = self._client.post(
                f"{self._base_url}/api/show",
                json={"model": self._model_name},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.info("Could not detect Ollama context limit: %s", exc)
            return None

        candidates: list[int] = []
        model_info = data.get("model_info")
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                if "context_length" not in str(key):
                    continue
                try:
                    candidates.append(int(value))
                except (TypeError, ValueError):
                    continue

        parameters = data.get("parameters")
        if isinstance(parameters, str):
            for match in re.finditer(r"\bnum_ctx\s+(\d+)\b", parameters):
                try:
                    candidates.append(int(match.group(1)))
                except (TypeError, ValueError):
                    continue

        return max(candidates) if candidates else None

    @staticmethod
    def _estimate_tokens_from_messages(messages: list[dict]) -> int:
        """Rough token estimate for choosing an Ollama context window."""
        chars = 0
        for msg in messages:
            chars += len(str(msg.get("content", "")))
        # Russian text often tokenizes denser than English. This is not
        # billing-grade tokenization; it only chooses 64K/128K/256K.
        return max(1, int(chars / 3.2))

    def _resolve_num_ctx(
        self,
        messages: list[dict],
        max_out: int,
    ) -> Optional[int]:
        """Resolve configured Ollama num_ctx for this request."""
        raw = str(
            getattr(self._config.summarization, "ollama_num_ctx", "auto")
        ).strip().lower()
        if raw in ("", "none", "off", "0"):
            return None

        configured_max = int(
            getattr(self._config.summarization, "ollama_num_ctx_auto_max", 262144)
        )
        model_cap = self._model_context_limit or configured_max
        cap = max(1024, min(configured_max, model_cap))

        if raw == "max":
            return cap

        if raw != "auto":
            try:
                requested = int(raw)
            except ValueError as exc:
                raise RuntimeError(
                    "EPAM_SUMMARIZATION_OLLAMA_NUM_CTX must be 'auto', "
                    "'max', 'off', or an integer token count"
                ) from exc
            if requested <= 0:
                return None
            if requested > cap:
                logger.warning(
                    "Configured num_ctx=%d exceeds detected cap=%d; using cap",
                    requested, cap,
                )
                return cap
            return requested

        min_ctx = int(
            getattr(self._config.summarization, "ollama_num_ctx_auto_min", 65536)
        )
        min_ctx = max(4096, min(min_ctx, cap))
        estimated_input = self._estimate_tokens_from_messages(messages)
        # Include output budget and headroom so the response can finish.
        needed = int((estimated_input + max_out) * 1.15)
        candidates = sorted({min_ctx, 131072, cap})
        for candidate in candidates:
            if candidate >= needed:
                return candidate

        logger.warning(
            "Estimated prompt+output needs %d tokens, but cap is %d. "
            "Ollama may truncate or lose attention on this run.",
            needed, cap,
        )
        return cap

    def _generate_openai_compatible(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a non-streaming request to vLLM/OpenAI-compatible API."""
        if self._client is None:
            raise RuntimeError("HTTP client not initialized")

        max_out = max_tokens or self._config.summarization.max_new_tokens
        request_body: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._config.summarization.temperature,
            "top_p": self._config.summarization.top_p,
            "max_tokens": max_out,
            "stream": False,
        }
        if response_format is not None:
            request_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "veritas_protocol",
                    "schema": response_format,
                },
            }

        retries = 3
        backoffs = [1.0, 3.0, 9.0]
        response = None
        for attempt in range(retries):
            try:
                response = self._client.post(
                    f"{self._base_url}/chat/completions",
                    json=request_body,
                    headers=self._openai_headers(),
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code in (408, 429, 500, 502, 503, 504) and attempt < retries - 1:
                    import time
                    time.sleep(backoffs[attempt])
                    continue
                raise RuntimeError(
                    f"OpenAI-compatible LLM HTTP error: {code} -- "
                    f"{exc.response.text}"
                ) from exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt < retries - 1:
                    import time
                    time.sleep(backoffs[attempt])
                    continue
                raise RuntimeError(
                    f"Lost connection to OpenAI-compatible LLM at {self._base_url}: {exc}"
                ) from exc

        if response is None:
            raise RuntimeError("OpenAI-compatible LLM did not return a response")
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI-compatible LLM returned no choices")
        message = choices[0].get("message") or {}
        text = str(message.get("content") or "").strip()
        if not text:
            raise RuntimeError("OpenAI-compatible LLM returned empty content")
        finish_reason = choices[0].get("finish_reason")
        if finish_reason == "length":
            raise RuntimeError(
                "LLM stopped because the output token limit was reached. "
                "Increase max output tokens and retry."
            )
        logger.info(
            "Generated %d chars via OpenAI-compatible provider, finish_reason=%s",
            len(text),
            finish_reason or "unknown",
        )
        return text

    def _generate(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request to Ollama (non-streaming).

        Args:
            messages: Chat messages (system + user).
            max_tokens: Override max output tokens.
            response_format: Optional Ollama structured-output schema.

        Returns:
            Generated text.
        """
        if self._client is None:
            raise RuntimeError("HTTP client not initialized")

        if self._provider == "openai_compatible":
            return self._generate_openai_compatible(
                messages,
                max_tokens=max_tokens,
                response_format=response_format,
            )

        max_out = max_tokens or self._config.summarization.max_new_tokens
        num_ctx = self._resolve_num_ctx(messages, max_out)

        request_body = {
            "model": self._model_name,
            "messages": messages,
            "options": {
                "temperature": self._config.summarization.temperature,
                "top_p": self._config.summarization.top_p,
                "num_predict": max_out,
            },
            "stream": False,
            "keep_alive": "10m",
            # Disable extended thinking — Gemma 4 and other thinking-capable
            # models may consume the entire token budget on internal reasoning
            # and return empty content. We want direct structured output.
            "think": False,
        }
        if num_ctx is not None:
            request_body["options"]["num_ctx"] = num_ctx
        if response_format is not None:
            request_body["format"] = response_format

        # T3.2 (2026-04-23): retry-with-backoff on transient failures.
        # Don't retry 4xx (programming errors). DO retry connection,
        # timeout, and 5xx — these are usually Ollama briefly busy or
        # the keep_alive window having just expired between calls.
        retries = 3
        backoffs = [1.0, 3.0, 9.0]
        last_exc: Optional[Exception] = None

        for attempt in range(retries):
            try:
                response = self._client.post(
                    f"{self._base_url}/api/chat",
                    json=request_body,
                )
                response.raise_for_status()
                break  # success
            except httpx.HTTPStatusError as e:
                # Retry only on 5xx and 408. 4xx other than 408 means we
                # sent something wrong — bailing out fast surfaces the bug.
                code = e.response.status_code
                if code >= 500 or code == 408:
                    last_exc = e
                    if attempt < retries - 1:
                        delay = backoffs[attempt]
                        logger.warning(
                            "Ollama %s on attempt %d/%d, retrying in %.1fs: %s",
                            code, attempt + 1, retries, delay, e.response.text[:200],
                        )
                        import time
                        time.sleep(delay)
                        continue
                raise RuntimeError(
                    f"Ollama HTTP error: {code} -- {e.response.text}"
                )
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e
                if attempt < retries - 1:
                    delay = backoffs[attempt]
                    logger.warning(
                        "Ollama %s on attempt %d/%d, retrying in %.1fs: %s",
                        type(e).__name__, attempt + 1, retries, delay, e,
                    )
                    import time
                    time.sleep(delay)
                    continue
                # Final attempt failed — surface the same error message
                # we used to use so callers see consistent messaging.
                if isinstance(e, httpx.TimeoutException):
                    raise RuntimeError(
                        f"Ollama request timed out after {self._timeout}s "
                        f"(after {retries} attempts). Consider increasing "
                        "EPAM_SUMMARIZATION_OLLAMA_TIMEOUT."
                    )
                raise RuntimeError(
                    f"Lost connection to Ollama at {self._base_url} "
                    f"(after {retries} attempts). "
                    "Is the service still running?"
                )
        else:
            # Exhausted all retries on retryable HTTP status codes.
            raise RuntimeError(
                f"Ollama failed after {retries} retries: {last_exc}"
            )

        result = response.json()

        if "error" in result:
            raise RuntimeError(f"Ollama error: {result['error']}")

        msg = result.get("message", {})
        text = msg.get("content", "").strip()

        # Fallback: if content is empty but thinking tokens were produced,
        # Ollama may have put everything in the "thinking" field.
        if not text:
            thinking = msg.get("thinking", "")
            if thinking:
                logger.warning(
                    "Ollama returned empty content but %d chars of thinking — "
                    "using thinking output as fallback", len(thinking)
                )
                text = thinking.strip()

        # Strip any <think>...</think> or unclosed <think> tags from output
        import re
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        text = re.sub(r"<think>[\s\S]*$", "", text).strip()

        # Strip chat-template markers that some GGUF/Ollama models leak into
        # their output (T-Pro 2.0 occasionally emits <|im_end|> at the tail,
        # which then pollutes the title and section headers downstream).
        for marker in (
            "<|im_end|>", "<|im_start|>", "<|endoftext|>",
            "<|assistant|>", "<|user|>", "<|system|>",
        ):
            text = text.replace(marker, "")
        text = text.strip()

        # Log generation stats
        eval_count = result.get("eval_count", "?")
        prompt_count = result.get("prompt_eval_count", "?")
        done_reason = result.get("done_reason", "")
        total_ns = result.get("total_duration", 0)
        total_s = total_ns / 1e9 if isinstance(total_ns, (int, float)) else 0
        logger.info(
            "Generated %s tokens (%d chars) in %.1fs, prompt=%s tokens, "
            "num_ctx=%s, done_reason=%s",
            eval_count, len(text), total_s, prompt_count,
            num_ctx or "default", done_reason or "unknown",
        )

        if done_reason == "length":
            raise RuntimeError(
                "Ollama stopped because the output token limit was reached. "
                "Increase EPAM_SUMMARIZATION_MAX_NEW_TOKENS or "
                "EPAM_SUMMARIZATION_ADMIN_MAX_NEW_TOKENS and retry."
            )
        if isinstance(prompt_count, int) and num_ctx is not None:
            if prompt_count > int(num_ctx * 0.92):
                logger.warning(
                    "Prompt used %d/%d context tokens. This run is near the "
                    "context limit and may miss details.",
                    prompt_count, num_ctx,
                )
        if isinstance(eval_count, int) and eval_count >= int(max_out * 0.98):
            logger.warning(
                "LLM used almost the full output budget (%d/%d tokens). "
                "The protocol may be incomplete even if Ollama did not "
                "report done_reason=length.",
                eval_count, max_out,
            )

        return text

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_protocol_prompt(
        self,
        transcript_text: str,
        language: str = "ru",
        meeting_context: str = "",
    ) -> list[dict]:
        """Build the full 7-section protocol prompt.

        Designed for Gemma 4 26B+ models with large context windows.
        All instructions in Russian (transliterated per project rules)
        for Russian meetings, English for English meetings.
        """
        if language == "ru":
            system = (
                "You are an experienced meeting secretary for a Russian law firm. "
                "Your task: produce a structured meeting protocol from the transcript below. "
                "The transcript is in Russian. Write ALL content in Russian, but use "
                "the EXACT English section headers shown below.\n\n"
                "STRICT ANTI-HALLUCINATION RULES:\n"
                "- Extract ONLY what is EXPLICITLY said in the transcript.\n"
                "- NEVER invent, add, or assume information not in the transcript.\n"
                "- NEVER add people, facts, dates, numbers, or events that are not mentioned.\n"
                "- When attributing a statement, you MUST name the speaker "
                "(e.g., 'Speaker_1 proposed...' or 'According to Speaker_3...').\n"
                "- If you cannot find a speaker for a claim, DO NOT include it.\n"
                "- If a section has no relevant content, write 'None' under it.\n"
                "- It is BETTER to write 'None' than to guess or invent content.\n"
                "- Use the EXACT format below. Do NOT add extra sections or headers.\n"
                "- Write summary and topics in Russian language.\n"
                "- Do NOT wrap output in markdown code blocks.\n\n"
                "RESPONSE FORMAT (use these exact headers):\n\n"
                "SUMMARY:\n"
                "(3-7 sentences in Russian: main topic, key positions of "
                "participants with their names, outcomes)\n\n"
                "TOPICS:\n"
                "- (topic 1 in Russian)\n"
                "- (topic 2 in Russian)\n\n"
                "DECISIONS:\n"
                "- (only if someone EXPLICITLY said 'we decided' or 'it was agreed'. "
                "Include WHO proposed it. Otherwise write: None)\n\n"
                "TASKS:\n"
                "- (what to do | who was assigned | deadline. "
                "Only if EXPLICITLY assigned in the transcript. Otherwise write: None)\n\n"
                "QUESTIONS:\n"
                "- (unresolved questions ACTUALLY raised by a named speaker. "
                "Otherwise write: None)"
            )
            if meeting_context:
                system += (
                    f"\n\nAdditional meeting context "
                    f"(provided by user):\n{meeting_context}"
                )
            user = f"Meeting transcript (in Russian):\n\n{transcript_text}"
        else:
            system = (
                "You are an experienced meeting secretary for a law firm. "
                "Your task is to produce a complete meeting protocol based "
                "on the transcript. Write in professional business English.\n\n"
                "RULES:\n"
                "- Extract ONLY what is EXPLICITLY stated in the transcript.\n"
                "- Do NOT invent, add, or assume anything.\n"
                "- If a section is empty, write 'None'.\n"
                "- Format the response EXACTLY as specified.\n\n"
                "RESPONSE FORMAT:\n"
                "BRIEF SUMMARY:\n"
                "(3-7 sentences: main topic, key positions, outcomes)\n\n"
                "KEY DISCUSSION POINTS:\n"
                "- (topic 1)\n"
                "- (topic 2)\n\n"
                "DECISIONS:\n"
                "- (only if EXPLICITLY decided. Otherwise: None)\n\n"
                "ACTION ITEMS:\n"
                "- (task | assignee | deadline. "
                "Only if EXPLICITLY assigned. Otherwise: None)\n\n"
                "OPEN QUESTIONS:\n"
                "- (unresolved questions. Otherwise: None)"
            )
            if meeting_context:
                system += (
                    f"\n\nMeeting context (provided by user):\n{meeting_context}"
                )
            user = f"Meeting transcript:\n\n{transcript_text}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def _apply_speaker_resolution(
        self,
        *,
        result,
        aligned,
        meeting_context: str,
    ) -> None:
        """Resolve SPEAKER_X labels to attendee names in place.

        One LLM call. Inputs: attendee list (from operator's checklist
        in meeting_context) + 1-3 sample turns per detected SPEAKER_X.
        Output: dict mapping SPEAKER_X → name. Applied across all
        speaker-bearing fields of the protocol payload (decisions,
        tasks, open_questions, risks, topic_summaries inner fields,
        turns, participants).

        Skipped silently if there's no attendee list to map against,
        or no SPEAKER_X labels in the aligned transcript. Failures
        are non-fatal — labels stay raw on error.
        """
        from backend.engine.protocols import speaker_resolution as sr
        from backend.engine.protocols.administrative import (
            _participants_from_context,
        )

        # Source the attendee list. Operator's pre-meeting picker is
        # authoritative; fall back to the LLM's participants if it
        # produced anything reasonable.
        attendees = _participants_from_context(meeting_context)
        if not attendees:
            payload = getattr(result, "payload", None)
            if payload is not None:
                payload_participants = getattr(payload, "participants", None)
                if isinstance(payload_participants, list):
                    attendees = [
                        str(n).strip() for n in payload_participants
                        if isinstance(n, str) and n.strip()
                    ]
        if not attendees:
            logger.info(
                "Speaker resolution: no attendee list available; "
                "skipping mapping pass"
            )
            return

        samples = sr._collect_speaker_samples(aligned)
        if not samples:
            logger.info(
                "Speaker resolution: no SPEAKER_X labels in aligned "
                "transcript; skipping"
            )
            return

        messages = sr.build_speaker_mapping_prompt(attendees, samples)
        try:
            raw = await asyncio.to_thread(self._generate, messages)
        except Exception as exc:
            logger.warning("Speaker mapping LLM call failed: %s", exc)
            return

        mapping = sr.parse_speaker_mapping(raw, valid_attendees=attendees)
        logger.info(
            "Speaker resolution: %d/%d labels mapped to attendees "
            "(unmapped will become 'Спикер #N')",
            len(mapping), len(samples),
        )

        payload = getattr(result, "payload", None)
        if payload is None:
            return
        sr.apply_mapping_to_protocol(payload, mapping)

    async def _expand_admin_practical_recall(
        self,
        *,
        result: ProtocolResult,
        transcript: list[AlignedSegment],
        meeting_context: str,
        progress_callback=None,
    ) -> ProtocolResult:
        """Add a recall pass for sparse practical admin protocols.

        Long-context single-pass extraction can compress a 90+ minute
        meeting into a neat top-10 list even when the transcript contains
        many useful local decisions. When the primary result is below a
        duration-based floor, re-scan chronological windows and merge
        non-duplicate candidates into the same practical `items` list.
        """
        from backend.engine.protocols import administrative as admin_proto
        from backend.engine.protocols.schemas import AdministrativeProtocol

        payload = result.payload
        if not isinstance(payload, AdministrativeProtocol):
            return result
        if not transcript:
            return result

        duration_s = self._admin_transcript_duration(transcript)
        if duration_s < 1800:
            return result

        rate = int(
            getattr(self._config.summarization, "admin_recall_min_items_per_hour", 15)
        )
        expected_min = max(12, int(math.ceil((duration_s / 3600.0) * rate)))
        current_count = len(payload.items or [])
        if current_count >= expected_min:
            logger.info(
                "Admin recall pass skipped: %d items meets expected minimum %d",
                current_count,
                expected_min,
            )
            return result

        window_minutes = int(
            getattr(self._config.summarization, "admin_recall_window_minutes", 12)
        )
        max_windows = int(
            getattr(self._config.summarization, "admin_recall_max_windows", 12)
        )
        windows = self._admin_recall_windows(
            transcript,
            window_minutes=window_minutes,
            max_windows=max_windows,
        )
        if not windows:
            return result

        logger.info(
            "Admin recall pass: %d primary items for %.1f min "
            "(expected >=%d); scanning %d windows",
            current_count,
            duration_s / 60.0,
            expected_min,
            len(windows),
        )

        response_format = None
        if bool(getattr(self._config.summarization, "structured_outputs", True)):
            response_format = admin_proto.practical_items_response_format_schema()

        total_added = 0
        for idx, (start_s, end_s, window_segments) in enumerate(windows, start=1):
            if progress_callback:
                progress = 0.72 + (idx - 1) / max(1, len(windows)) * 0.08
                progress_callback(
                    progress,
                    f"Expanding admin protocol coverage ({idx}/{len(windows)})",
                )

            existing_texts = [
                str(getattr(item, "text", "") or "")
                for item in (payload.items or [])
                if str(getattr(item, "text", "") or "").strip()
            ]
            messages = admin_proto.build_practical_recall_prompt(
                window_transcript=self._format_transcript(window_segments),
                existing_items=existing_texts,
                meeting_context=meeting_context,
                window_label=(
                    f"{self._format_timestamp(start_s)}-"
                    f"{self._format_timestamp(end_s)}"
                ),
            )
            try:
                raw = await asyncio.to_thread(
                    self._generate,
                    messages,
                    max_tokens=4096,
                    response_format=response_format,
                )
            except RuntimeError as exc:
                if response_format is None or "HTTP error: 400" not in str(exc):
                    logger.warning(
                        "Admin recall window %d/%d failed: %s",
                        idx,
                        len(windows),
                        exc,
                    )
                    continue
                logger.warning(
                    "Admin recall structured-output call failed (%s); "
                    "retrying window without schema",
                    exc,
                )
                try:
                    raw = await asyncio.to_thread(
                        self._generate,
                        messages,
                        max_tokens=4096,
                    )
                except Exception as retry_exc:  # noqa: BLE001
                    logger.warning(
                        "Admin recall schema-free retry failed for window %d/%d: %s",
                        idx,
                        len(windows),
                        retry_exc,
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Admin recall window %d/%d failed: %s",
                    idx,
                    len(windows),
                    exc,
                )
                continue

            candidates = admin_proto.parse_practical_items(raw)
            added = self._merge_admin_recall_items(payload, candidates)
            total_added += added
            logger.info(
                "Admin recall window %d/%d: %d candidates, %d added",
                idx,
                len(windows),
                len(candidates),
                added,
            )

        if total_added:
            payload.items.sort(key=self._admin_item_sort_key)
            logger.info(
                "Admin recall pass added %d items (%d total)",
                total_added,
                len(payload.items or []),
            )
        else:
            logger.info("Admin recall pass added no new items")
        return result

    @staticmethod
    def _admin_transcript_duration(transcript: list[AlignedSegment]) -> float:
        starts = [float(getattr(seg, "start", 0.0) or 0.0) for seg in transcript]
        ends = [float(getattr(seg, "end", 0.0) or 0.0) for seg in transcript]
        if not starts or not ends:
            return 0.0
        return max(0.0, max(ends) - min(starts))

    @staticmethod
    def _admin_recall_windows(
        transcript: list[AlignedSegment],
        *,
        window_minutes: int,
        max_windows: int,
    ) -> list[tuple[float, float, list[AlignedSegment]]]:
        if not transcript:
            return []
        first = min(float(getattr(seg, "start", 0.0) or 0.0) for seg in transcript)
        last = max(float(getattr(seg, "end", 0.0) or 0.0) for seg in transcript)
        duration = max(0.0, last - first)
        if duration <= 0:
            return []

        max_windows = max(1, max_windows)
        requested_window_s = max(300.0, float(max(1, window_minutes) * 60))
        window_s = max(requested_window_s, math.ceil(duration / max_windows))
        overlap_s = min(60.0, window_s * 0.1)

        windows: list[tuple[float, float, list[AlignedSegment]]] = []
        cursor = first
        while cursor < last and len(windows) < max_windows:
            start_s = max(first, cursor - (overlap_s if windows else 0.0))
            end_s = min(last, cursor + window_s)
            segs = [
                seg for seg in transcript
                if float(getattr(seg, "end", 0.0) or 0.0) > start_s
                and float(getattr(seg, "start", 0.0) or 0.0) < end_s
            ]
            if segs:
                windows.append((start_s, end_s, segs))
            cursor += window_s
        return windows

    @staticmethod
    def _normalise_admin_item_text(text: str) -> str:
        text = (text or "").lower().replace("ё", "е")
        text = re.sub(r"[^0-9a-z\u0400-\u04ff]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _admin_item_is_duplicate(cls, candidate, existing_items) -> bool:
        cand_text = cls._normalise_admin_item_text(
            str(getattr(candidate, "text", "") or "")
        )
        if not cand_text:
            return True
        cand_evidence = cls._normalise_admin_item_text(
            str(getattr(candidate, "evidence", "") or "")
        )
        cand_ts = cls._admin_timestamp_seconds(
            str(getattr(candidate, "timestamp", "") or "")
        )
        for item in existing_items or []:
            other_text = cls._normalise_admin_item_text(
                str(getattr(item, "text", "") or "")
            )
            if not other_text:
                continue
            if cand_text == other_text:
                return True
            ratio = SequenceMatcher(None, cand_text, other_text).ratio()
            if ratio >= 0.86:
                return True
            other_ts = cls._admin_timestamp_seconds(
                str(getattr(item, "timestamp", "") or "")
            )
            close_in_time = (
                cand_ts is not None
                and other_ts is not None
                and abs(cand_ts - other_ts) <= 180
            )
            if close_in_time and ratio >= 0.72:
                return True
            if cand_evidence:
                other_evidence = cls._normalise_admin_item_text(
                    str(getattr(item, "evidence", "") or "")
                )
                if other_evidence and cand_evidence == other_evidence:
                    return True
                evidence_ratio = SequenceMatcher(
                    None, cand_evidence, other_evidence
                ).ratio() if other_evidence else 0.0
                if close_in_time and evidence_ratio >= 0.78:
                    return True
        return False

    @classmethod
    def _merge_admin_recall_items(cls, payload, candidates) -> int:
        added = 0
        if not getattr(payload, "items", None):
            payload.items = []
        for item in candidates or []:
            if cls._admin_item_is_duplicate(item, payload.items):
                continue
            payload.items.append(item)
            kind = str(getattr(item, "kind", "") or "")
            if kind == "decision":
                payload.decisions.append(item)
            elif kind == "task":
                payload.tasks.append(item)
            elif kind == "open_question":
                payload.open_questions.append(item)
            elif kind == "risk":
                payload.risks.append(item)
            added += 1
        return added

    @staticmethod
    def _admin_item_sort_key(item) -> tuple[int, str]:
        seconds = SummarizationEngine._admin_timestamp_seconds(
            str(getattr(item, "timestamp", "") or "")
        )
        if seconds is None:
            return (10**9, str(getattr(item, "text", "") or ""))
        return (seconds, "")

    @staticmethod
    def _admin_timestamp_seconds(timestamp: str) -> Optional[int]:
        match = re.search(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})", timestamp or "")
        if not match:
            return None
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    async def _build_topic_segmented_admin(
        self,
        *,
        flat_result,
        aligned,
        language: str,
        meeting_context: str,
        progress_callback=None,
    ):
        """Map-reduce topic-segmented admin protocol (task #35).

        Two-stage pipeline:

        1. Topic boundary detection — one Gemma call asking "split
           the indexed transcript into N thematic segments". Result
           is a list of {name, start_idx, end_idx} ranges.

        2. Per-topic structured extraction — for each detected
           segment, one Gemma call producing a {discussion,
           decisions, tasks, open_questions} JSON. Smaller per-call
           prompts → better attention → more items recovered than
           the flat single-call path on long meetings.

        ``flat_result`` is the ProtocolResult from the regular flat
        9-section path. We pull meeting_date, meeting_goal, summary,
        participants, and risks from it (those are global / cross-
        topic) and replace the flat decisions/tasks/open_questions
        with the per-topic structure.

        Returns a new ProtocolResult on success, or None on failure
        so the caller falls back to the flat result.
        """
        from backend.engine.protocols import administrative as admin_proto
        from backend.engine.protocols.schemas import (
            AdministrativeProtocol,
            ProtocolResult,
        )

        # Pull global fields off the flat result. If the flat result
        # isn't an AdministrativeProtocol (shouldn't happen — guarded
        # at the call site by meeting_type check) we bail to fall back.
        flat_payload = getattr(flat_result, "payload", None)
        if not isinstance(flat_payload, AdministrativeProtocol):
            return None

        meeting_date = flat_payload.meeting_date
        meeting_goal = flat_payload.meeting_goal
        summary = flat_payload.summary
        participants = list(flat_payload.participants)
        risks = list(flat_payload.risks)

        # ----- Step 1: detect topic boundaries -----
        indexed_text, kept_segments = admin_proto._format_indexed_transcript(
            aligned
        )
        if not kept_segments:
            logger.info(
                "Topic-segmented admin: empty transcript, falling back to flat"
            )
            return None

        boundaries_messages = admin_proto.build_topic_boundaries_prompt(
            indexed_text
        )
        boundaries_raw = await asyncio.to_thread(
            self._generate, boundaries_messages
        )
        topics = admin_proto.parse_topic_boundaries(
            boundaries_raw, transcript_len=len(kept_segments)
        )
        if not topics:
            logger.info(
                "Topic detection returned no usable topics; falling back "
                "to flat 9-section format"
            )
            return None
        logger.info(
            "Detected %d topic segments: %s",
            len(topics), [t["name"] for t in topics],
        )

        # ----- Step 2: per-topic structured extraction -----
        topic_summaries = []
        n = len(topics)
        for i, t in enumerate(topics):
            start = int(t["start_idx"])
            end = int(t["end_idx"])
            slice_segs = kept_segments[start : end + 1]
            if not slice_segs:
                continue
            # Re-format the slice with local indices so the per-topic
            # prompt isn't confused by gaps in the global indexing.
            indexed_slice, _ = admin_proto._format_indexed_transcript(
                slice_segs
            )
            messages = admin_proto.build_topic_summary_prompt(
                topic_name=t["name"],
                indexed_segment=indexed_slice,
                meeting_context=meeting_context,
            )
            try:
                raw = await asyncio.to_thread(self._generate, messages)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Per-topic LLM call failed for '%s': %s; including "
                    "topic with empty body",
                    t["name"], exc,
                )
                from backend.engine.protocols.schemas import TopicSummary
                topic_summaries.append(TopicSummary(name=t["name"]))
                continue
            ts = admin_proto.parse_topic_summary(raw, topic_name=t["name"])
            topic_summaries.append(ts)
            if progress_callback:
                # Progress band 0.72-0.85 reserved for topic processing.
                progress_callback(
                    0.72 + 0.13 * ((i + 1) / max(1, n)),
                    f"Анализ темы: {t['name']}",
                )

        # All topics returned empty content → not worth replacing
        # the flat result. Fall back.
        any_content = any(
            t.discussion or t.decisions or t.tasks or t.open_questions
            for t in topic_summaries
        )
        if not any_content:
            logger.info(
                "Topic-segmented pass produced empty bodies; falling back"
            )
            return None

        return admin_proto.assemble_topic_segmented_protocol(
            aligned=aligned,
            meeting_date=meeting_date,
            meeting_goal=meeting_goal,
            summary=summary,
            participants=participants,
            risks=risks,
            topic_summaries=topic_summaries,
            meeting_context=meeting_context,
        )

    def _polish_admin_protocol(self, result):
        """T3.5 polish pass over admin protocol — format-only edits.

        Sends the protocol JSON to Gemma with a strict prompt that
        permits only formatting fixes (whitespace, capitalization,
        punctuation, deduplication) — never factual changes. Validates
        that counts (decisions/tasks/open_questions/risks/topics/
        participants) are preserved and speaker fields are identical
        before accepting.

        Returns the original `result` unchanged when validation fails
        or the LLM call errors. The polish pass is purely additive
        polish; we never let it regress quality.
        """
        from backend.engine.protocols.schemas import (
            AdministrativeProtocol,
            ProtocolResult,
        )

        payload = result.payload
        if not isinstance(payload, AdministrativeProtocol):
            return result

        # Serialize to JSON (Pydantic v2 mode='json' to handle Optional/
        # special types cleanly).
        original_json = payload.model_dump(mode="json")
        # Don't ship the stenogram turns to the polish pass — they're
        # verbatim ground truth, never to be touched, and they bloat
        # the prompt. Strip + restore.
        original_turns = original_json.pop("turns", [])

        system = (
            "Ты редактор-корректор протокола административного "
            "совещания. Тебе на вход — JSON протокола. Твоя задача "
            "— улучшить ТОЛЬКО форматирование: пробелы, "
            "капитализацию, пунктуацию, удаление дубликатов внутри "
            "массивов. НИКОГДА не меняй факты, имена, числа, "
            "speaker, owner, evidence.\n"
            "\n"
            "СТРОГИЕ ПРАВИЛА:\n"
            "- НЕ добавляй и НЕ удаляй элементы из массивов "
            "(decisions, tasks, open_questions, risks, topics, "
            "participants). Длина каждого массива должна остаться "
            "ровно такой же.\n"
            "- НЕ меняй значения полей speaker, owner, evidence в "
            "элементах. Они должны остаться идентичными.\n"
            "- НЕ переписывай text по смыслу. Можно поправить "
            "пунктуацию, исправить очевидные опечатки, но смысл "
            "должен сохраниться.\n"
            "- Можно удалить ОДИН элемент из массива, ТОЛЬКО если он "
            "буквально дублирует другой элемент в том же массиве. В "
            "этом случае оставь более полный.\n"
            "- Сохрани точный набор ключей верхнего уровня. Не "
            "добавляй и не удаляй ключи.\n"
            "\n"
            "Верни ТОЛЬКО исправленный JSON-объект. Никаких "
            "markdown-блоков, никакого текста вне JSON."
        )

        user = (
            "Отполируй форматирование протокола:\n\n"
            f"{json.dumps(original_json, ensure_ascii=False, indent=1)}"
        )

        try:
            raw = self._generate(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Polish-pass LLM call failed: %s", exc)
            return result

        # Strip code fences if model added them.
        raw_clean = raw.strip()
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean)
        raw_clean = re.sub(r"\s*```\s*$", "", raw_clean)
        try:
            polished_json = json.loads(raw_clean)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw_clean)
            if not match:
                logger.warning("Polish pass: JSON parse failed; keeping original")
                return result
            try:
                polished_json = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("Polish pass: JSON fallback parse failed")
                return result

        # Validate invariants before accepting.
        if not self._polish_validates(original_json, polished_json):
            logger.info("Polish pass: invariant check failed; keeping original")
            return result

        # Restore turns — the polish prompt never sees them.
        polished_json["turns"] = original_turns

        try:
            polished_payload = AdministrativeProtocol(**polished_json)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Polish pass: Pydantic re-validation failed (%s); keeping original",
                exc,
            )
            return result

        logger.info("Polish pass: applied (admin protocol)")
        return ProtocolResult(
            meeting_type=result.meeting_type,
            payload=polished_payload,
        )

    @staticmethod
    def _polish_validates(original: dict, polished) -> bool:
        """Check that polished JSON didn't break the invariants.

        The polish pass is allowed to:
          - Tweak text fields (whitespace, capitalization, punctuation).
          - Remove EXACT duplicates within an array (length may drop by
            up to 1 per array — but we currently require length match
            for simplicity).

        It is NOT allowed to:
          - Add/remove elements (current rule: lengths must match).
          - Change speaker/owner/evidence values.
          - Add/remove top-level keys.
        """
        if not isinstance(polished, dict):
            return False
        # Top-level key set must match.
        if set(polished.keys()) != set(original.keys()):
            return False
        # Length-preserving check on every array section. Strict for
        # this initial version — we can relax to "length may drop by
        # 1 per dedupe" later.
        for key in ("topics", "participants", "items", "decisions", "tasks",
                    "open_questions", "risks"):
            if key not in original:
                continue
            orig_list = original.get(key) or []
            polish_list = polished.get(key) or []
            if not isinstance(polish_list, list):
                return False
            if len(polish_list) != len(orig_list):
                return False
        # speaker / owner / evidence must match per-item across the
        # length-preserving arrays of dicts.
        for key, speaker_field in (
            ("items", "speaker"),
            ("decisions", "speaker"),
            ("tasks", "owner"),
            ("open_questions", "speaker"),
            ("risks", "speaker"),
        ):
            orig_items = original.get(key) or []
            polish_items = polished.get(key) or []
            for o, p in zip(orig_items, polish_items):
                if not isinstance(p, dict):
                    return False
                if (o or {}).get(speaker_field) != p.get(speaker_field):
                    return False
                if key == "items" and (o or {}).get("owner") != p.get("owner"):
                    return False
                # evidence stays exact too — it's the audit anchor.
                if (o or {}).get("evidence") != p.get("evidence"):
                    return False
        return True

    def _generate_topic(self, summary: str, language: str = "ru") -> str:
        """Generate a short meeting title from the summary."""
        if not summary:
            return ""

        if language == "ru":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Write ONLY a short meeting title in Russian (5-10 words). "
                        "No quotes, no period at the end. "
                        "Reply with ONE line only, in Russian."
                    ),
                },
                {"role": "user", "content": f"Meeting summary:\n{summary}"},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Write ONLY a short meeting title (5-10 words). "
                        "No quotes, no period. Reply with ONE line only."
                    ),
                },
                {"role": "user", "content": f"Summary:\n{summary}"},
            ]

        try:
            title = (
                self._generate(messages, max_tokens=80)
                .strip()
                .strip("\"'")
                .strip(".")
            )
            title = title.split("\n")[0].strip()
            if len(title) > 120:
                title = title[:117] + "..."
            if not title:
                raise ValueError("Empty title")
            return title
        except Exception:
            return summary.split(".")[0].strip()[:120]

    def _verify_protocol(
        self,
        protocol_text: str,
        transcript_text: str,
        language: str = "ru",
    ) -> str:
        """Second-pass verification: check protocol against transcript.

        Asks the LLM to review its own output and remove any claims that
        cannot be traced back to the transcript. This typically reduces
        hallucination by 30-50%.

        Args:
            protocol_text: The generated protocol to verify.
            transcript_text: The original transcript for fact-checking.
            language: Language code.

        Returns:
            Verified/corrected protocol text in the same format.
        """
        if not protocol_text or len(protocol_text.strip()) < 50:
            return protocol_text

        # Truncate transcript if too long — keep first 20K chars for verification
        # (the verification prompt + protocol + transcript must fit in context)
        max_transcript = 20000
        truncated_transcript = transcript_text[:max_transcript]

        if language == "ru":
            system = (
                "You are a STRICT fact-checker for legal meeting protocols. "
                "Your only job is to remove unsupported claims.\n\n"
                "VERIFICATION RULES (follow exactly):\n"
                "1. For EVERY sentence and bullet in the PROTOCOL, find a "
                "literal, word-level match in the TRANSCRIPT that supports it.\n"
                "2. If you cannot find clear support for a claim, DELETE the "
                "entire bullet or sentence. Do NOT annotate it, do NOT add "
                "notes, do NOT write '(not in transcript)' — just remove it "
                "silently.\n"
                "3. NEVER add commentary, explanations, or notes about what "
                "you removed. The output must look like a clean protocol, "
                "not a review document.\n"
                "4. NEVER add new information. You can only DELETE.\n"
                "5. If a speaker attribution is wrong, FIX it using the "
                "transcript. If no speaker can be attributed, DELETE the claim.\n"
                "6. If a section becomes empty after deletion, write ONLY the "
                "word 'None' on the bullet line.\n"
                "7. Output MUST be in the EXACT SAME FORMAT with the SAME "
                "section headers (SUMMARY, TOPICS, DECISIONS, TASKS, QUESTIONS).\n"
                "8. Content language: Russian.\n"
                "9. Never include placeholder bullets like '--', '-', or "
                "'(empty)'. Use 'None' instead.\n"
                "10. Never mention the verification process itself. The reader "
                "should not be able to tell this is a verified document."
            )
        else:
            system = (
                "You are a STRICT fact-checker. Your only job is to remove "
                "unsupported claims from the protocol.\n\n"
                "RULES:\n"
                "1. For every claim, find a literal match in the transcript. "
                "If you cannot, DELETE the bullet silently.\n"
                "2. Never annotate removals. Never add commentary or notes.\n"
                "3. Never add new information — only delete.\n"
                "4. Fix speaker attributions when wrong, or delete the claim.\n"
                "5. Empty sections: write 'None' on the bullet line.\n"
                "6. Output the same section headers (SUMMARY, TOPICS, DECISIONS, "
                "TASKS, QUESTIONS) in the same format.\n"
                "7. Never emit '--', '-', or '(empty)' — use 'None' instead.\n"
                "8. Never reveal that this is a verified document."
            )

        user = (
            f"PROTOCOL TO VERIFY:\n\n{protocol_text}\n\n"
            f"---\n\n"
            f"ORIGINAL TRANSCRIPT:\n\n{truncated_transcript}"
        )

        try:
            result = self._generate(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return result
        except Exception as e:
            logger.warning(f"Verification pass failed: {e}")
            return protocol_text

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_protocol_sections(text: str) -> dict:
        """Parse the structured 7-section LLM output.

        Handles both Russian (transliterated + Cyrillic) and English headers.

        Returns:
            dict with keys: summary, topics, decisions, tasks, questions.
        """
        result: dict = {
            "summary": "",
            "topics": [],
            "decisions": [],
            "tasks": [],
            "questions": [],
        }

        # Section header patterns (transliterated + Cyrillic Unicode + English)
        section_map = {
            # Transliterated Russian (matches prompt format)
            "KRATKOYE SODERZHANIYE": "summary",
            "KRATKOE SODERZHANIE": "summary",
            "KLYUCHEVYE TEMY OBSUZHDENIYA": "topics",
            "KLYUCHEVYE TEMY": "topics",
            "RESHENIYA": "decisions",
            "PORUCHENIYA": "tasks",
            "OTKRYTYE VOPROSY": "questions",
            # Cyrillic (LLM may respond in Cyrillic despite transliterated prompt)
            "КРАТКОЕ СОДЕРЖАНИЕ": "summary",
            "КЛЮЧЕВЫЕ ТЕМЫ ОБСУЖДЕНИЯ": "topics",
            "КЛЮЧЕВЫЕ ТЕМЫ": "topics",
            "РЕШЕНИЯ": "decisions",
            "ПОРУЧЕНИЯ": "tasks",
            "ЗАДАЧИ": "tasks",
            "ОТКРЫТЫЕ ВОПРОСЫ": "questions",
            # English
            "BRIEF SUMMARY": "summary",
            "KEY DISCUSSION POINTS": "topics",
            "DECISIONS": "decisions",
            "ACTION ITEMS": "tasks",
            "OPEN QUESTIONS": "questions",
            # Aliases
            "TOPICS": "topics",
            "TASKS": "tasks",
            "QUESTIONS": "questions",
            "SUMMARY": "summary",
            # Legacy from old prompt format
            "ТЕМЫ": "topics",
            "ВОПРОСЫ": "questions",
        }

        current_section: Optional[str] = None
        summary_lines: list[str] = []

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this is a section header.
            # LLM (T-Pro 2.0) emits headers as "### **SUMMARY:**" or similar:
            # markdown heading marker + bold emphasis + colon. Strip them all
            # before matching against section_map.
            header = stripped
            # 1. Strip leading markdown heading markers (#, ##, ###, etc.)
            header = re.sub(r"^#{1,6}\s*", "", header)
            # 2. Remove markdown bold markers (**...**)
            header = header.replace("**", "")
            # 3. Trim trailing colon and whitespace, then normalize to uppercase
            header = header.rstrip(":").strip().upper()

            if header in section_map:
                current_section = section_map[header]
                continue

            if current_section is None:
                continue

            # Summary section: collect as paragraph text
            if current_section == "summary":
                if stripped.lower() in ("net", "none", "net.", "none."):
                    continue
                summary_lines.append(stripped.lstrip("- ").lstrip("* "))
                continue

            # List sections: extract bullet items
            is_bullet = (
                stripped.startswith("-")
                or stripped.startswith("*")
                or re.match(r"^\d+[.)]\s", stripped)
            )
            if not is_bullet:
                continue

            item = re.sub(r"^[-*]\s*|^\d+[.)]\s*", "", stripped).strip()
            if not item or item.lower() in (
                "net", "none", "net.", "none.",
                "нет", "нет.",
            ):
                continue

            # Drop placeholder/separator items that the LLM sometimes emits
            # when a section has no content ("--", "---", em/en dashes, "**").
            if item in ("--", "---", "-", "—", "–", "**", "***"):
                continue

            # Drop verification-pass commentary that leaks into list sections.
            # These are meta-remarks the fact-checker produced instead of
            # actual protocol content (e.g., "**В разделе ...**",
            # "*Примечание*: ...", "Упоминание \"...\" не подтверждается...").
            noise_prefixes = (
                "**В разделе ",  # "**В разделе "
                "*Примечани",  # "*Примечани..."
                "Примечание:",  # "Примечание:"
                "Упоминание \"",  # "Упоминание \""
                "Note:", "*Note", "**Note",
                "(not in transcript",
                "(нет в транскрипт",  # "(нет в транскрипт..."
            )
            if any(item.startswith(p) for p in noise_prefixes):
                continue

            if current_section == "tasks":
                parts = [p.strip() for p in item.split("|")]
                task: dict[str, str] = {"text": parts[0]}
                if (
                    len(parts) > 1
                    and parts[1]
                    and parts[1].lower() not in ("net", "none", "-", "")
                ):
                    task["assignee"] = parts[1]
                if (
                    len(parts) > 2
                    and parts[2]
                    and parts[2].lower() not in ("net", "none", "-", "")
                ):
                    task["deadline"] = parts[2]
                result["tasks"].append(task)
            else:
                result[current_section].append(item)

        result["summary"] = " ".join(summary_lines)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_transcript(segments: list[AlignedSegment]) -> str:
        """Format aligned segments into readable text for the LLM."""
        lines = []
        for seg in segments:
            timestamp = SummarizationEngine._format_timestamp(seg.start)
            speaker = seg.speaker_name or seg.speaker_id
            lines.append(f"{speaker} [{timestamp}]: {seg.text}")
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _build_participants(
        self,
        transcript: list[AlignedSegment],
        participants: Optional[list[Participant]] = None,
    ) -> list[Participant]:
        """Build participants list with speaking stats from transcript."""
        if not participants:
            speakers: dict[str, Participant] = {}
            for seg in transcript:
                if seg.speaker_id not in speakers:
                    speakers[seg.speaker_id] = Participant(
                        speaker_id=seg.speaker_id,
                        speaker_name=seg.speaker_name,
                    )
            participants = list(speakers.values())

        total_duration = sum(seg.end - seg.start for seg in transcript)
        speaker_durations: dict[str, float] = {}
        for seg in transcript:
            speaker_durations[seg.speaker_id] = (
                speaker_durations.get(seg.speaker_id, 0.0) + seg.end - seg.start
            )
        for participant in participants:
            duration = speaker_durations.get(participant.speaker_id, 0.0)
            participant.speaking_time = duration
            if total_duration > 0:
                participant.speaking_share = round(
                    duration / total_duration * 100, 1
                )
        return participants
