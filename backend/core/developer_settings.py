"""Developer settings registry for controlled pipeline experiments.

This module deliberately avoids the "paste any model from the internet"
pattern. VERITAS is a legal on-premise pipeline, so model switches are
handled through a small manifest with explicit integration metadata.
The UI can still add draft entries later, but only validated entries are
eligible for production use.
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.app.paths import resolve_user_path


SETTINGS_VERSION = 1


def _settings_path() -> Path:
    path = resolve_user_path("data/developer/settings.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_MODEL_CATALOG: dict[str, list[dict[str, Any]]] = {
    "asr": [
        {
            "id": "asr-auto-quality",
            "name": "Auto: GigaAM for Russian, Whisper for English-heavy",
            "provider": "veritas",
            "status": "production",
            "engine": "auto",
            "recommended_for": ["court_hearing", "administrative", "generic"],
            "notes": "Quality default. Routes by detected language, not by meeting type.",
        },
        {
            "id": "gigaam-v3-rnnt",
            "name": "GigaAM v3 e2e RNNT",
            "provider": "sber",
            "status": "production",
            "engine": "gigaam",
            "recommended_for": ["court_hearing", "administrative"],
            "notes": "Primary Russian ASR. Do not silently downshift batch size.",
        },
        {
            "id": "faster-whisper-large-v3",
            "name": "faster-whisper large-v3",
            "provider": "ctranslate2",
            "status": "validated",
            "engine": "whisper",
            "whisper_model": "large-v3",
            "recommended_for": ["english_heavy", "mixed_language"],
            "notes": "Fallback for English-heavy or mixed-language recordings.",
        },
        {
            "id": "hf-whisper-ru-large-v3",
            "name": "HF Whisper large-v3 Russian",
            "provider": "huggingface",
            "status": "validated",
            "engine": "hf-whisper",
            "whisper_model": "antony66/whisper-large-v3-russian",
            "recommended_for": ["administrative"],
            "notes": "Russian Whisper fine-tune. Keep as comparison candidate.",
        },
        {
            "id": "nemo-ru-conformer",
            "name": "NVIDIA NeMo Russian Conformer",
            "provider": "nvidia",
            "status": "experimental",
            "engine": "nemo",
            "recommended_for": ["linux_server"],
            "notes": "Linux/Docker candidate. Requires server-side validation.",
        },
    ],
    "diarization": [
        {
            "id": "pyannote-community-1",
            "name": "pyannote speaker-diarization-community-1",
            "provider": "huggingface",
            "status": "production",
            "engine": "pyannote",
            "model": "pyannote/speaker-diarization-community-1",
            "recommended_for": ["court_hearing", "administrative", "generic"],
            "notes": "Production diarization. SpeechBrain is intentionally excluded.",
        },
        {
            "id": "pyannote-community-1-court",
            "name": "pyannote Community-1, court-tuned",
            "provider": "huggingface",
            "status": "validated",
            "engine": "pyannote",
            "model": "pyannote/speaker-diarization-community-1",
            "recommended_for": ["court_hearing"],
            "notes": "Same model with court-specific clustering/min-segment config.",
        },
        {
            "id": "pyannote-community-1-many-speakers",
            "name": "pyannote Community-1, many-speaker admin",
            "provider": "huggingface",
            "status": "validated",
            "engine": "pyannote",
            "model": "pyannote/speaker-diarization-community-1",
            "recommended_for": ["administrative"],
            "notes": "Use with explicit active-speaker range, e.g. 11-15 or 16-20.",
        },
    ],
    "llm": [
        {
            "id": "gemma4-26b-ollama",
            "name": "Gemma 4 26B via Ollama",
            "provider": "ollama",
            "status": "production",
            "model": "gemma4:26b",
            "base_url": "http://localhost:11434",
            "context_tokens": 262144,
            "structured_outputs": True,
            "notes": "Current local quality default on RTX 3090.",
        },
        {
            "id": "qwen3-32b-vllm",
            "name": "Qwen3 32B Instruct via vLLM",
            "provider": "openai_compatible",
            "status": "candidate",
            "model": "Qwen/Qwen3-32B",
            "base_url": "http://localhost:8000/v1",
            "context_tokens": 131072,
            "structured_outputs": True,
            "notes": "Ubuntu + 4090 48GB candidate. Validate VRAM and JSON quality.",
        },
        {
            "id": "qwen3-30b-a3b-vllm",
            "name": "Qwen3 30B-A3B via vLLM",
            "provider": "openai_compatible",
            "status": "candidate",
            "model": "Qwen/Qwen3-30B-A3B",
            "base_url": "http://localhost:8000/v1",
            "context_tokens": 131072,
            "structured_outputs": True,
            "notes": "Candidate when throughput matters. Needs real transcript tests.",
        },
        {
            "id": "mistral-small-vllm",
            "name": "Mistral Small Instruct via vLLM",
            "provider": "openai_compatible",
            "status": "experimental",
            "model": "mistralai/Mistral-Small-Instruct",
            "base_url": "http://localhost:8000/v1",
            "context_tokens": 32768,
            "structured_outputs": True,
            "notes": "Comparison candidate, not a Russian-default choice.",
        },
        {
            "id": "deepseek-r1-distill-qwen32b-vllm",
            "name": "DeepSeek-R1 Distill Qwen 32B via vLLM",
            "provider": "openai_compatible",
            "status": "experimental",
            "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "base_url": "http://localhost:8000/v1",
            "context_tokens": 131072,
            "structured_outputs": False,
            "notes": "Experimental only. Reasoning models can waste output budget.",
        },
    ],
}


DEFAULT_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "administrative.main",
        "meeting_type": "administrative",
        "stage": "main",
        "title": "Administrative protocol extraction",
        "enabled": False,
        "version": 1,
        "system_template": "",
        "user_template": "",
        "placeholders": ["transcript", "meeting_context", "language"],
        "notes": "Disabled means VERITAS uses the built-in production prompt.",
    },
    {
        "id": "administrative.recall",
        "meeting_type": "administrative",
        "stage": "recall",
        "title": "Administrative recall pass",
        "enabled": False,
        "version": 1,
        "system_template": "",
        "user_template": "",
        "placeholders": ["transcript_window", "existing_items", "meeting_context"],
        "notes": "Optional override for expanding missed decisions and tasks.",
    },
    {
        "id": "administrative.polish",
        "meeting_type": "administrative",
        "stage": "polish",
        "title": "Administrative polish pass",
        "enabled": False,
        "version": 1,
        "system_template": "",
        "user_template": "",
        "placeholders": ["protocol_json"],
        "notes": "Format-only pass. Disabled by default.",
    },
    {
        "id": "court_hearing.main",
        "meeting_type": "court_hearing",
        "stage": "main",
        "title": "Court protocol header and summary",
        "enabled": False,
        "version": 1,
        "system_template": "",
        "user_template": "",
        "placeholders": ["transcript", "meeting_context", "language"],
        "notes": "Court stenogram turns come from aligned transcript, not from LLM.",
    },
    {
        "id": "generic.main",
        "meeting_type": "generic",
        "stage": "main",
        "title": "Generic meeting protocol",
        "enabled": False,
        "version": 1,
        "system_template": "",
        "user_template": "",
        "placeholders": ["transcript", "meeting_context", "language"],
        "notes": "Fallback meeting protocol prompt.",
    },
]


def default_settings() -> dict[str, Any]:
    return {
        "version": SETTINGS_VERSION,
        "updated_at": datetime.utcnow().isoformat(),
        "active": {
            "profile": "rtx3090_quality",
            "asr": "asr-auto-quality",
            "diarization": "pyannote-community-1",
            "llm": "gemma4-26b-ollama",
        },
        "profiles": [
            {
                "id": "rtx3090_quality",
                "name": "RTX 3090 quality",
                "description": "Current local workstation profile. Max quality, no fallback degradation.",
            },
            {
                "id": "rtx4090_vllm_quality",
                "name": "Ubuntu RTX 4090 48GB + vLLM",
                "description": "Server target profile for vLLM/OpenAI-compatible LLM serving.",
            },
            {
                "id": "court_max_quality",
                "name": "Court hearing max quality",
                "description": "Exact speaker count, case dictionary, pyannote court tuning.",
            },
            {
                "id": "admin_max_recall",
                "name": "Administrative max recall",
                "description": "Full transcript, broad practical extraction, recall pass enabled.",
            },
            {
                "id": "experimental",
                "name": "Experimental",
                "description": "For IT testing only. Not for pilot production runs.",
            },
        ],
        "catalog": deepcopy(DEFAULT_MODEL_CATALOG),
        "prompts": deepcopy(DEFAULT_PROMPTS),
    }


def load_settings() -> dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        settings = default_settings()
        save_settings(settings)
        return settings
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        settings = default_settings()
        save_settings(settings)
        return settings

    # Merge in newly introduced defaults without overwriting local edits.
    changed = False
    if "catalog" not in settings:
        settings["catalog"] = deepcopy(DEFAULT_MODEL_CATALOG)
        changed = True
    if "prompts" not in settings:
        settings["prompts"] = deepcopy(DEFAULT_PROMPTS)
        changed = True
    existing_prompt_ids = {p.get("id") for p in settings.get("prompts", [])}
    for prompt in DEFAULT_PROMPTS:
        if prompt["id"] not in existing_prompt_ids:
            settings["prompts"].append(deepcopy(prompt))
            changed = True
    if changed:
        save_settings(settings)
    return settings


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    settings["version"] = SETTINGS_VERSION
    settings["updated_at"] = datetime.utcnow().isoformat()
    _settings_path().write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return settings


def find_catalog_entry(settings: dict[str, Any], category: str, model_id: str) -> dict[str, Any] | None:
    for entry in settings.get("catalog", {}).get(category, []):
        if entry.get("id") == model_id:
            return entry
    return None


def get_prompt_override(meeting_type: str, stage: str) -> dict[str, Any] | None:
    settings = load_settings()
    wanted = f"{meeting_type}.{stage}"
    for prompt in settings.get("prompts", []):
        if prompt.get("id") == wanted and prompt.get("enabled"):
            return prompt
    return None


def render_prompt_override(
    prompt: dict[str, Any],
    *,
    transcript: str = "",
    meeting_context: str = "",
    language: str = "ru",
    **extra: Any,
) -> list[dict[str, str]]:
    """Render a prompt override into Ollama/OpenAI chat messages."""
    values = {
        "transcript": transcript,
        "meeting_context": meeting_context,
        "language": language,
        **extra,
    }

    def render(template: str) -> str:
        text = template or ""
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", str(value))
        return text

    messages: list[dict[str, str]] = []
    system = render(str(prompt.get("system_template", ""))).strip()
    user = render(str(prompt.get("user_template", ""))).strip()
    if system:
        messages.append({"role": "system", "content": system})
    if user:
        messages.append({"role": "user", "content": user})
    return messages
