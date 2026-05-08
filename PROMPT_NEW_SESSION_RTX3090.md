# EPAM VERITAS — New Session Prompt: RTX 3090 SOTA Upgrade

## Context

You are continuing development of **EPAM VERITAS** (Verbal Intelligence, Transcription and Summarization) — an on-premise meeting transcription and protocol generation system for EPAM law firm (Egorov, Puginsky, Afanasiev & Partners — a top Russian law firm, NOT EPAM Systems the IT company).

The codebase has been copied from a laptop with RTX 4060 (8GB VRAM) to a new machine with **RTX 3090 (24GB VRAM)**. The system works end-to-end: audio upload → transcription → diarization → alignment → post-processing → QA → summarization → DOCX protocol. Frontend is React 19 + Vite, backend is Python 3.11 + FastAPI.

**Read `CLAUDE.md` first** — it contains the full architecture, all 50 bug fixes, file structure, and design decisions. Everything below is ADDITIVE context that CLAUDE.md doesn't have yet.

---

## Your Task: Block 6 — SOTA Model Upgrade

The goal is to replace the current ML engines with state-of-the-art models that fully utilize the 24GB VRAM of the RTX 3090. The current models were constrained by the 8GB RTX 4060 and produced mediocre quality for Russian legal speech.

### Three components to upgrade:

#### 1. ASR: Replace faster-whisper with Qwen3-ASR-1.7B

**Current**: `backend/engine/whisper_asr.py` — faster-whisper (CTranslate2) with Whisper large-v3.
**Target**: Qwen3-ASR-1.7B (released January 2026) — outperforms Whisper large-v3 on Russian benchmarks (MLS, Common Voice, MLC-SLM). Supports forced alignment (timestamps) for Russian among 11 languages.

**What to do:**
- Create `backend/engine/qwen_asr.py` as a new engine implementing `BaseEngine` (see `backend/engine/base.py` for the interface: `load()`, `unload()`, `process()`, `required_vram_gb`, `is_loaded`)
- Qwen3-ASR uses its own inference pipeline (NOT faster-whisper compatible). Check the official repo: https://github.com/QwenLM/Qwen3-ASR for the inference API
- Must produce `list[TranscriptionSegment]` with `start`, `end`, `text`, `confidence`, `words` (list of `WordInfo`) — same output format as `whisper_asr.py`
- VRAM: ~2GB. Load/unload via the existing `VRAMManager` pattern
- Add `engine: auto|nemo|whisper|qwen` to `ASRConfig` in `backend/app/config.py`
- Keep `whisper_asr.py` intact as Tier 2 fallback — config switch only
- Language: Russian primary. The system processes Russian legal meetings.

**Fallback tier**:
- Tier 1: Qwen3-ASR-1.7B (new, SOTA)
- Tier 2: faster-whisper with Whisper large-v3 at float16 (current, works)
- Tier 3: faster-whisper with Whisper large-v3 at int8_float32 (for weak GPUs)

#### 2. Diarization: Upgrade pyannote-audio from 3.1 to 4.0

**Current**: `backend/engine/pyannote_diarization.py` — pyannote-audio 3.1 (optional, not default). Default is SpeechBrain (`backend/engine/diarization.py`) which under-segments speakers (detected 7 instead of 11 real speakers).
**Target**: pyannote-audio 4.0 with `community-1` model — significantly better speaker assignment and counting.

**What to do:**
- Update `pyannote_diarization.py` to use pyannote-audio 4.0 API (check for breaking changes from 3.1 → 4.0)
- The `community-1` model is the recommended open model for 4.0
- VRAM: ~9.5GB (up from ~600MB in 3.1). This is fine — models load sequentially, and 9.5GB fits easily in 24GB
- Make pyannote the **default** diarization engine in config (`diarization.engine: pyannote` in `DiarizationConfig`)
- Keep SpeechBrain engine as Tier 2 fallback
- Update `pyannote_model` config field to point to the 4.0 community-1 model
- Ensure it still produces `list[DiarizationSegment]` with `start`, `end`, `speaker_id`
- Test with min_speakers/max_speakers constraints (native pyannote feature)

**Fallback tier**:
- Tier 1: pyannote-audio 4.0 community-1 (~9.5GB VRAM)
- Tier 2: pyannote-audio 3.1 (~600MB VRAM)
- Tier 3: SpeechBrain ECAPA-TDNN (current default, weakest)

#### 3. Summarization: Replace llama-cpp-python with Ollama + Gemma 4

**Current**: `backend/engine/summarization.py` — llama-cpp-python with Qwen3-8B GGUF Q4_K_M. Produces 1-paragraph summaries. Has Windows CUDA DLL hacks (`_setup_cuda_dll_paths`). Complex VRAM management with manual `gc.collect()` + `torch.cuda.empty_cache()`.
**Target**: Ollama HTTP API with Gemma 4 26B MoE (A4B) as the primary model.

**What to do:**
- **Rewrite `backend/engine/summarization.py`** to use Ollama's HTTP API (`http://localhost:11434/api/generate` or `/api/chat`) instead of llama-cpp-python
- Remove ALL llama-cpp-python code, CUDA DLL hacks, GGUF file detection, torch-based VRAM management
- The engine becomes a thin HTTP client — Ollama manages GPU memory
- VRAMManager is NOT needed for summarization anymore (Ollama handles its own VRAM). Update orchestrator accordingly.
- Ollama must be running as a separate service (document in README / start script)
- Config changes in `SummarizationConfig`:
  - Remove: `gguf_model_path`, `context_size` (Ollama manages this), `chunk_size`, `chunk_overlap`
  - Add: `ollama_base_url: str = "http://localhost:11434"`, `ollama_model: str = "gemma4:26b-a4b-it-q4_K_M"` (or however Ollama names it), `ollama_timeout: int = 300`
  - Keep: `enabled`, `temperature`, `top_p`, `max_new_tokens`
- Gemma 4 26B MoE: ~16GB VRAM at Q4_K_M, 128K context window. With 24GB total, plenty of headroom.
- **Re-enable the full 7-section structured Russian legal protocol**. The old prompts (`_build_messages_ru`) were removed in MVP stabilization because Qwen3-4B/8B couldn't handle them. Gemma 4 at 26B should handle complex structured output. Build new prompts optimized for Gemma 4.
- The 7 protocol sections are: TOPIC, PARTICIPANTS, BRIEF SUMMARY, KEY DISCUSSION POINTS, DECISIONS, ACTION ITEMS (who/what/when), OPEN QUESTIONS
- All prompts in Russian (the meetings are in Russian)
- Add `/no_think` or equivalent if Gemma 4 has thinking tokens (Qwen3 had `<think>` tags that leaked into output — we stripped them)
- Multi-chunk approach is likely unnecessary with 128K context — try single-pass first, fall back to chunking only if context is exceeded
- Anti-hallucination instructions are critical — the LLM must NOT invent decisions or action items that weren't discussed
- `load()` should verify Ollama connectivity and model availability (`GET /api/tags`)
- `unload()` can be a no-op or call Ollama's model unload endpoint if available

**Fallback tier**:
- Tier 1: Gemma 4 26B MoE via Ollama (~16GB VRAM)
- Tier 2: T-Pro 2.0 32B via Ollama (~20GB VRAM, Russian-optimized, based on Qwen3-32B with custom Cyrillic tokenizer). May need GGUF conversion from HuggingFace — include instructions.
- Tier 3: Qwen3-32B via Ollama (~20GB VRAM, proven multilingual)

Model switching should be a config change only (`ollama_model` field) — no code changes.

---

### VRAM Budget (RTX 3090, 24GB, sequential loading)

Models load and unload one at a time via VRAMManager (except summarization which is now Ollama-managed):

| Stage | Model | VRAM | Notes |
|-------|-------|------|-------|
| ASR | Qwen3-ASR-1.7B | ~2GB | Load → process → unload |
| Diarization | pyannote 4.0 | ~9.5GB | Load → process → unload |
| Summarization | Gemma 4 26B (Ollama) | ~16GB | Ollama manages lifecycle |

Peak VRAM at any stage: ~16GB. Comfortable margin on 24GB.

**Optimization opportunity**: ASR (~2GB) + Diarization (~9.5GB) = 11.5GB. They could potentially run in parallel to speed up the pipeline. Consider this as an optimization after the basic sequential flow works.

---

### Orchestrator Changes

`backend/core/orchestrator.py` needs updates:
- Summarization engine no longer needs VRAMManager load/unload cycle — it's an HTTP call to Ollama
- Remove `torch.cuda.empty_cache()` / `gc.collect()` calls around summarization
- ASR engine selection: add `qwen` option alongside `auto|nemo|whisper`
- Diarization: change default from `speechbrain` to `pyannote`
- The pipeline order stays the same: Preprocess → ASR → Diarization → Align → Post-process → QA → Summarize → Format

---

### Dependencies Changes

**Add:**
- `httpx` (already in requirements.txt) — for Ollama HTTP calls
- `qwen3-asr` or whatever the Qwen3-ASR pip package is (check the repo)
- `pyannote-audio>=4.0` (upgrade from 3.3.2)

**Remove:**
- `llama-cpp-python` — replaced by Ollama
- `speechbrain` — can be kept as optional fallback but no longer required

**Keep:**
- `faster-whisper` — Tier 2 ASR fallback
- All other dependencies unchanged

---

### Formatter Changes

`backend/core/formatter.py` currently generates:
- Title page + participants table + 1-paragraph summary + transcript

With the full protocol restored, it should generate:
- Title page + participants table + topic + brief summary + key discussion points + decisions + action items + open questions + full transcript
- The formatter already had code for this (removed in MVP stabilization). Check git history or rebuild from the CLAUDE.md description.
- Always Russian headers, DD.MM.YYYY dates, EPAM branding (#B2001F red, Georgia headings, Arial Narrow body)

---

### Config Changes Summary

In `backend/app/config.py`:

```python
class ASRConfig(BaseSettings):
    engine: str = "qwen"  # auto|nemo|whisper|qwen — qwen is new default
    # ... keep existing whisper fields for fallback
    qwen_model: str = "Qwen/Qwen3-ASR-1.7B"  # or correct HF path

class DiarizationConfig(BaseSettings):
    engine: str = "pyannote"  # pyannote is now default (was speechbrain)
    pyannote_model: str = "pyannote/speaker-diarization"  # Update to 4.0 model name
    # ... keep existing speechbrain fields for fallback

class SummarizationConfig(BaseSettings):
    enabled: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:26b-a4b-it"  # Verify exact Ollama model name
    ollama_timeout: int = 300  # seconds, LLM can be slow on long transcripts
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 4096  # Gemma 4 can handle more output
    # Remove: gguf_model_path, context_size, chunk_size, chunk_overlap, quantization, model
```

---

### Setup Instructions for the New Machine

1. **Install Ollama**: `curl -fsSL https://ollama.com/install.sh | sh` (Linux) or download from ollama.com
2. **Pull Gemma 4**: `ollama pull gemma4:26b-a4b-it` (verify exact model name in Ollama library)
3. **If Gemma 4 not in Ollama library**: Create a Modelfile from GGUF weights downloaded from HuggingFace (Unsloth provides quantized versions: `unsloth/gemma-4-26B-A4B-it-GGUF`)
4. **Pull fallback models**: `ollama pull t-pro2:32b` or equivalent, `ollama pull qwen3:32b`
5. **Python environment**: Python 3.11, CUDA 12.x, install requirements.txt
6. **Frontend**: `cd frontend && npm install && npm run dev`
7. **Start**: Ollama service + `python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`

---

### Critical Rules (from CLAUDE.md, still apply)

1. **Privacy**: Law firm data. All on-premise. No data leaves the machine. No cloud APIs.
2. **Language**: All code in English, no Cyrillic in source files. Docstrings in English.
3. **Production-ready**: No stubs, no placeholders. Every line of code must work.
4. **Security**: JWT auth, AES-256-GCM encryption at rest, audit logging — all stay.
5. **Branding**: EPAM red #B2001F, Georgia headings, Arial Narrow body.
6. **Testing**: Update existing tests. Mock the Ollama HTTP calls in tests. Keep test count at or above 241.
7. **CLAUDE.md**: Update it at the end of the session with ALL changes made.

---

### What NOT to change

- Frontend (React 19 + Vite) — works fine, no changes needed in this block
- Authentication (JWT) — works fine
- Audio preprocessing (`backend/core/audio.py`) — works fine
- Aligner (`backend/core/aligner.py`) — works fine
- Post-processor (`backend/core/postprocessor.py`) — works fine
- QA (`backend/core/qa.py`) — works fine
- Encryption, audit, cleanup modules — all fine
- API routes structure — keep same endpoints

---

### Order of Work

1. **Summarization first** (Ollama + Gemma 4) — biggest impact, simplifies architecture
2. **Diarization second** (pyannote 4.0) — fixes the 7→11 speaker problem
3. **ASR third** (Qwen3-ASR) — transcription quality improvement
4. **Integration test** — run full pipeline end-to-end with a real Russian audio file
5. **Formatter update** — restore 7-section protocol now that LLM can handle it

Each step should be independently testable. Commit after each step works.
