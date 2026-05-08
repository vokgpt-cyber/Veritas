# Handoff Prompt: Fix ASR Engine Performance (Block 6 Step 4)

**Date**: 2026-04-16
**Project**: EPAM VERITAS v2.0 — On-premise meeting transcription for Russian law firm
**Hardware**: RTX 3090 (24GB VRAM), Windows

## READ FIRST
Read `CLAUDE.md` for full project context.

## The Problem

The antony66/whisper-large-v3-russian model (fine-tuned for Russian, WER 6.39%) needs to work for **transcription** (not translation), but:

1. **faster-whisper + BatchedInferencePipeline** (CTranslate2 backend) — transcribes a 30-min file in **~30 seconds** on RTX 3090 with batch_size=16. Excellent speed. BUT: when the antony66 model is loaded via CTranslate2, it **translates to English** instead of transcribing in Russian. Even explicit `task="transcribe"` doesn't fix it. The CTranslate2 conversion breaks the model's fine-tuned task configuration. This is Bug 56 in CLAUDE.md.

2. **HFWhisperASREngine** (HuggingFace transformers pipeline) — loads the model in original format, preserves task="transcribe", produces Russian output. BUT: takes **20+ minutes** for the same file, uses 23.7/24GB VRAM, and stalls the computer. Unacceptable. The HF pipeline's chunk-based batching is fundamentally slower than CTranslate2 + BatchedInferencePipeline.

The user wants antony66's quality (Russian fine-tuned) at faster-whisper's speed (~30 seconds). The current situation is: fast engine produces wrong language, correct-language engine is 40x too slow.

## Files Involved

- `backend/engine/hf_whisper_asr.py` — NEW file from this session. HuggingFace transformers pipeline. Works correctly (Russian output) but far too slow. **This is the engine currently configured as default (`engine: "hf-whisper"`).**
- `backend/engine/whisper_asr.py` — faster-whisper with BatchedInferencePipeline. Fast but translates to English with antony66 model.
- `backend/core/orchestrator.py` — selects engine based on `config.asr.engine` setting. Lines ~85-105.
- `config/settings.yaml` — `asr.engine: "hf-whisper"`, `asr.whisper_model: "large-v3"`, `asr.beam_size: 1`
- `models/whisper-large-v3-russian/` — local copy of antony66 model in original HF format (model.safetensors, 3.09GB)
- `models/whisper-large-v3-russian-ct2/` — CTranslate2-converted version (the one that translates instead of transcribing). May have config issues — check config.json and preprocessor_config.json.

## Research Directions

The new session should research and implement ONE of these solutions:

### Option A: Fix CTranslate2 conversion to preserve task="transcribe"
The ct2 conversion lost the model's default task config. Research whether:
- CTranslate2/faster-whisper has a way to force transcription mode at the model level (not just at inference time)
- The converted model's config files need patching (config.json, tokenizer files)
- The `forced_decoder_ids` in the tokenizer/generation config can be set to force transcription
- Look at `generation_config.json` in the original model — it likely has `forced_decoder_ids` that set `task="transcribe"`. These may need to be preserved in the ct2 conversion.

If this works, we get antony66 quality + faster-whisper speed. **This is the ideal solution.**

### Option B: Use HF transformers model.generate() directly instead of pipeline()
The HF warning in the logs said: *"Using 'chunk_length_s' is very experimental with seq2seq models. Use Whisper for long-form transcription using the model's .generate method directly."*

This means:
- Load model via `WhisperForConditionalGeneration.from_pretrained()` + `WhisperProcessor`
- Use `model.generate()` with Whisper's native long-form decoding (sequential, no chunking)
- This is the officially recommended HF approach for long audio
- Should be faster than the pipeline's chunk batching, though still slower than CTranslate2

### Option C: Hybrid — load antony66 via HF for short audio, faster-whisper for long
Not ideal but practical. Use HF engine only for audio under 5 minutes, faster-whisper with standard large-v3 (non-antony66) for longer files.

### Option D: Re-examine the CTranslate2 model directory
The `models/whisper-large-v3-russian-ct2/` directory was created during this session with manual conversion. Check:
- Does `generation_config.json` exist? If not, copy from original and ensure `forced_decoder_ids` includes the transcribe task token.
- Does the tokenizer properly map the `<|transcribe|>` token?
- Can we add a `generation_config.json` that forces `task="transcribe"` at the CTranslate2 level?

## Other Changes Made This Session (all working, keep them)

1. **Punctuation restoration** — `deepmultilingualpunctuation` integrated into `backend/core/postprocessor.py`. Works on CPU, lazy-loaded. Keep this.

2. **Gemma 4 anti-hallucination** (3 measures in `backend/engine/summarization.py`):
   - Temperature 0.2, top_p 0.85 (in settings.yaml)
   - Speaker citation requirements in protocol prompt
   - `_verify_protocol()` — second LLM call to fact-check protocol against transcript
   - **These haven't been tested yet** — the pipeline keeps failing at ASR stage.

3. **BatchedInferencePipeline** in `whisper_asr.py` — processes 16 chunks in parallel. Working great with standard large-v3.

4. **Bug 57 fix** in `whisper_asr.py` — `getattr(segment, "avg_log_prob", 0.0) or 0.0`

## What NOT to Do

- Do NOT suggest whisper-large-v3-turbo — user explicitly rejected it, wants quality over speed
- Do NOT suggest Qwen3-ASR — user already tested it, poor quality on Russian legal speech
- Do NOT increase diarization step_duration — meetings have overlapping speakers, needs fine granularity (0.5s)
- Do NOT reduce max_speakers below current setting — meetings can have 10-15 speakers

## Success Criteria

- antony66/whisper-large-v3-russian model producing Russian transcription (not English translation)
- Transcription of 30-min audio completes in under 2 minutes on RTX 3090
- VRAM usage stays under 20GB during ASR stage
- After ASR fix, run full pipeline end-to-end to test all stages including Gemma 4 anti-hallucination
