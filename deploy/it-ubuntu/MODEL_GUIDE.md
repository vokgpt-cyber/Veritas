# VERITAS 1.0 LLM Guide for IT

VERITAS 1.0 has one baseline LLM path and several future comparison candidates.
For the pilot handoff, do not replace the baseline unless there is a separate
A/B test on the same gold transcripts.

## Baseline for pilot deployment

### Gemma 4 26B through Ollama

This is the current validated VERITAS summarization setup. It produced the best
practical protocol quality during the Windows RTX 3090 tuning sprint, so the
Ubuntu server handoff should start here as well.

Server `.env.server` baseline:

```bash
COMPOSE_PROFILES=
EPAM_SUMMARIZATION_PROVIDER=ollama
EPAM_SUMMARIZATION_OLLAMA_BASE_URL=http://ollama:11434
EPAM_SUMMARIZATION_OLLAMA_MODEL=gemma4:26b
OLLAMA_IMAGE=ollama/ollama:0.13.4
OLLAMA_BIND=127.0.0.1
OLLAMA_PORT=11435
```

The launcher runs:

```bash
docker exec veritas-ollama ollama pull gemma4:26b
```

on first start when the `ollama` profile is enabled.

## Optional future experiments

### Qwen/Qwen3.6-27B through vLLM

This is the first alternative candidate for later IT experiments, not the pilot
baseline.

```bash
COMPOSE_PROFILES=vllm
EPAM_SUMMARIZATION_PROVIDER=openai_compatible
EPAM_SUMMARIZATION_OPENAI_BASE_URL=http://vllm:8000/v1
EPAM_SUMMARIZATION_OLLAMA_MODEL=Qwen/Qwen3.6-27B
VLLM_IMAGE=vllm/vllm-openai:v0.18.2
VLLM_MODEL=Qwen/Qwen3.6-27B
VLLM_SERVED_MODEL_NAME=Qwen/Qwen3.6-27B
VLLM_GPU_MEMORY_UTILIZATION=0.62
VLLM_MAX_MODEL_LEN=65536
```

Use it only after Gemma 4 is running and there is a stable comparison dataset.

### Larger SOTA candidates

Keep these in a watchlist until IT validates memory, chat template, JSON
stability, speed, and security:

- Qwen/Qwen3.6-35B-A3B
- Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
- moonshotai/Kimi-K2.6
- DeepSeek V4 / V4-Flash
- GLM-5.1
- MiniMax M2.7

Some of these may need more than one 48 GB GPU, reduced context, quantization,
or `trust_remote_code`. Do not enable `trust_remote_code` unless the exact model
repository has been reviewed as executable code and pinned.

## VRAM rule

Docker on RTX/GeForce usually does not hard-partition VRAM per container.

For Gemma 4 through Ollama, the operational rule is simple: keep the GPU free
for VERITAS during processing and expect an explicit error if the model cannot
fit.

For later vLLM experiments, use:

```bash
VLLM_GPU_MEMORY_UTILIZATION=<budget_gb / total_vram_gb>
```

For RTX 4090 48 GB:

- 30 GB budget: `0.62`
- 36 GB budget: `0.75`
- 40 GB budget: `0.83`

If a model does not fit, do not silently fall back to a weaker model just to
complete the run. Quality is the priority.

## How to compare models

1. Keep ASR and diarization fixed.
2. Run the same gold transcripts through Gemma 4 and the candidate model.
3. Compare practical output, not just style:
   - number of real decisions found;
   - number of action items found;
   - responsible party;
   - deadline;
   - quote/timecode support;
   - absence of invented tasks.
4. Promote a candidate only if it beats Gemma 4 on the actual EPAM meeting and
   court-hearing material.
