# VERITAS Benchmarks

Stage-wise evaluation of ASR, diarization, and punctuation for Russian
courtroom + meeting audio. Every artefact produced by a benchmark lives
under this folder so the whole tree can be deleted once a choice is locked.

## Folder layout

```
benchmarks/
  audio/                           symlinks or copies of the input audio
                                   (kept out of git; sources live in data/test/)
  gold/                            ground-truth transcripts, parsed + normalised
    court_hearing_129.jsonl        [{turn, speaker, text}] — from the "TRUTH" docx
    court_hearing_129_plain.txt    text only, one utterance per line (WER input)
    court_hearing_129_speakers.txt "[Speaker] text" — easy visual comparison
    court_hearing_129_metadata.json turn/speaker counts, char totals

  asr/<engine>/                    one folder per candidate engine
    <base>_transcript.md           prose transcript for human reading
    <base>_segments.json           list[{start, end, text, avg_logprob?, words?}]
    <base>_plain.txt               detokenised text, no speaker labels
    <base>_run.json                {load_s, transcribe_s, vram_peak_gb, engine_version}

  diarization/<engine>/            one folder per candidate diarizer
    <base>_rttm                    canonical RTTM for DER scoring
    <base>_segments.json           list[{start, end, speaker}]
    <base>_run.json                timing + VRAM

  punctuation/<engine>/            punctuation-restoration candidates run on the
                                   raw (unpunctuated) ASR output of the chosen ASR
    <base>_punctuated.txt          text after restoration
    <base>_run.json                timing + basic diff stats

  preprocess/<engine>/             optional pre-ASR denoise/VAD experiments
    <base>_<variant>.wav           cleaned audio
    <base>_<variant>_notes.md

  reports/                         human-facing side-by-side reports
    asr_comparison.md
    asr_comparison.html            (optional) rendered HTML diff vs gold
    diarization_comparison.md
    punctuation_comparison.md

  metrics/                         raw numbers only — one JSON per run
    asr_<engine>_<base>.json       { wer, cer, substitutions, deletions, insertions }
    diarization_<engine>_<base>.json  { der, fa, miss, confusion, num_speakers }

  scripts/                         benchmark-local Python helpers
    parse_gold_docx.py             docx -> jsonl/plain/speakers/metadata
    score_wer.py                   compare an ASR plain.txt to a gold plain.txt
    score_der.py                   compare an RTTM to gold RTTM (once we have one)
    render_asr_diff.py             side-by-side HTML diff per line
```

## Naming convention

`<base>` is the short audio tag, e.g. `court_hearing_129`, `admin_13_04_2026`,
`meeting_2speakers_10min`. Keep this consistent so an ASR run and the gold
transcript can be joined on the `<base>` token alone.

## Cleanup

The entire `benchmarks/` tree is transient. After we lock the production stack
(ASR + diarizer + punctuation), delete the whole folder. Nothing here is
imported by `backend/` at runtime.

`benchmarks/` is added to `.gitignore` so large transcripts and audio don't
end up in git.

## What we are benchmarking

### ASR (6 candidates)

| Engine | Variant | Rationale |
|---|---|---|
| GigaAM v3 | `gigaam/v3_e2e_rnnt` | Current default. SOTA on Russian benchmarks. |
| Whisper large-v3 | base, via `faster-whisper` | User's previous-favourite on RTX 4060 — needs a fair RTX 3090 rerun. |
| antony66/whisper-large-v3-russian | HF transformers | User wants empirical data, not a priori drop. |
| dvislobokov/whisper-large-v3-turbo-russian | faster-whisper or HF | New 2026 Russian turbo fine-tune surfaced in research. |
| Qwen3-ASR-1.7B | `qwen-asr` pip package + ForcedAligner | Jan 2026 release. Previously flopped "naked"; retest with VAD + proper pipeline. |
| (Optional) NVIDIA Canary-1b-v2 | NeMo | 2025 SOTA multilingual (25 langs, Russian-capable) — only if setup is cheap. |

Dropped after research: `bond005/whisper-large-v3-ru` and
`waveletdeboshir/whisper-large-v3-russian-ties-podlodka` — both return 404 on
HuggingFace (hallucinated model IDs).

### Diarization (3-4 candidates)

| Engine | Variant | Rationale |
|---|---|---|
| pyannote 3.1 | `pyannote/speaker-diarization-3.1` | User's previous-favourite — baseline we're trying to beat. |
| pyannote 4.0 community-1 | `pyannote/speaker-diarization-community-1` | Current default. User wants it re-tested on court audio. |
| NVIDIA Sortformer | `nvidia/diar_sortformer_4spk-v1` | 2024-2025 end-to-end neural diarization, strong on overlap. |

### Punctuation (2 candidates)

| Engine | Notes |
|---|---|
| deepmultilingualpunctuation (current) | BERT-based, CPU, 300 MB, 2022. Baseline. |
| Silero te_models (punc+caps) | Russian-native. No 2026 update but clean ONNX weights. |

### Optional preprocessing

| Tool | When to use |
|---|---|
| Silero VAD (already deployed in GigaAM) | Always; no change needed. |
| DeepFilterNet v3 | Only on distant-mic court audio (measured SNR < 15 dB). |
| NeMo inverse text normalisation | Post-punctuation; formats numbers/dates for the DOCX. |

## Evaluation protocol

1. **Preprocess audio once** — single 16 kHz mono WAV per base, shared by all ASR runs.
2. **Run each ASR** sequentially (one engine at a time, VRAM released between).
3. **Score WER** against `gold/<base>_plain.txt` with normalisation: lowercase,
   strip punctuation, normalise ё → е, collapse whitespace.
4. **Render side-by-side diff** per turn in `reports/asr_comparison.md`.
5. Pick ASR winner before touching diarization.
6. Run each diarizer against the same audio, score DER against a gold RTTM
   (to be created — probably by converting `gold/<base>.jsonl` + manual
   timestamp alignment to the winning ASR).
7. Run each punctuation restorer on the winning ASR's unpunctuated output.
8. Lock the stack and update `CLAUDE.md`.
