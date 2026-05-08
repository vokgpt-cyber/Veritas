"""Multi-engine ASR benchmark runner for VERITAS.

Runs each requested engine on the SAME preprocessed audio and writes per-engine
outputs into `benchmarks/asr/<engine>/` so they can be scored against a gold
transcript with `benchmarks/scripts/score_wer.py`.

Usage (Windows, repo-root venv activated):

    venv\\Scripts\\python scripts\\run_asr_benchmark.py ^
        data\\test\\Court hearing 129_01.mp4 ^
        --base court_hearing_129 ^
        --engines gigaam,whisper_base,antony66,whisper_turbo_ru,qwen

Supported engines (keys are what you pass to --engines):

    gigaam              Sber GigaAM v3_e2e_rnnt          (engine key: gigaam)
    whisper_base        OpenAI Whisper large-v3          (engine key: whisper, whisper_model=large-v3)
    antony66            antony66/whisper-large-v3-russian (engine key: hf-whisper)
    whisper_turbo_ru    dvislobokov/whisper-large-v3-turbo-russian (engine key: whisper)
    qwen                Qwen/Qwen3-ASR-1.7B + ForcedAligner (engine key: qwen)
    canary              NVIDIA Canary-1b-v2              (engine key: nemo, model=nvidia/canary-1b-v2)

Per-engine output files (in benchmarks/asr/<engine>/):

    <base>_transcript.md   prose transcript with timestamps
    <base>_segments.json   list[TranscriptionSegment.dict()]
    <base>_plain.txt       text only, one utterance per line (WER input)
    <base>_run.json        {load_s, transcribe_s, unload_s, vram_peak_gb, ...}

Consolidated report:

    benchmarks/reports/asr_comparison_<base>_<timestamp>.md
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from backend.app.config import AppConfig
from backend.app.models import TranscriptionSegment
from backend.core.audio import AudioPreprocessor

BENCH_ROOT = REPO_ROOT / "benchmarks"
ASR_ROOT = BENCH_ROOT / "asr"
REPORTS_ROOT = BENCH_ROOT / "reports"
METRICS_ROOT = BENCH_ROOT / "metrics"


# ------------------------------------------------------------------
# Engine registry — maps a CLI key to (display_name, folder, engine_key,
# config-override callable).
# ------------------------------------------------------------------


def _override_whisper_base(cfg: AppConfig) -> None:
    cfg.asr.whisper_model = "large-v3"


def _override_whisper_turbo_ru(cfg: AppConfig) -> None:
    # Bug 55 workaround: HF cache symlinks are broken on Windows even with
    # Developer Mode enabled, which prevents CTranslate2 from opening model.bin.
    # Solution: materialise the repo into <project_root>/models/<basename>/ via
    # snapshot_download (same pattern used for antony66), then point faster-whisper
    # at that absolute local path. faster-whisper picks it up through the
    # `Path(model_size).is_dir()` branch in FasterWhisperASREngine.load().
    repo_id = "dvislobokov/whisper-large-v3-turbo-russian"
    local_dir = REPO_ROOT / "models" / "whisper-large-v3-turbo-russian"
    if not (local_dir / "model.bin").is_file():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required to materialise dvislobokov model. "
                "Install with: pip install huggingface_hub"
            ) from e
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[whisper_turbo_ru] Downloading {repo_id} -> {local_dir} ...")
        # huggingface_hub >= 0.24 copies files by default when local_dir is set;
        # older versions need local_dir_use_symlinks=False to avoid broken symlinks
        # on Windows. Try the explicit flag first, fall back without it.
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
        except TypeError:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
            )
        print(f"[whisper_turbo_ru] Download complete: {local_dir}")
    cfg.asr.whisper_model = str(local_dir)


def _override_canary(cfg: AppConfig) -> None:
    cfg.asr.model = "nvidia/canary-1b-v2"


ENGINE_REGISTRY: dict[str, dict] = {
    "gigaam": {
        "display": "GigaAM_v3",
        "folder": "gigaam_v3",
        "engine_key": "gigaam",
        "override": None,
    },
    "whisper_base": {
        "display": "Whisper_large_v3_base",
        "folder": "whisper_large_v3_base",
        "engine_key": "whisper",
        "override": _override_whisper_base,
    },
    "antony66": {
        "display": "antony66_whisper_v3_russian",
        "folder": "antony66_whisper_large_v3_russian",
        "engine_key": "hf-whisper",
        "override": None,
    },
    "whisper_turbo_ru": {
        "display": "Whisper_turbo_russian",
        "folder": "whisper_large_v3_turbo_russian",
        # dvislobokov repo ships only model.safetensors + ggml-model.bin (no CT2 model.bin),
        # so we must use the HF transformers path (same as antony66) not faster-whisper.
        # Our previous `engine_key: "whisper"` run silently produced Whisper-base output.
        "engine_key": "hf-whisper",
        "override": _override_whisper_turbo_ru,
    },
    "qwen": {
        "display": "Qwen3_ASR_1_7B",
        "folder": "qwen3_asr_1_7b",
        "engine_key": "qwen",
        "override": None,
    },
    "canary": {
        "display": "Canary_1b_v2",
        "folder": "canary_1b_v2",
        "engine_key": "nemo",
        "override": _override_canary,
    },
}


@dataclass
class EngineRun:
    key: str                       # CLI key (e.g. "whisper_base")
    display: str                   # human-readable name
    folder: str                    # subfolder under benchmarks/asr/
    engine_key: str                # backend engine key
    load_s: float = 0.0
    transcribe_s: float = 0.0
    unload_s: float = 0.0
    vram_peak_gb: float = 0.0
    segments: list[TranscriptionSegment] = field(default_factory=list)
    error: Optional[str] = None


# ------------------------------------------------------------------
# VRAM / GC helpers
# ------------------------------------------------------------------


def _vram_used_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _full_gc() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def format_timestamp(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h}:{m:02d}:{s:02d}"


# ------------------------------------------------------------------
# Engine runner
# ------------------------------------------------------------------


def _instantiate_engine(engine_key: str, config: AppConfig):
    if engine_key == "gigaam":
        from backend.engine.gigaam_asr import GigaAMASREngine
        return GigaAMASREngine(config)
    if engine_key == "hf-whisper":
        from backend.engine.hf_whisper_asr import HFWhisperASREngine
        return HFWhisperASREngine(config)
    if engine_key == "whisper":
        from backend.engine.whisper_asr import FasterWhisperASREngine
        return FasterWhisperASREngine(config)
    if engine_key == "qwen":
        from backend.engine.qwen_asr import Qwen3ASREngine
        return Qwen3ASREngine(config)
    if engine_key == "nemo":
        from backend.engine.asr import NeMoASREngine
        return NeMoASREngine(config)
    raise ValueError(f"Unsupported engine key: {engine_key}")


async def run_one(
    key: str,
    audio_path: str,
    config: AppConfig,
) -> EngineRun:
    entry = ENGINE_REGISTRY[key]
    run = EngineRun(
        key=key,
        display=entry["display"],
        folder=entry["folder"],
        engine_key=entry["engine_key"],
    )

    print(f"\n{'='*70}")
    print(f"[bench] Engine: {run.display}  ({run.engine_key})")
    print(f"{'='*70}")

    # Snapshot the bits of config we may mutate so we can restore after.
    orig_engine = config.asr.engine
    orig_whisper_model = config.asr.whisper_model
    orig_model = config.asr.model
    config.asr.engine = run.engine_key
    if entry["override"]:
        entry["override"](config)

    try:
        engine = _instantiate_engine(run.engine_key, config)

        # --- load ---
        t0 = time.time()
        engine.load()
        run.load_s = time.time() - t0
        after_load = _vram_used_gb()
        run.vram_peak_gb = max(run.vram_peak_gb, after_load)
        print(f"[{run.display}] loaded in {run.load_s:.1f}s (VRAM {after_load:.2f} GB)")

        # --- transcribe ---
        t0 = time.time()
        last_pct = [0]

        def on_progress(pct: float, _msg: str = "") -> None:
            shown = int(pct * 100)
            if shown >= last_pct[0] + 10:
                last_pct[0] = shown
                vram = _vram_used_gb()
                run.vram_peak_gb = max(run.vram_peak_gb, vram)
                print(f"[{run.display}] {shown}% (VRAM {vram:.2f} GB)")

        run.segments = await engine.process(audio_path, progress_callback=on_progress)
        run.transcribe_s = time.time() - t0
        run.vram_peak_gb = max(run.vram_peak_gb, _vram_used_gb())
        print(
            f"[{run.display}] {len(run.segments)} segments in {run.transcribe_s:.1f}s "
            f"(peak VRAM {run.vram_peak_gb:.2f} GB)"
        )

        # --- unload ---
        t0 = time.time()
        engine.unload()
        del engine
        _full_gc()
        run.unload_s = time.time() - t0
        print(f"[{run.display}] unloaded in {run.unload_s:.1f}s "
              f"(VRAM now {_vram_used_gb():.2f} GB)")

    except Exception as e:
        import traceback
        run.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[{run.display}] FAILED: {e}")
        _full_gc()
    finally:
        config.asr.engine = orig_engine
        config.asr.whisper_model = orig_whisper_model
        config.asr.model = orig_model

    return run


# ------------------------------------------------------------------
# Output writers
# ------------------------------------------------------------------


def _out_dir(run: EngineRun) -> Path:
    d = ASR_ROOT / run.folder
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_segments_json(run: EngineRun, base: str) -> Path:
    path = _out_dir(run) / f"{base}_segments.json"
    data = [s.model_dump() for s in run.segments]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def write_plain_txt(run: EngineRun, base: str) -> Path:
    """One utterance per line, no timestamps, no speaker labels. WER input."""
    path = _out_dir(run) / f"{base}_plain.txt"
    lines = []
    for seg in run.segments:
        text = (getattr(seg, "text", "") or "").strip()
        if text:
            lines.append(text)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_transcript_md(run: EngineRun, base: str, audio_name: str) -> Path:
    path = _out_dir(run) / f"{base}_transcript.md"
    duration = max((float(s.end) for s in run.segments), default=0.0)
    rtf = (duration / run.transcribe_s) if run.transcribe_s > 0 else 0.0

    lines: list[str] = []
    lines.append(f"# {run.display} — {base.replace('_', ' ')}")
    lines.append("")
    lines.append(f"**Audio**: `{audio_name}`  ")
    lines.append(
        f"**Duration**: {format_timestamp(duration)} ({duration/60:.1f} min)  "
    )
    lines.append(f"**Engine**: {run.display}  (`{run.engine_key}`)  ")
    lines.append(f"**Segments**: {len(run.segments)}  ")
    lines.append(
        f"**Transcribe**: {run.transcribe_s:.1f}s ({rtf:.1f}x realtime)  "
    )
    lines.append(f"**VRAM peak**: {run.vram_peak_gb:.2f} GB  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    for seg in run.segments:
        start = float(getattr(seg, "start", 0.0) or 0.0)
        end = float(getattr(seg, "end", 0.0) or 0.0)
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        ts = f"[{format_timestamp(start)} \u2013 {format_timestamp(end)}]"
        lines.append(f"`{ts}` {text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_run_json(run: EngineRun, base: str, audio_info: dict) -> Path:
    path = _out_dir(run) / f"{base}_run.json"
    data = {
        "engine_key": run.engine_key,
        "display": run.display,
        "folder": run.folder,
        "load_s": round(run.load_s, 2),
        "transcribe_s": round(run.transcribe_s, 2),
        "unload_s": round(run.unload_s, 2),
        "vram_peak_gb": round(run.vram_peak_gb, 3),
        "num_segments": len(run.segments),
        "realtime_factor": (
            round(audio_info["duration_s"] / run.transcribe_s, 2)
            if run.transcribe_s > 0 else 0.0
        ),
        "audio": audio_info,
        "error": run.error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


# ------------------------------------------------------------------
# Consolidated report
# ------------------------------------------------------------------


def write_comparison_report(
    runs: list[EngineRun],
    base: str,
    audio_info: dict,
) -> Path:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_ROOT / f"asr_comparison_{base}_{ts}.md"
    duration = audio_info["duration_s"]

    lines: list[str] = []
    lines.append(f"# ASR comparison — {base}")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}  ")
    lines.append(
        f"Audio: `{audio_info['path']}` "
        f"({audio_info['size_mb']:.1f} MB, {duration/60:.1f} min)  "
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Engine | Load (s) | Transcribe (s) | Realtime | Unload (s) | "
        "VRAM peak (GB) | Segments | Status |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---|"
    )
    for r in runs:
        rtf = (duration / r.transcribe_s) if r.transcribe_s > 0 else 0.0
        status = "FAILED" if r.error else "OK"
        lines.append(
            f"| {r.display} | {r.load_s:.1f} | {r.transcribe_s:.1f} | "
            f"{rtf:.1f}x | {r.unload_s:.1f} | {r.vram_peak_gb:.2f} | "
            f"{len(r.segments)} | {status} |"
        )
    lines.append("")

    lines.append("## Score WER")
    lines.append("")
    lines.append(
        "After all runs complete, score each engine against the gold "
        "transcript:"
    )
    lines.append("")
    lines.append("```")
    for r in runs:
        if r.error or not r.segments:
            continue
        hyp = f"benchmarks/asr/{r.folder}/{base}_plain.txt"
        metrics = f"benchmarks/metrics/asr_{r.folder}_{base}.json"
        lines.append(
            f"python benchmarks/scripts/score_wer.py \\\n"
            f"  --ref benchmarks/gold/{base}_plain.txt \\\n"
            f"  --hyp {hyp} \\\n"
            f"  --out {metrics} --tag {r.folder}"
        )
    lines.append("```")
    lines.append("")

    # First 10 segments side-by-side
    lines.append("## First 10 segments (side-by-side, raw ASR, no speakers)")
    lines.append("")
    header = ["#", "Time"] + [r.display for r in runs]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for i in range(10):
        row_cells = [str(i + 1)]
        time_cell = ""
        engine_cells: list[str] = []
        for r in runs:
            if i < len(r.segments):
                seg = r.segments[i]
                start = float(getattr(seg, "start", 0.0) or 0.0)
                end = float(getattr(seg, "end", 0.0) or 0.0)
                text = (getattr(seg, "text", "") or "")
                text = text.replace("|", "\\|").replace("\n", " ").strip()
                if not time_cell:
                    time_cell = f"{start:.1f}-{end:.1f}"
                if len(text) > 160:
                    text = text[:160] + "\u2026"
                engine_cells.append(text)
            else:
                engine_cells.append("—")
        row_cells.append(time_cell or "—")
        row_cells.extend(engine_cells)
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    for r in runs:
        if r.error:
            lines.append(f"## {r.display} — error")
            lines.append("")
            lines.append("```")
            lines.append(r.error)
            lines.append("```")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------


def load_config() -> AppConfig:
    cfg_path = REPO_ROOT / "config" / "settings.yaml"
    if cfg_path.exists():
        return AppConfig.load_from_yaml(cfg_path)
    return AppConfig()


async def run(
    audio_path: Path,
    base: str,
    engines: list[str],
) -> dict:
    config = load_config()
    print(f"[bench] Audio:   {audio_path}")
    print(f"[bench] Base:    {base}")
    print(f"[bench] Engines: {engines}")
    print(f"[bench] Out:     {ASR_ROOT}")

    # Preprocess once
    print("\n[bench] Preprocessing audio (resample, mono)...")
    t0 = time.time()
    preprocessed_path, metadata = await AudioPreprocessor.process(
        str(audio_path), str(REPO_ROOT / "data" / "meetings" / "_bench_tmp")
    )
    preprocess_s = time.time() - t0
    print(
        f"[bench] Preprocessed in {preprocess_s:.1f}s "
        f"({metadata.sample_rate}Hz, {metadata.channels}ch, "
        f"{metadata.duration:.1f}s)"
    )

    audio_info = {
        "path": str(audio_path),
        "name": audio_path.name,
        "size_mb": round(audio_path.stat().st_size / (1024 ** 2), 1),
        "duration_s": round(metadata.duration, 2),
        "sample_rate": metadata.sample_rate,
        "channels": metadata.channels,
    }

    runs: list[EngineRun] = []
    for key in engines:
        r = await run_one(key, preprocessed_path, config)
        runs.append(r)
        if r.segments:
            write_segments_json(r, base)
            write_plain_txt(r, base)
            write_transcript_md(r, base, audio_path.name)
        write_run_json(r, base, audio_info)

    report = write_comparison_report(runs, base, audio_info)
    return {"report": report, "runs": runs, "preprocess_s": preprocess_s}


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-engine ASR benchmark runner")
    ap.add_argument("audio", type=Path, help="Path to audio file")
    ap.add_argument(
        "--base", required=True,
        help="Short ASCII tag for this audio (e.g. court_hearing_129)",
    )
    ap.add_argument(
        "--engines",
        default="gigaam,whisper_base,antony66,whisper_turbo_ru,qwen",
        help="Comma-separated engine keys. "
             f"Valid: {', '.join(ENGINE_REGISTRY)}",
    )
    ap.add_argument(
        "--skip", default="",
        help="Comma-separated engine keys to skip",
    )
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"ERROR: audio not found: {args.audio}", file=sys.stderr)
        return 2

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    skip = {e.strip() for e in args.skip.split(",") if e.strip()}
    engines = [e for e in engines if e not in skip]
    unknown = [e for e in engines if e not in ENGINE_REGISTRY]
    if unknown:
        print(f"ERROR: unknown engine keys: {unknown}", file=sys.stderr)
        print(f"Valid: {sorted(ENGINE_REGISTRY)}", file=sys.stderr)
        return 2
    if not engines:
        print("ERROR: no engines after --skip", file=sys.stderr)
        return 2

    result = asyncio.run(run(args.audio.resolve(), args.base, engines))

    print()
    print("=" * 72)
    print(f"Report: {result['report']}")
    print("=" * 72)
    for r in result["runs"]:
        print(
            f"  {r.display:35s}  load={r.load_s:5.1f}s  "
            f"transcribe={r.transcribe_s:7.1f}s  "
            f"VRAM={r.vram_peak_gb:5.2f}GB  "
            f"{'OK' if not r.error else 'FAILED'}"
        )
    print()
    print("Next step — score WER:")
    for r in result["runs"]:
        if r.error or not r.segments:
            continue
        print(
            f"  python benchmarks/scripts/score_wer.py "
            f"--ref benchmarks/gold/{args.base}_plain.txt "
            f"--hyp benchmarks/asr/{r.folder}/{args.base}_plain.txt "
            f"--out benchmarks/metrics/asr_{r.folder}_{args.base}.json "
            f"--tag {r.folder}"
        )

    any_err = any(r.error for r in result["runs"])
    return 1 if any_err else 0


if __name__ == "__main__":
    sys.exit(main())
