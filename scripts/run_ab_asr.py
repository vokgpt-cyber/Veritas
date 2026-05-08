"""A/B ASR comparison harness.

Runs two ASR engines (GigaAM v3 and HF Whisper antony66/whisper-large-v3-russian)
on the SAME preprocessed audio and emits two parallel transcript .md files into
`deliverables/` so they can be reviewed side-by-side.

Scope: ASR ONLY. No diarization, no summarization. Speakers are unknown in the
output — the point of this harness is to let the user judge raw transcription
quality independent of downstream stages.

Usage (Windows, with venv activated at repo root):

    venv\\Scripts\\python scripts\\run_ab_asr.py path\\to\\audio.mp3
    venv\\Scripts\\python scripts\\run_ab_asr.py audio.mp3 --engines gigaam,hf-whisper
    venv\\Scripts\\python scripts\\run_ab_asr.py audio.mp3 --base Admin_13-04-2026
    venv\\Scripts\\python scripts\\run_ab_asr.py audio.mp3 --skip gigaam   # only HF

Output:
    deliverables/{base}_GigaAM_TRANSCRIPT.md
    deliverables/{base}_HFWhisper_TRANSCRIPT.md
    scripts/ab_asr_report_YYYYMMDD_HHMMSS.md  -- timing, VRAM, side-by-side summary
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import re
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

DELIVERABLES = REPO_ROOT / "deliverables"


# ---------- basename helper (duplicated from build_deliverables to avoid circular import) ----------


def sanitize_basename(s: str) -> str:
    """Convert an arbitrary filename (incl. Cyrillic) into ASCII-safe base."""
    base = Path(s).stem
    cyr = "\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f"
    lat = ["a", "b", "v", "g", "d", "e", "zh", "z", "i", "y", "k", "l", "m", "n",
           "o", "p", "r", "s", "t", "u", "f", "h", "c", "ch", "sh", "sch", "",
           "y", "", "e", "yu", "ya"]
    cmap = dict(zip(cyr, lat))
    cmap.update({c.upper(): v.capitalize() for c, v in cmap.items()})
    out = "".join(cmap.get(ch, ch) for ch in base)
    out = re.sub(r"[^\w\-.]", "_", out, flags=re.ASCII)
    out = re.sub(r"_+", "_", out).strip("_.")
    return out or "meeting"


def format_timestamp(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h}:{m:02d}:{s:02d}"


# ---------- engine runners ----------


@dataclass
class EngineRun:
    name: str  # display name (e.g. "GigaAM")
    engine_key: str  # config key (e.g. "gigaam")
    load_s: float = 0.0
    transcribe_s: float = 0.0
    unload_s: float = 0.0
    vram_peak_gb: float = 0.0
    segments: list[TranscriptionSegment] = field(default_factory=list)
    error: Optional[str] = None


def _vram_used_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            # allocated is what this process holds; reserved includes cache
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


async def run_one_engine(
    engine_name: str,
    engine_key: str,
    audio_path: str,
    config: AppConfig,
) -> EngineRun:
    """Load engine, transcribe, unload. Returns timings + segments.

    engine_key controls which class to instantiate. The underlying config
    object is mutated in-place so the engine reads the right model name
    from config.asr — but we restore the original on exit.
    """
    run = EngineRun(name=engine_name, engine_key=engine_key)

    print(f"\n{'='*60}")
    print(f"[ab-asr] Engine: {engine_name} ({engine_key})")
    print(f"{'='*60}")

    # Set engine key on the config so engine __init__/load reads correct fields
    original_engine = config.asr.engine
    config.asr.engine = engine_key

    try:
        if engine_key == "gigaam":
            from backend.engine.gigaam_asr import GigaAMASREngine
            engine = GigaAMASREngine(config)
        elif engine_key == "hf-whisper":
            from backend.engine.hf_whisper_asr import HFWhisperASREngine
            engine = HFWhisperASREngine(config)
        elif engine_key == "whisper":
            from backend.engine.whisper_asr import FasterWhisperASREngine
            engine = FasterWhisperASREngine(config)
        else:
            raise ValueError(f"Unsupported engine key: {engine_key}")

        # --- load ---
        t0 = time.time()
        engine.load()
        run.load_s = time.time() - t0
        after_load = _vram_used_gb()
        run.vram_peak_gb = max(run.vram_peak_gb, after_load)
        print(f"[{engine_name}] loaded in {run.load_s:.1f}s (VRAM {after_load:.2f} GB)")

        # --- transcribe ---
        t0 = time.time()

        last_pct = [0]

        def on_progress(pct: float) -> None:
            # Report every 10% crossing so log output is readable
            shown = int(pct * 100)
            if shown >= last_pct[0] + 10:
                last_pct[0] = shown
                vram = _vram_used_gb()
                run.vram_peak_gb = max(run.vram_peak_gb, vram)
                print(f"[{engine_name}] {shown}% (VRAM {vram:.2f} GB)")

        run.segments = await engine.process(audio_path, progress_callback=on_progress)
        run.transcribe_s = time.time() - t0
        vram_after = _vram_used_gb()
        run.vram_peak_gb = max(run.vram_peak_gb, vram_after)
        print(
            f"[{engine_name}] transcribed {len(run.segments)} segments in "
            f"{run.transcribe_s:.1f}s (peak VRAM {run.vram_peak_gb:.2f} GB)"
        )

        # --- unload ---
        t0 = time.time()
        engine.unload()
        del engine
        _full_gc()
        run.unload_s = time.time() - t0
        print(f"[{engine_name}] unloaded in {run.unload_s:.1f}s "
              f"(VRAM now {_vram_used_gb():.2f} GB)")
    except Exception as e:
        import traceback
        run.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"[{engine_name}] FAILED: {e}")
        _full_gc()
    finally:
        config.asr.engine = original_engine

    return run


# ---------- transcript markdown writer ----------


def write_transcript_md(
    run: EngineRun,
    audio_name: str,
    base: str,
    out_dir: Path,
) -> Optional[Path]:
    """Write a flat transcript .md for one engine."""
    if not run.segments:
        return None

    lines: list[str] = []
    lines.append(f"# Transcript ({run.name}) \u2014 {base.replace('_', ' ')}")
    lines.append("")
    lines.append(f"**Audio**: `{audio_name}`")
    duration = max((float(s.end) for s in run.segments), default=0.0)
    lines.append(
        f"**Duration**: {format_timestamp(duration)} ({duration/60:.1f} min)"
    )
    lines.append(f"**ASR Engine**: {run.name}  (`{run.engine_key}`)")
    lines.append(f"**Segments**: {len(run.segments)}")
    if run.transcribe_s > 0:
        rtf = duration / run.transcribe_s if run.transcribe_s > 0 else 0.0
        lines.append(
            f"**Transcription time**: {run.transcribe_s:.1f}s "
            f"({rtf:.1f}x realtime)"
        )
    lines.append(f"**VRAM peak**: {run.vram_peak_gb:.2f} GB")
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

    out_path = out_dir / f"{base}_{run.name}_TRANSCRIPT.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------- side-by-side report ----------


def write_ab_report(
    audio_path: Path,
    audio_size_mb: float,
    audio_duration_s: float,
    runs: list[EngineRun],
    transcript_paths: dict[str, Optional[Path]],
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPO_ROOT / "scripts" / f"ab_asr_report_{ts}.md"

    lines: list[str] = []
    lines.append("# A/B ASR Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- **Audio**: `{audio_path}` ({audio_size_mb:.1f} MB, "
                 f"{audio_duration_s/60:.1f} min)")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Engine | Load (s) | Transcribe (s) | Realtime | Unload (s) | "
                 "VRAM peak (GB) | Segments | Status |")
    lines.append("|--------|---------:|---------------:|---------:|-----------:|"
                 "---------------:|---------:|--------|")
    for r in runs:
        rtf = (audio_duration_s / r.transcribe_s) if r.transcribe_s > 0 else 0.0
        status = "FAILED" if r.error else "OK"
        lines.append(
            f"| {r.name} | {r.load_s:.1f} | {r.transcribe_s:.1f} | "
            f"{rtf:.1f}x | {r.unload_s:.1f} | {r.vram_peak_gb:.2f} | "
            f"{len(r.segments)} | {status} |"
        )
    lines.append("")

    lines.append("## Transcripts")
    lines.append("")
    for r in runs:
        p = transcript_paths.get(r.name)
        if p:
            lines.append(f"- **{r.name}**: `{p}`")
        else:
            lines.append(f"- **{r.name}**: (no transcript emitted)")
    lines.append("")

    # First 10 segments side-by-side for quick visual diff
    lines.append("## First 10 segments (side-by-side)")
    lines.append("")
    header = "| # | Time |"
    sep = "|--:|------|"
    for r in runs:
        header += f" {r.name} |"
        sep += "-------|"
    lines.append(header)
    lines.append(sep)
    max_preview = 10
    for i in range(max_preview):
        cells_ok = False
        row = f"| {i+1} |"
        time_cell = ""
        engine_cells: list[str] = []
        for r in runs:
            if i < len(r.segments):
                seg = r.segments[i]
                start = float(getattr(seg, "start", 0.0) or 0.0)
                end = float(getattr(seg, "end", 0.0) or 0.0)
                text = (getattr(seg, "text", "") or "").replace("|", "\\|").strip()
                if not time_cell:
                    time_cell = f"{start:.1f}-{end:.1f}"
                if len(text) > 200:
                    text = text[:200] + "\u2026"
                engine_cells.append(text)
                cells_ok = True
            else:
                engine_cells.append("—")
        if not cells_ok:
            break
        row = f"| {i+1} | {time_cell} | " + " | ".join(engine_cells) + " |"
        lines.append(row)
    lines.append("")

    for r in runs:
        if r.error:
            lines.append(f"## {r.name} — Error")
            lines.append("")
            lines.append("```")
            lines.append(r.error)
            lines.append("```")
            lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------- main ----------


def load_config() -> AppConfig:
    cfg_path = REPO_ROOT / "config" / "settings.yaml"
    if cfg_path.exists():
        return AppConfig.load_from_yaml(cfg_path)
    return AppConfig()


async def run(
    audio_path: Path,
    engines: list[tuple[str, str]],
    base_override: Optional[str] = None,
) -> dict:
    config = load_config()
    print(f"[ab-asr] Audio:      {audio_path}")
    print(f"[ab-asr] Engines:    {[e[0] for e in engines]}")
    print(f"[ab-asr] Deliver:    {DELIVERABLES}")

    # Preprocess once
    print("\n[ab-asr] Preprocessing audio (resample, mono)...")
    t0 = time.time()
    preprocessed_path, metadata = await AudioPreprocessor.process(
        str(audio_path), str(REPO_ROOT / "data" / "meetings" / "_ab_asr_tmp")
    )
    preprocess_s = time.time() - t0
    print(
        f"[ab-asr] Preprocessed in {preprocess_s:.1f}s "
        f"({metadata.sample_rate}Hz, {metadata.channels}ch, "
        f"{metadata.duration:.1f}s)"
    )

    base = base_override or sanitize_basename(audio_path.name)
    DELIVERABLES.mkdir(parents=True, exist_ok=True)

    runs: list[EngineRun] = []
    transcript_paths: dict[str, Optional[Path]] = {}
    for engine_name, engine_key in engines:
        run_result = await run_one_engine(
            engine_name, engine_key, preprocessed_path, config
        )
        runs.append(run_result)
        path = write_transcript_md(
            run_result, audio_path.name, base, DELIVERABLES
        )
        transcript_paths[engine_name] = path
        if path:
            print(f"[ab-asr] Wrote {path}")
        # Also persist raw JSON for reproducibility
        json_path = DELIVERABLES / f"{base}_{engine_name}_segments.json"
        try:
            json_path.write_text(
                json.dumps(
                    [s.model_dump() for s in run_result.segments],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[ab-asr] JSON dump failed for {engine_name}: {e}")

    report = write_ab_report(
        audio_path,
        audio_path.stat().st_size / (1024 * 1024),
        metadata.duration,
        runs,
        transcript_paths,
    )

    return {
        "report": report,
        "transcripts": transcript_paths,
        "runs": runs,
        "preprocess_s": preprocess_s,
    }


def parse_engines(arg: str) -> list[tuple[str, str]]:
    """Convert 'gigaam,hf-whisper' into display-name/key tuples."""
    mapping = {
        "gigaam": ("GigaAM", "gigaam"),
        "hf-whisper": ("HFWhisper", "hf-whisper"),
        "hf": ("HFWhisper", "hf-whisper"),
        "whisper": ("FasterWhisper", "whisper"),
    }
    result: list[tuple[str, str]] = []
    for tok in arg.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok not in mapping:
            raise SystemExit(
                f"Unknown engine '{tok}'. "
                f"Valid: {', '.join(mapping.keys())}"
            )
        result.append(mapping[tok])
    if not result:
        raise SystemExit("At least one engine is required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B ASR comparison harness")
    parser.add_argument("audio", type=Path, help="Path to audio file")
    parser.add_argument(
        "--engines",
        default="gigaam,hf-whisper",
        help="Comma-separated engine list (default: gigaam,hf-whisper)",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Base filename for deliverables (ASCII-safe, overrides auto)",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated engines to skip (applied after --engines)",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    engines = parse_engines(args.engines)
    if args.skip:
        skip_keys = {t.strip().lower() for t in args.skip.split(",") if t.strip()}
        engines = [e for e in engines if e[1] not in skip_keys]
    if not engines:
        print("ERROR: no engines left after --skip", file=sys.stderr)
        return 2

    result = asyncio.run(run(args.audio.resolve(), engines, args.base))

    print()
    print("=" * 70)
    print(f"A/B ASR report: {result['report']}")
    print("=" * 70)
    for r in result["runs"]:
        print(f"  {r.name:15s}  load={r.load_s:5.1f}s  "
              f"transcribe={r.transcribe_s:6.1f}s  "
              f"VRAM={r.vram_peak_gb:5.2f}GB  "
              f"{'OK' if not r.error else 'FAILED'}")
    print()
    print("Deliverables:")
    for name, path in result["transcripts"].items():
        if path:
            print(f"  {name}: {path}")

    # Success only if all engines produced segments
    any_error = any(r.error for r in result["runs"])
    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
