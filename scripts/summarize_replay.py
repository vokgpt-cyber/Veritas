"""
EPAM VERITAS — Summarization Replay Harness (LLM A/B testing)

Replays ONLY the summarization stage on an existing aligned.json transcript,
against one or more Ollama models. Skips ASR / diarization / alignment /
post-processing entirely, so the transcript is held constant and the only
variable is the LLM.

This is the right shape for isolating "did Gemma 4 vs T-Pro 2.0 make a
difference?" — if you re-run the full pipeline you conflate ASR jitter
with LLM differences.

Usage (Windows, venv activated at repo root):

    # Default: both today's aligned.json files, both Gemma + T-Pro
    venv\\Scripts\\python scripts\\summarize_replay.py

    # Explicit files and models
    venv\\Scripts\\python scripts\\summarize_replay.py \
        --aligned data/archive/2026-04-20_Court_hearing_129_01/aligned.json \
        --aligned data/archive/2026-04-20_Админ.../aligned.json \
        --model gemma4:26b --model t-tech/T-pro-it-2.0:q4_K_M

    # With shared context
    venv\\Scripts\\python scripts\\summarize_replay.py --context-file ctx.txt

Outputs:
    scripts/llm_compare/{aligned_stem}__{model_slug}.protocol.json
    scripts/phase5_llm_compare_YYYYMMDD_HHMMSS.md

The outer loop is models (not files) so Ollama only swaps its loaded
model once per model, not once per (file, model) cell. Each model is
unloaded via keep_alive:0 before the next one loads.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import sys
import time
import traceback
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
from backend.app.models import AlignedSegment
from backend.engine.summarization import SummarizationEngine


DEFAULT_ALIGNED = [
    REPO_ROOT / "data" / "archive" / "2026-04-20_Court_hearing_129_01" / "aligned.json",
    REPO_ROOT / "data" / "archive" / "2026-04-20_Админ_13-04-2026_без_мусора_в_начале" / "aligned.json",
]

DEFAULT_MODELS = [
    "gemma4:26b",
    "t-tech/T-pro-it-2.0:q4_K_M",
]

OUT_DIR = REPO_ROOT / "scripts" / "llm_compare"


# --- Helpers ------------------------------------------------------------------


def _slug(s: str) -> str:
    """Filesystem-safe slug for a model tag."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def load_aligned(path: Path) -> list[AlignedSegment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected list or dict with 'segments', got {type(data).__name__}")
    return [AlignedSegment(**d) for d in data]


def load_config() -> AppConfig:
    cfg_path = REPO_ROOT / "config" / "settings.yaml"
    if cfg_path.exists():
        return AppConfig.load_from_yaml(cfg_path)
    fallback = REPO_ROOT / "settings.yaml"
    if fallback.exists():
        return AppConfig.load_from_yaml(fallback)
    print("[replay] WARNING: no settings.yaml found, using hardcoded defaults")
    return AppConfig()


def _proto_progress(progress: float, message: str = "") -> None:
    # Keep stage callbacks terse in the log
    print(f"  [llm] {progress*100:5.1f}%  {message}")


# --- Replay core --------------------------------------------------------------


async def run_one(
    aligned_path: Path,
    model_tag: str,
    context: str,
    base_config: AppConfig,
) -> dict:
    """Run summarization once on the given aligned.json with the given Ollama model."""
    # Clone config and override the model tag. AppConfig is Pydantic v2 so
    # deepcopy is safe.
    cfg = copy.deepcopy(base_config)
    cfg.summarization.ollama_model = model_tag

    print(f"\n--- {aligned_path.name}  |  model={model_tag} ---")
    t_load0 = time.time()
    engine = SummarizationEngine(cfg)
    engine.load()
    t_load = time.time() - t_load0

    segments = load_aligned(aligned_path)
    transcript_chars = sum(len(s.text or "") for s in segments)
    print(f"  transcript: {len(segments)} segments, {transcript_chars} chars")
    print(f"  load:       {t_load:.1f}s")

    error: Optional[str] = None
    protocol = None
    t0 = time.time()
    try:
        protocol = await engine.process(
            transcript=segments,
            language="ru",
            progress_callback=_proto_progress,
            meeting_context=context,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"  FAILED: {exc}")
    t_process = time.time() - t0

    # Release model from Ollama VRAM before swapping
    t_unload0 = time.time()
    try:
        engine.unload()
    except Exception as exc:
        print(f"  unload warning: {exc}")
    t_unload = time.time() - t_unload0

    # Write protocol JSON
    out_path: Optional[Path] = None
    if protocol is not None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = aligned_path.parent.name or aligned_path.stem
        out_path = OUT_DIR / f"{stem}__{_slug(model_tag)}.protocol.json"
        out_path.write_text(
            json.dumps(protocol.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  wrote: {out_path}")

    print(f"  process:    {t_process:.1f}s   unload: {t_unload:.1f}s")

    return {
        "aligned_path": str(aligned_path),
        "aligned_stem": aligned_path.parent.name or aligned_path.stem,
        "model_tag": model_tag,
        "model_slug": _slug(model_tag),
        "transcript_segments": len(segments),
        "transcript_chars": transcript_chars,
        "load_s": t_load,
        "process_s": t_process,
        "unload_s": t_unload,
        "error": error,
        "protocol": protocol.model_dump() if protocol is not None else None,
        "protocol_path": str(out_path) if out_path else None,
    }


# --- Comparison report --------------------------------------------------------


def _count(x) -> int:
    try:
        return len(x or [])
    except TypeError:
        return 0


def _status(r: dict) -> str:
    return "FAILED" if r.get("error") else "OK"


def _short_model(tag: str) -> str:
    """A compact display name for tables."""
    if "gemma" in tag.lower():
        return "gemma4:26b"
    if "t-pro" in tag.lower() or "tpro" in tag.lower():
        return "T-Pro 2.0 Q4"
    return tag


def _short_file(stem: str) -> str:
    if "Court_hearing" in stem or "court_hearing" in stem.lower():
        return "Court 129"
    if "Админ" in stem or "admin" in stem.lower():
        return "Админ 13-04"
    return stem[:16]


def write_comparison(results: list[dict], audios: list[Path], models: list[str]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = REPO_ROOT / "scripts" / f"phase5_llm_compare_{ts}.md"

    # Index by (aligned_stem, model_tag)
    by_cell: dict[tuple[str, str], dict] = {
        (r["aligned_stem"], r["model_tag"]): r for r in results
    }
    aligned_stems = [a.parent.name or a.stem for a in audios]

    lines: list[str] = []
    lines.append("# Phase 5 LLM Comparison Report (summarization replay)")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Transcript held constant (aligned.json from today's E2E run). "
                 "Only the Ollama model varied. ASR / diarization / alignment / "
                 "post-processing NOT re-run.")
    lines.append("")

    lines.append("## Inputs")
    lines.append("")
    for a in audios:
        lines.append(f"- `{a}`")
    lines.append("")
    lines.append("Models:")
    for m in models:
        lines.append(f"- `{m}`")
    lines.append("")

    # --- Timing table: rows=files, cols=models ---
    lines.append("## Summarization wall-clock (s) — process() only")
    lines.append("")
    header = "| File | " + " | ".join(_short_model(m) for m in models) + " |"
    sep = "|------|" + "|".join(["-----:" for _ in models]) + "|"
    lines.append(header)
    lines.append(sep)
    for stem in aligned_stems:
        cells = []
        for m in models:
            r = by_cell.get((stem, m))
            cells.append(f"{r['process_s']:.1f}" if r else "—")
        lines.append(f"| {_short_file(stem)} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- Protocol stats ---
    lines.append("## Protocol stats per (file x model)")
    lines.append("")
    lines.append("| File | Model | Status | Summary chars | Topics | Decisions | Action items | Open qs | Time (s) |")
    lines.append("|------|-------|--------|--------------:|-------:|----------:|-------------:|--------:|---------:|")
    for stem in aligned_stems:
        for m in models:
            r = by_cell.get((stem, m))
            if not r:
                lines.append(f"| {_short_file(stem)} | {_short_model(m)} | — | — | — | — | — | — | — |")
                continue
            p = r.get("protocol") or {}
            summary_chars = len((p.get("summary") or "").strip())
            lines.append(
                f"| {_short_file(stem)} | {_short_model(m)} | {_status(r)} | "
                f"{summary_chars} | {_count(p.get('key_topics'))} | "
                f"{_count(p.get('decisions'))} | {_count(p.get('tasks'))} | "
                f"{_count(p.get('open_questions'))} | {r['process_s']:.1f} |"
            )
    lines.append("")

    # --- Topic + participant count check ---
    lines.append("## Topic strings (quick eyeball)")
    lines.append("")
    for stem in aligned_stems:
        lines.append(f"### {_short_file(stem)}")
        lines.append("")
        for m in models:
            r = by_cell.get((stem, m))
            if not r or not r.get("protocol"):
                continue
            topic = (r["protocol"].get("topic") or "").strip() or "—"
            lines.append(f"- **{_short_model(m)}**: {topic}")
        lines.append("")

    # --- Full summary text side by side per file ---
    for stem in aligned_stems:
        lines.append(f"## Full summaries — {_short_file(stem)}")
        lines.append("")
        for m in models:
            r = by_cell.get((stem, m))
            if not r:
                continue
            lines.append(f"### {_short_model(m)}  ({r['process_s']:.1f}s)")
            lines.append("")
            if r.get("error"):
                lines.append("```")
                lines.append(r["error"])
                lines.append("```")
                lines.append("")
                continue
            p = r.get("protocol") or {}
            summary = (p.get("summary") or "").strip()
            lines.append(summary or "_(empty)_")
            lines.append("")

            decisions = p.get("decisions") or []
            if decisions:
                lines.append("**Decisions:**")
                lines.append("")
                for d in decisions[:30]:
                    text = d.get("text") if isinstance(d, dict) else str(d)
                    lines.append(f"- {text}")
                lines.append("")
            tasks = p.get("tasks") or []
            if tasks:
                lines.append("**Action items:**")
                lines.append("")
                for t in tasks[:30]:
                    text = t.get("text") if isinstance(t, dict) else str(t)
                    owner = (t.get("assignee") if isinstance(t, dict) else "") or ""
                    due = (t.get("deadline") if isinstance(t, dict) else "") or ""
                    meta = " / ".join(x for x in (owner, due) if x)
                    lines.append(f"- [{meta}] {text}" if meta else f"- {text}")
                lines.append("")
            questions = p.get("open_questions") or []
            if questions:
                lines.append("**Open questions:**")
                lines.append("")
                for q in questions[:30]:
                    text = q if isinstance(q, str) else (q.get("text") if isinstance(q, dict) else str(q))
                    lines.append(f"- {text}")
                lines.append("")

    # --- Output file index ---
    lines.append("## Output files")
    lines.append("")
    for r in results:
        if r.get("protocol_path"):
            lines.append(f"- `{r['protocol_path']}`")
    lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")
    return report


# --- Entry point --------------------------------------------------------------


async def main_async(args) -> int:
    audios: list[Path] = [Path(a).resolve() for a in (args.aligned or [])]
    if not audios:
        audios = [a for a in DEFAULT_ALIGNED if a.exists()]
    models: list[str] = args.model or list(DEFAULT_MODELS)

    missing = [a for a in audios if not a.exists()]
    if missing:
        for m in missing:
            print(f"ERROR: aligned.json not found: {m}", file=sys.stderr)
        return 2
    if not audios:
        print("ERROR: no aligned.json paths provided and no defaults exist on disk.", file=sys.stderr)
        return 2
    if not models:
        print("ERROR: no models specified.", file=sys.stderr)
        return 2

    context = args.context or ""
    if args.context_file and args.context_file.exists():
        context = args.context_file.read_text(encoding="utf-8")

    base_config = load_config()

    print(f"[replay] aligned files ({len(audios)}):")
    for a in audios:
        print(f"  - {a}")
    print(f"[replay] models ({len(models)}):")
    for m in models:
        print(f"  - {m}")
    print(f"[replay] context chars: {len(context)}")
    print(f"[replay] output dir:    {OUT_DIR}")
    print()

    results: list[dict] = []
    t_start = time.time()

    # Outer loop: models, inner loop: files.
    # This minimizes Ollama model swaps (load once per model, process all files).
    for model_tag in models:
        for aligned_path in audios:
            result = await run_one(aligned_path, model_tag, context, base_config)
            results.append(result)

    total = time.time() - t_start

    report = write_comparison(results, audios, models)

    print("\n" + "=" * 70)
    print(f"Comparison report: {report}")
    print(f"Wall-clock total:  {total:.1f}s  ({total/60:.2f} min)")
    print(f"Runs:              {len(results)}  "
          f"({sum(1 for r in results if r.get('error'))} failed)")
    print("=" * 70)
    for r in results:
        print(f"  {_short_file(r['aligned_stem']):16s}  {_short_model(r['model_tag']):14s}  "
              f"{_status(r):7s}  {r['process_s']:>6.1f}s"
              + (f"  -> {Path(r['protocol_path']).name}" if r.get("protocol_path") else ""))

    return 1 if any(r.get("error") for r in results) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay summarization on aligned.json with different Ollama models")
    parser.add_argument("--aligned", action="append", type=Path, default=None,
                        help="Path to an aligned.json file. Repeat for multiple. "
                             "Defaults to today's two E2E archive outputs.")
    parser.add_argument("--model", action="append", default=None,
                        help="Ollama model tag. Repeat for multiple. "
                             "Defaults to gemma4:26b + T-Pro 2.0 Q4_K_M.")
    parser.add_argument("--context", default="", help="Meeting context applied to ALL runs")
    parser.add_argument("--context-file", type=Path, default=None,
                        help="Read context from a UTF-8 file")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
