"""
EPAM VERITAS — Phase 5 End-to-End Comparison Harness (2 files back-to-back)

Runs the full pipeline on both target audios sequentially, writes the
per-run markdown report (via run_e2e_test.write_report) for each, and
then emits a consolidated side-by-side comparison report:

    scripts/phase5_compare_YYYYMMDD_HHMMSS.md

Usage (Windows, with venv activated at repo root):

    venv\\Scripts\\python scripts\\run_e2e_both.py

By default it targets the two files the user named:
    data/test/Court hearing 129_01.mp4
    data/test/Админ 13-04-2026 без мусора в начале.mp3

Optional: pass --context-file / --expected-speakers — applied to BOTH runs.
To use different settings per file, run run_e2e_test.py twice instead.

Between runs the script does a best-effort VRAM release (VRAMManager
.release_all()) so the second file starts clean. Ollama is asked to
unload its current model via keep_alive:0 on the summarization side,
but a lingering `ollama serve` process will still hold its own VRAM;
that's outside this harness's control.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# Reuse the single-file harness wholesale so we stay in lockstep with it
from scripts.run_e2e_test import (  # type: ignore
    StageStats,
    run_pipeline,
    write_report,
)

# Defaults: the two files the user explicitly asked about
DEFAULT_AUDIOS = [
    REPO_ROOT / "data" / "test" / "Court hearing 129_01.mp4",
    REPO_ROOT / "data" / "test" / "Админ 13-04-2026 без мусора в начале.mp3",
]


# --- Comparison report --------------------------------------------------------


def _fmt_int(x: Optional[int]) -> str:
    return "—" if x is None else f"{x}"


def _fmt_float(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def _stage_lookup(stages: list[StageStats]) -> dict[str, StageStats]:
    return {s.name: s for s in stages}


def _status(result: dict) -> str:
    return "FAILED" if result.get("error") else "OK"


def _count(coll) -> int:
    try:
        return len(coll or [])
    except TypeError:
        return 0


def write_comparison(results: list[dict], per_run_reports: list[Path]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPO_ROOT / "scripts" / f"phase5_compare_{ts}.md"

    # Pre-compute labels short enough for table headers
    labels: list[str] = []
    for r in results:
        name = Path(r["audio_path"]).stem
        if len(name) > 32:
            name = name[:29] + "…"
        labels.append(name)

    lines: list[str] = []
    lines.append("# Phase 5 E2E Comparison Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for idx, (r, rpt) in enumerate(zip(results, per_run_reports), start=1):
        lines.append(f"{idx}. `{r['audio_path']}` ({r['audio_size_mb']:.1f} MB) — "
                     f"status **{_status(r)}** — per-run report: `{rpt.name}`")
    lines.append("")

    # --- Top-line timings ---
    lines.append("## Top-line timings")
    lines.append("")
    header = "| Metric | " + " | ".join(labels) + " |"
    sep = "|--------|" + "|".join(["-----:" for _ in labels]) + "|"
    lines.append(header)
    lines.append(sep)

    def row(metric: str, values: list[str]) -> str:
        return "| " + metric + " | " + " | ".join(values) + " |"

    lines.append(row("Audio size (MB)", [f"{r['audio_size_mb']:.1f}" for r in results]))
    lines.append(row("Total wall-clock (s)", [f"{r['total_s']:.1f}" for r in results]))
    lines.append(row("Total wall-clock (min)", [f"{r['total_s']/60:.2f}" for r in results]))
    lines.append(row("Status", [_status(r) for r in results]))
    lines.append(row("Expected speakers", [_fmt_int(r.get("expected_speakers")) for r in results]))
    lines.append(row("Context chars", [str(r.get("context_chars", 0)) for r in results]))
    lines.append("")

    # --- Per-stage timing comparison ---
    # Collect union of stage names, preserving the order from the first run
    # that actually has them.
    seen: list[str] = []
    seen_set: set[str] = set()
    for r in results:
        for s in r["stages"]:
            if s.name not in seen_set:
                seen.append(s.name)
                seen_set.add(s.name)

    lines.append("## Per-stage duration (s)")
    lines.append("")
    lines.append("| Stage | " + " | ".join(labels) + " |")
    lines.append("|-------|" + "|".join(["-----:" for _ in labels]) + "|")
    lookups = [_stage_lookup(r["stages"]) for r in results]
    for name in seen:
        cells = []
        for lk in lookups:
            s = lk.get(name)
            cells.append(f"{s.duration_s:.1f}" if s else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- Per-stage VRAM peak ---
    lines.append("## Per-stage VRAM peak (GB)")
    lines.append("")
    lines.append("| Stage | " + " | ".join(labels) + " |")
    lines.append("|-------|" + "|".join(["-----:" for _ in labels]) + "|")
    for name in seen:
        cells = []
        for lk in lookups:
            s = lk.get(name)
            cells.append(f"{s.vram_peak_gb:.2f}" if s else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- Protocol quality snapshot ---
    lines.append("## Protocol quality snapshot")
    lines.append("")
    lines.append("| Metric | " + " | ".join(labels) + " |")
    lines.append("|--------|" + "|".join(["-----:" for _ in labels]) + "|")

    def proto_cell(result: dict, key: str) -> str:
        p = result.get("protocol")
        if not p:
            return "—"
        return str(_count(p.get(key)))

    def proto_topic(result: dict) -> str:
        p = result.get("protocol") or {}
        t = (p.get("topic") or "").strip()
        if not t:
            return "—"
        if len(t) > 60:
            t = t[:57] + "…"
        # Escape pipes for markdown table
        return t.replace("|", "\\|")

    def proto_participants(result: dict) -> str:
        p = result.get("protocol") or {}
        return str(len(p.get("participants") or []))

    def proto_summary_chars(result: dict) -> str:
        p = result.get("protocol") or {}
        return str(len((p.get("summary") or "").strip()))

    def proto_transcript_segs(result: dict) -> str:
        p = result.get("protocol") or {}
        return str(len(p.get("transcript") or []))

    lines.append(row("Topic", [proto_topic(r) for r in results]))
    lines.append(row("Participants", [proto_participants(r) for r in results]))
    lines.append(row("Transcript segments", [proto_transcript_segs(r) for r in results]))
    lines.append(row("Summary chars", [proto_summary_chars(r) for r in results]))
    lines.append(row("Decisions", [proto_cell(r, "decisions") for r in results]))
    lines.append(row("Action items", [proto_cell(r, "tasks") for r in results]))
    lines.append(row("Open questions", [proto_cell(r, "open_questions") for r in results]))
    lines.append("")

    # --- Config snapshot (should be identical across runs, but worth logging) ---
    lines.append("## Config snapshot (should match across runs)")
    lines.append("")
    cfg_keys = sorted({k for r in results for k in (r.get("config") or {}).keys()})
    lines.append("| Key | " + " | ".join(labels) + " |")
    lines.append("|-----|" + "|".join(["------" for _ in labels]) + "|")
    for k in cfg_keys:
        cells = []
        for r in results:
            v = (r.get("config") or {}).get(k)
            cells.append(f"`{v}`")
        lines.append(f"| `{k}` | " + " | ".join(cells) + " |")
    lines.append("")

    # --- Per-run protocol text blocks ---
    for label, r in zip(labels, results):
        p = r.get("protocol")
        if not p:
            continue
        lines.append(f"## Protocol text — {label}")
        lines.append("")
        summary = (p.get("summary") or "").strip()
        if summary:
            lines.append("### Summary")
            lines.append("")
            lines.append(summary[:4000])
            if len(summary) > 4000:
                lines.append(f"\n_... ({len(summary) - 4000} more chars truncated — see per-run report)_")
            lines.append("")
        decisions = p.get("decisions") or []
        if decisions:
            lines.append("### Decisions")
            lines.append("")
            for d in decisions[:30]:
                text = d.get("text") or d.get("description") or str(d)
                speaker = d.get("speaker_id") or d.get("speaker") or ""
                lines.append(f"- **{speaker}**: {text}" if speaker else f"- {text}")
            lines.append("")
        tasks = p.get("tasks") or []
        if tasks:
            lines.append("### Action items")
            lines.append("")
            for t in tasks[:30]:
                text = t.get("text") or t.get("description") or str(t)
                owner = t.get("owner") or t.get("assignee") or ""
                due = t.get("due_date") or t.get("deadline") or ""
                meta = " / ".join(x for x in (owner, due) if x)
                lines.append(f"- [{meta}] {text}" if meta else f"- {text}")
            lines.append("")
        questions = p.get("open_questions") or []
        if questions:
            lines.append("### Open questions")
            lines.append("")
            for q in questions[:30]:
                text = q if isinstance(q, str) else (q.get("text") or str(q))
                lines.append(f"- {text}")
            lines.append("")

    # --- Errors ---
    any_error = any(r.get("error") for r in results)
    if any_error:
        lines.append("## Errors")
        lines.append("")
        for label, r in zip(labels, results):
            if r.get("error"):
                lines.append(f"### {label}")
                lines.append("")
                lines.append("```")
                lines.append(r["error"])
                lines.append("```")
                lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# --- VRAM cleanup between runs ------------------------------------------------


def _release_vram_between_runs() -> None:
    """Best-effort: ask VRAMManager to release everything it tracked, and
    poke CUDA's allocator. Ollama's own VRAM (the LLM) is released by the
    summarization engine's unload() via keep_alive:0 during the first run,
    so we just need to catch any stragglers on the Python side."""
    try:
        from backend.core.vram_manager import VRAMManager
        VRAMManager().release_all()
    except Exception as exc:
        print(f"[between] VRAMManager.release_all failed: {exc}")
    try:
        import gc
        gc.collect()
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --- Entry point --------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VERITAS E2E on two audios back-to-back")
    parser.add_argument("audios", type=Path, nargs="*",
                        help="Audio files to process (default: the two files in data/test)")
    parser.add_argument("--context", default="", help="Context applied to ALL runs")
    parser.add_argument("--context-file", type=Path, default=None,
                        help="Read context from a UTF-8 file")
    parser.add_argument("--expected-speakers", type=int, default=None,
                        help="Expected speaker count applied to ALL runs")
    args = parser.parse_args()

    audios = args.audios if args.audios else DEFAULT_AUDIOS
    audios = [Path(a).resolve() for a in audios]

    missing = [a for a in audios if not a.exists()]
    if missing:
        for m in missing:
            print(f"ERROR: audio not found: {m}", file=sys.stderr)
        return 2

    context = args.context
    if args.context_file and args.context_file.exists():
        context = args.context_file.read_text(encoding="utf-8")

    results: list[dict] = []
    per_run_reports: list[Path] = []
    t0 = time.time()

    for idx, audio in enumerate(audios, start=1):
        banner = f"RUN {idx}/{len(audios)}: {audio.name}"
        print("\n" + "=" * 70)
        print(banner)
        print("=" * 70)

        result = asyncio.run(run_pipeline(audio, context, args.expected_speakers))
        report = write_report(result)
        results.append(result)
        per_run_reports.append(report)

        print(f"\n[run {idx}] report: {report}")
        print(f"[run {idx}] status: {_status(result)}  total: {result['total_s']:.1f}s")

        if idx < len(audios):
            print(f"\n[between] releasing VRAM before next run...")
            _release_vram_between_runs()

    compare = write_comparison(results, per_run_reports)
    total = time.time() - t0

    print("\n" + "=" * 70)
    print(f"Comparison report: {compare}")
    print(f"Wall-clock total:  {total:.1f}s ({total/60:.2f} min)")
    for label, r, rpt in zip(
        [Path(r["audio_path"]).stem for r in results],
        results,
        per_run_reports,
    ):
        print(f"  {label[:40]:40s}  {_status(r):7s}  {r['total_s']:>7.1f}s  -> {rpt.name}")
    print("=" * 70)

    return 1 if any(r.get("error") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
