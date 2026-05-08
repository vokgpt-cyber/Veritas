"""
EPAM VERITAS — Phase 5 End-to-End Test Harness

Runs the full pipeline on a given audio file, measuring per-stage wall-clock
time and VRAM peaks. Preserves all intermediate artifacts and writes a markdown
report with timings and quality-review excerpts.

Usage (Windows, with venv activated at repo root):

    venv\\Scripts\\python scripts\\run_e2e_test.py path\\to\\audio.wav
    venv\\Scripts\\python scripts\\run_e2e_test.py audio.wav --context "Agenda: ..."
    venv\\Scripts\\python scripts\\run_e2e_test.py audio.wav --expected-speakers 3

Output:
    scripts/phase5_report_YYYYMMDD_HHMMSS.md   - timings, VRAM, quality excerpts
    data/meetings/{job_id}/                    - all intermediate + final files
    data/archive/YYYY-MM-DD_{filename}/        - archived outputs (DOCX, JSON)

The harness runs the orchestrator directly (no FastAPI/WebSocket layer) so the
timings isolate ML pipeline cost, not API overhead.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make backend.* importable and anchor cwd to repo root regardless of invocation
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# Load .env so HF_TOKEN, EPAM_* overrides, and pyannote auth are honored
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

from backend.app.config import AppConfig
from backend.app.models import MeetingJob
from backend.core.orchestrator import Orchestrator
from backend.core.vram_manager import VRAMManager


# --- Stage tracking -----------------------------------------------------------


@dataclass
class StageStats:
    name: str
    order: int
    start_wall: float
    progress_at_start: float
    end_wall: Optional[float] = None
    vram_start_gb: float = 0.0
    vram_end_gb: float = 0.0
    vram_peak_gb: float = 0.0

    @property
    def duration_s(self) -> float:
        return (self.end_wall if self.end_wall is not None else time.time()) - self.start_wall


class VRAMSampler(threading.Thread):
    """Background thread that polls VRAM and tracks the peak for the active stage."""

    def __init__(self, interval_s: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._vram = VRAMManager()
        # NOTE: do not name this `_stop` — collides with threading.Thread._stop
        self._stop_event = threading.Event()
        self._current: Optional[StageStats] = None
        self._lock = threading.Lock()

    def _read_vram(self) -> float:
        try:
            status = self._vram.get_status() or {}
            return float(status.get("vram_used_gb", 0.0) or 0.0)
        except Exception:
            return 0.0

    def begin_stage(self, stage: StageStats) -> None:
        with self._lock:
            self._current = stage
            used = self._read_vram()
            stage.vram_start_gb = used
            if used > stage.vram_peak_gb:
                stage.vram_peak_gb = used

    def end_stage(self) -> None:
        with self._lock:
            stage = self._current
            if stage is not None:
                stage.vram_end_gb = self._read_vram()
                if stage.end_wall is None:
                    stage.end_wall = time.time()
            self._current = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                stage = self._current
            if stage is not None:
                used = self._read_vram()
                with self._lock:
                    if used > stage.vram_peak_gb:
                        stage.vram_peak_gb = used
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop_event.set()


# --- Pipeline invocation ------------------------------------------------------


def load_config() -> AppConfig:
    cfg_path = REPO_ROOT / "config" / "settings.yaml"
    if cfg_path.exists():
        return AppConfig.load_from_yaml(cfg_path)
    fallback = REPO_ROOT / "settings.yaml"
    if fallback.exists():
        return AppConfig.load_from_yaml(fallback)
    print("[harness] WARNING: no settings.yaml found, using hardcoded defaults")
    return AppConfig()


async def run_pipeline(
    audio_path: Path,
    context: str,
    expected_speakers: Optional[int],
) -> dict:
    config = load_config()
    print(f"[harness] ASR engine:        {config.asr.engine}")
    print(f"[harness] Diarization engine: {config.diarization.engine}")
    print(f"[harness] Summarization:     enabled={config.summarization.enabled} "
          f"model={config.summarization.ollama_model}")
    print(f"[harness] Audio:             {audio_path}")
    print(f"[harness] Context chars:     {len(context)}")
    print(f"[harness] Expected speakers: {expected_speakers}")

    orchestrator = Orchestrator(config)
    job = MeetingJob(filename=audio_path.name, context=context)
    print(f"[harness] Job ID:            {job.id}")
    print()

    stages: list[StageStats] = []
    stage_index: dict[str, StageStats] = {}
    order_counter = 0
    current_name: Optional[str] = None

    # 1 Hz polling is plenty for stage-level peak detection and keeps the
    # nvidia-smi subprocess cost low when pynvml is unavailable.
    sampler = VRAMSampler(interval_s=1.0)
    sampler.start()

    def on_progress(updated_job: MeetingJob) -> None:
        nonlocal current_name, order_counter
        name = (updated_job.current_stage or "").strip() or "unknown"
        if name == current_name:
            return
        # Stage transition
        if current_name is not None:
            sampler.end_stage()
            prev = stage_index.get(current_name)
            if prev is not None:
                print(f"[stage-done]  {prev.name:30s} {prev.duration_s:>7.1f}s  "
                      f"peak={prev.vram_peak_gb:.2f}GB  end={prev.vram_end_gb:.2f}GB")
        order_counter += 1
        stage = StageStats(
            name=name,
            order=order_counter,
            start_wall=time.time(),
            progress_at_start=float(updated_job.progress or 0.0),
        )
        stages.append(stage)
        stage_index[name] = stage
        sampler.begin_stage(stage)
        current_name = name
        print(f"[stage-start] {name:30s} @ progress {updated_job.progress:>5.1f}%")

    total_start = time.time()
    error: Optional[str] = None
    protocol = None

    try:
        protocol = await orchestrator.process_meeting(
            audio_path=str(audio_path),
            job=job,
            on_progress=on_progress,
            expected_speakers=expected_speakers,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(f"\n[harness] PIPELINE FAILED: {exc}")
    finally:
        # Close out any in-flight stage
        if current_name is not None:
            sampler.end_stage()
            last = stage_index.get(current_name)
            if last is not None and last.end_wall is not None:
                print(f"[stage-done]  {last.name:30s} {last.duration_s:>7.1f}s  "
                      f"peak={last.vram_peak_gb:.2f}GB  end={last.vram_end_gb:.2f}GB")
        sampler.stop()
        sampler.join(timeout=2.0)

    total_s = time.time() - total_start

    # Collect output artifacts
    meeting_dir = REPO_ROOT / "data" / "meetings" / job.id
    outputs: dict[str, str] = {}
    for fname in (
        "transcription.json",
        "diarization.json",
        "aligned_raw.json",
        "aligned.json",
        "protocol.json",
    ):
        p = meeting_dir / fname
        if p.exists():
            outputs[fname] = str(p)
    for docx in meeting_dir.glob("*.docx"):
        outputs[docx.name] = str(docx)

    return {
        "job_id": job.id,
        "audio_path": str(audio_path),
        "audio_size_mb": audio_path.stat().st_size / (1024 * 1024),
        "context_chars": len(context),
        "expected_speakers": expected_speakers,
        "total_s": total_s,
        "stages": stages,
        "outputs": outputs,
        "meeting_dir": str(meeting_dir),
        "protocol": protocol.model_dump() if protocol is not None else None,
        "error": error,
        "config": {
            "asr_engine": config.asr.engine,
            "asr_model": getattr(config.asr, "whisper_model", None) or "n/a",
            "diarization_engine": config.diarization.engine,
            "summarization_enabled": config.summarization.enabled,
            "ollama_model": config.summarization.ollama_model,
            "ollama_base_url": config.summarization.ollama_base_url,
            "summarization_temperature": getattr(config.summarization, "temperature", None),
            "summarization_top_p": getattr(config.summarization, "top_p", None),
        },
    }


# --- Reporting ----------------------------------------------------------------


def _load_transcript_preview(outputs: dict[str, str], max_segments: int = 30) -> list[dict]:
    """Load the aligned transcript for a quality-review preview."""
    path = outputs.get("aligned.json") or outputs.get("aligned_raw.json")
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, list):
        return data[:max_segments]
    if isinstance(data, dict) and "segments" in data:
        return data["segments"][:max_segments]
    return []


def write_report(result: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPO_ROOT / "scripts" / f"phase5_report_{ts}.md"
    stages: list[StageStats] = result["stages"]

    lines: list[str] = []
    lines.append(f"# Phase 5 End-to-End Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- **Audio**: `{result['audio_path']}` ({result['audio_size_mb']:.1f} MB)")
    lines.append(f"- **Job ID**: `{result['job_id']}`")
    lines.append(f"- **Expected speakers**: {result['expected_speakers']}")
    lines.append(f"- **Context chars**: {result['context_chars']}")
    lines.append(f"- **Total wall-clock**: {result['total_s']:.1f}s "
                 f"({result['total_s']/60:.2f} min)")
    if result.get("error"):
        lines.append(f"- **STATUS**: FAILED")
    else:
        lines.append(f"- **STATUS**: OK")
    lines.append("")

    lines.append("## Config snapshot")
    lines.append("")
    for k, v in result["config"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")

    lines.append("## Per-stage timing + VRAM")
    lines.append("")
    lines.append("| # | Stage | Duration (s) | % of total | VRAM start (GB) | VRAM peak (GB) | VRAM end (GB) |")
    lines.append("|--:|-------|-------------:|-----------:|----------------:|---------------:|--------------:|")
    for s in stages:
        pct = (s.duration_s / result["total_s"] * 100.0) if result["total_s"] > 0 else 0.0
        lines.append(
            f"| {s.order} | {s.name} | {s.duration_s:.1f} | {pct:.1f}% | "
            f"{s.vram_start_gb:.2f} | {s.vram_peak_gb:.2f} | {s.vram_end_gb:.2f} |"
        )
    lines.append("")

    lines.append("## Output artifacts")
    lines.append("")
    if result["outputs"]:
        for name, path in result["outputs"].items():
            lines.append(f"- `{name}` → `{path}`")
    else:
        lines.append("_(none — pipeline failed before any artifact was written)_")
    lines.append(f"- Meeting dir: `{result['meeting_dir']}`")
    lines.append("")

    if result.get("error"):
        lines.append("## Error")
        lines.append("")
        lines.append("```")
        lines.append(result["error"])
        lines.append("```")
        lines.append("")

    protocol = result.get("protocol")
    if protocol:
        lines.append("## Protocol snapshot (quality review)")
        lines.append("")
        lines.append(f"- Topic: `{protocol.get('topic', '—')}`")
        lines.append(f"- Participants: {len(protocol.get('participants', []))}")
        for p in protocol.get("participants", []):
            sid = p.get("speaker_id") or p.get("name") or "?"
            share = p.get("speaking_share") or p.get("speaking_time_pct") or 0.0
            lines.append(f"  - `{sid}` — {share:.1f}% speaking share")
        lines.append(f"- Transcript segments: {len(protocol.get('transcript', []))}")
        lines.append(f"- Decisions: {len(protocol.get('decisions', []))}")
        lines.append(f"- Tasks: {len(protocol.get('tasks', []))}")
        lines.append(f"- Open questions: {len(protocol.get('open_questions', []))}")
        lines.append("")

        summary = (protocol.get("summary") or "").strip()
        if summary:
            lines.append("### Summary")
            lines.append("")
            lines.append(summary[:3000])
            if len(summary) > 3000:
                lines.append(f"\n_... ({len(summary) - 3000} more chars truncated)_")
            lines.append("")

        decisions = protocol.get("decisions") or []
        if decisions:
            lines.append("### Decisions")
            lines.append("")
            for d in decisions[:30]:
                text = d.get("text") or d.get("description") or str(d)
                speaker = d.get("speaker_id") or d.get("speaker") or ""
                lines.append(f"- **{speaker}**: {text}" if speaker else f"- {text}")
            lines.append("")

        tasks = protocol.get("tasks") or []
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

        questions = protocol.get("open_questions") or []
        if questions:
            lines.append("### Open questions")
            lines.append("")
            for q in questions[:30]:
                text = q if isinstance(q, str) else (q.get("text") or str(q))
                lines.append(f"- {text}")
            lines.append("")

    preview = _load_transcript_preview(result["outputs"], max_segments=40)
    if preview:
        lines.append("## Transcript preview (first 40 segments)")
        lines.append("")
        lines.append("| Start | End | Speaker | Text |")
        lines.append("|------:|----:|---------|------|")
        for seg in preview:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            spk = seg.get("speaker_id") or seg.get("speaker") or "?"
            text = (seg.get("text") or "").replace("|", "\\|")
            if len(text) > 250:
                text = text[:250] + "…"
            lines.append(f"| {start:.1f} | {end:.1f} | {spk} | {text} |")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# --- Entry point --------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="VERITAS Phase 5 end-to-end harness")
    parser.add_argument("audio", type=Path, help="Path to audio file (any format supported by ffmpeg/torchaudio)")
    parser.add_argument("--context", default="", help="Meeting context/agenda passed to the LLM prompt")
    parser.add_argument("--context-file", type=Path, default=None,
                        help="Read context from a UTF-8 text file instead of --context")
    parser.add_argument("--expected-speakers", type=int, default=None,
                        help="Expected speaker count (constrains diarization)")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        return 2

    context = args.context
    if args.context_file and args.context_file.exists():
        context = args.context_file.read_text(encoding="utf-8")

    result = asyncio.run(run_pipeline(args.audio.resolve(), context, args.expected_speakers))

    report = write_report(result)

    print()
    print("=" * 70)
    print(f"Report:  {report}")
    print(f"Total:   {result['total_s']:.1f}s ({result['total_s']/60:.2f} min)")
    print(f"Status:  {'FAILED' if result.get('error') else 'OK'}")
    print("=" * 70)
    print(f"{'Stage':30s} {'Duration':>10s}  {'VRAM peak':>10s}")
    for s in result["stages"]:
        print(f"{s.name:30s} {s.duration_s:>9.1f}s  {s.vram_peak_gb:>8.2f}GB")

    return 1 if result.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
