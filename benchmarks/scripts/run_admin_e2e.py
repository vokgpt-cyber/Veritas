#!/usr/bin/env python3
"""Run the full VERITAS pipeline on the Админ 13-04 audio and dump
every intermediate artifact into a fixed comparison directory.

What this does (in order):
  1. Loads config/settings.yaml.
  2. Instantiates the live Orchestrator — real ASR (GigaAM by default),
     real diarization (pyannote-community-1), real LLM (T-Pro 2.0 via
     Ollama). All three have GPU/external-service requirements that must
     be satisfied on the host.
  3. Builds a MeetingJob with meeting_type=ADMINISTRATIVE (matches the
     real character of the file so the Block 7 admin DOCX renderer fires).
  4. Awaits process_meeting() — this is the same entry point the API
     uses, so output matches a production run byte-for-byte.
  5. Copies the meeting's intermediate JSONs + final DOCX into
     benchmarks/data/admin_13_04_fresh_{YYYY-MM-DD_HHMMSS}/ where the
     comparison-HTML builder expects to find them.

Fails loud if Ollama is unreachable (summarization depends on it) or if
the audio file cannot be located, rather than silently producing a
half-pipeline result.

Usage:
    python benchmarks/scripts/run_admin_e2e.py
    python benchmarks/scripts/run_admin_e2e.py --audio path/to/audio.mp3
    python benchmarks/scripts/run_admin_e2e.py --meeting-type generic

Output:
    benchmarks/data/admin_13_04_fresh_<stamp>/
        transcription.json      — raw ASR output
        diarization.json        — raw speaker segments
        aligned.json            — post-aligner, post-postprocessor turns
        protocol.json           — structured protocol
        <job_id>.docx           — final Word doc
        run_metadata.json       — timings, VRAM peak, config digest
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# noqa: E402 — must set sys.path before importing backend
from backend.app.config import AppConfig  # noqa: E402
from backend.app.models import MeetingJob, MeetingType, PipelineState  # noqa: E402
from backend.core.orchestrator import Orchestrator  # noqa: E402

# Default audio location on the VERITAS machine.
DEFAULT_AUDIO = REPO_ROOT / "data" / "test" / "Админ 13-04-2026 без мусора в начале.mp3"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("admin_e2e")


def load_config() -> AppConfig:
    """Load settings.yaml the same way the live API does."""
    for candidate in (REPO_ROOT / "config" / "settings.yaml", REPO_ROOT / "settings.yaml"):
        if candidate.exists():
            logger.info("Loading config: %s", candidate)
            return AppConfig.load_from_yaml(candidate)
    logger.warning("No settings.yaml found — using hardcoded defaults")
    return AppConfig()


def preflight_ollama(config: AppConfig) -> None:
    """Fail early if Ollama isn't reachable. The summarization stage
    will fail 70% into the run otherwise, which is a waste of 15 minutes
    of ASR + diarization compute."""
    if not config.summarization.enabled:
        logger.info("Summarization disabled in config — skipping Ollama preflight")
        return
    try:
        import httpx
    except ImportError:
        logger.error("httpx is required for Ollama preflight")
        sys.exit(2)

    url = config.summarization.ollama_base_url.rstrip("/") + "/api/tags"
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url)
            r.raise_for_status()
            tags = r.json()
    except Exception as exc:  # noqa: BLE001 — preflight is best-effort
        logger.error("Ollama not reachable at %s: %s", url, exc)
        logger.error("Start Ollama (`ollama serve`) before running this script.")
        sys.exit(3)

    models = {m.get("name", "") for m in tags.get("models", [])}
    wanted = config.summarization.ollama_model
    if wanted not in models:
        logger.error(
            "Model %r not installed in Ollama. Installed: %s",
            wanted,
            sorted(models),
        )
        logger.error("Run: ollama pull %s", wanted)
        sys.exit(4)
    logger.info("Ollama OK — %s available", wanted)


def resolve_audio(candidate: Path) -> Path:
    """Normalise the audio path. Accept a bare filename and fall back to
    the project's data/test/ directory."""
    if candidate.is_file():
        return candidate
    fallback = REPO_ROOT / "data" / "test" / candidate.name
    if fallback.is_file():
        logger.info("Resolved audio via data/test/: %s", fallback)
        return fallback
    logger.error("Audio file not found: %s (also tried %s)", candidate, fallback)
    sys.exit(5)


def make_progress_printer():
    """Progress callback that prints stage changes + pct deltas on a
    single line to stderr so the user sees liveness."""
    state: dict[str, object] = {"last_state": None, "last_pct": -1.0}

    def _cb(job: MeetingJob) -> None:
        cur = job.state.value if isinstance(job.state, PipelineState) else str(job.state)
        pct = float(job.progress)
        if cur != state["last_state"]:
            print()  # newline on stage change
            state["last_state"] = cur
            state["last_pct"] = -1.0
        # Print on every 1% delta or stage change
        if pct - float(state["last_pct"]) >= 1.0 or cur != state["last_state"]:
            state["last_pct"] = pct
            msg = job.current_stage or ""
            sys.stderr.write(f"\r[{cur:>16}] {pct:5.1f}%  {msg:<60}")
            sys.stderr.flush()

    return _cb


async def run_pipeline(
    config: AppConfig,
    audio: Path,
    meeting_type: MeetingType,
    context: str,
) -> tuple[MeetingJob, Path]:
    orchestrator = Orchestrator(config)
    job = MeetingJob(
        filename=audio.name,
        context=context,
        meeting_type=meeting_type,
    )
    meeting_dir = orchestrator.get_meeting_dir(job.id)
    logger.info("Job id:        %s", job.id)
    logger.info("Meeting type:  %s", meeting_type.value)
    logger.info("Meeting dir:   %s", meeting_dir)
    logger.info("Audio:         %s", audio)

    on_progress = make_progress_printer()
    protocol = await orchestrator.process_meeting(
        audio_path=str(audio),
        job=job,
        on_progress=on_progress,
    )
    sys.stderr.write("\n")

    if protocol is None:
        logger.error("Pipeline FAILED. Job state=%s error=%r", job.state, job.error)
        sys.exit(6)
    logger.info("Pipeline OK. Final state=%s progress=%.1f%%", job.state, job.progress)
    return job, meeting_dir


def archive_fresh(
    meeting_dir: Path,
    job: MeetingJob,
    started_at: float,
    prefix: str = "admin_13_04_fresh",
) -> Path:
    """Copy the intermediate JSONs and final DOCX into a benchmarks
    folder the comparison builder can find. Folder name is
    `{prefix}_{stamp}` so multiple comparison runs don't collide.
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    target = REPO_ROOT / "benchmarks" / "data" / f"{prefix}_{stamp}"
    target.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for name in (
        "transcription.json",
        "diarization.json",
        "aligned.json",
        "aligned_raw.json",
        "protocol.json",
    ):
        src = meeting_dir / name
        if src.exists():
            shutil.copy2(src, target / name)
            copied.append(name)

    # DOCX: <job_id>.docx, sometimes multiple
    for docx in meeting_dir.glob("*.docx"):
        shutil.copy2(docx, target / docx.name)
        copied.append(docx.name)

    # Also stash a copy into a stable "_latest" symlink/dir so the batch
    # script doesn't need to know the timestamp. We write a marker file
    # instead of a symlink because Windows symlinks need Developer Mode.
    latest_marker = REPO_ROOT / "benchmarks" / "data" / "admin_13_04_fresh_LATEST.txt"
    latest_marker.write_text(target.name + "\n", encoding="utf-8")

    meta = {
        "job_id": job.id,
        "filename": job.filename,
        "meeting_type": job.meeting_type.value if hasattr(job.meeting_type, "value") else str(job.meeting_type),
        "state": job.state.value if hasattr(job.state, "value") else str(job.state),
        "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "wall_time_s": round(time.time() - started_at, 2),
        "copied_files": copied,
        "meeting_dir": str(meeting_dir),
    }
    (target / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("Archived %d files → %s", len(copied), target)
    logger.info("Latest marker: %s", latest_marker)
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO,
                    help=f"Audio path (default: {DEFAULT_AUDIO.name})")
    ap.add_argument("--meeting-type", default="administrative",
                    choices=[t.value for t in MeetingType],
                    help="Meeting type for Block 7 dispatch (default: administrative)")
    ap.add_argument("--context", default="",
                    help="Optional meeting context string passed to the LLM")
    ap.add_argument("--skip-ollama-check", action="store_true",
                    help="Skip the Ollama preflight (run anyway if summarization disabled)")
    ap.add_argument(
        "--engine",
        default=None,
        choices=["auto", "gigaam", "hf-whisper", "whisper", "whisperx", "qwen", "nemo"],
        help=(
            "Force a specific ASR engine for this run, overriding the "
            "language-detection auto-routing. Used by the ASR engine "
            "comparison harness (compare_asr_engines.py)."
        ),
    )
    ap.add_argument(
        "--archive-prefix",
        default="admin_13_04_fresh",
        help=(
            "Prefix for the archive folder name under benchmarks/data/. "
            "Default 'admin_13_04_fresh' matches the existing comparison "
            "HTML builder. Use a different prefix when you want isolated "
            "runs (e.g. 'compare_gigaam', 'compare_whisperx')."
        ),
    )
    ap.add_argument(
        "--skip-summarization",
        action="store_true",
        help=(
            "Disable the LLM summarization stage for this run (saves "
            "~3-5 min). Useful for ASR-only comparison harnesses where "
            "the protocol summary doesn't matter."
        ),
    )
    args = ap.parse_args()

    config = load_config()

    # Apply CLI overrides BEFORE Ollama preflight so we don't fail the
    # check when summarization is disabled for this run.
    if args.engine is not None:
        logger.info("Forcing ASR engine: %s (was: %s)", args.engine, config.asr.engine)
        config.asr.engine = args.engine
    if args.skip_summarization:
        logger.info("Disabling summarization for this run (--skip-summarization)")
        config.summarization.enabled = False

    if not args.skip_ollama_check:
        preflight_ollama(config)

    audio = resolve_audio(args.audio)
    meeting_type = MeetingType(args.meeting_type)

    started = time.time()
    job, meeting_dir = asyncio.run(
        run_pipeline(config, audio, meeting_type, args.context)
    )
    archive_fresh(meeting_dir, job, started, prefix=args.archive_prefix)

    elapsed = time.time() - started
    logger.info("Total wall time: %.1fs (%.1f min)", elapsed, elapsed / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
