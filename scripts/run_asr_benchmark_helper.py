"""UTF-8 helper for run_asr_benchmark.bat.

cmd.exe on Russian Windows (CP866) garbles Cyrillic in .bat files, so the .bat
stays pure ASCII and this helper resolves the audio filename in Python.

Resolves the court-hearing audio file and runs scripts/run_asr_benchmark.py with:
  --base    court_hearing_129
  --engines gigaam,whisper_base,antony66,whisper_turbo_ru,qwen

Override via env vars:
  BENCH_AUDIO   absolute path to a specific file (skips auto-discovery)
  BENCH_BASE    short ASCII tag (default: court_hearing_129)
  BENCH_ENGINES comma-separated engine keys
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TEST_DIR = REPO_ROOT / "data" / "test"

# Preferred audio for the ASR benchmark. User confirmed 2026-04-17:
#   "Court hearing 129_01.mp4" and "Новая запись 129_01.mp4" are identical files.
DEFAULT_AUDIO_NAME = "Court hearing 129_01.mp4"
DEFAULT_BASE = "court_hearing_129"
DEFAULT_ENGINES = "gigaam,whisper_base,antony66,whisper_turbo_ru,qwen"


def resolve_audio() -> Path:
    override = os.environ.get("BENCH_AUDIO", "").strip()
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            print(f"ERROR: BENCH_AUDIO does not exist: {p}", file=sys.stderr)
            sys.exit(2)
        return p

    primary = TEST_DIR / DEFAULT_AUDIO_NAME
    if primary.exists():
        return primary

    # Fallback: search for any .mp4/.mp3/.wav that starts with "Court hearing"
    candidates = sorted(
        p for p in TEST_DIR.iterdir()
        if p.is_file()
        and p.name.lower().startswith("court hearing")
        and p.suffix.lower() in {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    )
    if candidates:
        return candidates[0]

    print(f"ERROR: no court-hearing audio found in {TEST_DIR}", file=sys.stderr)
    for p in TEST_DIR.iterdir():
        print(f"  - {p.name}", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    audio = resolve_audio()
    base = os.environ.get("BENCH_BASE", DEFAULT_BASE).strip() or DEFAULT_BASE
    engines = os.environ.get("BENCH_ENGINES", DEFAULT_ENGINES).strip() or DEFAULT_ENGINES

    size_mb = audio.stat().st_size / (1024 * 1024)
    print(f"[helper] Audio:   {audio}")
    print(f"[helper] Size:    {size_mb:.1f} MB")
    print(f"[helper] Base:    {base}")
    print(f"[helper] Engines: {engines}")

    sys.argv = [
        "run_asr_benchmark.py",
        str(audio),
        "--base", base,
        "--engines", engines,
    ]
    import runpy
    try:
        runpy.run_path(
            str(REPO_ROOT / "scripts" / "run_asr_benchmark.py"),
            run_name="__main__",
        )
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0


if __name__ == "__main__":
    sys.exit(main())
