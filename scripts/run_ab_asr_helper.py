"""ASCII-only wrapper that hands the Cyrillic-named audio to the A/B harness.

cmd.exe garbles Cyrillic chars in .bat files on Russian Windows (CP866 default),
so we keep the .bat pure ASCII and resolve the filename here in Python (UTF-8).

Resolves the Admin audio file and runs scripts/run_ab_asr.py with:
  --engines gigaam,hf-whisper
  --base   Admin_13-04-2026
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TEST_DIR = REPO_ROOT / "data" / "test"
candidates = []
for p in TEST_DIR.iterdir():
    name = p.name
    # Cyrillic "\u0410\u0434\u043c\u0438\u043d" (Admin) prefix
    if name.startswith("\u0410\u0434\u043c\u0438\u043d") and name.lower().endswith(
        (".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac")
    ):
        candidates.append(p)

if not candidates:
    print("ERROR: no Admin audio file found in data/test/", file=sys.stderr)
    for p in TEST_DIR.iterdir():
        print(f"  - {p.name}", file=sys.stderr)
    sys.exit(2)

audio_path = candidates[0]
print(f"[helper] Resolved audio: {audio_path}")
print(f"[helper] Size:           {audio_path.stat().st_size / (1024*1024):.1f} MB")

# Run the A/B harness via runpy (same interpreter, keeps imports hot).
sys.argv = [
    "run_ab_asr.py",
    str(audio_path),
    "--engines", "gigaam,hf-whisper",
    "--base", "Admin_13-04-2026",
]

import runpy
try:
    runpy.run_path(str(REPO_ROOT / "scripts" / "run_ab_asr.py"), run_name="__main__")
except SystemExit as _exit:
    harness_rc = _exit.code if isinstance(_exit.code, int) else 0
else:
    harness_rc = 0

sys.exit(harness_rc)
