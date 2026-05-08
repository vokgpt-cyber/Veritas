"""ASCII-only wrapper that hands the Cyrillic-named audio to the E2E harness.

cmd.exe garbles Cyrillic chars in .bat files on Russian Windows (CP866 default),
so we keep the .bat pure ASCII and resolve the filename + context here in Python,
which is UTF-8 aware.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Glob for the admin audio file — matches any file under data/test/ whose name
# starts with "Admin" (Latin) or contains Cyrillic Admin prefix.
TEST_DIR = REPO_ROOT / "data" / "test"
candidates = []
for p in TEST_DIR.iterdir():
    name = p.name
    # Cyrillic "Админ" starts with 0x0410 0x0434 0x043c 0x0438 0x043d
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

# Russian context for an admin meeting — defined here (UTF-8 source) not in .bat
context = (
    "\u0410\u0434\u043c\u0438\u043d\u0438\u0441\u0442\u0440\u0430\u0442\u0438\u0432\u043d\u043e\u0435 "
    "\u0441\u043e\u0432\u0435\u0449\u0430\u043d\u0438\u0435 EPAM, 13.04.2026. "
    "\u0412\u043d\u0443\u0442\u0440\u0435\u043d\u043d\u0435\u0435 \u043e\u0431\u0441\u0443\u0436\u0434\u0435\u043d\u0438\u0435 "
    "\u0442\u0435\u043a\u0443\u0449\u0438\u0445 \u0440\u0430\u0431\u043e\u0447\u0438\u0445 \u0432\u043e\u043f\u0440\u043e\u0441\u043e\u0432."
)

# Now run the harness by importing main() and patching sys.argv.
# This avoids spawning a new python process (simpler, same interpreter).
# We also need to capture the job_id it creates so the deliverables step can
# find the output dir afterwards. The harness doesn't return the job_id, so
# we snapshot data/meetings/ before + after and take the diff.
MEETINGS_DIR = REPO_ROOT / "data" / "meetings"
MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
existing_jobs = {p.name for p in MEETINGS_DIR.iterdir() if p.is_dir()}

sys.argv = [
    "run_e2e_test.py",
    str(audio_path),
    "--context", context,
]

import runpy
try:
    runpy.run_path(str(REPO_ROOT / "scripts" / "run_e2e_test.py"), run_name="__main__")
except SystemExit as _exit:
    # The harness calls sys.exit(...) at the end; swallow so we can also
    # build deliverables even on partial failure (meeting dir may still exist).
    harness_rc = _exit.code if isinstance(_exit.code, int) else 0
else:
    harness_rc = 0

# Identify the job that was just created
new_jobs = [p for p in MEETINGS_DIR.iterdir() if p.is_dir() and p.name not in existing_jobs]
if new_jobs:
    job_dir = max(new_jobs, key=lambda p: p.stat().st_mtime)
    job_id = job_dir.name
    print()
    print("=" * 60)
    print(f"[helper] Building deliverables for job {job_id}")
    print("=" * 60)
    try:
        from scripts.build_deliverables import build as build_deliverables  # type: ignore
    except ImportError:
        # scripts/ isn't a package; fall back to runpy
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from build_deliverables import build as build_deliverables  # type: ignore

    base_override = "Admin_13-04-2026"
    result = build_deliverables(job_id, base_override=base_override)
    print(f"[helper] DELIVERABLES WRITTEN TO: {REPO_ROOT / 'deliverables'}")
    for k in ("md", "docx", "json"):
        v = result.get(k)
        if v:
            print(f"  {k.upper():4s}: {v}")
else:
    print("[helper] WARNING: no new meeting directory detected; skipping deliverables.")

sys.exit(harness_rc)
