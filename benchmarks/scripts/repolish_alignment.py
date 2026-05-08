#!/usr/bin/env python3
"""Regenerate aligned_v2_polished.json from aligned_v2.json.

Reads the aligner output (aligned_v2.json) from an archive directory,
runs it through postprocess_transcript() with the current production
postprocessor config (no DMP, fillers on, repetitions on, cap on), and
writes aligned_v2_polished.json next to it.

Use this to re-validate seam-polish / dash-strip changes without
running the full pipeline end-to-end.

Usage:
    python benchmarks/scripts/repolish_alignment.py \
        "data/archive/2026-04-20_.../"

No network, no models (punctuation restoration stays OFF), no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.models import AlignedSegment  # noqa: E402
from backend.core.postprocessor import postprocess_transcript  # noqa: E402


def load_aligned_v2(path: Path) -> list[AlignedSegment]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    out: list[AlignedSegment] = []
    for s in segs:
        out.append(
            AlignedSegment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=s.get("text", "") or "",
                speaker_id=s.get("speaker_id") or s.get("speaker", "SPEAKER_00"),
                speaker_name=s.get("speaker_name"),
                confidence=s.get("confidence"),
                attribution_confidence=s.get("attribution_confidence"),
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive", type=Path,
                    help="Archive directory containing aligned_v2.json")
    ap.add_argument("--input-name", default="aligned_v2.json",
                    help="Source file name (default: aligned_v2.json)")
    ap.add_argument("--output-name", default="aligned_v2_polished.json",
                    help="Output file name (default: aligned_v2_polished.json)")
    args = ap.parse_args()

    arch = args.archive
    src = arch / args.input_name
    dst = arch / args.output_name
    if not src.exists():
        print(f"ERROR: {src} not found", file=sys.stderr)
        return 2

    print(f"Archive:    {arch}")
    print(f"Reading:    {src.name}")
    segs = load_aligned_v2(src)
    print(f"Loaded:     {len(segs)} aligned segments")

    polished, lang = postprocess_transcript(
        segs,
        restore_punctuation_enabled=False,
        remove_fillers=True,
        remove_repetitions=True,
        capitalize_sentences_enabled=True,
    )

    print(f"Polished:   {len(polished)} turns (language: {lang})")

    # Leading-dash audit: count turns that still start with em/en/hyphen + space
    stubborn = [
        p for p in polished
        if p.text and p.text[:2] in ("\u2014 ", "\u2013 ", "- ")
    ]
    print(f"Leading-dash turns remaining: {len(stubborn)}")
    for s in stubborn[:5]:
        print(f"  {s.start:7.2f}s  {s.speaker_id}  {s.text[:60]!r}")

    payload = {"segments": [p.model_dump() for p in polished]}
    with dst.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote:      {dst} ({dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
