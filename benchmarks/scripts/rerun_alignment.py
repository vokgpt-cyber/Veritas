#!/usr/bin/env python3
"""Re-run alignment on an archived VERITAS run with the updated aligner.

Loads transcription.json + diarization.json from a data/archive/... folder,
runs the new sentence-split + sum-aggregate aligner, and diffs the result
against the original aligned.json that was produced by the old aligner.

Reports:
  - total segment count before/after sentence splitting
  - how many segments changed speaker attribution
  - attribution_confidence distribution
  - the specific "Саш, не говори..." Админ 13-04 failing case, if present

Usage:
    python benchmarks/scripts/rerun_alignment.py \
        "data/archive/2026-04-20_Админ_13-04-2026_без_мусора_в_начале"

No network, no models, no GPU. Pure reprocessing of saved JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make "backend" importable when run from repo root
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.app.models import (  # noqa: E402
    AlignedSegment,
    DiarizationSegment,
    TranscriptionSegment,
    WordInfo,
)
from backend.core.aligner import TranscriptAligner  # noqa: E402


def load_transcription(path: Path) -> list[TranscriptionSegment]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    out = []
    for s in segs:
        words = []
        for w in s.get("words") or []:
            words.append(
                WordInfo(
                    start=float(w["start"]),
                    end=float(w["end"]),
                    word=w.get("word") or w.get("text") or "",
                    confidence=float(w.get("confidence", 1.0)),
                )
            )
        out.append(
            TranscriptionSegment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=s.get("text", ""),
                confidence=float(s.get("confidence", 1.0)),
                words=words,
            )
        )
    return out


def load_diarization(path: Path) -> list[DiarizationSegment]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    segs = raw.get("segments", raw) if isinstance(raw, dict) else raw
    return [
        DiarizationSegment(
            start=float(s["start"]),
            end=float(s["end"]),
            speaker_id=s.get("speaker_id") or s.get("speaker") or "SPEAKER_?",
            speaker_name=s.get("speaker_name"),
        )
        for s in segs
    ]


def load_aligned_old(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("segments", raw) if isinstance(raw, dict) else raw


def find_target_phrase(segments, substrings: list[str]) -> list[tuple[int, object]]:
    """Return (idx, seg) for every segment whose text contains any substring."""
    hits = []
    for i, s in enumerate(segments):
        text = s.text if isinstance(s, AlignedSegment) else s.get("text", "")
        for sub in substrings:
            if sub in text:
                hits.append((i, s))
                break
    return hits


def summarize(new_aligned: list[AlignedSegment], old_aligned: list[dict]) -> None:
    print(f"Old aligner output: {len(old_aligned)} segments")
    print(f"New aligner output: {len(new_aligned)} segments "
          f"(delta {len(new_aligned) - len(old_aligned):+d})")

    # Confidence distribution
    confs = [
        a.attribution_confidence for a in new_aligned
        if a.attribution_confidence is not None
    ]
    if confs:
        buckets = {"[1.00]": 0, "[0.80, 1.00)": 0, "[0.60, 0.80)": 0, "[0.00, 0.60)": 0}
        for c in confs:
            if c >= 1.0:
                buckets["[1.00]"] += 1
            elif c >= 0.80:
                buckets["[0.80, 1.00)"] += 1
            elif c >= 0.60:
                buckets["[0.60, 0.80)"] += 1
            else:
                buckets["[0.00, 0.60)"] += 1
        avg = sum(confs) / len(confs)
        print(f"\nAttribution confidence (n={len(confs)}, avg={avg:.3f}):")
        for k, v in buckets.items():
            print(f"  {k:<15s} {v:4d}  ({100 * v / len(confs):5.1f}%)")

        low = [a for a in new_aligned
               if a.attribution_confidence is not None
               and a.attribution_confidence < 0.6]
        if low:
            print(f"\nLow-confidence (<0.60) segments: {len(low)}")
            for a in low[:15]:
                preview = a.text[:90].replace("\n", " ")
                print(f"  [{a.start:7.2f}-{a.end:7.2f}] conf={a.attribution_confidence:.2f} "
                      f"{a.speaker_id} :: {preview}")
            if len(low) > 15:
                print(f"  ... +{len(low) - 15} more")

    # Speaker share comparison
    def share(segs_like, getter):
        tot = defaultdict(float)
        grand = 0.0
        for s in segs_like:
            dur = getter(s, "end") - getter(s, "start")
            tot[getter(s, "speaker_id") or "?"] += dur
            grand += dur
        if grand <= 0:
            return {}
        return {k: round(100 * v / grand, 1) for k, v in sorted(tot.items())}

    def g_new(s, k): return getattr(s, k)
    def g_old(s, k): return s.get(k)

    old_share = share(old_aligned, g_old)
    new_share = share(new_aligned, g_new)
    all_keys = sorted(set(old_share) | set(new_share))
    print("\nSpeaker share (% of total aligned time):")
    print(f"  {'speaker':<14s} {'old':>8s} {'new':>8s} {'delta':>8s}")
    for k in all_keys:
        o = old_share.get(k, 0.0)
        n = new_share.get(k, 0.0)
        print(f"  {k:<14s} {o:7.1f}% {n:7.1f}% {n - o:+7.1f}%")


def check_target_case(new_aligned: list[AlignedSegment],
                      old_aligned: list[dict]) -> None:
    """Админ 13-04 regression check: 'Саш, не говори, пожалуйста...'."""
    phrases = ["Саш, не говори", "не говори, пожалуйста"]

    print("\n" + "=" * 70)
    print("TARGET CASE: Админ 13-04 interjection 'Саш, не говори, пожалуйста...'")
    print("=" * 70)

    old_hits = find_target_phrase(old_aligned, phrases)
    new_hits = find_target_phrase(new_aligned, phrases)

    print("\nOld aligner:")
    for i, s in old_hits[:5]:
        text = s.get("text", "")
        print(f"  [{i}] {s.get('start', 0):7.2f}-{s.get('end', 0):7.2f} "
              f"dur={s.get('end', 0) - s.get('start', 0):5.2f}s  "
              f"spk={s.get('speaker_id') or s.get('speaker') or '?'}")
        print(f"       {text[:180]}")

    print("\nNew aligner:")
    for i, s in new_hits[:5]:
        conf = f"{s.attribution_confidence:.2f}" if s.attribution_confidence is not None else "--"
        print(f"  [{i}] {s.start:7.2f}-{s.end:7.2f} dur={s.end - s.start:5.2f}s  "
              f"spk={s.speaker_id}  conf={conf}")
        print(f"       {s.text[:180]}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path,
                   help="Path to data/archive/<run> folder")
    p.add_argument("--no-sentence-split", action="store_true",
                   help="Disable the new sentence-boundary splitter (to test aligner-rule change alone)")
    args = p.parse_args()

    arch = args.archive
    if not arch.exists():
        print(f"ERROR: archive not found: {arch}", file=sys.stderr)
        return 2

    trans_path = arch / "transcription.json"
    dia_path = arch / "diarization.json"
    aligned_old_path = arch / "aligned_raw.json"
    if not aligned_old_path.exists():
        aligned_old_path = arch / "aligned.json"
    for req in (trans_path, dia_path, aligned_old_path):
        if not req.exists():
            print(f"ERROR: missing {req}", file=sys.stderr)
            return 2

    print(f"Archive: {arch}")
    print(f"  transcription: {trans_path.name}")
    print(f"  diarization:   {dia_path.name}")
    print(f"  baseline:      {aligned_old_path.name}")

    transcription = load_transcription(trans_path)
    diarization = load_diarization(dia_path)
    aligned_old = load_aligned_old(aligned_old_path)

    print(f"\nLoaded {len(transcription)} transcription segments "
          f"({sum(len(s.words) for s in transcription)} words)")
    print(f"Loaded {len(diarization)} diarization segments")
    print(f"Loaded {len(aligned_old)} baseline aligned segments")

    new_aligned = TranscriptAligner.align(
        transcription, diarization,
        sentence_split=not args.no_sentence_split,
    )

    summarize(new_aligned, aligned_old)
    check_target_case(new_aligned, aligned_old)

    # Save new alignment for inspection
    out_path = arch / "aligned_v2.json"
    payload = {"segments": [a.model_dump() for a in new_aligned]}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
